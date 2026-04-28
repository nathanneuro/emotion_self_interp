"""Model-agnostic abstraction over HF causal LMs for residual-stream access.

Modern decoder-only families (Llama, Qwen, Gemma, Mistral, OLMo) expose decoder
blocks at `model.model.layers[i]` and a block's hidden-state output is the
residual stream after that block's contribution. This module hides the family
detection so callers can speak in `(layer_idx, site)` terms.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

_SUPPORTED_FAMILIES = {
    "llama",
    "qwen2",
    "qwen3",
    "gemma",
    "gemma2",
    "gemma3",
    "mistral",
    "olmo",
    "olmo2",
}


def _detect_family(config) -> str:
    mt = getattr(config, "model_type", "").lower()
    if mt in _SUPPORTED_FAMILIES:
        return mt
    raise ValueError(
        f"Unsupported model_type {mt!r}. "
        f"Add it to _SUPPORTED_FAMILIES after verifying `model.model.layers[i]` "
        f"exposes decoder blocks with residual-stream output."
    )


@dataclass
class ModelAdapter:
    """Wraps a HF causal LM with hooks for activation cache and residual steering."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    family: str
    name: str

    @classmethod
    def load(
        cls,
        name: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict | None = "auto",
        **kwargs,
    ) -> "ModelAdapter":
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Left-padding so position=-1 always indexes the last real token.
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map=device_map, **kwargs
        )
        model.eval()
        family = _detect_family(model.config)
        return cls(model=model, tokenizer=tok, family=family, name=name)

    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _layers_module(self) -> nn.ModuleList:
        # CausalLM wrappers expose the backbone at `.model`; the decoder blocks
        # live at `.model.layers` for every family in _SUPPORTED_FAMILIES.
        return self.model.model.layers  # type: ignore[attr-defined]

    def get_block(self, layer_idx: int) -> nn.Module:
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")
        return self._layers_module()[layer_idx]

    @contextlib.contextmanager
    def cache_residual(self, layer_idxs: list[int]) -> Iterator[dict[int, torch.Tensor]]:
        """Capture residual stream after each named decoder block.

        Stored as float32 CPU tensors of shape (batch, seq, d_model) for the most
        recent forward pass. Detached from the autograd graph.
        """
        cache: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(idx: int):
            def hook(_mod: nn.Module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cache[idx] = h.detach().to("cpu", dtype=torch.float32)
            return hook

        for li in layer_idxs:
            handles.append(self.get_block(li).register_forward_hook(make_hook(li)))
        try:
            yield cache
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def steer_residual(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float,
        token_mask: torch.Tensor | None = None,
    ) -> Iterator[None]:
        """Add `alpha * vector` to the residual after `layer_idx` during forward.

        token_mask: optional bool tensor of shape (batch, seq) selecting which
        positions get the addition. None = all positions.
        """
        block = self.get_block(layer_idx)
        v = vector.detach()

        def hook(_mod: nn.Module, _inp, out):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            v_dev = v.to(device=h.device, dtype=h.dtype)
            if token_mask is None:
                h_new = h + alpha * v_dev
            else:
                m = token_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
                h_new = h + alpha * m * v_dev
            return (h_new, *out[1:]) if tup else h_new

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
