"""Model-agnostic abstraction over HF causal LMs for residual-stream access.

Modern decoder-only families (Llama, Qwen, Gemma, Mistral, OLMo) expose decoder
blocks at `model.model.layers[i]` and a block's hidden-state output is the
residual stream after that block's contribution. This module hides the family
detection so callers can speak in `(layer_idx, site)` terms.
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

_SUPPORTED_FAMILIES = {
    "llama",
    "qwen2",
    "qwen3",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",  # text-only Gemma 3 variants report this as model_type
    "mistral",
    "olmo",
    "olmo2",
    "ouro",   # universal-transformer-style: same layers looped total_ut_steps times
    "monet",  # sparse-expert transformer (Monet-VD); standard model.layers path
}

# Families where each layer's forward is called multiple times per forward pass
# (universal-transformer style). We need an appending hook rather than overwriting.
_LOOPING_FAMILIES = {"ouro"}


def _compat_compute_default_rope_parameters(config=None, device=None, seq_len=None):
    """Shim for `compute_default_rope_parameters` removed from transformers 5.x.

    Computes inverse frequencies for the original RoPE per the standard
    formula. Kept as a free function so it can be injected into the globals
    of remote-code modeling modules that reference the bare name.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def _patch_remote_modeling_modules(symbol_name: str, value) -> None:
    """Inject `value` into every loaded `transformers_modules.*` module that
    doesn't already define `symbol_name`. Used to retro-fit free-function
    references that newer transformers versions removed.
    """
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("transformers_modules.") and not hasattr(mod, symbol_name):
            setattr(mod, symbol_name, value)


def _patch_remote_rotary_classes(method_name: str, fn) -> None:
    """Add `method_name` (as a static method) to any `*RotaryEmbedding` class
    in loaded `transformers_modules.*`. Used because transformers 5.x's
    `_init_weights` looks up `compute_default_rope_parameters` directly on
    rotary-embedding modules — older custom-modeling repos (Ouro base) don't
    define it as a class method.
    """
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("transformers_modules."):
            continue
        for attr_name in dir(mod):
            if "RotaryEmbedding" not in attr_name:
                continue
            cls = getattr(mod, attr_name, None)
            if isinstance(cls, type) and not hasattr(cls, method_name):
                setattr(cls, method_name, staticmethod(fn))


def _ensure_llama_attention_classes_shim() -> None:
    """transformers 5.x removed the LLAMA_ATTENTION_CLASSES dict in favour of
    a unified `LlamaAttention` with `_attn_implementation` switching. Old
    custom-modeling repos (Monet) still import the dict. Add a back-compat
    alias before the remote-code module loads.
    """
    try:
        from transformers.models.llama import modeling_llama as _ll
    except Exception:
        return
    if not hasattr(_ll, "LLAMA_ATTENTION_CLASSES"):
        _ll.LLAMA_ATTENTION_CLASSES = {
            "eager": _ll.LlamaAttention,
            "sdpa": _ll.LlamaAttention,
            "flash_attention_2": _ll.LlamaAttention,
        }


def _ensure_rope_default_shim() -> None:
    """transformers 5.x removed the ``"default"`` key from ROPE_INIT_FUNCTIONS
    (it now contains only the *named* RoPE variants — linear/dynamic/yarn/
    longrope/llama3/proportional). Some custom-modeling repos (Ouro base)
    look up ROPE_INIT_FUNCTIONS["default"] directly. Inject the shim here.
    """
    try:
        from transformers import modeling_rope_utils as _ru
    except Exception:
        return
    table = getattr(_ru, "ROPE_INIT_FUNCTIONS", None)
    if isinstance(table, dict) and "default" not in table:
        table["default"] = _compat_compute_default_rope_parameters


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
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "ModelAdapter":
        if trust_remote_code:
            _ensure_llama_attention_classes_shim()
            _ensure_rope_default_shim()
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        except Exception:
            # Some old custom-modeling repos (Monet) ship a SentencePiece
            # tokenizer.model that newer transformers auto-detects as tiktoken
            # and fails to convert. Fall back to the slow tokenizer; if that
            # also fails, re-raise the original error.
            tok = AutoTokenizer.from_pretrained(
                name, trust_remote_code=trust_remote_code, use_fast=False,
            )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Left-padding so position=-1 always indexes the last real token.
        tok.padding_side = "left"

        # Some custom-modeling repos (e.g. Ouro) read attributes their config
        # doesn't always set. Patch missing pad_token_id from the tokenizer
        # before instantiating the model.
        config = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
        if getattr(config, "pad_token_id", None) is None:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else getattr(config, "eos_token_id", 0)
            config.pad_token_id = pad_id

        def _from_pretrained():
            return AutoModelForCausalLM.from_pretrained(
                name, config=config, dtype=dtype, device_map=device_map,
                trust_remote_code=trust_remote_code, **kwargs,
            )

        try:
            model = _from_pretrained()
        except (NameError, AttributeError) as e:
            # Some custom-modeling repos (Ouro 1.4B Thinking and base in
            # particular) hit transformers 5.x compat issues around
            # `compute_default_rope_parameters`:
            #   * Thinking version raises NameError because its modeling
            #     code references the bare free name (removed in 5.x).
            #   * Base version raises AttributeError during _init_weights
            #     because the function isn't defined as a class method on
            #     OuroRotaryEmbedding (5.x's _init_weights accesses it as
            #     module.compute_default_rope_parameters).
            # By the time the exception fires the modeling module is
            # already in sys.modules, so we can patch and retry.
            if "compute_default_rope_parameters" not in str(e):
                raise
            _patch_remote_modeling_modules(
                "compute_default_rope_parameters",
                _compat_compute_default_rope_parameters,
            )
            _patch_remote_rotary_classes(
                "compute_default_rope_parameters",
                _compat_compute_default_rope_parameters,
            )
            model = _from_pretrained()
        model.eval()
        family = _detect_family(model.config)
        return cls(model=model, tokenizer=tok, family=family, name=name)

    @property
    def is_looping(self) -> bool:
        """True if each decoder layer is called multiple times per forward pass."""
        return self.family in _LOOPING_FAMILIES

    @property
    def n_loop_steps(self) -> int:
        """For looping families, how many times each layer fires per forward.
        Returns 1 for standard families.
        """
        if self.family == "ouro":
            return int(getattr(self.model.config, "total_ut_steps", 1))
        return 1

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
    def cache_residual_looped(
        self, layer_idxs: list[int]
    ) -> Iterator[dict[int, list[torch.Tensor]]]:
        """Like `cache_residual` but appends per call instead of overwriting.

        For universal-transformer-style models (Ouro), each decoder layer's
        forward is called `n_loop_steps` times per forward pass. The returned
        cache is `{layer_idx: [tensor_at_call_0, tensor_at_call_1, ...]}`
        with one entry per call, in call order. For non-looping models,
        each list has length 1.
        """
        cache: dict[int, list[torch.Tensor]] = {li: [] for li in layer_idxs}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(idx: int):
            def hook(_mod: nn.Module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cache[idx].append(h.detach().to("cpu", dtype=torch.float32))
            return hook

        for li in layer_idxs:
            handles.append(self.get_block(li).register_forward_hook(make_hook(li)))
        try:
            yield cache
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def ablate_residual(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> Iterator[None]:
        """Project the residual at `layer_idx` onto the null space of `vector`.

        For unit u = vector / ‖vector‖, this replaces h with h − (h·u) u at
        every token position (or only the masked positions). Tests "remove
        this direction and see if behavior changes" — the counterpart to
        steer_residual, which adds a direction.
        """
        block = self.get_block(layer_idx)
        v = vector.detach()
        v = v / (v.norm() + 1e-12)

        def hook(_mod: nn.Module, _inp, out):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            v_dev = v.to(device=h.device, dtype=h.dtype)
            # Project out the v direction. h: (B, S, d), v_dev: (d,).
            coeff = (h @ v_dev).unsqueeze(-1)  # (B, S, 1)
            removal = coeff * v_dev            # (B, S, d)
            if token_mask is None:
                h_new = h - removal
            else:
                m = token_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
                h_new = h - m * removal
            return (h_new, *out[1:]) if tup else h_new

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextlib.contextmanager
    def steer_residual_at_ut_step(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float,
        target_ut_step: int,
        n_ut: int | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> Iterator[None]:
        """Steering for looping families: add `alpha · vector` only at the
        layer's call corresponding to `target_ut_step`, leaving the other
        n_ut−1 calls unmodified.

        Each forward of the model fires the layer's hook `n_ut` times (once
        per ut step). We track call count modulo `n_ut`, applying the steer
        only when the count equals `target_ut_step`. This lets us test how
        a single-step intervention propagates through subsequent loop
        iterations vs. how it acts when there's no further processing.
        """
        if n_ut is None:
            n_ut = self.n_loop_steps
        if not 0 <= target_ut_step < n_ut:
            raise ValueError(f"target_ut_step {target_ut_step} not in [0, {n_ut})")

        block = self.get_block(layer_idx)
        v = vector.detach()
        # Closure-mutable counter; resets implicitly via modulo each forward.
        state = {"call_count": 0}

        def hook(_mod: nn.Module, _inp, out):
            cur_ut = state["call_count"] % n_ut
            state["call_count"] += 1
            if cur_ut != target_ut_step:
                return out  # pass-through on non-target ut steps
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
