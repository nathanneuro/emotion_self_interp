"""Activation extraction at chosen (layer, token_position) sites."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.models.adapter import ModelAdapter


@dataclass
class ActivationRequest:
    """A request for residual-stream activations at specific layers and positions.

    position semantics:
        int: a fixed index into the sequence dimension (-1 = last token).
              With left-padded tokenization (the adapter default), -1 is always
              the last real token.
        "last_real": last non-pad token per row (works under any padding side).
    """
    layer_idxs: list[int]
    position: int | str = -1
    extra: dict = field(default_factory=dict)


def _select_position(
    h: torch.Tensor,  # (B, S, d) cpu float32
    attention_mask: torch.Tensor,  # (B, S) on device — passed in cpu form
    position: int | str,
) -> torch.Tensor:
    if isinstance(position, int):
        return h[:, position, :]
    if position == "last_real":
        # Largest position index where attention_mask == 1; works under either
        # padding side. (Counting `seq_len - 1` would only work for right-padding.)
        am = attention_mask.cpu().long()
        positions = torch.arange(am.shape[1]).unsqueeze(0).expand_as(am).clone()
        positions[am == 0] = -1
        last_real = positions.max(dim=1).values
        return torch.stack([h[i, int(last_real[i]), :] for i in range(h.shape[0])])
    raise ValueError(f"Unknown position spec {position!r}")


@torch.no_grad()
def extract(model: ModelAdapter, prompt: str, req: ActivationRequest) -> dict[int, torch.Tensor]:
    """Run a forward pass on `prompt` and return {layer: (d_model,) cpu float32}."""
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
    with model.cache_residual(req.layer_idxs) as cache:
        model.model(**inputs)
    return {
        li: _select_position(cache[li], inputs["attention_mask"], req.position).squeeze(0)
        for li in req.layer_idxs
    }


@torch.no_grad()
def extract_batch(
    model: ModelAdapter, prompts: list[str], req: ActivationRequest, batch_size: int = 8
) -> dict[int, torch.Tensor]:
    """Batched extraction. Returns {layer: (n_prompts, d_model) cpu float32}."""
    out: dict[int, list[torch.Tensor]] = {li: [] for li in req.layer_idxs}
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        inputs = model.tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with model.cache_residual(req.layer_idxs) as cache:
            model.model(**inputs)
        for li in req.layer_idxs:
            vecs = _select_position(cache[li], inputs["attention_mask"], req.position)
            out[li].append(vecs)
    return {li: torch.cat(parts, dim=0) for li, parts in out.items()}
