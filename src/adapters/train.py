"""Training loop for Pepper-style self-interpretation adapters.

The setup:
- Frozen base model + tokenizer.
- Adapter parameters (`α`, `b`, optionally `W`) are the only trainable params.
- For each training example (vector h, label "calm"):
    1. Tokenize a probe prompt that ends in a position where the model is
       expected to predict the label token.
    2. Run forward; at the chosen layer L, hook the residual to *replace*
       the residual at the probe position with `adapter(h)`.
    3. Compute cross-entropy on the label token id at the final position.
    4. Backprop through the adapter only.

The hook replaces the residual at one token position rather than adding to
all positions — this matches Pepper's "self-interpretation" framing where
the activation under question is being injected into a context that asks the
model to describe it.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.adapters.scalar_affine import _AdapterBase
from src.models.adapter import ModelAdapter


# Probe prompt: places the activation under a "what is this concept" frame.
# `{ACT}` is a single placeholder token whose residual gets replaced by the
# adapter output at the target layer.
DEFAULT_PROBE = (
    "The following hidden state encodes a concept.\n"
    "Concept: {ACT}\n"
    "In one word, this concept is:"
)
ACT_TOKEN = " <ACT>"  # leading space helps tokenize as a single piece


@dataclass
class TrainExample:
    vector: torch.Tensor  # (d_model,) cpu float32
    label: str            # raw label string ("calm", "desperate", ...)


@dataclass
class TrainConfig:
    layer_idx: int        # which residual-stream layer to inject at
    label_position: int = -1  # position to read logits from (default: last)
    learning_rate: float = 1e-2
    n_epochs: int = 10
    batch_size: int = 16
    weight_decay: float = 0.0
    seed: int = 0


class _VecLabelDataset(Dataset):
    def __init__(self, examples: list[TrainExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> TrainExample:
        return self.examples[i]


def _collate(batch: list[TrainExample]) -> tuple[torch.Tensor, list[str]]:
    vecs = torch.stack([b.vector for b in batch], dim=0)
    labels = [b.label for b in batch]
    return vecs, labels


def _label_first_token_id(tokenizer, label: str) -> int:
    """First token id of " <label>" — leading space handles BPE word-boundary
    behavior on most tokenizers (Llama, Qwen, GPT-2 family, Mistral)."""
    ids = tokenizer.encode(" " + label, add_special_tokens=False)
    if not ids:
        raise ValueError(f"label {label!r} produced no tokens")
    return int(ids[0])


def _find_act_position(input_ids: torch.Tensor, act_token_id: int) -> int:
    """Find the (single) position in input_ids where `act_token_id` appears."""
    matches = (input_ids == act_token_id).nonzero(as_tuple=False)
    if matches.numel() == 0:
        raise RuntimeError(
            f"<ACT> sentinel (id={act_token_id}) not found in tokenized prompt"
        )
    return int(matches[0, -1])


@contextlib.contextmanager
def _residual_replace_hook(
    block: nn.Module,
    new_residual: torch.Tensor,  # (B, d_model) — replacement vectors per batch row
    position: int,                # token position to overwrite
) -> Iterator[None]:
    """Replace the residual at `position` with `new_residual` per batch row."""
    def hook(_mod, _inp, out):
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        h = h.clone()
        # h: (B, S, d). Coerce dtype/device of the replacement to match h.
        repl = new_residual.to(device=h.device, dtype=h.dtype)
        h[:, position, :] = repl
        return (h, *out[1:]) if tup else h

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _build_probe_inputs(
    tokenizer, model_device: torch.device, batch_size: int,
) -> tuple[torch.Tensor, int]:
    """Tokenize the probe prompt once. Returns (input_ids batched, act_pos)."""
    # Add the ACT_TOKEN as a special token if it isn't already a single piece.
    if " <ACT>" not in tokenizer.get_vocab() and "<ACT>" not in tokenizer.get_vocab():
        # Add as a special additional token so the tokenizer reserves an id.
        tokenizer.add_special_tokens({"additional_special_tokens": ["<ACT>"]})
    text = DEFAULT_PROBE.replace("{ACT}", "<ACT>")
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model_device)  # (1, S)
    ids = ids.expand(batch_size, -1).contiguous()
    act_token_id = tokenizer.convert_tokens_to_ids("<ACT>")
    act_pos = _find_act_position(ids[0], act_token_id)
    return ids, act_pos


def train_adapter(
    model: ModelAdapter,
    adapter: _AdapterBase,
    train: list[TrainExample],
    val: list[TrainExample] | None,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> dict:
    """Train `adapter` on (vector, label) pairs against a frozen `model`.

    Returns a dict with per-epoch train/val loss and top-1 accuracy.
    """
    torch.manual_seed(cfg.seed)
    if device is None:
        device = model.device

    # Freeze the base model.
    for p in model.model.parameters():
        p.requires_grad = False
    model.model.eval()
    adapter.to(device).train()

    # Resize embeddings if we just added <ACT>.
    if model.model.get_input_embeddings().num_embeddings != len(model.tokenizer):
        model.model.resize_token_embeddings(len(model.tokenizer))

    # One static probe-prompt batch we replicate per minibatch.
    probe_ids, act_pos = _build_probe_inputs(model.tokenizer, device, cfg.batch_size)

    block = model.get_block(cfg.layer_idx)

    opt = torch.optim.Adam(
        [p for p in adapter.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    train_loader = DataLoader(
        _VecLabelDataset(train), batch_size=cfg.batch_size,
        shuffle=True, collate_fn=_collate, drop_last=True,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(cfg.n_epochs):
        adapter.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        for vecs, labels in train_loader:
            vecs = vecs.to(device)
            label_ids = torch.tensor(
                [_label_first_token_id(model.tokenizer, lab) for lab in labels],
                device=device, dtype=torch.long,
            )

            injected = adapter(vecs)  # (B, d_model)
            B = injected.shape[0]
            ids = probe_ids[:B]

            with _residual_replace_hook(block, injected, act_pos):
                logits = model.model(input_ids=ids).logits  # (B, S, V)
            pred_logits = logits[:, cfg.label_position, :]
            loss = F.cross_entropy(pred_logits, label_ids)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = pred_logits.argmax(dim=-1)
                ep_correct += int((preds == label_ids).sum())
                ep_total += int(B)
                ep_loss += float(loss) * B

        history["train_loss"].append(ep_loss / max(ep_total, 1))
        history["train_acc"].append(ep_correct / max(ep_total, 1))

        if val:
            adapter.eval()
            val_results = evaluate_adapter(model, adapter, val, cfg, device)
            history["val_loss"].append(val_results["loss"])
            history["val_acc"].append(val_results["top1"])

    return history


@torch.no_grad()
def evaluate_adapter(
    model: ModelAdapter,
    adapter: _AdapterBase,
    examples: list[TrainExample],
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> dict:
    """Top-1 token-prediction accuracy + mean cross-entropy on `examples`."""
    if device is None:
        device = model.device
    adapter.to(device).eval()
    block = model.get_block(cfg.layer_idx)
    probe_ids, act_pos = _build_probe_inputs(model.tokenizer, device, cfg.batch_size)

    total = 0
    correct = 0
    loss_sum = 0.0
    for i in range(0, len(examples), cfg.batch_size):
        batch = examples[i : i + cfg.batch_size]
        if not batch:
            continue
        B = len(batch)
        vecs = torch.stack([b.vector for b in batch], dim=0).to(device)
        label_ids = torch.tensor(
            [_label_first_token_id(model.tokenizer, b.label) for b in batch],
            device=device, dtype=torch.long,
        )

        injected = adapter(vecs)
        ids = probe_ids[:B]
        with _residual_replace_hook(block, injected, act_pos):
            logits = model.model(input_ids=ids).logits
        pred_logits = logits[:, cfg.label_position, :]
        loss = F.cross_entropy(pred_logits, label_ids, reduction="sum")

        preds = pred_logits.argmax(dim=-1)
        correct += int((preds == label_ids).sum())
        total += B
        loss_sum += float(loss)

    return {
        "loss": loss_sum / max(total, 1),
        "top1": correct / max(total, 1),
        "n": total,
    }
