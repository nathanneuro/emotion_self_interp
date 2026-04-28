"""Small reusable building blocks shared across Phase 1 / 2 / 5 scripts.

The intent is the *protocol* layer: given a model, a layer, and a stimulus
set, produce the per-prompt residual cache and group it by (emotion, level).
Everything downstream (probes, adapters, experiments) reads from these.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.adapters.scalar_affine import (
    AdapterConfig,
    ScalarAffineAdapter,
    _AdapterBase,
    make_adapter,
)
from src.adapters.train import TrainConfig, TrainExample, train_adapter
from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set
from src.data.stimuli import Stimulus
from src.hooks.extract import ActivationRequest, extract_batch
from src.models.adapter import ModelAdapter
from src.probes.diff_means import diff_of_means


@dataclass
class StimulusResiduals:
    """Per-prompt residuals at one layer, with row-index lookups."""
    stimuli: list[Stimulus]
    residuals: torch.Tensor  # (N, d) cpu fp32
    layer: int

    @property
    def d_model(self) -> int:
        return int(self.residuals.shape[1])

    def rows_by_key(self) -> dict[tuple[str, str], list[int]]:
        out: dict[tuple[str, str], list[int]] = {}
        for i, s in enumerate(self.stimuli):
            out.setdefault((s.emotion, s.level), []).append(i)
        return out


def extract_stimulus_residuals(
    model: ModelAdapter,
    layer: int,
    stims: list[Stimulus] | None = None,
    per_cell: int = 30,
    batch_size: int = 16,
) -> StimulusResiduals:
    """Run all stimuli through the model and cache last-token residuals at `layer`."""
    stims = stims if stims is not None else build_stimulus_set(per_cell=per_cell)
    prompts = [s.prompt for s in stims]
    req = ActivationRequest(layer_idxs=[layer], position=-1)
    H = extract_batch(model, prompts, req, batch_size=batch_size)[layer]
    return StimulusResiduals(stimuli=stims, residuals=H, layer=layer)


def build_emotion_vectors(
    res: StimulusResiduals,
    contrast_level: str = "euphoric",
) -> dict[str, np.ndarray]:
    """Build a per-emotion direction via diff-of-means against neutral.

    `contrast_level`: which level of the emotion's stimuli to use as the
    in-class set. Default "euphoric" matches Phase 1 v0.
    """
    rows = res.rows_by_key()
    H = res.residuals.numpy()
    neutral_rows = rows.get(("neutral", "neutral"), [])
    if not neutral_rows:
        raise ValueError("no neutral stimuli found in residual cache")
    out: dict[str, np.ndarray] = {}
    for emo in EMOTIONS:
        in_class = rows.get((emo, contrast_level), [])
        if not in_class:
            raise ValueError(f"no {emo}/{contrast_level} stimuli found")
        out[emo] = diff_of_means(H[in_class], H[neutral_rows]).astype(np.float32)
    return out


def train_pepper_on_residuals(
    model: ModelAdapter,
    res: StimulusResiduals,
    kind: str = "full_rank",
    epochs: int = 20,
    lr: float = 5e-3,
    batch_size: int = 8,
    train_level: str = "euphoric",
) -> tuple[_AdapterBase, dict]:
    """Train a Pepper-style adapter on (residual, emotion-label) pairs from
    the chosen level. Returns (trained_adapter, history)."""
    train_examples: list[TrainExample] = []
    for i, s in enumerate(res.stimuli):
        if s.emotion == "neutral" or s.level != train_level:
            continue
        train_examples.append(TrainExample(vector=res.residuals[i].clone(), label=s.emotion))
    if not train_examples:
        raise ValueError(f"no training examples for level={train_level!r}")

    cfg = TrainConfig(
        layer_idx=res.layer, batch_size=batch_size, n_epochs=epochs, learning_rate=lr,
    )
    adapter = make_adapter(AdapterConfig(kind=kind, d_model=res.d_model)).to(model.device)
    history = train_adapter(model, adapter, train_examples, val=None, cfg=cfg)
    return adapter, history


def make_untrained_selfie_adapter(d_model: int) -> ScalarAffineAdapter:
    """Pepper's untrained-SelfIE baseline: α=1, b=0 — the residual passes
    through the residual-replace hook unchanged."""
    return ScalarAffineAdapter(d_model=d_model)
