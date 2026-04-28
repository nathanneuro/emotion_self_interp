"""Pepper-style self-interpretation adapters.

Three variants for Experiment 2's bias-prior decomposition:
    ScalarAffineAdapter   h ↦ α·h + b           (d_model + 1 params, headline)
    BiasOnlyAdapter       h ↦ b                  (d_model params, format prior only)
    FullRankAdapter       h ↦ W·h + b            (d_model² + d_model params, ceiling)

The adapter's output is a residual-stream replacement: at the chosen layer L,
we hook the block and add the adapter output (or replace the residual at the
target token position). Training is standard cross-entropy on the next-token
emotion label, with the base model frozen.

`forward(h)` returns the modified residual; `apply_in_residual(h_resid)`
returns the residual after injecting the adapter output at a target position.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class _AdapterBase(nn.Module):
    """Common interface for the three adapter variants."""

    name: str = "base"
    n_params: int = 0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Map an input vector h (..., d_model) to an injected vector."""
        raise NotImplementedError


class ScalarAffineAdapter(_AdapterBase):
    """h ↦ α·h + b. Pepper's headline variant: d_model + 1 trainable params."""

    name = "scalar_affine"

    def __init__(self, d_model: int, alpha_init: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        self.n_params = d_model + 1

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.alpha * h + self.bias


class BiasOnlyAdapter(_AdapterBase):
    """h ↦ b. Bias-only ablation: d_model params, ignores the input vector.

    Tests how much of the headline result comes from a layer-agnostic format
    prior rather than activation-conditional content.
    """

    name = "bias_only"

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        self.n_params = d_model

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.bias.expand_as(h)


class ScaleOnlyAdapter(_AdapterBase):
    """h ↦ α·h. Scale-only ablation: 1 param, no bias.

    Tests whether magnitude-along-the-input-direction alone is enough for the
    LM head to produce the right label distribution. Pepper's "1 param scale,
    no learned bias" condition lives here.
    """

    name = "scale_only"

    def __init__(self, d_model: int, alpha_init: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.n_params = 1

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.alpha * h


class FullRankAdapter(_AdapterBase):
    """h ↦ W·h + b. Full-rank ceiling: d_model² + d_model params.

    For Wikipedia/SAE features Pepper found this overfits at small data.
    For our emotion vectors with stronger geometric structure, full-rank
    might genuinely buy something — that's the Experiment 2 question.
    """

    name = "full_rank"

    def __init__(self, d_model: int, init_scale: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.W = nn.Parameter(torch.eye(d_model, dtype=torch.float32) + init_scale * torch.randn(d_model, d_model))
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        self.n_params = d_model * d_model + d_model

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., d). Apply W^T so output keeps shape (..., d).
        return h @ self.W.T + self.bias


@dataclass
class AdapterConfig:
    kind: str  # "scalar_affine" | "bias_only" | "full_rank"
    d_model: int
    alpha_init: float = 1.0
    full_rank_init_scale: float = 1e-3


def make_adapter(cfg: AdapterConfig) -> _AdapterBase:
    if cfg.kind == "scalar_affine":
        return ScalarAffineAdapter(cfg.d_model, alpha_init=cfg.alpha_init)
    if cfg.kind == "bias_only":
        return BiasOnlyAdapter(cfg.d_model)
    if cfg.kind == "scale_only":
        return ScaleOnlyAdapter(cfg.d_model, alpha_init=cfg.alpha_init)
    if cfg.kind == "full_rank":
        return FullRankAdapter(cfg.d_model, init_scale=cfg.full_rank_init_scale)
    raise ValueError(f"unknown adapter kind: {cfg.kind}")
