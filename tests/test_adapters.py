"""Unit tests for the Pepper-style adapter variants — pure math, no model."""
from __future__ import annotations

import torch

from src.adapters.scalar_affine import (
    AdapterConfig,
    BiasOnlyAdapter,
    FullRankAdapter,
    ScalarAffineAdapter,
    make_adapter,
)


def test_scalar_affine_param_count():
    a = ScalarAffineAdapter(d_model=128)
    n_trainable = sum(p.numel() for p in a.parameters() if p.requires_grad)
    assert n_trainable == 128 + 1
    assert a.n_params == 128 + 1


def test_bias_only_param_count():
    a = BiasOnlyAdapter(d_model=64)
    assert sum(p.numel() for p in a.parameters() if p.requires_grad) == 64
    h = torch.randn(3, 64)
    out = a(h)
    # bias_only ignores h: output rows must all equal the bias.
    assert torch.allclose(out[0], a.bias.detach())
    assert torch.allclose(out[1], a.bias.detach())


def test_full_rank_param_count():
    a = FullRankAdapter(d_model=32)
    n = sum(p.numel() for p in a.parameters() if p.requires_grad)
    assert n == 32 * 32 + 32


def test_scalar_affine_init_is_identity_plus_zero():
    a = ScalarAffineAdapter(d_model=16, alpha_init=1.0)
    h = torch.randn(5, 16)
    out = a(h)
    assert torch.allclose(out, h, atol=1e-6)


def test_full_rank_init_is_near_identity():
    torch.manual_seed(0)
    a = FullRankAdapter(d_model=16, init_scale=1e-3)
    h = torch.randn(5, 16)
    out = a(h)
    # near-identity means most of h is preserved.
    cos = torch.nn.functional.cosine_similarity(out, h, dim=-1)
    assert (cos > 0.99).all(), f"expected near-identity, got cos={cos}"


def test_make_adapter_dispatch():
    cfg = AdapterConfig(kind="scalar_affine", d_model=8)
    assert isinstance(make_adapter(cfg), ScalarAffineAdapter)
    assert isinstance(make_adapter(AdapterConfig(kind="bias_only", d_model=8)), BiasOnlyAdapter)
    assert isinstance(make_adapter(AdapterConfig(kind="full_rank", d_model=8)), FullRankAdapter)
