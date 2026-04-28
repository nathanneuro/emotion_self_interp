"""Smoke tests for the ModelAdapter abstraction."""
from __future__ import annotations

import pytest
import torch


def test_basic_attrs(adapter):
    assert adapter.n_layers > 0
    assert adapter.d_model > 0
    assert adapter.family in {"qwen2", "qwen3", "llama", "gemma2"}  # tolerate model swaps
    assert adapter.tokenizer.padding_side == "left"


def test_get_block_bounds(adapter):
    assert adapter.get_block(0) is not None
    assert adapter.get_block(adapter.n_layers - 1) is not None
    with pytest.raises(IndexError):
        adapter.get_block(adapter.n_layers)
    with pytest.raises(IndexError):
        adapter.get_block(-1)


def test_cache_residual_shape(adapter):
    inputs = adapter.tokenizer("Hello world.", return_tensors="pt").to(adapter.device)
    layers = [0, adapter.n_layers // 2, adapter.n_layers - 1]
    with torch.no_grad(), adapter.cache_residual(layers) as cache:
        adapter.model(**inputs)
    for li in layers:
        assert li in cache
        assert cache[li].shape[-1] == adapter.d_model
        assert cache[li].dtype == torch.float32  # cast in hook
        assert cache[li].device.type == "cpu"


def test_steer_changes_logits(adapter):
    inputs = adapter.tokenizer("The capital of France is", return_tensors="pt").to(adapter.device)
    with torch.no_grad():
        base = adapter.model(**inputs).logits.detach().cpu().float()
    torch.manual_seed(0)
    v = torch.randn(adapter.d_model)
    layer = adapter.n_layers // 2
    with torch.no_grad(), adapter.steer_residual(layer, v, alpha=2.0):
        steered = adapter.model(**inputs).logits.detach().cpu().float()
    diff = (base - steered).abs().mean().item()
    assert diff > 1e-3, f"Steering had no effect on logits (mean abs diff {diff})"


def test_ablate_residual_kills_the_targeted_direction(adapter):
    """After ablate(v), the residual at the chosen layer must be orthogonal to v."""
    inputs = adapter.tokenizer("The forest was very quiet.", return_tensors="pt").to(adapter.device)
    layer = adapter.n_layers // 2
    torch.manual_seed(2)
    v = torch.randn(adapter.d_model)
    v = v / v.norm()

    # Forward-hook execution order = registration order, so the ablate hook
    # must enter BEFORE the cache hook for the cache to see the post-modified
    # residual rather than the pre-modified one.
    with torch.no_grad(), adapter.ablate_residual(layer, v), adapter.cache_residual([layer]) as cache:
        adapter.model(**inputs)
    h_ablated = cache[layer]  # (1, S, d) cpu fp32

    # Projection onto v should be ~0 everywhere along the sequence. The bf16
    # forward + cast-to-fp32 cache leaves O(1e-1) roundoff on a d≈900 vector
    # whose unmodified projection would be tens; we check both an absolute
    # residue bound and that it's a tiny fraction of the residual norm.
    proj_abs = (h_ablated @ v).abs().max().item()
    norm = h_ablated.norm(dim=-1).max().item()
    assert proj_abs < 0.5, f"residual still has component along v: max |h·v| = {proj_abs}"
    assert proj_abs / max(norm, 1e-9) < 0.01, (
        f"projection-to-norm ratio too high: {proj_abs / norm}"
    )


def test_ablate_changes_logits(adapter):
    inputs = adapter.tokenizer("The capital of Brazil is", return_tensors="pt").to(adapter.device)
    with torch.no_grad():
        base = adapter.model(**inputs).logits.detach().cpu().float()
    torch.manual_seed(3)
    v = torch.randn(adapter.d_model)
    layer = adapter.n_layers // 2
    with torch.no_grad(), adapter.ablate_residual(layer, v):
        ablated = adapter.model(**inputs).logits.detach().cpu().float()
    diff = (base - ablated).abs().mean().item()
    assert diff > 1e-4, f"ablation had no effect on logits (mean abs diff {diff})"


def test_steer_with_token_mask_only_affects_masked_positions(adapter):
    """Mask out all positions → output equals baseline. Mask only one → only later positions differ."""
    inputs = adapter.tokenizer("The quick brown fox", return_tensors="pt").to(adapter.device)
    seq_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        base = adapter.model(**inputs).logits.detach().cpu().float()

    torch.manual_seed(1)
    v = torch.randn(adapter.d_model)
    layer = adapter.n_layers // 2

    zero_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    with torch.no_grad(), adapter.steer_residual(layer, v, alpha=5.0, token_mask=zero_mask):
        masked_off = adapter.model(**inputs).logits.detach().cpu().float()
    assert torch.allclose(base, masked_off, atol=1e-4), "zero mask should be a no-op"

    # Steer only at the first position. Causal attention means later positions
    # should be affected; logits over the full sequence should differ from base.
    one_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    one_mask[0, 0] = True
    with torch.no_grad(), adapter.steer_residual(layer, v, alpha=5.0, token_mask=one_mask):
        partial = adapter.model(**inputs).logits.detach().cpu().float()
    diff = (base - partial).abs().mean().item()
    assert diff > 1e-3, "first-token steering should propagate via causal attention"
