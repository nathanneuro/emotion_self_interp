"""Smoke tests for activation extraction."""
from __future__ import annotations

import torch

from src.hooks.extract import ActivationRequest, extract, extract_batch


def test_single_extract_shape(adapter):
    req = ActivationRequest(layer_idxs=[0, adapter.n_layers - 1], position=-1)
    out = extract(adapter, "Once upon a time", req)
    for li in req.layer_idxs:
        assert out[li].shape == (adapter.d_model,)
        assert out[li].dtype == torch.float32


def test_batch_extract_shape(adapter):
    prompts = [
        "The character felt utterly desperate.",
        "She was perfectly calm.",
        "Nothing of note occurred today.",
    ]
    layer = adapter.n_layers // 2
    req = ActivationRequest(layer_idxs=[layer], position=-1)
    out = extract_batch(adapter, prompts, req, batch_size=2)
    assert out[layer].shape == (len(prompts), adapter.d_model)


def test_batch_last_real_matches_position_minus_one_with_left_padding(adapter):
    """With left-padding, position=-1 and position='last_real' should agree."""
    prompts = ["Short.", "A somewhat longer sentence with more tokens to encode."]
    layer = adapter.n_layers // 2
    req_neg = ActivationRequest(layer_idxs=[layer], position=-1)
    req_real = ActivationRequest(layer_idxs=[layer], position="last_real")
    out_neg = extract_batch(adapter, prompts, req_neg, batch_size=4)[layer]
    out_real = extract_batch(adapter, prompts, req_real, batch_size=4)[layer]
    assert torch.allclose(out_neg, out_real, atol=1e-5)


def test_extract_distinguishes_emotional_prompts(adapter):
    """Sanity: activations for an emotional vs neutral prompt should differ.

    This is a substrate-level smoke check — not a claim about emotion-vector
    geometry. Just verifies the pipeline reads something prompt-conditional.
    """
    layer = adapter.n_layers // 2
    req = ActivationRequest(layer_idxs=[layer], position=-1)
    a = extract(adapter, "He was filled with desperate panic.", req)[layer]
    b = extract(adapter, "He sorted the books alphabetically.", req)[layer]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert cos < 0.999, f"prompts produced near-identical activations (cos {cos})"
