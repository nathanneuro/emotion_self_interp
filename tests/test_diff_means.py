"""Unit tests for the linear-probe utilities. Pure math, no model load."""
from __future__ import annotations

import numpy as np
import pytest

from src.probes.diff_means import diff_of_means, fit_lda, probe_separation, project


def _two_clusters(d=8, n=50, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=d)
    direction = direction / np.linalg.norm(direction)
    pos = rng.normal(size=(n, d)) + sep * direction
    neg = rng.normal(size=(n, d))
    return pos.astype(np.float32), neg.astype(np.float32), direction.astype(np.float32)


def test_diff_of_means_recovers_direction():
    pos, neg, true_dir = _two_clusters(d=16, n=200, sep=4.0, seed=1)
    v = diff_of_means(pos, neg)
    cos = float(v @ true_dir)
    assert cos > 0.95, f"recovered direction cos = {cos}"
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_lda_recovers_direction_with_isotropic_noise():
    pos, neg, true_dir = _two_clusters(d=16, n=200, sep=4.0, seed=2)
    v = fit_lda(pos, neg)
    cos = float(v @ true_dir)
    assert cos > 0.9, f"LDA recovered direction cos = {cos}"


def test_separation_signs():
    pos, neg, true_dir = _two_clusters(d=8, n=100, sep=3.0, seed=3)
    v = diff_of_means(pos, neg)
    res = probe_separation(pos, neg, v)
    assert res.d_prime > 1.0
    assert 0.9 < res.auroc <= 1.0
    assert res.mean_pos > res.mean_neg


def test_separation_zero_when_classes_identical():
    rng = np.random.default_rng(4)
    H = rng.normal(size=(200, 8)).astype(np.float32)
    v = rng.normal(size=8).astype(np.float32)
    v /= np.linalg.norm(v)
    res = probe_separation(H[:100], H[100:], v)
    assert abs(res.d_prime) < 0.5
    assert 0.4 < res.auroc < 0.6


def test_diff_of_means_rejects_dim_mismatch():
    with pytest.raises(ValueError):
        diff_of_means(np.zeros((10, 4)), np.zeros((10, 5)))


def test_project_shape():
    H = np.random.default_rng(5).normal(size=(7, 4)).astype(np.float32)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p = project(H, v)
    assert p.shape == (7,)
    assert np.allclose(p, H[:, 0])
