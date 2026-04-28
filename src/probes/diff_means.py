"""Linear probes for emotion-vector extraction (Sofroniew-style).

Given activations at a single (model, layer, position) site, build a direction
that separates emotion-bearing prompts from a contrast set:

    diff_of_means: v = mean(H_emotion) - mean(H_contrast)
    fit_lda:        v = (Σ_pooled^{-1}) (μ_emotion - μ_contrast)

Both return a unit vector in the direction of higher emotion-class projection.
`probe_separation` reports d-prime and AUROC for held-out activations.

These probes are deliberately simple — single-direction linear separators —
because that is what the program is testing the limits of (linear-direction
assumption, methodological caveat in docs/planning.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _to_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32).numpy()
    return np.asarray(x, dtype=np.float32)


def diff_of_means(
    H_pos: torch.Tensor | np.ndarray,
    H_neg: torch.Tensor | np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Mean(H_pos, axis=0) - Mean(H_neg, axis=0). Optionally L2-normalized."""
    pos = _to_np(H_pos)
    neg = _to_np(H_neg)
    if pos.ndim != 2 or neg.ndim != 2:
        raise ValueError(f"expected (N, d) inputs, got {pos.shape} / {neg.shape}")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(f"d_model mismatch: {pos.shape[1]} vs {neg.shape[1]}")
    v = pos.mean(axis=0) - neg.mean(axis=0)
    if normalize:
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
    return v.astype(np.float32)


def fit_lda(
    H_pos: torch.Tensor | np.ndarray,
    H_neg: torch.Tensor | np.ndarray,
    shrinkage: float = 1e-2,
    normalize: bool = True,
) -> np.ndarray:
    """Fisher LDA direction. Pooled within-class covariance + ridge shrinkage.

    Falls back to diff-of-means if the inversion is unstable. Shrinkage keeps
    the small-sample regime well-conditioned (d_model >> n is common here).
    """
    pos = _to_np(H_pos)
    neg = _to_np(H_neg)
    mu_p, mu_n = pos.mean(0), neg.mean(0)
    Cp = np.cov(pos, rowvar=False, bias=False) if pos.shape[0] > 1 else 0.0
    Cn = np.cov(neg, rowvar=False, bias=False) if neg.shape[0] > 1 else 0.0
    n_p, n_n = pos.shape[0], neg.shape[0]
    pooled = ((n_p - 1) * Cp + (n_n - 1) * Cn) / max(n_p + n_n - 2, 1)
    d = pooled.shape[0] if hasattr(pooled, "shape") else pos.shape[1]
    pooled = pooled + shrinkage * np.eye(d, dtype=np.float64)
    try:
        v = np.linalg.solve(pooled, (mu_p - mu_n))
    except np.linalg.LinAlgError:
        v = mu_p - mu_n
    if normalize:
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
    return v.astype(np.float32)


def project(H: torch.Tensor | np.ndarray, v: np.ndarray) -> np.ndarray:
    """Scalar projections H · v. Returns shape (N,)."""
    return _to_np(H) @ v.astype(np.float32)


@dataclass
class SeparationResult:
    d_prime: float
    auroc: float
    mean_pos: float
    mean_neg: float
    std_pos: float
    std_neg: float


def probe_separation(
    H_pos: torch.Tensor | np.ndarray,
    H_neg: torch.Tensor | np.ndarray,
    v: np.ndarray,
) -> SeparationResult:
    """Project both groups onto v and report d-prime + AUROC.

    d' = (μ_pos − μ_neg) / sqrt(0.5 (σ_pos² + σ_neg²))
    AUROC = P(proj_pos > proj_neg) over all pairs (Mann-Whitney form).
    """
    p = project(H_pos, v)
    n = project(H_neg, v)
    mu_p, mu_n = float(p.mean()), float(n.mean())
    s_p, s_n = float(p.std(ddof=1)) if len(p) > 1 else 0.0, float(n.std(ddof=1)) if len(n) > 1 else 0.0
    pooled_var = 0.5 * (s_p**2 + s_n**2)
    d_prime = (mu_p - mu_n) / float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0

    # AUROC via rank statistics — equivalent to mean of 1[p > n] over all pairs.
    all_vals = np.concatenate([p, n])
    labels = np.concatenate([np.ones_like(p), np.zeros_like(n)])
    order = np.argsort(all_vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_vals) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        auroc = float("nan")
    else:
        sum_ranks_pos = ranks[labels == 1].sum()
        u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
        auroc = float(u / (n_pos * n_neg))

    return SeparationResult(
        d_prime=d_prime, auroc=auroc,
        mean_pos=mu_p, mean_neg=mu_n,
        std_pos=s_p, std_neg=s_n,
    )
