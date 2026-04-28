"""Experiment 2: disentangling format prior from activation-conditional readout.

For each adapter variant we measure two extra signals beyond Phase 2's
held-out accuracy:

    zero_vector_decoding   Feed h=0 to the adapter, run the standard
                           probe + residual-replace + score-each-emotion-
                           label pipeline. Returns the predicted label
                           distribution given "no input." If the adapter
                           is mostly a format prior, this distribution will
                           closely match the held-out distribution at
                           α=0 / b alone. If activation-conditional content
                           is meaningful, the distributions will diverge.

    input_shuffle_test     Pair every test residual with the WRONG label
                           and rescore. If the adapter's prediction
                           depends on input, accuracy collapses (correct
                           label is no longer associated with the right
                           residual). If it's mostly bias, accuracy stays
                           near the baseline.

The Phase 2 v0 already trains scalar_affine / bias_only / full_rank — this
module adds the activation-vs-format-prior decomposition on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.adapters.scalar_affine import _AdapterBase
from src.data.emotion_stimuli import EMOTIONS
from src.experiments.experiment1 import (
    _adapter_scores_batched,
    _argmax_label,
    _emotion_label_token_seqs,
)
from src.models.adapter import ModelAdapter


@dataclass
class BiasPriorReport:
    adapter_kind: str
    held_out_top1: float
    zero_vector_pred: str
    zero_vector_probs: dict[str, float]
    held_out_label_distribution: dict[str, float]   # mean prob per emotion across test
    shuffle_top1: float                              # accuracy under random label-pairings
    n_test: int
    extras: dict = field(default_factory=dict)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    """Softmax across emotion-keyed log-probs."""
    keys = list(scores.keys())
    arr = np.array([scores[k] for k in keys], dtype=np.float64)
    arr = arr - arr.max()
    p = np.exp(arr)
    p = p / p.sum()
    return {k: float(v) for k, v in zip(keys, p)}


def _chunked_scores(
    model: ModelAdapter,
    adapter: _AdapterBase,
    residuals: torch.Tensor,
    layer: int,
    label_seqs: dict[str, list[int]],
    chunk_size: int = 8,
) -> list[dict[str, float]]:
    """Wrapper around _adapter_scores_batched that processes in chunks of
    `chunk_size` stimuli. With 6 emotions per stimulus, batch=8 means 48
    rows per forward — comfortable on a single 4090 at d≈900."""
    out: list[dict[str, float]] = []
    for i in range(0, residuals.shape[0], chunk_size):
        out.extend(_adapter_scores_batched(
            model, adapter, residuals[i : i + chunk_size], layer, label_seqs,
        ))
    return out


@torch.no_grad()
def evaluate_adapter_bias_prior(
    model: ModelAdapter,
    adapter: _AdapterBase,
    layer: int,
    test_residuals: torch.Tensor,                  # (N, d) cpu fp32
    test_labels: list[str],                        # one per row, in EMOTIONS
    seed: int = 0,
    chunk_size: int = 8,
) -> BiasPriorReport:
    """Run the three Experiment-2 measurements for one adapter variant."""
    label_seqs = _emotion_label_token_seqs(model.tokenizer, EMOTIONS)
    N = test_residuals.shape[0]
    d = test_residuals.shape[1]

    # 1. Held-out scores: argmax → top-1 + per-emotion mean prob distribution.
    scores_per_row = _chunked_scores(
        model, adapter, test_residuals, layer, label_seqs, chunk_size=chunk_size,
    )
    correct = 0
    sum_probs: dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for i, scores in enumerate(scores_per_row):
        if _argmax_label(scores) == test_labels[i]:
            correct += 1
        p = _softmax(scores)
        for e, v in p.items():
            sum_probs[e] += v
    held_out_top1 = correct / max(N, 1)
    held_out_dist = {e: v / N for e, v in sum_probs.items()}

    # 2. Zero-vector decoding: feed h=0, see what the adapter outputs.
    zero_h = torch.zeros(1, d, dtype=torch.float32)
    zero_scores = _adapter_scores_batched(
        model, adapter, zero_h, layer, label_seqs,
    )[0]
    zero_pred = _argmax_label(zero_scores)
    zero_probs = _softmax(zero_scores)

    # 3. Shuffle test: pair each residual with a deterministically-shuffled
    # incorrect label, score top-1 against the *original* correct label. If
    # the adapter's predictions actually depend on h, top-1 should collapse;
    # if they don't (pure format prior), top-1 stays near held-out.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    # Ensure the perm is a derangement (no row maps to itself) so the test
    # is meaningful even on small N.
    for _ in range(3):
        if not any(perm[i] == i for i in range(N)):
            break
        rng.shuffle(perm)
    shuffled_inputs = test_residuals[perm]
    shuffled_scores = _chunked_scores(
        model, adapter, shuffled_inputs, layer, label_seqs, chunk_size=chunk_size,
    )
    shuffle_correct = 0
    for i, scores in enumerate(shuffled_scores):
        if _argmax_label(scores) == test_labels[i]:
            shuffle_correct += 1
    shuffle_top1 = shuffle_correct / max(N, 1)

    return BiasPriorReport(
        adapter_kind=adapter.name,
        held_out_top1=held_out_top1,
        zero_vector_pred=zero_pred,
        zero_vector_probs=zero_probs,
        held_out_label_distribution=held_out_dist,
        shuffle_top1=shuffle_top1,
        n_test=N,
    )


def summarize_reports(reports: list[BiasPriorReport]) -> dict:
    """Compact dict view for JSON serialization + console printing."""
    return {
        r.adapter_kind: {
            "held_out_top1": r.held_out_top1,
            "zero_vector_pred": r.zero_vector_pred,
            "shuffle_top1": r.shuffle_top1,
            "shuffle_minus_held_out": r.shuffle_top1 - r.held_out_top1,
            "zero_vector_probs": r.zero_vector_probs,
            "held_out_label_dist": r.held_out_label_distribution,
            "n_test": r.n_test,
        }
        for r in reports
    }
