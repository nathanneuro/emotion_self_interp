"""Experiment 4: veridical introspection — does behavior follow substrate or report?

The clean operationalization in our framework:

  1. Train two adapters on the same residual cache:
       honest      (h_E, "E")        — usual training
       deceptive   (h_E, swap("E"))  — labels swapped per a fixed pairing

     The deceptive adapter learns to *say* "calm" when given a desperate
     residual (and vice versa, etc.). This decouples REPORT (adapter output)
     from ACTIVATION (the residual itself).

  2. On held-out stimuli, measure four channels per stimulus:
       deceptive_pred   adapter output (the report)
       honest_pred      adapter output (the report)
       likert_valence   substrate-driven (reads residual via the model's
                         normal forward, not via the adapter)
       substrate_score  cosine(h, v_E) per emotion (substrate-direct)

  3. The key test: do Likert and substrate track the *true* emotion (the
     activation pattern), while the deceptive adapter tracks its *swap*
     of the true emotion? If yes, we have the program's clean operational
     definition of veridical introspection: the report is a separable
     channel that can be made non-veridical by training, while the
     substrate-driven channels remain causally tied to the activation.

The standard SWAP_PAIRING below pairs emotions across the valence axis
when possible, so deceptive predictions are not just "permuted noise"
but specifically valence-flipped — making the divergence test crisp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.adapters.scalar_affine import _AdapterBase
from src.adapters.train import TrainConfig, TrainExample, train_adapter
from src.adapters.scalar_affine import AdapterConfig, make_adapter
from src.data.emotion_stimuli import EMOTIONS
from src.experiments.experiment1 import (
    VALENCE_TARGET,
    _argmax_label,
    _emotion_label_token_seqs,
    _substrate_scores,
)
from src.experiments.experiment2 import _chunked_scores
from src.experiments.protocol import StimulusResiduals
from src.models.adapter import ModelAdapter

# Swap pairs that flip valence: (calm,desperate), (blissful,sad), and
# (afraid,hostile). The first two are clean +/- valence flips; the third
# pairs two same-valence emotions, which makes the deceptive adapter's
# output *not* trivially valence-correlated with truth — a sharper test.
SWAP_PAIRING = {
    "calm": "desperate",
    "desperate": "calm",
    "blissful": "sad",
    "sad": "blissful",
    "afraid": "hostile",
    "hostile": "afraid",
}


@dataclass
class IntrospectionRow:
    stimulus_id: str
    true_emotion: str
    level: str
    swap_target: str
    honest_pred: str
    deceptive_pred: str
    substrate_pred: str
    honest_scores: dict[str, float]
    deceptive_scores: dict[str, float]
    substrate_scores: dict[str, float]


def train_honest_and_deceptive_adapters(
    model: ModelAdapter,
    res: StimulusResiduals,
    kind: str = "full_rank",
    epochs: int = 20,
    lr: float = 5e-3,
    batch_size: int = 8,
    train_level: str = "euphoric",
) -> tuple[_AdapterBase, _AdapterBase]:
    """Train one adapter on truthful (h, label) pairs and one on labels
    swapped via SWAP_PAIRING. Both share the same residual cache."""
    truthful_examples: list[TrainExample] = []
    deceptive_examples: list[TrainExample] = []
    for i, s in enumerate(res.stimuli):
        if s.emotion == "neutral" or s.level != train_level:
            continue
        truthful_examples.append(
            TrainExample(vector=res.residuals[i].clone(), label=s.emotion)
        )
        deceptive_examples.append(
            TrainExample(vector=res.residuals[i].clone(), label=SWAP_PAIRING[s.emotion])
        )

    cfg = TrainConfig(
        layer_idx=res.layer, batch_size=batch_size, n_epochs=epochs, learning_rate=lr,
    )
    honest = make_adapter(AdapterConfig(kind=kind, d_model=res.d_model)).to(model.device)
    train_adapter(model, honest, truthful_examples, val=None, cfg=cfg)

    deceptive = make_adapter(AdapterConfig(kind=kind, d_model=res.d_model)).to(model.device)
    train_adapter(model, deceptive, deceptive_examples, val=None, cfg=cfg)

    return honest, deceptive


@torch.no_grad()
def measure_introspection(
    model: ModelAdapter,
    layer: int,
    honest_adapter: _AdapterBase,
    deceptive_adapter: _AdapterBase,
    emotion_vectors: dict[str, np.ndarray],
    test_residuals: torch.Tensor,                 # (N, d) cpu fp32
    test_stimuli_meta: list[tuple[str, str, str]],  # (id, true_emotion, level) per row
    chunk_size: int = 8,
) -> list[IntrospectionRow]:
    """Run honest + deceptive adapters and substrate scoring on each test
    residual. Returns per-stimulus rows ready for cross-channel comparison."""
    label_seqs = _emotion_label_token_seqs(model.tokenizer, EMOTIONS)
    honest_scores = _chunked_scores(model, honest_adapter, test_residuals, layer, label_seqs, chunk_size=chunk_size)
    deceptive_scores = _chunked_scores(model, deceptive_adapter, test_residuals, layer, label_seqs, chunk_size=chunk_size)

    rows: list[IntrospectionRow] = []
    for i, (sid, true_emo, level) in enumerate(test_stimuli_meta):
        sub = _substrate_scores(test_residuals[i], emotion_vectors)
        rows.append(IntrospectionRow(
            stimulus_id=sid, true_emotion=true_emo, level=level,
            swap_target=SWAP_PAIRING.get(true_emo, "unknown"),
            honest_pred=_argmax_label(honest_scores[i]),
            deceptive_pred=_argmax_label(deceptive_scores[i]),
            substrate_pred=_argmax_label(sub),
            honest_scores=honest_scores[i],
            deceptive_scores=deceptive_scores[i],
            substrate_scores=sub,
        ))
    return rows


def _signed_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def summarize_introspection(
    rows: list[IntrospectionRow],
    likert_valences: list[float] | None = None,
) -> dict:
    """Per-channel match rates against (a) the actual emotion of the stimulus
    and (b) the deceptive-adapter's swap target. Plus continuous valence
    correlations against the target if Likert valences are supplied.

    A veridical introspection result looks like:
      honest match-true ≈ substrate match-true > deceptive match-true (≈ chance)
      deceptive match-swap ≈ honest match-true (deceptive learned the swap)
      Likert / substrate r vs target valence remain positive while
      deceptive r vs target valence flips negative.
    """
    n = len(rows)
    honest_true = sum(r.honest_pred == r.true_emotion for r in rows) / max(n, 1)
    honest_swap = sum(r.honest_pred == r.swap_target for r in rows) / max(n, 1)
    decep_true = sum(r.deceptive_pred == r.true_emotion for r in rows) / max(n, 1)
    decep_swap = sum(r.deceptive_pred == r.swap_target for r in rows) / max(n, 1)
    sub_true = sum(r.substrate_pred == r.true_emotion for r in rows) / max(n, 1)
    sub_swap = sum(r.substrate_pred == r.swap_target for r in rows) / max(n, 1)

    out = {
        "n": n,
        "match_true": {
            "honest_adapter": honest_true,
            "deceptive_adapter": decep_true,
            "substrate": sub_true,
        },
        "match_swap": {
            "honest_adapter": honest_swap,
            "deceptive_adapter": decep_swap,
            "substrate": sub_swap,
        },
    }

    # Continuous: each channel's "valence-like" projection vs target valence.
    pos = ("calm", "blissful")
    neg = ("desperate", "sad", "afraid", "hostile")

    def val_proj(scores_attr: str) -> np.ndarray:
        return np.array([
            sum(getattr(r, scores_attr).get(e, 0.0) for e in pos)
            - sum(getattr(r, scores_attr).get(e, 0.0) for e in neg)
            for r in rows
        ], dtype=np.float64)

    target = np.array([VALENCE_TARGET[r.true_emotion] for r in rows], dtype=np.float64)
    out["correlations"] = {
        "honest_vs_target": _signed_r(val_proj("honest_scores"), target),
        "deceptive_vs_target": _signed_r(val_proj("deceptive_scores"), target),
        "substrate_vs_target": _signed_r(val_proj("substrate_scores"), target),
    }
    if likert_valences is not None and len(likert_valences) == n:
        lv = np.array(likert_valences, dtype=np.float64)
        out["correlations"]["likert_vs_target"] = _signed_r(lv, target)
        out["correlations"]["honest_vs_likert"] = _signed_r(val_proj("honest_scores"), lv)
        out["correlations"]["deceptive_vs_likert"] = _signed_r(val_proj("deceptive_scores"), lv)
        out["correlations"]["substrate_vs_likert"] = _signed_r(val_proj("substrate_scores"), lv)
    return out
