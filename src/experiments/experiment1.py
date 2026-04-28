"""Experiment 1: cross-method convergence on the v0 emotion stimulus set.

For each stimulus we measure four channels at the canonical layer:

    substrate     dot(residual, v_E) for each E in EMOTIONS — argmax = predicted
                  emotion. Equivalent to the Phase 1 diff-of-means probe scored
                  per stimulus.
    adapter       Pepper-style trained adapter (α·h + b or W·h + b) injected
                  via residual-replace at a probe position; score each emotion
                  by full-token-sequence log-prob of " <emotion>" at the
                  answer position.
    untrained     Same adapter machinery with α=1, b=0 (or W=I, b=0). The
                  "training matters" comparison from Pepper et al.
    behavior      Likert valence + arousal ratings of the stimulus passage.

The convergence question is whether all four channels track each other across
stimuli. Per-channel 6-class accuracy + pairwise channel-prediction agreement
+ continuous correlations (substrate score ↔ Likert valence) give the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.adapters.scalar_affine import _AdapterBase
from src.adapters.train import (
    _build_probe_inputs,
    _residual_replace_hook,
)
from src.behaviors.likert import LikertConfig, likert_rating
from src.data.emotion_stimuli import EMOTIONS
from src.data.stimuli import Stimulus
from src.hooks.extract import ActivationRequest, extract_batch
from src.models.adapter import ModelAdapter

VALENCE_TARGET = {
    "calm": +1, "blissful": +1,
    "desperate": -1, "sad": -1, "afraid": -1, "hostile": -1,
    "neutral": 0,
}
AROUSAL_TARGET = {
    "calm": -1, "sad": -1,
    "blissful": +1, "desperate": +1, "afraid": +1, "hostile": +1,
    "neutral": 0,
}


@dataclass
class PerStimulus:
    """Channel readouts for one stimulus."""
    stimulus_id: str
    true_emotion: str
    level: str

    substrate_scores: dict[str, float] = field(default_factory=dict)
    substrate_pred: str = ""

    adapter_scores: dict[str, float] = field(default_factory=dict)
    adapter_pred: str = ""

    untrained_scores: dict[str, float] = field(default_factory=dict)
    untrained_pred: str = ""

    likert_valence: float = float("nan")
    likert_arousal: float = float("nan")


def _emotion_label_token_seqs(tokenizer, emotions: Iterable[str]) -> dict[str, list[int]]:
    """Tokenize " <emotion>" for each label as a token sequence (multi-token allowed)."""
    out: dict[str, list[int]] = {}
    for e in emotions:
        ids = tokenizer.encode(" " + e, add_special_tokens=False)
        if not ids:
            raise ValueError(f"emotion label {e!r} produced no tokens")
        out[e] = ids
    return out


@torch.no_grad()
def _substrate_scores(
    residual: torch.Tensor,                  # (d,) cpu fp32
    emotion_vectors: dict[str, np.ndarray],  # {E: (d,)}
) -> dict[str, float]:
    """Project the residual onto each emotion vector. Cosine similarity (unit-
    normalized) so different emotion vectors are comparable when their norms
    differ.
    """
    h = residual.detach().cpu().to(torch.float32).numpy()
    h_norm = h / (np.linalg.norm(h) + 1e-12)
    out: dict[str, float] = {}
    for E, v in emotion_vectors.items():
        v_n = v / (np.linalg.norm(v) + 1e-12)
        out[E] = float(h_norm @ v_n)
    return out


@torch.no_grad()
def _adapter_scores_batched(
    model: ModelAdapter,
    adapter: _AdapterBase,
    residuals: torch.Tensor,                 # (N, d) cpu fp32
    layer: int,
    label_token_seqs: dict[str, list[int]],
) -> list[dict[str, float]]:
    """Score N stimuli × E emotions in a single forward by tiling.

    Builds an (N·E, P+max_lab) batch where row (n, e) has the probe followed
    by emotion e's label tokens, and the residual-replace hook is fed an
    (N·E, d) tensor with stimulus n's injected residual repeated E times.
    Returns a list of {emotion → log P(label | probe + adapter(h_n))} per n.
    """
    device = model.device
    probe_ids, act_pos = _build_probe_inputs(model.tokenizer, device, batch_size=1)
    P = probe_ids.shape[1]
    label_items = list(label_token_seqs.items())
    E = len(label_items)
    N = residuals.shape[0]
    max_lab = max(len(seq) for _, seq in label_items)
    pad_id = model.tokenizer.pad_token_id or 0
    block = model.get_block(layer)

    # Per-emotion (P + max_lab) row template; we'll repeat each row N times.
    base_rows: list[list[int]] = []
    label_lens: list[int] = []
    for _, seq in label_items:
        base_rows.append(probe_ids[0].tolist() + seq + [pad_id] * (max_lab - len(seq)))
        label_lens.append(len(seq))
    base_input_ids = torch.tensor(base_rows, device=device, dtype=torch.long)  # (E, P+max_lab)

    # Tile to (N·E, P+max_lab). Order: stimulus 0 emotions, stimulus 1 emotions, ...
    input_ids = base_input_ids.unsqueeze(0).expand(N, E, -1).reshape(N * E, -1).contiguous()

    # Per-row attention mask: padding only the trailing slots beyond each
    # row's actual emotion-label length.
    attn_mask = torch.ones_like(input_ids)
    for e_idx, L in enumerate(label_lens):
        if L < max_lab:
            attn_mask[e_idx::E, P + L:] = 0

    # Build the (N·E, d) injected-residual tensor. Each chunk of E rows shares
    # the residual for stimulus n, processed once through the adapter.
    h_dev = residuals.to(device=device, dtype=next(adapter.parameters()).dtype)
    injected = adapter(h_dev)                              # (N, d)
    injected_NE = injected.unsqueeze(1).expand(N, E, -1).reshape(N * E, -1).contiguous()

    with _residual_replace_hook(block, injected_NE, act_pos):
        out = model.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    log_probs = F.log_softmax(out.logits.float(), dim=-1)

    results: list[dict[str, float]] = []
    for n in range(N):
        scores: dict[str, float] = {}
        for e_idx, (emo, seq) in enumerate(label_items):
            row = n * E + e_idx
            s = 0.0
            for j, tok in enumerate(seq):
                s += float(log_probs[row, P + j - 1, tok])
            scores[emo] = s
        results.append(scores)
    return results


def _argmax_label(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def run_experiment1(
    model: ModelAdapter,
    layer: int,
    emotion_vectors: dict[str, np.ndarray],
    trained_adapter: _AdapterBase,
    untrained_adapter: _AdapterBase,
    stimuli: list[Stimulus],
    likert_cfg: LikertConfig | None = None,
    adapter_batch_size: int = 4,
    progress: bool = True,
) -> list[PerStimulus]:
    """Run the four convergence channels on each stimulus.

    Internally batches the adapter-scoring forward (B stimuli × 6 emotions
    per call). Likert is still per-stimulus because the prompt text varies
    per item and batching across them would require padding the probe text.
    """
    likert_cfg = likert_cfg or LikertConfig()
    label_seqs = _emotion_label_token_seqs(model.tokenizer, EMOTIONS)

    # Substrate channel: one batched forward pass over all prompts.
    prompts = [s.prompt for s in stimuli]
    req = ActivationRequest(layer_idxs=[layer], position=-1)
    H = extract_batch(model, prompts, req, batch_size=16)[layer]  # (N, d) cpu fp32

    rows: list[PerStimulus] = [
        PerStimulus(stimulus_id=s.id, true_emotion=s.emotion, level=s.level)
        for s in stimuli
    ]

    # Substrate scores per stimulus.
    for i, rec in enumerate(rows):
        rec.substrate_scores = _substrate_scores(H[i], emotion_vectors)
        rec.substrate_pred = _argmax_label(rec.substrate_scores)

    # Adapter + untrained scores: process in chunks for memory; each chunk
    # makes ONE forward pass producing (B × 6) (stimulus, emotion) scores.
    chunk_iter = range(0, len(stimuli), adapter_batch_size)
    if progress:
        chunk_iter = tqdm(list(chunk_iter), desc="adapter scoring (batched)")
    for start in chunk_iter:
        end = min(start + adapter_batch_size, len(stimuli))
        chunk_H = H[start:end]
        trained_scores = _adapter_scores_batched(
            model, trained_adapter, chunk_H, layer, label_seqs,
        )
        untrained_scores = _adapter_scores_batched(
            model, untrained_adapter, chunk_H, layer, label_seqs,
        )
        for j, idx in enumerate(range(start, end)):
            rows[idx].adapter_scores = trained_scores[j]
            rows[idx].adapter_pred = _argmax_label(trained_scores[j])
            rows[idx].untrained_scores = untrained_scores[j]
            rows[idx].untrained_pred = _argmax_label(untrained_scores[j])

    # Likert per stimulus (one prompt per stimulus, two forwards each — so
    # batching across stimuli would require padded prompts; keep simple).
    likert_iter = stimuli
    if progress:
        likert_iter = tqdm(stimuli, desc="likert")
    for i, s in enumerate(likert_iter):
        if s.emotion == "neutral":
            continue
        lk = likert_rating(model, s.prompt, likert_cfg)
        rows[i].likert_valence = float(lk.valence.expected)
        rows[i].likert_arousal = float(lk.arousal.expected)

    return rows


def summarize_experiment1(rows: list[PerStimulus]) -> dict:
    """Compute per-channel accuracy, pairwise prediction agreement, and
    channel-vs-target correlations.

    Returns a dict with keys:
      `accuracy_<channel>`               6-class top-1 accuracy on rows where
                                         true_emotion is in EMOTIONS.
      `pairwise_agreement`               {(c1, c2): agreement_rate} on top-1.
      `correlations.<channel>_vs_target` Pearson r between channel score and
                                         the target valence per stimulus.
    """
    eval_rows = [r for r in rows if r.true_emotion in EMOTIONS]
    n = len(eval_rows)
    if n == 0:
        return {"n": 0}

    def _acc(channel: str) -> float:
        return sum(getattr(r, channel) == r.true_emotion for r in eval_rows) / n

    pred_channels = ["substrate_pred", "adapter_pred", "untrained_pred"]
    accuracy = {c.replace("_pred", ""): _acc(c) for c in pred_channels}

    # Pairwise top-1 agreement.
    pairwise: dict[str, float] = {}
    for i, c1 in enumerate(pred_channels):
        for c2 in pred_channels[i + 1:]:
            agree = sum(getattr(r, c1) == getattr(r, c2) for r in eval_rows) / n
            pairwise[f"{c1.replace('_pred','')}__{c2.replace('_pred','')}"] = agree

    # Continuous: per-channel "valence-like" score per stimulus, vs target valence.
    val_target = np.array([VALENCE_TARGET[r.true_emotion] for r in eval_rows], dtype=np.float64)

    # Substrate valence-like score = (calm + blissful) − (desperate + sad + afraid + hostile)
    pos = ("calm", "blissful")
    neg = ("desperate", "sad", "afraid", "hostile")
    sub_val = np.array([
        sum(r.substrate_scores.get(e, 0.0) for e in pos)
        - sum(r.substrate_scores.get(e, 0.0) for e in neg)
        for r in eval_rows
    ], dtype=np.float64)
    adp_val = np.array([
        sum(r.adapter_scores.get(e, 0.0) for e in pos)
        - sum(r.adapter_scores.get(e, 0.0) for e in neg)
        for r in eval_rows
    ], dtype=np.float64)
    unt_val = np.array([
        sum(r.untrained_scores.get(e, 0.0) for e in pos)
        - sum(r.untrained_scores.get(e, 0.0) for e in neg)
        for r in eval_rows
    ], dtype=np.float64)
    lik_val = np.array([r.likert_valence for r in eval_rows], dtype=np.float64)

    def _r(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return float("nan")
        a, b = a[mask], b[mask]
        a = a - a.mean()
        b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / denom) if denom > 0 else 0.0

    correlations = {
        "substrate_vs_target": _r(sub_val, val_target),
        "adapter_vs_target": _r(adp_val, val_target),
        "untrained_vs_target": _r(unt_val, val_target),
        "likert_vs_target": _r(lik_val, val_target),
        "substrate_vs_likert": _r(sub_val, lik_val),
        "adapter_vs_likert": _r(adp_val, lik_val),
        "untrained_vs_likert": _r(unt_val, lik_val),
        "substrate_vs_adapter": _r(sub_val, adp_val),
    }

    return {
        "n": n,
        "accuracy": accuracy,
        "pairwise_agreement": pairwise,
        "correlations": correlations,
    }
