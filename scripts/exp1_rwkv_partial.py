"""Partial Exp 1 v1 on RWKV-7 (substrate + Likert).

The rwkv pip package's `RWKV_x070` class doesn't fit the HF transformers
interface our ModelAdapter is built on (flat-weight forward, no `model.layers`,
distinct kwarg conventions). So we run a focused script that:

  1. Extracts per-prompt last-token residuals at the canonical layer (L7 from
     Phase 1's PC1↔valence peak).
  2. Builds within-emotion-contrast emotion vectors.
  3. Computes substrate cosine per stimulus per emotion.
  4. Scores Likert valence per stimulus with a custom RWKV-compatible scorer
     that uses RWKV's recurrent state to teacher-force multi-token rating
     sequences efficiently (one prompt forward + one extra forward per
     multi-token rating).
  5. Reports r vs target valence + substrate↔Likert correlation.

Trained-adapter and untrained-SelfIE channels are skipped here — they would
need a custom RWKV residual-replace + LM-head readout, which is significant
extra work. Substrate + Likert are the highest-information channels for the
cross-paradigm convergence claim.

Run:
    uv run python scripts/exp1_rwkv_partial.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Required before importing rwkv.model.
os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.behaviors.likert import VALENCE_PROMPT_TEMPLATE  # noqa: E402
from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.experiments.experiment1 import VALENCE_TARGET  # noqa: E402
from src.models.rwkv7_adapter import RWKV7Adapter  # noqa: E402
from src.probes.diff_means import diff_of_means  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402

DEFAULT_SCALE = [-3, -2, -1, 0, 1, 2, 3]


@torch.no_grad()
def rwkv_likert_valence(
    adapter: RWKV7Adapter,
    prompt: str,
    scale: list[int] = None,
) -> tuple[float, float, dict[int, float]]:
    """Score the next-token distribution over numeric ratings on RWKV-7.

    Tokenizers split " -3" as [' -', '3'] (negatives) but " 3" as a single
    token (positives). Uses full-sequence log-prob: prompt forward gives
    last-token logits; for multi-token ratings, append the prefix tokens
    via RWKV's stateful one-token forward to get the conditional log-prob
    of each subsequent piece.
    """
    scale = scale if scale is not None else DEFAULT_SCALE
    ids = adapter.encode(prompt)
    # forward returns (logits, new_state) for the last token by default
    logits, state = adapter.model.forward(ids, state=None)
    log_probs_after_prompt = F.log_softmax(logits.float(), dim=-1)

    rating_log_probs: dict[int, float] = {}
    for v in scale:
        ids_v = adapter.encode(f" {v}")
        if not ids_v:
            raise ValueError(f"rating {v!r} produced no tokens")
        if len(ids_v) == 1:
            rating_log_probs[v] = float(log_probs_after_prompt[ids_v[0]])
            continue
        # Multi-token rating. First-token log-prob comes from the prompt's
        # ending-state logits. For each subsequent token, feed the previous
        # token (advances state) and read the next logits — that gives the
        # conditional log-prob of the next token.
        log_p = float(log_probs_after_prompt[ids_v[0]])
        cur_state = [s.clone() if isinstance(s, torch.Tensor) else s for s in state]
        for i in range(1, len(ids_v)):
            next_logits, cur_state = adapter.model.forward([ids_v[i - 1]], state=cur_state)
            lp = F.log_softmax(next_logits.float(), dim=-1)
            log_p += float(lp[ids_v[i]])
        rating_log_probs[v] = log_p

    # Softmax across the rating values to a normalized distribution
    keys = list(rating_log_probs.keys())
    scores = torch.tensor([rating_log_probs[k] for k in keys], dtype=torch.float32)
    probs = F.softmax(scores, dim=-1).tolist()
    prob_dict = {k: float(p) for k, p in zip(keys, probs)}
    expected = sum(k * p for k, p in prob_dict.items())
    argmax_value = max(prob_dict.items(), key=lambda kv: kv[1])[0]
    return float(expected), float(argmax_value), prob_dict


def _signed_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default="/media/drive2/projects2/self_awareness/sae_self_model/models/RWKV-x070-World-2.9B-v3-20250211-ctx4096",
    )
    ap.add_argument("--strategy", default="cuda bf16")
    ap.add_argument("--layer", type=int, default=7,
                    help="L7 was the Phase 1 PC1↔valence peak for RWKV-7 World 2.9B.")
    ap.add_argument("--per-cell", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading RWKV-7 from {args.model_path}.pth ...")
    adapter = RWKV7Adapter.load(args.model_path, strategy=args.strategy)
    print(f"  family={adapter.family} n_layers={adapter.n_layers} d_model={adapter.d_model}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    nick = Path(args.model_path).name
    rd = make_run_dir(
        f"phase5_exp1_partial_{nick}",
        config={
            "model_path": args.model_path, "strategy": args.strategy,
            "layer": args.layer, "per_cell": args.per_cell,
            "stimulus_set_size": len(stims),
            "framework": "rwkv-pip-0.8.32",
            "channels": ["substrate", "likert"],
            "skipped": ["trained_adapter", "untrained_selfie"],
        },
    )
    print(f"  run dir: {rd}")

    # 1. Per-prompt residuals at the chosen layer.
    print(f"\n[1/3] Extracting per-prompt residuals at L{args.layer} ...")
    H_per_prompt: list[torch.Tensor] = []
    for s in tqdm(stims):
        ids = adapter.encode(s.prompt)
        cache = adapter.forward_with_residuals(ids, layer_idxs=[args.layer])
        H_per_prompt.append(cache[args.layer][-1])  # last-token residual
    H = torch.stack(H_per_prompt, dim=0)  # (N, d) cpu fp32
    print(f"  residuals: {tuple(H.shape)}")

    # Index stimuli by (emotion, level).
    rows_by_key: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(stims):
        rows_by_key.setdefault((s.emotion, s.level), []).append(i)

    # 2. Within-emotion-contrast emotion vectors.
    print("\n[2/3] Building emotion vectors (within-emotion contrast) ...")
    H_np = H.numpy()
    emotion_vectors: dict[str, np.ndarray] = {}
    for emo in EMOTIONS:
        in_class = rows_by_key.get((emo, "euphoric"), [])
        out_rows: list[int] = []
        for other in EMOTIONS:
            if other == emo:
                continue
            out_rows.extend(rows_by_key.get((other, "euphoric"), []))
        emotion_vectors[emo] = diff_of_means(H_np[in_class], H_np[out_rows]).astype(np.float32)

    # 3. Per-stimulus measurements.
    print("\n[3/3] Per-stimulus substrate + Likert ...")
    test_indices = [
        i for i, s in enumerate(stims) if s.level == "naturalistic" and s.emotion != "neutral"
    ]
    test_stims = [stims[i] for i in test_indices]
    test_H = H[test_indices]
    print(f"  test stimuli: {len(test_stims)} naturalistic")

    pos = ("calm", "blissful")
    neg = ("desperate", "sad", "afraid", "hostile")

    substrate_val_proj = []
    substrate_pred = []
    likert_valences = []
    rows_out: list[dict] = []
    for i, s in enumerate(tqdm(test_stims)):
        h = test_H[i].numpy()
        h_n = h / (np.linalg.norm(h) + 1e-12)
        scores = {}
        for emo, v in emotion_vectors.items():
            v_n = v / (np.linalg.norm(v) + 1e-12)
            scores[emo] = float(h_n @ v_n)
        sub_val = sum(scores[e] for e in pos) - sum(scores[e] for e in neg)
        sub_pred = max(scores.items(), key=lambda kv: kv[1])[0]
        substrate_val_proj.append(sub_val)
        substrate_pred.append(sub_pred)

        prompt = VALENCE_PROMPT_TEMPLATE.format(stimulus=s.prompt)
        likert_v, _, _ = rwkv_likert_valence(adapter, prompt)
        likert_valences.append(likert_v)
        rows_out.append({
            "id": s.id, "emotion": s.emotion, "level": s.level,
            "substrate_pred": sub_pred,
            "substrate_val_proj": sub_val,
            "likert_valence": likert_v,
        })

    target_valence = np.array([VALENCE_TARGET[s.emotion] for s in test_stims], dtype=np.float64)
    sub_val_arr = np.array(substrate_val_proj, dtype=np.float64)
    likert_arr = np.array(likert_valences, dtype=np.float64)

    sub_acc = sum(substrate_pred[i] == test_stims[i].emotion for i in range(len(test_stims))) / len(test_stims)
    summary = {
        "n": len(test_stims),
        "substrate_accuracy_6class": sub_acc,
        "substrate_vs_target": _signed_r(sub_val_arr, target_valence),
        "likert_vs_target": _signed_r(likert_arr, target_valence),
        "substrate_vs_likert": _signed_r(sub_val_arr, likert_arr),
    }
    print(f"\n=== Partial Exp 1 v1 on RWKV-7 (n={summary['n']}) ===")
    print(f"  substrate 6-class accuracy:      {summary['substrate_accuracy_6class']:.3f}")
    print(f"  substrate r vs target valence:   {summary['substrate_vs_target']:+.3f}")
    print(f"  Likert r vs target valence:      {summary['likert_vs_target']:+.3f}")
    print(f"  substrate ↔ Likert r:            {summary['substrate_vs_likert']:+.3f}")

    (rd / "rows.json").write_text(json.dumps(rows_out, indent=2))
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {emo: torch.from_numpy(v) for emo, v in emotion_vectors.items()},
        rd / "emotion_vectors.pt",
    )
    print(f"\nSaved {rd}")


if __name__ == "__main__":
    main()
