"""Per-ut-step Likert valence on Ouro.

Phase 1's per-(layer, ut_step) finding was that valence-vector PCA structure
builds across loop iterations: max |PC1↔valence| climbs 0.348 → 0.976 →
0.982 → 0.984 from ut=0 to ut=3. Question: does the *behavioral readout*
follow the same trajectory? We ask Likert valence with `exit_at_step=N` for
each N ∈ {0, 1, 2, 3} on the same stimulus, capturing the model's reported
emotion-rating after each loop pass.

Ouro's `OuroForCausalLM.forward` accepts `exit_at_step`, which selects the
post-norm hidden states from `hidden_states_list[N]` and applies `lm_head`
to them. So we get the model's actual generation distribution conditioned
on stopping at ut step N — no probing/cosine, just the LM head.

Run:
    uv run python scripts/per_ut_step_likert_ouro.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.behaviors.likert import LikertConfig, likert_rating  # noqa: E402
from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402

VALENCE_TARGET = {
    "calm": +1, "blissful": +1,
    "desperate": -1, "sad": -1, "afraid": -1, "hostile": -1,
}


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
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    ap.add_argument("--per-cell", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map, trust_remote_code=True)
    n_ut = int(getattr(model.model.config, "total_ut_steps", 4))
    print(f"  family={model.family} d_model={model.d_model} n_ut={n_ut}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    test_stims = [s for s in stims if s.level == "naturalistic" and s.emotion != "neutral"]
    print(f"  test stimuli: {len(test_stims)} naturalistic")

    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"per_ut_likert_{nick}",
        config={"model": args.model, "n_ut": n_ut, "per_cell": args.per_cell, "n_test": len(test_stims)},
    )
    print(f"  run dir: {rd}")

    # Collect per-ut-step Likert valence per stimulus.
    likert_cfg = LikertConfig()
    rows: list[dict] = []
    for s in tqdm(test_stims, desc="stimuli"):
        for ut in range(n_ut):
            lk = likert_rating(model, s.prompt, likert_cfg, forward_kwargs={"exit_at_step": ut})
            rows.append({
                "id": s.id, "emotion": s.emotion, "level": s.level, "ut": ut,
                "likert_valence": float(lk.valence.expected),
                "likert_arousal": float(lk.arousal.expected),
                "target_valence": VALENCE_TARGET[s.emotion],
            })
        # Also baseline: full forward (no exit_at_step) — should match ut=last in
        # most cases but use_weighted_exit isn't passed so the model falls back
        # to last_hidden_state.
        lk = likert_rating(model, s.prompt, likert_cfg)
        rows.append({
            "id": s.id, "emotion": s.emotion, "level": s.level, "ut": "full",
            "likert_valence": float(lk.valence.expected),
            "likert_arousal": float(lk.arousal.expected),
            "target_valence": VALENCE_TARGET[s.emotion],
        })

    (rd / "rows.json").write_text(json.dumps(rows, indent=2))

    # Per-ut summary: r vs target, mean per emotion, direction-correct count.
    print("\nPer-ut-step Likert valence summary:\n")
    print(f"  {'ut':<6} {'r vs tgt':>10} {'r aro vs tgt':>14} {'dir-correct/6':>16}")
    summary: list[dict] = []
    for ut in list(range(n_ut)) + ["full"]:
        ur = [r for r in rows if r["ut"] == ut]
        if not ur:
            continue
        vals = np.array([r["likert_valence"] for r in ur], dtype=np.float64)
        aros = np.array([r["likert_arousal"] for r in ur], dtype=np.float64)
        targets = np.array([r["target_valence"] for r in ur], dtype=np.float64)
        r_val = _signed_r(vals, targets)
        r_aro = _signed_r(aros, targets)

        # direction-correct: per emotion, does mean rating sign match expected sign?
        per_emo = defaultdict(list)
        for r in ur:
            per_emo[r["emotion"]].append(r["likert_valence"])
        dir_correct = 0
        means_by_emo = {}
        for emo, lst in per_emo.items():
            m = float(np.mean(lst))
            means_by_emo[emo] = m
            if np.sign(m) == np.sign(VALENCE_TARGET[emo]):
                dir_correct += 1

        summary.append({
            "ut": ut, "n": len(ur),
            "r_val_vs_target": r_val, "r_aro_vs_target": r_aro,
            "dir_correct": dir_correct, "n_emotions": len(per_emo),
            "means_by_emotion": means_by_emo,
        })
        print(
            f"  {str(ut):<6} {r_val:>+10.3f} {r_aro:>+14.3f} "
            f"{dir_correct:>10}/{len(per_emo)}"
        )

    print("\nPer-ut-step mean Likert valence by emotion:")
    print(f"  {'ut':<6} " + " ".join(f"{e:>10}" for e in EMOTIONS))
    for s in summary:
        means = s["means_by_emotion"]
        print(f"  {str(s['ut']):<6} " + " ".join(
            f"{means.get(e, float('nan')):>+10.2f}" for e in EMOTIONS
        ))

    (rd / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plot: r vs target per ut step.
    try:
        import matplotlib.pyplot as plt
        steps = [s["ut"] for s in summary if s["ut"] != "full"]
        rs = [s["r_val_vs_target"] for s in summary if s["ut"] != "full"]
        full_r = next((s["r_val_vs_target"] for s in summary if s["ut"] == "full"), None)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(steps, rs, marker="o", linewidth=2, color="C0", label="exit_at_step=N")
        if full_r is not None:
            ax.axhline(full_r, ls="--", color="C3", lw=0.8, label=f"full forward (r={full_r:+.3f})")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(steps)
        ax.set_xlabel("ut step (early-exit)")
        ax.set_ylabel("Likert valence r vs target valence")
        ax.set_title(f"Per-ut-step Likert — {nick}")
        ax.set_ylim(-0.2, 1.0)
        ax.legend()
        for x, y in zip(steps, rs):
            ax.text(x, y + 0.02, f"{y:+.2f}", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(rd / "per_ut_likert.png", dpi=140)
        print(f"\nSaved {rd / 'per_ut_likert.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
