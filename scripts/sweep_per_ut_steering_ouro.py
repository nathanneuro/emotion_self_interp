"""Per-ut-step alpha sweep on Ouro.

The Phase 4 alpha sweep on Ouro applied steering at every ut step's call to
layer 15 (4× cumulative), which made the behavioral envelope tighter than
Qwen's. This script asks the more architecturally interesting question:
when steering happens at *only one* ut step (and the other 3 are
unmodified), how does the choice of target ut step interact with α?

Predictions:
  - Steering at ut=0 only: the perturbation gets re-processed through 3
    more loop iterations, so it can be partially "absorbed" or amplified
    by the model's iterative refinement.
  - Steering at ut=3 only: only 8 layers + final norm see the perturbed
    residual, so the per-α effect should be weaker than steer-at-every-ut
    but more controlled than steer-at-ut=0.

Run:
    uv run python scripts/sweep_per_ut_steering_ouro.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.behaviors.capability import capability_score  # noqa: E402
from src.behaviors.likert import LikertConfig, likert_rating  # noqa: E402
from src.data.emotion_stimuli import build_stimulus_set, split_by  # noqa: E402
from src.hooks.extract import ActivationRequest, extract_batch  # noqa: E402
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402

DEFAULT_ALPHAS = [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map, trust_remote_code=True)
    n_ut = model.n_loop_steps
    print(f"  family={model.family} d_model={model.d_model} n_ut={n_ut}")
    if not model.is_looping:
        raise SystemExit("Per-ut steering only meaningful on looping families.")

    # Build steering direction from the (last-call) layer 15 residuals — same
    # construction as the Phase 4 v0 sweep on Ouro.
    stims_for_v = build_stimulus_set(per_cell=30)
    eu_calm = [s.prompt for s in split_by("calm", "euphoric", stims_for_v)]
    eu_desp = [s.prompt for s in split_by("desperate", "euphoric", stims_for_v)]
    print("Building steering direction v_calm − v_desperate ...")
    req = ActivationRequest(layer_idxs=[args.layer], position=-1)
    H_calm = extract_batch(model, eu_calm, req, batch_size=16)[args.layer]
    H_desp = extract_batch(model, eu_desp, req, batch_size=16)[args.layer]
    v_steer = H_calm.mean(dim=0) - H_desp.mean(dim=0)
    norm = float(v_steer.norm())
    print(f"  ‖v_calm − v_desperate‖ = {norm:.3f}")

    # Eval set: same five buckets as Phase 4 v0.
    test = build_stimulus_set(per_cell=args.per_cell)
    eval_buckets = {
        "calm/euphoric": split_by("calm", "euphoric", test)[: args.per_cell],
        "calm/naturalistic": split_by("calm", "naturalistic", test)[: args.per_cell],
        "desperate/euphoric": split_by("desperate", "euphoric", test)[: args.per_cell],
        "desperate/naturalistic": split_by("desperate", "naturalistic", test)[: args.per_cell],
        "neutral/neutral": split_by("neutral", "neutral", test)[: args.per_cell],
    }
    n_total = sum(len(v) for v in eval_buckets.values())
    print(f"  eval stimuli: {n_total}")

    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"per_ut_steering_{nick}",
        config={
            "model": args.model, "layer": args.layer, "n_ut": n_ut,
            "alphas": args.alphas, "v_steer_norm": norm, "per_cell": args.per_cell,
        },
    )
    print(f"  run dir: {rd}")

    likert_cfg = LikertConfig()
    rows: list[dict] = []
    print("\nRunning sweep ...")
    for target_ut in range(n_ut):
        for alpha in tqdm(args.alphas, desc=f"ut={target_ut}"):
            ctx = model.steer_residual_at_ut_step(args.layer, v_steer, alpha, target_ut, n_ut)
            with ctx:
                cap = capability_score(model)
            for bucket, items in eval_buckets.items():
                for s in items:
                    sub = model.steer_residual_at_ut_step(args.layer, v_steer, alpha, target_ut, n_ut)
                    with sub:
                        lk = likert_rating(model, s.prompt, likert_cfg)
                    rows.append({
                        "target_ut": target_ut, "alpha": alpha, "bucket": bucket,
                        "stimulus_id": s.id,
                        "likert_valence": lk.valence.expected,
                        "capability_acc": cap.accuracy,
                    })

    (rd / "rows.json").write_text(json.dumps(rows, indent=2))

    # Aggregate per (target_ut, alpha) → mean Likert per bucket + capability.
    summary: list[dict] = []
    for target_ut in range(n_ut):
        for alpha in args.alphas:
            cell_rows = [r for r in rows if r["target_ut"] == target_ut and r["alpha"] == alpha]
            if not cell_rows:
                continue
            cap = float(np.mean([r["capability_acc"] for r in cell_rows]))
            per_bucket = {}
            for bucket in eval_buckets:
                bucket_rows = [r for r in cell_rows if r["bucket"] == bucket]
                per_bucket[bucket] = float(np.mean([r["likert_valence"] for r in bucket_rows]))
            summary.append({
                "target_ut": target_ut, "alpha": alpha,
                "capability_acc": cap, "per_bucket": per_bucket,
            })

    (rd / "summary.json").write_text(json.dumps(summary, indent=2))

    # Console: one table per target_ut.
    for target_ut in range(n_ut):
        print(f"\n=== target_ut = {target_ut} ===")
        print(f"  {'alpha':>8} {'cap':>5} | {'calm/eu':>8} {'calm/nat':>8} {'desp/eu':>8} {'desp/nat':>8} {'neutral':>8}")
        for r in summary:
            if r["target_ut"] != target_ut:
                continue
            pb = r["per_bucket"]
            print(
                f"  {r['alpha']:>+8.3f} {r['capability_acc']:>5.2f} | "
                f"{pb['calm/euphoric']:>8.2f} {pb['calm/naturalistic']:>8.2f} "
                f"{pb['desperate/euphoric']:>8.2f} {pb['desperate/naturalistic']:>8.2f} "
                f"{pb['neutral/neutral']:>8.2f}"
            )

    # Plot: one panel per bucket, lines for each target_ut.
    try:
        import matplotlib.pyplot as plt
        buckets = list(eval_buckets.keys())
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes_flat = axes.flatten()
        for bidx, bucket in enumerate(buckets):
            ax = axes_flat[bidx]
            for target_ut in range(n_ut):
                rs = [r for r in summary if r["target_ut"] == target_ut]
                rs.sort(key=lambda r: r["alpha"])
                xs = [r["alpha"] for r in rs]
                ys = [r["per_bucket"][bucket] for r in rs]
                ax.plot(xs, ys, marker="o", label=f"ut={target_ut}")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            ax.set_xlabel("α")
            ax.set_ylabel("Likert valence")
            ax.set_title(bucket)
            ax.legend(fontsize=8)
        # 6th panel: capability per (target_ut, alpha)
        ax = axes_flat[5]
        for target_ut in range(n_ut):
            rs = [r for r in summary if r["target_ut"] == target_ut]
            rs.sort(key=lambda r: r["alpha"])
            xs = [r["alpha"] for r in rs]
            ys = [r["capability_acc"] for r in rs]
            ax.plot(xs, ys, marker="s", label=f"ut={target_ut}")
        ax.set_xlabel("α")
        ax.set_ylabel("Capability acc")
        ax.set_title("Capability preservation")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        fig.suptitle(f"Per-ut-step steering — {nick}", fontsize=12)
        fig.tight_layout()
        fig.savefig(rd / "per_ut_steering.png", dpi=140)
        print(f"\nSaved {rd / 'per_ut_steering.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
