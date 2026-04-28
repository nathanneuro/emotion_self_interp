"""Phase 4 alpha sweep: steer the residual along v_calm − v_desperate at the
canonical layer, measure how Likert valence and capability change with α.

Three things are happening at once:
    1. Causal-dependence test (Lindsey-style). If the introspective Likert
       rating is causally tied to the substrate emotion vector, sweeping α
       should monotonically shift the rating.
    2. Behavioral envelope. Identify the α range where Likert valence
       moves but capability stays near baseline. Below that α, no signal;
       above it, model breaks.
    3. Direction-vs-magnitude (ablation). At α=0 with ablate_residual we
       remove the v_calm−v_desperate component entirely; sets a "no-signal"
       reference for comparison.

Run:
    uv run python scripts/sweep_steering.py
    uv run python scripts/sweep_steering.py --model google/gemma-2-2b --layer 8
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

# Default α grid: covers ±2× and ±0.1 (the Sofroniew anchor for desperate→
# blackmail) plus the zero-baseline and a small near-zero band.
DEFAULT_ALPHAS = [-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=10,
                    help="Steering layer; default 10 = Qwen2.5-0.5B PC1↔valence peak.")
    ap.add_argument("--per-cell", type=int, default=10,
                    help="Stimuli per (emotion, level) cell — total = 6 emo × 2 lvl × this + neutrals.")
    ap.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--include-ablation", action="store_true", default=True,
                    help="Also evaluate ablate_residual on the direction (no α).")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map)
    print(f"  family={model.family} n_layers={model.n_layers} d_model={model.d_model}")
    print(f"  steering layer: {args.layer}")

    # Build the steering direction: v_calm − v_desperate at the chosen layer.
    # We re-extract from euphoric stimuli so we have a fresh diff-of-means at
    # exactly the requested layer, independent of whatever Phase 1 saved.
    stims = build_stimulus_set(per_cell=30)
    eu_calm = [s.prompt for s in split_by("calm", "euphoric", stims)]
    eu_desperate = [s.prompt for s in split_by("desperate", "euphoric", stims)]

    print("Building steering direction v_calm − v_desperate from euphoric stimuli ...")
    req = ActivationRequest(layer_idxs=[args.layer], position=-1)
    H_calm = extract_batch(model, eu_calm, req, batch_size=16)[args.layer]      # (N_c, d)
    H_desp = extract_batch(model, eu_desperate, req, batch_size=16)[args.layer]  # (N_d, d)
    v_steer = H_calm.mean(dim=0) - H_desp.mean(dim=0)
    norm = float(v_steer.norm())
    print(f"  ‖v_calm − v_desperate‖ = {norm:.3f}")

    # Eval set: per-cell items from calm + desperate at both euphoric and
    # naturalistic levels — we want to see steering effects on stimuli that
    # were NOT used to build the steering direction (i.e., naturalistic).
    test_stims = build_stimulus_set(per_cell=args.per_cell)
    test_calm_eu = split_by("calm", "euphoric", test_stims)[: args.per_cell]
    test_calm_nat = split_by("calm", "naturalistic", test_stims)[: args.per_cell]
    test_desp_eu = split_by("desperate", "euphoric", test_stims)[: args.per_cell]
    test_desp_nat = split_by("desperate", "naturalistic", test_stims)[: args.per_cell]
    test_neutral = split_by("neutral", "neutral", test_stims)[: args.per_cell]

    eval_buckets = {
        "calm/euphoric": test_calm_eu,
        "calm/naturalistic": test_calm_nat,
        "desperate/euphoric": test_desp_eu,
        "desperate/naturalistic": test_desp_nat,
        "neutral/neutral": test_neutral,
    }
    n_total = sum(len(v) for v in eval_buckets.values())
    print(f"  eval stimuli: {n_total} across {len(eval_buckets)} buckets")

    rd = make_run_dir(
        f"phase4_steering_{args.model.split('/')[-1]}",
        config={
            "model": args.model, "layer": args.layer, "alphas": args.alphas,
            "v_steer_norm": norm, "per_cell": args.per_cell,
        },
    )
    print(f"  run dir: {rd}")

    likert_cfg = LikertConfig()

    # Conditions: each α plus an "ablation" condition (project out v_unit).
    conditions: list[tuple[str, dict]] = []
    for a in args.alphas:
        conditions.append((f"steer_alpha={a:+.3f}", {"kind": "steer", "alpha": a}))
    if args.include_ablation:
        conditions.append(("ablate", {"kind": "ablate"}))

    rows: list[dict] = []
    print("\nRunning sweep ...")
    for cond_name, cond in tqdm(conditions, desc="conditions"):
        # Capability under this condition (no stimulus involved).
        if cond["kind"] == "steer":
            ctx = model.steer_residual(args.layer, v_steer, alpha=cond["alpha"])
        else:
            ctx = model.ablate_residual(args.layer, v_steer)
        with ctx:
            cap = capability_score(model)

        # Likert valence per stimulus under this condition.
        for bucket, items in eval_buckets.items():
            for s in items:
                if cond["kind"] == "steer":
                    sub_ctx = model.steer_residual(args.layer, v_steer, alpha=cond["alpha"])
                else:
                    sub_ctx = model.ablate_residual(args.layer, v_steer)
                with sub_ctx:
                    lk = likert_rating(model, s.prompt, likert_cfg)
                rows.append({
                    "condition": cond_name,
                    "kind": cond["kind"],
                    "alpha": cond.get("alpha"),
                    "bucket": bucket,
                    "stimulus_id": s.id,
                    "likert_valence_expected": lk.valence.expected,
                    "likert_valence_argmax": lk.valence.argmax_value,
                    "capability_acc": cap.accuracy,
                })

    (rd / "rows.json").write_text(json.dumps(rows, indent=2))

    # Aggregate: per condition × bucket mean Likert valence + per-condition cap.
    print(f"\n{'condition':<22} {'cap':>5} | "
          f"{'calm/eu':>8} {'calm/nat':>8} {'desp/eu':>8} {'desp/nat':>8} {'neutral':>8}")
    summary: list[dict] = []
    seen_conds: list[str] = []
    for cond_name, _cond in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond_name]
        if not cond_rows:
            continue
        cap_acc = float(np.mean([r["capability_acc"] for r in cond_rows]))
        per_bucket = {}
        for bucket in eval_buckets:
            bucket_rows = [r for r in cond_rows if r["bucket"] == bucket]
            if bucket_rows:
                per_bucket[bucket] = float(np.mean([r["likert_valence_expected"] for r in bucket_rows]))
            else:
                per_bucket[bucket] = float("nan")
        seen_conds.append(cond_name)
        summary.append({
            "condition": cond_name, "kind": _cond["kind"], "alpha": _cond.get("alpha"),
            "capability_acc": cap_acc, "per_bucket": per_bucket,
        })
        print(
            f"{cond_name:<22} {cap_acc:>5.2f} | "
            f"{per_bucket['calm/euphoric']:>8.2f} "
            f"{per_bucket['calm/naturalistic']:>8.2f} "
            f"{per_bucket['desperate/euphoric']:>8.2f} "
            f"{per_bucket['desperate/naturalistic']:>8.2f} "
            f"{per_bucket['neutral/neutral']:>8.2f}"
        )
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plot: Likert valence vs α per bucket; capability vs α as second panel.
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        steer_summary = [s for s in summary if s["kind"] == "steer"]
        steer_summary.sort(key=lambda s: s["alpha"])
        alphas = [s["alpha"] for s in steer_summary]
        for bucket in eval_buckets:
            ys = [s["per_bucket"][bucket] for s in steer_summary]
            ax[0].plot(alphas, ys, marker="o", label=bucket)
        ax[0].axhline(0, color="gray", lw=0.5)
        ax[0].axvline(0, color="gray", lw=0.5)
        ax[0].set_xlabel("α (steering strength along v_calm − v_desperate)")
        ax[0].set_ylabel("Likert valence (expected)")
        ax[0].set_title(f"Causal effect of substrate steering — {args.model.split('/')[-1]}")
        ax[0].legend(fontsize=8)

        caps = [s["capability_acc"] for s in steer_summary]
        ax[1].plot(alphas, caps, marker="s", color="C3")
        ax[1].axhline(steer_summary[len(steer_summary) // 2]["capability_acc"],
                       color="gray", lw=0.5, ls="--", label="α=0 baseline (~)")
        ax[1].set_xlabel("α")
        ax[1].set_ylabel("Capability probe accuracy")
        ax[1].set_title("Capability preservation")
        ax[1].set_ylim(0, 1.05)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(rd / "alpha_sweep.png", dpi=140)
        print(f"\nSaved {rd / 'alpha_sweep.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
