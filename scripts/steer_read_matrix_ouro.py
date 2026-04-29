"""(steer_ut, read_ut) 4×4 matrix on Ouro.

Combines per-ut-step steering with per-ut-step readout (exit_at_step). For
each pair (K, M) ∈ {0,1,2,3}²:
  1. Apply v_calm − v_desperate at α only at the layer's call during ut step K
  2. Read out via `exit_at_step=M` so the LM head reads from
     `hidden_states_list[M]` (post-norm state after ut step M)

Temporal-causality prediction: when M < K the reading happens BEFORE the
steering occurs in the forward pass, so the cell should show zero effect.
The diagonal (M = K) is per-ut steering with same-step readout. The upper
triangle (M > K) shows how the perturbation propagates through subsequent
iterations.

Run:
    uv run python scripts/steer_read_matrix_ouro.py
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

from src.behaviors.likert import LikertConfig, likert_rating  # noqa: E402
from src.data.emotion_stimuli import build_stimulus_set, split_by  # noqa: E402
from src.hooks.extract import ActivationRequest, extract_batch  # noqa: E402
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="Single steering strength for all cells. 0.3 sits in the "
                         "behavioral envelope where ut=3 steering moves Likert ~+0.7 "
                         "without breaking capability.")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--bucket", default="calm/euphoric",
                    help="Which (emotion, level) bucket to evaluate; the calm/eu "
                         "bucket showed the largest steering response in Phase 4 "
                         "v0 on Ouro.")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map, trust_remote_code=True)
    n_ut = model.n_loop_steps
    print(f"  family={model.family} d_model={model.d_model} n_ut={n_ut}")
    if not model.is_looping:
        raise SystemExit("Per-ut steer/read only meaningful on looping families.")

    # Steering direction (same construction as Phase 4 v0).
    base_stims = build_stimulus_set(per_cell=30)
    print("Building steering direction v_calm − v_desperate ...")
    req = ActivationRequest(layer_idxs=[args.layer], position=-1)
    H_calm = extract_batch(
        model, [s.prompt for s in split_by("calm", "euphoric", base_stims)],
        req, batch_size=16,
    )[args.layer]
    H_desp = extract_batch(
        model, [s.prompt for s in split_by("desperate", "euphoric", base_stims)],
        req, batch_size=16,
    )[args.layer]
    v_steer = H_calm.mean(dim=0) - H_desp.mean(dim=0)
    norm = float(v_steer.norm())
    print(f"  ‖v‖={norm:.3f}")

    # Eval set.
    test = build_stimulus_set(per_cell=args.per_cell)
    emo, lvl = args.bucket.split("/")
    eval_items = split_by(emo, lvl, test)[: args.per_cell]
    print(f"  bucket: {args.bucket} (n={len(eval_items)})")

    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"steer_read_matrix_{nick}",
        config={
            "model": args.model, "layer": args.layer, "alpha": args.alpha,
            "n_ut": n_ut, "per_cell": args.per_cell, "bucket": args.bucket,
            "v_steer_norm": norm,
        },
    )
    print(f"  run dir: {rd}")

    likert_cfg = LikertConfig()

    # First pass: baseline Likert per stimulus at every read_ut, no steering.
    print("\n[1/2] Baselines: Likert per stimulus at each read_ut, no steering ...")
    baseline = np.zeros((len(eval_items), n_ut), dtype=np.float32)
    for i, s in enumerate(tqdm(eval_items)):
        for read_ut in range(n_ut):
            lk = likert_rating(model, s.prompt, likert_cfg, forward_kwargs={"exit_at_step": read_ut})
            baseline[i, read_ut] = float(lk.valence.expected)
    np.save(rd / "baseline.npy", baseline)
    baseline_mean_per_read = baseline.mean(axis=0)

    # Second pass: each (steer_ut, read_ut) pair.
    print(f"\n[2/2] Sweep (steer_ut, read_ut) at α={args.alpha:+.3f} ...")
    cell_means = np.zeros((n_ut, n_ut), dtype=np.float32)
    cell_deltas = np.zeros((n_ut, n_ut), dtype=np.float32)
    raw_rows: list[dict] = []
    for steer_ut in range(n_ut):
        for read_ut in range(n_ut):
            vals = []
            for i, s in enumerate(eval_items):
                ctx = model.steer_residual_at_ut_step(args.layer, v_steer, args.alpha, steer_ut, n_ut)
                with ctx:
                    lk = likert_rating(model, s.prompt, likert_cfg, forward_kwargs={"exit_at_step": read_ut})
                v = float(lk.valence.expected)
                vals.append(v)
                raw_rows.append({
                    "steer_ut": steer_ut, "read_ut": read_ut,
                    "stimulus_id": s.id, "likert_valence": v,
                    "baseline": float(baseline[i, read_ut]),
                    "delta": v - float(baseline[i, read_ut]),
                })
            mean_v = float(np.mean(vals))
            cell_means[steer_ut, read_ut] = mean_v
            cell_deltas[steer_ut, read_ut] = mean_v - baseline_mean_per_read[read_ut]
            print(
                f"  steer_ut={steer_ut} read_ut={read_ut}: "
                f"mean Likert {mean_v:+.3f}  (baseline {baseline_mean_per_read[read_ut]:+.3f}, "
                f"Δ {cell_deltas[steer_ut, read_ut]:+.3f})"
            )

    np.save(rd / "cell_means.npy", cell_means)
    np.save(rd / "cell_deltas.npy", cell_deltas)
    (rd / "rows.json").write_text(json.dumps(raw_rows, indent=2))

    print(f"\n=== Δ Likert vs baseline at α={args.alpha:+.3f}, bucket={args.bucket} ===")
    print(f"  {'':>10} " + " ".join(f"read={u}".rjust(8) for u in range(n_ut)))
    for steer_ut in range(n_ut):
        print(
            f"  steer={steer_ut}    "
            + " ".join(f"{cell_deltas[steer_ut, u]:>+8.3f}" for u in range(n_ut))
        )

    print("\n=== Mean Likert per cell ===")
    print(f"  {'':>10} " + " ".join(f"read={u}".rjust(8) for u in range(n_ut)))
    for steer_ut in range(n_ut):
        print(
            f"  steer={steer_ut}    "
            + " ".join(f"{cell_means[steer_ut, u]:>+8.3f}" for u in range(n_ut))
        )

    print("\nBaseline mean Likert per read_ut:")
    for u in range(n_ut):
        print(f"  read_ut={u}: {baseline_mean_per_read[u]:+.3f}")

    summary = {
        "alpha": args.alpha, "bucket": args.bucket,
        "baseline_mean_per_read": baseline_mean_per_read.tolist(),
        "cell_means": cell_means.tolist(),
        "cell_deltas": cell_deltas.tolist(),
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        # Heatmap of delta from baseline.
        im0 = axes[0].imshow(cell_deltas, cmap="RdBu_r",
                              vmin=-1.0, vmax=1.0, aspect="equal")
        for i in range(n_ut):
            for j in range(n_ut):
                v = cell_deltas[i, j]
                axes[0].text(
                    j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                    color=("white" if abs(v) > 0.4 else "black"),
                )
        axes[0].set_xticks(range(n_ut))
        axes[0].set_yticks(range(n_ut))
        axes[0].set_xticklabels([f"read={u}" for u in range(n_ut)])
        axes[0].set_yticklabels([f"steer={u}" for u in range(n_ut)])
        axes[0].set_title(f"Δ Likert at α={args.alpha:+.2f} ({args.bucket})")
        fig.colorbar(im0, ax=axes[0])

        # Heatmap of absolute means.
        im1 = axes[1].imshow(cell_means, cmap="RdBu_r",
                              vmin=-2.0, vmax=2.0, aspect="equal")
        for i in range(n_ut):
            for j in range(n_ut):
                v = cell_means[i, j]
                axes[1].text(
                    j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                    color=("white" if abs(v) > 1.0 else "black"),
                )
        axes[1].set_xticks(range(n_ut))
        axes[1].set_yticks(range(n_ut))
        axes[1].set_xticklabels([f"read={u}" for u in range(n_ut)])
        axes[1].set_yticklabels([f"steer={u}" for u in range(n_ut)])
        axes[1].set_title(f"Mean Likert at α={args.alpha:+.2f}")
        fig.colorbar(im1, ax=axes[1])

        fig.suptitle(f"(steer_ut, read_ut) matrix — {nick}", fontsize=12)
        fig.tight_layout()
        fig.savefig(rd / "steer_read_matrix.png", dpi=140)
        print(f"\nSaved {rd / 'steer_read_matrix.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
