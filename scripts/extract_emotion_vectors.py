"""Phase 1: extract per-emotion direction vectors and run a layer sweep.

For each layer of the chosen model, we
  1. Extract last-token residuals for the v0 stimulus set.
  2. Build a diff-of-means direction per emotion vs. the neutral set, using the
     'euphoric'-level prompts as the in-class set (concentrated emotion signal).
  3. Score each direction's separation on a held-out partition of the
     'naturalistic' stimuli — these were *never* used to fit the direction, so
     transfer from euphoric → naturalistic is a real out-of-distribution test.
  4. Save vectors, scores, and a summary CSV. Also save a layer-sweep figure.

Run:
    uv run python scripts/extract_emotion_vectors.py
    uv run python scripts/extract_emotion_vectors.py --model meta-llama/Llama-3.2-1B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set, split_by  # noqa: E402
from src.hooks.extract import ActivationRequest, extract_batch  # noqa: E402
from src.models.adapter import ModelAdapter  # noqa: E402
from src.probes.diff_means import diff_of_means, probe_separation  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--position", default=-1, help="int or 'last_real'")
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    position: int | str = int(args.position) if str(args.position).lstrip("-").isdigit() else args.position

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map)
    print(f"  family={model.family} n_layers={model.n_layers} d_model={model.d_model}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    print(f"  stimuli: {len(stims)} total ({len(EMOTIONS)} emotions × 2 levels + neutral)")

    layer_idxs = list(range(model.n_layers))
    rd = make_run_dir(
        f"phase1_vectors_{args.model.split('/')[-1]}",
        config={
            "model": args.model, "n_layers": model.n_layers, "d_model": model.d_model,
            "position": position, "per_cell": args.per_cell, "batch_size": args.batch_size,
            "stimulus_set_size": len(stims),
        },
    )
    print(f"  run dir: {rd}")

    # One forward pass per stimulus, caching every layer.
    print("Extracting activations across all layers ...")
    prompts = [s.prompt for s in stims]
    req = ActivationRequest(layer_idxs=layer_idxs, position=position)
    acts = extract_batch(model, prompts, req, batch_size=args.batch_size)
    # acts[layer]: (N, d_model) cpu float32

    # Index stimuli by (emotion, level).
    rows_by_key: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(stims):
        rows_by_key.setdefault((s.emotion, s.level), []).append(i)

    neutral_rows = rows_by_key[("neutral", "neutral")]
    summary_rows: list[dict] = []
    vectors: dict[str, dict[int, np.ndarray]] = {emo: {} for emo in EMOTIONS}

    for emo in EMOTIONS:
        eu_rows = rows_by_key[(emo, "euphoric")]
        nat_rows = rows_by_key[(emo, "naturalistic")]
        for li in layer_idxs:
            H = acts[li].numpy()
            v = diff_of_means(H[eu_rows], H[neutral_rows])
            vectors[emo][li] = v
            sep = probe_separation(H[nat_rows], H[neutral_rows], v)
            summary_rows.append({
                "emotion": emo, "layer": li,
                "n_eu": len(eu_rows), "n_nat": len(nat_rows), "n_neu": len(neutral_rows),
                "d_prime_nat_vs_neu": sep.d_prime,
                "auroc_nat_vs_neu": sep.auroc,
                "mean_proj_nat": sep.mean_pos,
                "mean_proj_neu": sep.mean_neg,
            })

    # Persist.
    torch.save(
        {emo: {li: torch.from_numpy(v) for li, v in vs.items()} for emo, vs in vectors.items()},
        rd / "vectors.pt",
    )
    (rd / "summary.json").write_text(json.dumps(summary_rows, indent=2))

    # Pick best-separation layer per emotion (transfer-evaluated).
    best_per_emo: dict[str, dict] = {}
    for emo in EMOTIONS:
        rows = [r for r in summary_rows if r["emotion"] == emo]
        best = max(rows, key=lambda r: r["auroc_nat_vs_neu"])
        best_per_emo[emo] = best
    (rd / "best_layer_per_emotion.json").write_text(json.dumps(best_per_emo, indent=2))

    print("\nBest-separation layer per emotion (euphoric→naturalistic transfer):")
    print(f"  {'emotion':<10} {'layer':>5}  {'AUROC':>6}  {'d_prime':>8}")
    for emo, r in best_per_emo.items():
        print(f"  {emo:<10} {r['layer']:>5}  {r['auroc_nat_vs_neu']:>6.3f}  {r['d_prime_nat_vs_neu']:>8.3f}")

    if not args.no_plot:
        try:
            import matplotlib  # noqa: F401
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
            for emo in EMOTIONS:
                rows = [r for r in summary_rows if r["emotion"] == emo]
                rows.sort(key=lambda r: r["layer"])
                xs = [r["layer"] for r in rows]
                ys = [r["auroc_nat_vs_neu"] for r in rows]
                ax.plot(xs, ys, marker="o", label=emo)
            ax.axhline(0.5, color="gray", lw=0.7, ls="--")
            ax.set_xlabel("layer")
            ax.set_ylabel("AUROC (naturalistic vs neutral, on euphoric-fit direction)")
            ax.set_title(f"Layer sweep — {args.model}")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(rd / "layer_sweep.png", dpi=140)
            print(f"\nSaved layer sweep plot to {rd / 'layer_sweep.png'}")
        except ImportError:
            print("\n(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
