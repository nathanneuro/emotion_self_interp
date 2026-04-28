"""Phase 1 extraction for RWKV-7 models via the rwkv pip package.

Mirrors scripts/extract_emotion_vectors.py but uses RWKV7Adapter, which has a
different forward signature (no HF tokenizer, no batch dim). One forward pass
per stimulus, last-token residual at every block.

Run:
    uv run python scripts/extract_emotion_vectors_rwkv.py \
        --model-path /media/drive2/projects2/self_awareness/sae_self_model/models/RWKV-x070-World-2.9B-v3-20250211-ctx4096
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must be set before any rwkv.model import.
os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.models.rwkv7_adapter import RWKV7Adapter  # noqa: E402
from src.probes.diff_means import diff_of_means, probe_separation  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default="/media/drive2/projects2/self_awareness/sae_self_model/models/RWKV-x070-World-2.9B-v3-20250211-ctx4096",
        help="Path to the .pth file *without* the .pth extension.",
    )
    ap.add_argument("--strategy", default="cuda bf16")
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print(f"Loading RWKV-7 model from {args.model_path}.pth (strategy={args.strategy!r}) ...")
    adapter = RWKV7Adapter.load(args.model_path, strategy=args.strategy)
    print(f"  family={adapter.family} n_layers={adapter.n_layers} d_model={adapter.d_model}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    print(f"  stimuli: {len(stims)} total")

    nick = Path(args.model_path).name
    rd = make_run_dir(
        f"phase1_vectors_{nick}",
        config={
            "model_path": args.model_path, "strategy": args.strategy,
            "n_layers": adapter.n_layers, "d_model": adapter.d_model,
            "per_cell": args.per_cell, "stimulus_set_size": len(stims),
            "framework": "rwkv-pip-0.8.32",
        },
    )
    print(f"  run dir: {rd}")

    layer_idxs = list(range(adapter.n_layers))
    prompts = [s.prompt for s in stims]
    print("Extracting last-token residuals across all layers (one prompt at a time) ...")
    acts: dict[int, list[torch.Tensor]] = {li: [] for li in layer_idxs}
    for p in tqdm(prompts):
        ids = adapter.encode(p)
        if len(ids) < 1:
            raise ValueError(f"empty token sequence for prompt {p!r}")
        cache = adapter.forward_with_residuals(ids, layer_idxs=layer_idxs)
        for li in layer_idxs:
            acts[li].append(cache[li][-1])  # last-token residual
    H_by_layer = {li: torch.stack(parts, dim=0).numpy() for li, parts in acts.items()}

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
            H = H_by_layer[li]
            v = diff_of_means(H[eu_rows], H[neutral_rows])
            vectors[emo][li] = v
            sep = probe_separation(H[nat_rows], H[neutral_rows], v)
            summary_rows.append({
                "emotion": emo, "layer": li,
                "n_eu": len(eu_rows), "n_nat": len(nat_rows), "n_neu": len(neutral_rows),
                "d_prime_nat_vs_neu": sep.d_prime,
                "auroc_nat_vs_neu": sep.auroc,
                "mean_proj_nat": sep.mean_pos, "mean_proj_neu": sep.mean_neg,
            })

    torch.save(
        {emo: {li: torch.from_numpy(v) for li, v in vs.items()} for emo, vs in vectors.items()},
        rd / "vectors.pt",
    )
    (rd / "summary.json").write_text(json.dumps(summary_rows, indent=2))

    best_per_emo = {}
    for emo in EMOTIONS:
        rows = [r for r in summary_rows if r["emotion"] == emo]
        best_per_emo[emo] = max(rows, key=lambda r: r["auroc_nat_vs_neu"])
    (rd / "best_layer_per_emotion.json").write_text(json.dumps(best_per_emo, indent=2))

    print("\nBest-separation layer per emotion (euphoric→naturalistic transfer):")
    print(f"  {'emotion':<10} {'layer':>5}  {'AUROC':>6}  {'d_prime':>8}")
    for emo, r in best_per_emo.items():
        print(f"  {emo:<10} {r['layer']:>5}  {r['auroc_nat_vs_neu']:>6.3f}  {r['d_prime_nat_vs_neu']:>8.3f}")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
            for emo in EMOTIONS:
                rows = [r for r in summary_rows if r["emotion"] == emo]
                rows.sort(key=lambda r: r["layer"])
                ax.plot([r["layer"] for r in rows], [r["auroc_nat_vs_neu"] for r in rows],
                        marker="o", label=emo)
            ax.axhline(0.5, color="gray", lw=0.7, ls="--")
            ax.set_xlabel("layer")
            ax.set_ylabel("AUROC (naturalistic vs neutral, on euphoric-fit direction)")
            ax.set_title(f"Layer sweep — {nick}")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(rd / "layer_sweep.png", dpi=140)
            print(f"\nSaved {rd / 'layer_sweep.png'}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
