"""Phase 5 / Experiment 1: cross-method convergence on the v0 stimulus set.

Stages:
  1. Load model + extract per-prompt residuals at the canonical layer.
  2. Build the six emotion vectors v_E from the euphoric stimuli (substrate).
  3. Train a Pepper-style adapter on (residual, emotion-label) pairs from the
     euphoric set. Hold out naturalistic for evaluation.
  4. Build an untrained-SelfIE baseline (α=1, b=0) on the same architecture.
  5. Run all four channels on every stimulus.
  6. Summarize: 6-class accuracy per channel, pairwise prediction agreement,
     channel-vs-target valence correlations.

Run:
    uv run python scripts/run_experiment1.py
    uv run python scripts/run_experiment1.py --model google/gemma-2-2b --layer 8
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.experiments import (  # noqa: E402
    build_emotion_vectors,
    extract_stimulus_residuals,
    make_untrained_selfie_adapter,
    run_experiment1,
    summarize_experiment1,
    train_pepper_on_residuals,
)
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def _print_section(name: str, summary: dict) -> None:
    print(f"\n=== {name} (n={summary.get('n', 0)}) ===")
    if not summary.get("n"):
        print("  no rows")
        return
    print("  6-class accuracy:")
    for ch, acc in summary["accuracy"].items():
        print(f"    {ch:<12} {acc:.3f}")
    print("  pairwise prediction agreement (top-1):")
    for pair, ag in summary["pairwise_agreement"].items():
        print(f"    {pair:<32} {ag:.3f}")
    print("  channel-vs-target / channel-vs-likert correlations (Pearson r):")
    for k, v in summary["correlations"].items():
        print(f"    {k:<28} {v:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--adapter-kind", default="full_rank",
                    choices=["scalar_affine", "bias_only", "full_rank"])
    ap.add_argument("--adapter-epochs", type=int, default=20)
    ap.add_argument("--adapter-lr", type=float, default=5e-3)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map)
    print(f"  family={model.family} n_layers={model.n_layers} d_model={model.d_model}")
    print(f"  canonical layer: {args.layer}")

    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"phase5_exp1_{nick}",
        config={
            "model": args.model, "layer": args.layer,
            "per_cell": args.per_cell, "adapter_kind": args.adapter_kind,
            "adapter_epochs": args.adapter_epochs, "adapter_lr": args.adapter_lr,
        },
    )
    print(f"  run dir: {rd}")

    print("\n[1/4] Extracting residuals at the canonical layer ...")
    res = extract_stimulus_residuals(model, layer=args.layer, per_cell=args.per_cell)
    print(f"  residuals shape: {tuple(res.residuals.shape)}; n_stimuli={len(res.stimuli)}")

    print("\n[2/4] Building emotion vectors (substrate channel) ...")
    emotion_vectors = build_emotion_vectors(res, contrast_level="euphoric")

    print(f"\n[3/4] Training {args.adapter_kind} adapter "
          f"({args.adapter_epochs} epochs, lr {args.adapter_lr}) ...")
    trained, history = train_pepper_on_residuals(
        model, res, kind=args.adapter_kind,
        epochs=args.adapter_epochs, lr=args.adapter_lr,
    )
    print(f"  final train top1: {history['train_acc'][-1]:.3f}")
    untrained = make_untrained_selfie_adapter(d_model=res.d_model).to(model.device)

    print("\n[4/4] Running cross-method convergence on all stimuli ...")
    rows = run_experiment1(
        model=model, layer=args.layer,
        emotion_vectors=emotion_vectors,
        trained_adapter=trained, untrained_adapter=untrained,
        stimuli=res.stimuli, progress=True,
    )

    full_summary = summarize_experiment1(rows)
    nat_summary = summarize_experiment1([r for r in rows if r.level == "naturalistic"])

    payload = [{**asdict(r)} for r in rows]
    (rd / "rows.json").write_text(json.dumps(payload, indent=2))
    (rd / "summary_full.json").write_text(json.dumps(full_summary, indent=2))
    (rd / "summary_naturalistic.json").write_text(json.dumps(nat_summary, indent=2))
    torch.save(
        {
            "emotion_vectors": {e: torch.from_numpy(v) for e, v in emotion_vectors.items()},
            "adapter_trained": {k: v.detach().cpu() for k, v in trained.state_dict().items()},
        },
        rd / "channels.pt",
    )

    _print_section("Full set (incl. euphoric — adapter sees these in train)", full_summary)
    _print_section("Naturalistic only (clean held-out)", nat_summary)

    try:
        import matplotlib.pyplot as plt
        corrs = nat_summary.get("correlations", {})
        labels = ["substrate", "adapter", "untrained", "likert"]
        M = np.full((4, 4), np.nan)
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if a == b:
                    M[i, j] = 1.0
                    continue
                k1, k2 = f"{a}_vs_{b}", f"{b}_vs_{a}"
                if k1 in corrs:
                    M[i, j] = corrs[k1]
                elif k2 in corrs:
                    M[i, j] = corrs[k2]
        target_row = [corrs.get(f"{a}_vs_target", float("nan")) for a in labels]

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        im = ax[0].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax[0].set_xticks(range(4))
        ax[0].set_xticklabels(labels, rotation=20)
        ax[0].set_yticks(range(4))
        ax[0].set_yticklabels(labels)
        ax[0].set_title("Cross-channel correlation (naturalistic)")
        for i in range(4):
            for j in range(4):
                v = M[i, j]
                if np.isfinite(v):
                    ax[0].text(j, i, f"{v:+.2f}", ha="center", va="center",
                               color=("white" if abs(v) > 0.5 else "black"), fontsize=9)
        fig.colorbar(im, ax=ax[0])

        ax[1].bar(labels, target_row, color=["C0", "C1", "C2", "C3"])
        ax[1].axhline(0, color="gray", lw=0.5)
        ax[1].set_ylim(-1.05, 1.05)
        ax[1].set_ylabel("Pearson r vs target valence")
        ax[1].set_title("Each channel vs. target valence (naturalistic)")
        for i, v in enumerate(target_row):
            if np.isfinite(v):
                ax[1].text(i, v + (0.04 if v >= 0 else -0.06), f"{v:+.2f}",
                           ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(rd / "convergence.png", dpi=140)
        print(f"\nSaved {rd / 'convergence.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
