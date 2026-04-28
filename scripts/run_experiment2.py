"""Phase 6 / Experiment 2: bias-prior decomposition.

For each adapter variant (bias_only, scale_only, scalar_affine, full_rank),
train on euphoric stimuli at the canonical layer, then run three diagnostics
on the naturalistic held-out set:

    1. held_out_top1            6-class accuracy
    2. zero_vector_pred         what the adapter says when h = 0
    3. shuffle_top1             accuracy when test residuals are
                                permuted vs their labels

If the adapter is mostly a format prior:
    - zero_vector_pred mode-class equals the held-out modal prediction
    - shuffle_top1 ≈ held_out_top1 (input doesn't matter)

If the adapter is activation-conditional:
    - zero_vector predictions diverge from the held-out distribution
    - shuffle_top1 collapses well below held_out_top1

Run:
    uv run python scripts/run_experiment2.py
    uv run python scripts/run_experiment2.py --model google/gemma-2-2b --layer 8
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.adapters.train import TrainConfig, TrainExample, train_adapter  # noqa: E402
from src.adapters.scalar_affine import AdapterConfig, make_adapter  # noqa: E402
from src.experiments import (  # noqa: E402
    evaluate_adapter_bias_prior,
    extract_stimulus_residuals,
    summarize_reports,
)
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--variants", nargs="*",
                    default=["bias_only", "scale_only", "scalar_affine", "full_rank"])
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map)
    print(f"  family={model.family} d_model={model.d_model}")

    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"phase6_exp2_{nick}",
        config={
            "model": args.model, "layer": args.layer, "per_cell": args.per_cell,
            "epochs": args.epochs, "lr": args.lr, "variants": args.variants,
        },
    )
    print(f"  run dir: {rd}")

    print("\nExtracting per-prompt residuals at canonical layer ...")
    res = extract_stimulus_residuals(model, layer=args.layer, per_cell=args.per_cell)
    d_model = res.d_model

    train_examples: list[TrainExample] = []
    test_residuals: list[torch.Tensor] = []
    test_labels: list[str] = []
    for i, s in enumerate(res.stimuli):
        if s.emotion == "neutral":
            continue
        if s.level == "euphoric":
            train_examples.append(TrainExample(vector=res.residuals[i].clone(), label=s.emotion))
        elif s.level == "naturalistic":
            test_residuals.append(res.residuals[i].clone())
            test_labels.append(s.emotion)
    test_H = torch.stack(test_residuals, dim=0)
    print(f"  train: {len(train_examples)} euphoric  test: {len(test_labels)} naturalistic")

    cfg = TrainConfig(
        layer_idx=args.layer, batch_size=8,
        n_epochs=args.epochs, learning_rate=args.lr,
    )

    reports = []
    for kind in args.variants:
        print(f"\n=== {kind} ===")
        adapter = make_adapter(AdapterConfig(kind=kind, d_model=d_model)).to(model.device)
        history = train_adapter(model, adapter, train_examples, val=None, cfg=cfg)
        print(f"  final train top1: {history['train_acc'][-1]:.3f}")

        report = evaluate_adapter_bias_prior(
            model=model, adapter=adapter, layer=args.layer,
            test_residuals=test_H, test_labels=test_labels,
        )
        print(f"  held-out top1: {report.held_out_top1:.3f}")
        print(f"  zero-vector pred: {report.zero_vector_pred}")
        print(f"  shuffle top1: {report.shuffle_top1:.3f}  (Δ vs held-out: "
              f"{report.shuffle_top1 - report.held_out_top1:+.3f})")
        reports.append(report)

        torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                   rd / f"adapter_{kind}.pt")

    summary = summarize_reports(reports)
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    (rd / "reports.json").write_text(json.dumps([asdict(r) for r in reports], indent=2))

    print("\n=== Bias-prior decomposition (naturalistic held-out) ===")
    print(f"  {'variant':<14} {'held_out':>10} {'shuffle':>10} {'Δ':>8} {'zero→':>14}")
    for r in reports:
        delta = r.shuffle_top1 - r.held_out_top1
        print(
            f"  {r.adapter_kind:<14} {r.held_out_top1:>10.3f} "
            f"{r.shuffle_top1:>10.3f} {delta:>+8.3f} {r.zero_vector_pred:>14}"
        )
    print(f"\nSaved {rd}")


if __name__ == "__main__":
    main()
