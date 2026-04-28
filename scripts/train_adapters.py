"""Phase 2 tiny end-to-end training: fit all three adapter variants
(scalar-affine, bias-only, full-rank) on emotion vectors from Phase 1.

Per-prompt residuals at the canonical layer become the (vector, label)
training pairs. We hold out the naturalistic stimuli for evaluation, so
the train ↔ eval split mirrors the off-policy → on-policy generalization
the program cares about.

Run:
    uv run python scripts/train_adapters.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.adapters.scalar_affine import AdapterConfig, make_adapter  # noqa: E402
from src.adapters.train import (  # noqa: E402
    TrainConfig,
    TrainExample,
    evaluate_adapter,
    train_adapter,
)
from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.hooks.extract import ActivationRequest, extract_batch  # noqa: E402
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=10,
                    help="Probe layer (Qwen2.5-0.5B PC1↔valence peak is L10).")
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map)
    print(f"  family={model.family} n_layers={model.n_layers} d_model={model.d_model}")

    # Phase 1 stimuli; we'll extract per-prompt residuals at the chosen layer.
    stims = build_stimulus_set(per_cell=args.per_cell)
    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"phase2_train_{nick}",
        config={
            "model": args.model, "layer": args.layer,
            "per_cell": args.per_cell,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "stimulus_set_size": len(stims),
        },
    )
    print(f"  run dir: {rd}")

    print("Extracting per-prompt residuals at canonical layer ...")
    prompts = [s.prompt for s in stims]
    req = ActivationRequest(layer_idxs=[args.layer], position=-1)
    H = extract_batch(model, prompts, req, batch_size=16)[args.layer]  # (N, d) cpu fp32
    d_model = int(H.shape[1])

    # Train on euphoric, evaluate on naturalistic. (Neutral isn't an emotion
    # label, so it's only used for sanity contrasts — not for adapter fit.)
    train_examples: list[TrainExample] = []
    val_examples: list[TrainExample] = []
    for i, s in enumerate(stims):
        if s.emotion == "neutral":
            continue
        ex = TrainExample(vector=H[i].clone(), label=s.emotion)
        if s.level == "euphoric":
            train_examples.append(ex)
        elif s.level == "naturalistic":
            val_examples.append(ex)
    print(f"  train: {len(train_examples)} (euphoric)  val: {len(val_examples)} (naturalistic)")

    # Untrained-baseline accuracy: a fresh ScalarAffineAdapter at α=1, b=0
    # is the identity; the residual-replace hook drops `h` directly into the
    # probe position. Whatever accuracy that has is the SelfIE-style untrained
    # baseline that Pepper's "training matters" claim is supposed to beat.
    print("\nUntrained SelfIE baseline (α=1, b=0):")
    base = make_adapter(AdapterConfig(kind="scalar_affine", d_model=d_model)).to(model.device)
    base_eval = evaluate_adapter(model, base, val_examples,
                                 TrainConfig(layer_idx=args.layer, batch_size=args.batch_size))
    print(f"  top1 = {base_eval['top1']:.3f}  loss = {base_eval['loss']:.3f}  n = {base_eval['n']}")

    results = {"untrained_baseline": base_eval}

    for kind in ("scalar_affine", "bias_only", "full_rank"):
        print(f"\n=== Training {kind} ===")
        cfg = TrainConfig(
            layer_idx=args.layer, batch_size=args.batch_size,
            n_epochs=args.epochs, learning_rate=args.lr,
        )
        adapter = make_adapter(AdapterConfig(kind=kind, d_model=d_model)).to(model.device)
        history = train_adapter(model, adapter, train_examples, val_examples, cfg)
        final_val = evaluate_adapter(model, adapter, val_examples, cfg)
        print(f"  final val top1 = {final_val['top1']:.3f}  loss = {final_val['loss']:.3f}")
        results[kind] = {
            "history": history, "final_val": final_val,
            "n_params": adapter.n_params,
        }
        torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                   rd / f"adapter_{kind}.pt")

    (rd / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== Summary ===")
    print(f"  {'variant':<18} {'n_params':>10}  {'val_top1':>10}  {'val_loss':>10}")
    print(f"  {'untrained':<18} {'(α=1,b=0)':>10}  {results['untrained_baseline']['top1']:>10.3f}  {results['untrained_baseline']['loss']:>10.3f}")
    for kind in ("bias_only", "scalar_affine", "full_rank"):
        r = results[kind]
        print(f"  {kind:<18} {r['n_params']:>10}  {r['final_val']['top1']:>10.3f}  {r['final_val']['loss']:>10.3f}")
    print(f"\nSaved {rd}")


if __name__ == "__main__":
    main()
