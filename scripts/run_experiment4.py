"""Phase 6 / Experiment 4: veridical introspection — does behavior follow
substrate or report?

Trains an honest and a deceptive (swap-labeled) adapter on the same residual
cache. On naturalistic held-out, measures whether each channel tracks the
true emotion or the deceptive adapter's swap target. The substrate-driven
channels (substrate score, Likert) should follow the true emotion while the
deceptive adapter follows its swap.

Run:
    uv run python scripts/run_experiment4.py
    uv run python scripts/run_experiment4.py --model google/gemma-2-2b --layer 8
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.behaviors.likert import LikertConfig, likert_rating  # noqa: E402
from src.experiments import (  # noqa: E402
    SWAP_PAIRING,
    build_emotion_vectors,
    extract_stimulus_residuals,
    measure_introspection,
    summarize_introspection,
    train_honest_and_deceptive_adapters,
)
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--adapter-kind", default="full_rank",
                    choices=["scalar_affine", "scale_only", "bias_only", "full_rank"])
    ap.add_argument("--adapter-epochs", type=int, default=20)
    ap.add_argument("--adapter-lr", type=float, default=5e-3)
    ap.add_argument("--no-likert", action="store_true",
                    help="Skip the Likert measurement (faster sanity check).")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="Required for custom-modeling repos (e.g. Ouro).")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(
        args.model, dtype=dtype, device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"  family={model.family} d_model={model.d_model}")
    print(f"  swap pairing: {SWAP_PAIRING}")

    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"phase6_exp4_{nick}",
        config={
            "model": args.model, "layer": args.layer, "per_cell": args.per_cell,
            "adapter_kind": args.adapter_kind, "adapter_epochs": args.adapter_epochs,
            "adapter_lr": args.adapter_lr, "swap_pairing": SWAP_PAIRING,
        },
    )
    print(f"  run dir: {rd}")

    print("\n[1/4] Extracting residuals ...")
    res = extract_stimulus_residuals(model, layer=args.layer, per_cell=args.per_cell)

    print("\n[2/4] Building emotion vectors (within-emotion contrast) ...")
    emotion_vectors = build_emotion_vectors(res, contrast="other_emotions")

    print(f"\n[3/4] Training honest + deceptive {args.adapter_kind} adapters ...")
    honest, deceptive = train_honest_and_deceptive_adapters(
        model, res, kind=args.adapter_kind,
        epochs=args.adapter_epochs, lr=args.adapter_lr,
    )

    # Naturalistic test set.
    test_residuals_list: list[torch.Tensor] = []
    test_meta: list[tuple[str, str, str]] = []
    test_prompts: list[str] = []
    for i, s in enumerate(res.stimuli):
        if s.level != "naturalistic" or s.emotion == "neutral":
            continue
        test_residuals_list.append(res.residuals[i].clone())
        test_meta.append((s.id, s.emotion, s.level))
        test_prompts.append(s.prompt)
    test_H = torch.stack(test_residuals_list, dim=0)
    print(f"  test stimuli: {len(test_meta)} naturalistic")

    print("\n[4/4] Measuring honest + deceptive + substrate per stimulus ...")
    rows = measure_introspection(
        model=model, layer=args.layer,
        honest_adapter=honest, deceptive_adapter=deceptive,
        emotion_vectors=emotion_vectors,
        test_residuals=test_H, test_stimuli_meta=test_meta,
    )

    likert_valences: list[float] | None = None
    if not args.no_likert:
        print("\n[bonus] Likert valence per stimulus ...")
        likert_cfg = LikertConfig()
        likert_valences = []
        for i, prompt in enumerate(test_prompts):
            lk = likert_rating(model, prompt, likert_cfg)
            likert_valences.append(float(lk.valence.expected))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(test_prompts)}")

    summary = summarize_introspection(rows, likert_valences=likert_valences)

    (rd / "rows.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    if likert_valences is not None:
        (rd / "likert_valences.json").write_text(json.dumps(likert_valences))
    torch.save(
        {
            "honest": {k: v.detach().cpu() for k, v in honest.state_dict().items()},
            "deceptive": {k: v.detach().cpu() for k, v in deceptive.state_dict().items()},
        },
        rd / "adapters.pt",
    )

    print(f"\n=== Veridical introspection (n={summary['n']}) ===")
    print("\n  Match rate vs TRUE emotion:")
    for ch, v in summary["match_true"].items():
        print(f"    {ch:<20} {v:.3f}")
    print("\n  Match rate vs SWAP target (deceptive learned the swap?):")
    for ch, v in summary["match_swap"].items():
        print(f"    {ch:<20} {v:.3f}")
    print("\n  Continuous valence-projection correlations:")
    for k, v in summary["correlations"].items():
        print(f"    {k:<32} {v:+.3f}")
    print(f"\nSaved {rd}")


if __name__ == "__main__":
    main()
