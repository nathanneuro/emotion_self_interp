"""Smallest end-to-end pipeline: load, extract, steer, save.

Run:
    uv run python scripts/smoke_pipeline.py
    uv run python scripts/smoke_pipeline.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.hooks.extract import ActivationRequest, extract  # noqa: E402
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--prompt-emotional", default="He felt utterly desperate, on the edge of breaking.")
    ap.add_argument("--prompt-neutral", default="He sorted the books on the shelf alphabetically.")
    ap.add_argument("--alpha", type=float, default=5.0)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    adapter = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map)
    print(f"  family={adapter.family} n_layers={adapter.n_layers} d_model={adapter.d_model}")

    layer = args.layer if args.layer is not None else adapter.n_layers // 2
    print(f"Probing at layer {layer}")

    req = ActivationRequest(layer_idxs=[layer], position=-1)
    v_emo = extract(adapter, args.prompt_emotional, req)[layer]
    v_neu = extract(adapter, args.prompt_neutral, req)[layer]
    diff = v_emo - v_neu
    cos = torch.nn.functional.cosine_similarity(v_emo, v_neu, dim=0).item()
    print(f"  ‖v_emo‖={v_emo.norm():.2f}  ‖v_neu‖={v_neu.norm():.2f}  cos(emo,neu)={cos:.4f}")

    # Steering check: confirm the steering hook actually changes outputs.
    inputs = adapter.tokenizer("The next thing I noticed was", return_tensors="pt").to(adapter.device)
    with torch.no_grad():
        base_logits = adapter.model(**inputs).logits.detach().cpu().float()
    with torch.no_grad(), adapter.steer_residual(layer, diff, alpha=args.alpha):
        steered_logits = adapter.model(**inputs).logits.detach().cpu().float()
    logit_diff = (base_logits - steered_logits).abs().mean().item()
    print(f"  steering @ layer {layer}, alpha {args.alpha}: mean |Δlogit| = {logit_diff:.4f}")

    # Sample a generation under steering for qualitative inspection.
    gen_kwargs = dict(max_new_tokens=40, do_sample=False, pad_token_id=adapter.tokenizer.pad_token_id)
    prompt = "Tell me, briefly, how you feel right now."
    inp = adapter.tokenizer(prompt, return_tensors="pt").to(adapter.device)
    with torch.no_grad():
        base_out = adapter.model.generate(**inp, **gen_kwargs)
    with torch.no_grad(), adapter.steer_residual(layer, diff, alpha=args.alpha):
        steer_out = adapter.model.generate(**inp, **gen_kwargs)
    base_text = adapter.tokenizer.decode(base_out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    steer_text = adapter.tokenizer.decode(steer_out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n--- baseline generation ---")
    print(base_text)
    print("\n--- steered (emo − neu) generation ---")
    print(steer_text)

    rd = make_run_dir(
        f"smoke_{args.model.split('/')[-1]}",
        config={
            "model": args.model,
            "layer": layer,
            "alpha": args.alpha,
            "prompt_emotional": args.prompt_emotional,
            "prompt_neutral": args.prompt_neutral,
        },
    )
    torch.save(
        {"v_emo": v_emo, "v_neu": v_neu, "diff": diff},
        rd / "vectors.pt",
    )
    (rd / "generations.txt").write_text(
        f"PROMPT: {prompt}\n\n--- baseline ---\n{base_text}\n\n--- steered ---\n{steer_text}\n"
    )
    print(f"\nSaved to {rd}")


if __name__ == "__main__":
    main()
