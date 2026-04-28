"""Phase 1 extraction inside the Monet compat env (transformers 4.42).

Reuses the main project's stimulus set, probe code, and run-dir helper via
sys.path injection. The only thing that lives here is the model-loading
path — kept narrow on purpose so the same runner works for any other legacy
custom-code model that needs transformers 4.42.

Run:
    cd compat_envs/monet
    uv sync
    uv run python run.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Inject the project root so we can import the shared src/ modules. parents[2]
# = compat_envs/monet/ -> compat_envs/ -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.probes.diff_means import diff_of_means, probe_separation  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


VALENCE = {"calm": +1, "blissful": +1, "desperate": -1, "sad": -1, "afraid": -1, "hostile": -1}
AROUSAL = {"calm": -1, "sad": -1, "blissful": +1, "desperate": +1, "afraid": +1, "hostile": +1}


def _signed_r(scores_col: np.ndarray, target: np.ndarray) -> float:
    sc = scores_col - scores_col.mean()
    tg = target - target.mean()
    denom = np.linalg.norm(sc) * np.linalg.norm(tg)
    return float(sc @ tg / denom) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MonetLLM/monet-vd-850M-100BT-hf")
    ap.add_argument("--per-cell", type=int, default=30)
    args = ap.parse_args()

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/media/external-drive/huggingface"

    print(f"Loading {args.model} (transformers 4.42 compat env) ...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Slow tokenizer keeps sentencepiece pathway, avoiding the 5.x
    # tiktoken-conversion regression. trust_remote_code=True is required.
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tok.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()

    n_layers = int(model.config.num_hidden_layers)
    d_model = int(model.config.hidden_size)
    print(f"  family={model.config.model_type} n_layers={n_layers} d_model={d_model}")

    # Standard model.layers structure (verified in modeling_monet.py).
    layers = model.model.layers

    stims = build_stimulus_set(per_cell=args.per_cell)
    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"phase1_vectors_{nick}",
        config={
            "model": args.model, "n_layers": n_layers, "d_model": d_model,
            "per_cell": args.per_cell, "stimulus_set_size": len(stims),
            "framework": "transformers-4.42-compat",
        },
    )
    print(f"  run dir: {rd}")
    print(f"  stimuli: {len(stims)}")

    # Forward hooks on each block — same pattern as the main ModelAdapter,
    # written inline so this runner has zero coupling to our adapter code
    # (which is built around transformers 5.x).
    layer_idxs = list(range(n_layers))
    cache: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[idx] = h.detach().to("cpu", dtype=torch.float32)
        return hook

    handles = [layers[li].register_forward_hook(make_hook(li)) for li in layer_idxs]

    print("Extracting last-token residuals ...")
    H_per_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layer_idxs}
    try:
        for s in tqdm(stims):
            inputs = tok(s.prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs, use_cache=False)
            for li in layer_idxs:
                H_per_layer[li].append(cache[li][0, -1].clone())
    finally:
        for h in handles:
            h.remove()

    H_by_layer = {li: torch.stack(parts, dim=0).numpy() for li, parts in H_per_layer.items()}

    rows_by_key: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(stims):
        rows_by_key.setdefault((s.emotion, s.level), []).append(i)
    neutral_rows = rows_by_key[("neutral", "neutral")]

    summary_rows: list[dict] = []
    vectors: dict[str, dict[int, np.ndarray]] = {emo: {} for emo in EMOTIONS}
    for emo in EMOTIONS:
        eu = rows_by_key[(emo, "euphoric")]
        nat = rows_by_key[(emo, "naturalistic")]
        for li in layer_idxs:
            H = H_by_layer[li]
            v = diff_of_means(H[eu], H[neutral_rows])
            vectors[emo][li] = v
            sep = probe_separation(H[nat], H[neutral_rows], v)
            summary_rows.append({
                "emotion": emo, "layer": li,
                "auroc_nat_vs_neu": sep.auroc,
                "d_prime_nat_vs_neu": sep.d_prime,
            })

    # PCA across emotions per layer.
    val_target = np.array([VALENCE[e] for e in EMOTIONS], dtype=np.float32)
    aro_target = np.array([AROUSAL[e] for e in EMOTIONS], dtype=np.float32)
    pca_rows = []
    for li in layer_idxs:
        V = np.stack([vectors[e][li] for e in EMOTIONS])
        Vc = V - V.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Vc, full_matrices=False)
        scores = U * S
        pca_rows.append({
            "layer": li,
            "PC1_valence_r": _signed_r(scores[:, 0], val_target),
            "PC2_valence_r": _signed_r(scores[:, 1], val_target),
            "PC1_arousal_r": _signed_r(scores[:, 0], aro_target),
            "PC2_arousal_r": _signed_r(scores[:, 1], aro_target),
        })

    torch.save(
        {emo: {li: torch.from_numpy(v) for li, v in vs.items()} for emo, vs in vectors.items()},
        rd / "vectors.pt",
    )
    (rd / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    (rd / "pca_summary.json").write_text(json.dumps(pca_rows, indent=2))

    best = max(pca_rows, key=lambda r: abs(r["PC1_valence_r"]))
    print(f"\nBest |PC1↔valence|: layer {best['layer']} = {best['PC1_valence_r']:+.3f}")
    best_aro = max(pca_rows, key=lambda r: abs(r["PC2_arousal_r"]))
    print(f"Best |PC2↔arousal|: layer {best_aro['layer']} = {best_aro['PC2_arousal_r']:+.3f}")


if __name__ == "__main__":
    main()
