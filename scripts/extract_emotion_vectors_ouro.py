"""Phase 1 extraction for the Ouro universal-transformer model.

Ouro reuses the same 24 decoder layers `total_ut_steps` (=4) times per forward
pass, like a Universal Transformer. The natural probe space is therefore
24 × 4 = 96 (layer, ut_step) sites rather than 24.

We hook each layer with `cache_residual_looped`, which appends one tensor per
forward-call rather than overwriting. We then build a diff-of-means direction
at every (layer, ut_step) site and run the PCA-vs-valence sanity check across
all 96 sites.

Run:
    uv run python scripts/extract_emotion_vectors_ouro.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.models.adapter import ModelAdapter  # noqa: E402
from src.probes.diff_means import diff_of_means, probe_separation  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402

VALENCE = {"calm": +1, "blissful": +1, "desperate": -1, "sad": -1, "afraid": -1, "hostile": -1}
AROUSAL = {"calm": -1, "sad": -1, "blissful": +1, "desperate": +1, "afraid": +1, "hostile": +1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if "HF_HOME" not in os.environ:
        # Default to the external-drive cache where Ouro is staged.
        os.environ["HF_HOME"] = "/media/external-drive/huggingface"

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    adapter = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map, trust_remote_code=True)
    n_layers = adapter.n_layers
    n_ut = adapter.n_loop_steps
    print(f"  family={adapter.family} n_layers={n_layers} n_ut={n_ut} d_model={adapter.d_model}")
    if not adapter.is_looping:
        raise RuntimeError("Ouro should be detected as a looping family.")

    stims = build_stimulus_set(per_cell=args.per_cell)
    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"phase1_vectors_{nick}",
        config={
            "model": args.model, "n_layers": n_layers, "n_ut": n_ut,
            "d_model": adapter.d_model, "per_cell": args.per_cell,
            "stimulus_set_size": len(stims),
        },
    )
    print(f"  run dir: {rd}")
    print(f"  stimuli: {len(stims)} total; will probe {n_layers * n_ut} (layer, ut) sites")

    layer_idxs = list(range(n_layers))
    prompts = [s.prompt for s in stims]
    # acts[layer][ut][prompt_idx] = (d_model,) cpu fp32
    acts: dict[int, list[list[torch.Tensor]]] = {li: [[] for _ in range(n_ut)] for li in layer_idxs}

    print("Extracting (layer, ut, prompt) residuals ...")
    for prompt in tqdm(prompts):
        inputs = adapter.tokenizer(prompt, return_tensors="pt").to(adapter.device)
        with torch.no_grad(), adapter.cache_residual_looped(layer_idxs) as cache:
            adapter.model(**inputs, use_cache=False)
        # cache[li] is a list of length n_ut, each a (1, S, d) tensor on cpu fp32.
        for li in layer_idxs:
            calls = cache[li]
            if len(calls) != n_ut:
                raise RuntimeError(
                    f"layer {li} fired {len(calls)} times, expected {n_ut} — "
                    f"is the model actually looped?"
                )
            for ut_step, h in enumerate(calls):
                acts[li][ut_step].append(h[0, -1].clone())  # last-token residual

    H_by_site: dict[tuple[int, int], np.ndarray] = {}
    for li in layer_idxs:
        for ut in range(n_ut):
            H_by_site[(li, ut)] = torch.stack(acts[li][ut], dim=0).numpy()

    rows_by_key: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(stims):
        rows_by_key.setdefault((s.emotion, s.level), []).append(i)
    neutral_rows = rows_by_key[("neutral", "neutral")]

    # Build diff-of-means at every site, score on naturalistic transfer.
    summary_rows: list[dict] = []
    vectors: dict[str, dict[tuple[int, int], np.ndarray]] = {emo: {} for emo in EMOTIONS}
    for emo in EMOTIONS:
        eu_rows = rows_by_key[(emo, "euphoric")]
        nat_rows = rows_by_key[(emo, "naturalistic")]
        for site, H in H_by_site.items():
            v = diff_of_means(H[eu_rows], H[neutral_rows])
            vectors[emo][site] = v
            sep = probe_separation(H[nat_rows], H[neutral_rows], v)
            li, ut = site
            summary_rows.append({
                "emotion": emo, "layer": li, "ut_step": ut,
                "effective_layer": ut * n_layers + li,
                "auroc_nat_vs_neu": sep.auroc,
                "d_prime_nat_vs_neu": sep.d_prime,
            })

    # PCA across emotions per site → which (layer, ut) gives best PC1↔valence?
    valence_target = np.array([VALENCE[e] for e in EMOTIONS], dtype=np.float32)
    arousal_target = np.array([AROUSAL[e] for e in EMOTIONS], dtype=np.float32)

    def signed_r(scores_col, target):
        sc = scores_col - scores_col.mean()
        tg = target - target.mean()
        denom = np.linalg.norm(sc) * np.linalg.norm(tg)
        return float(sc @ tg / denom) if denom > 0 else 0.0

    pca_rows: list[dict] = []
    for site in H_by_site.keys():
        V = np.stack([vectors[e][site] for e in EMOTIONS])  # (6, d)
        Vc = V - V.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Vc, full_matrices=False)
        scores = U * S
        li, ut = site
        pca_rows.append({
            "layer": li, "ut_step": ut, "effective_layer": ut * n_layers + li,
            "PC1_valence_r": signed_r(scores[:, 0], valence_target),
            "PC2_valence_r": signed_r(scores[:, 1], valence_target) if scores.shape[1] > 1 else 0.0,
            "PC1_arousal_r": signed_r(scores[:, 0], arousal_target),
            "PC2_arousal_r": signed_r(scores[:, 1], arousal_target) if scores.shape[1] > 1 else 0.0,
        })

    # Persist.
    torch.save(
        {emo: {f"{li}/{ut}": torch.from_numpy(v) for (li, ut), v in vs.items()}
         for emo, vs in vectors.items()},
        rd / "vectors.pt",
    )
    (rd / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    (rd / "pca_summary.json").write_text(json.dumps(pca_rows, indent=2))

    best_pc1 = max(pca_rows, key=lambda r: abs(r["PC1_valence_r"]))
    best_pc2 = max(pca_rows, key=lambda r: abs(r["PC2_arousal_r"]))
    print(f"\nBest |PC1↔valence|: layer {best_pc1['layer']} ut={best_pc1['ut_step']} "
          f"(eff L{best_pc1['effective_layer']}/{n_layers*n_ut}) "
          f"= {best_pc1['PC1_valence_r']:+.3f}")
    print(f"Best |PC2↔arousal|: layer {best_pc2['layer']} ut={best_pc2['ut_step']} "
          f"(eff L{best_pc2['effective_layer']}/{n_layers*n_ut}) "
          f"= {best_pc2['PC2_arousal_r']:+.3f}")

    # Per-ut-step best layer (4 numbers) — does later thinking improve geometry?
    print("\nBest |PC1↔valence| per ut step:")
    for ut in range(n_ut):
        rows = [r for r in pca_rows if r["ut_step"] == ut]
        b = max(rows, key=lambda r: abs(r["PC1_valence_r"]))
        print(f"  ut={ut}: layer {b['layer']:>2}  r={b['PC1_valence_r']:+.3f}")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            for ut in range(n_ut):
                rows = [r for r in pca_rows if r["ut_step"] == ut]
                rows.sort(key=lambda r: r["layer"])
                xs = [r["layer"] for r in rows]
                ax.plot(xs, [abs(r["PC1_valence_r"]) for r in rows],
                        marker="o", label=f"ut={ut}")
            ax.axhline(0.81, color="gray", lw=0.6, ls="--")
            ax.set_xlabel("layer index (within the 24-layer stack)")
            ax.set_ylabel("|PC1↔valence|")
            ax.set_title(f"Ouro layer × ut-step sweep — {nick}")
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            fig.savefig(rd / "ouro_layer_ut_sweep.png", dpi=140)
            print(f"\nSaved {rd / 'ouro_layer_ut_sweep.png'}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
