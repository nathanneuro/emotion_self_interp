"""Phase 1 sanity check: PCA over per-emotion vectors should yield valence-like PC1.

Sofroniew et al. report PC1 of their emotion-vector geometry correlates r=0.81
with human valence ratings. Here we check whether the structure exists at all
in our small-model, small-stimulus replication.

Approach:
  - Load vectors saved by extract_emotion_vectors.py.
  - At each layer, stack the six per-emotion vectors and run PCA.
  - Score PC1 against a hand-coded valence target for our six emotions:
        calm:+1, blissful:+1, desperate:-1, sad:-1, afraid:-1, hostile:-1
    (signs are arbitrary; we report |pearson r|.)
  - Also score PC2 against a hand-coded arousal target:
        calm:-1, sad:-1, blissful:+1, desperate:+1, afraid:+1, hostile:+1.
  - Report the layer where each correlation peaks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.emotion_stimuli import EMOTIONS  # noqa: E402

VALENCE = {"calm": +1, "blissful": +1, "desperate": -1, "sad": -1, "afraid": -1, "hostile": -1}
AROUSAL = {"calm": -1, "sad": -1, "blissful": +1, "desperate": +1, "afraid": +1, "hostile": +1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    rd: Path = args.run_dir
    vectors = torch.load(rd / "vectors.pt", weights_only=False)
    # vectors: {emotion: {layer: tensor(d_model,)}}
    n_layers = len(next(iter(vectors.values())))
    d = next(iter(vectors["calm"].values())).shape[0]
    print(f"Loaded vectors: {len(vectors)} emotions × {n_layers} layers × {d} dims")

    valence_target = np.array([VALENCE[e] for e in EMOTIONS], dtype=np.float32)
    arousal_target = np.array([AROUSAL[e] for e in EMOTIONS], dtype=np.float32)

    rows = []
    for li in range(n_layers):
        V = np.stack([vectors[e][li].numpy() for e in EMOTIONS])  # (6, d)
        # Center across emotions, then SVD.
        Vc = V - V.mean(axis=0, keepdims=True)
        # Right-singular vectors (Vt) are PCs in feature space; left-singular
        # vectors (U) give per-emotion scores along each PC.
        U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
        scores = U * S  # (6, k); columns are PC scores for each emotion.

        def signed_r(scores_col, target):
            sc = scores_col - scores_col.mean()
            tg = target - target.mean()
            denom = np.linalg.norm(sc) * np.linalg.norm(tg)
            return float(sc @ tg / denom) if denom > 0 else 0.0

        r1_val = signed_r(scores[:, 0], valence_target)
        r2_val = signed_r(scores[:, 1], valence_target) if scores.shape[1] > 1 else 0.0
        r1_aro = signed_r(scores[:, 0], arousal_target)
        r2_aro = signed_r(scores[:, 1], arousal_target) if scores.shape[1] > 1 else 0.0
        rows.append({
            "layer": li,
            "sv": S.tolist(),
            "PC1_valence_r": r1_val,
            "PC2_valence_r": r2_val,
            "PC1_arousal_r": r1_aro,
            "PC2_arousal_r": r2_aro,
        })

    (rd / "pca_summary.json").write_text(json.dumps(rows, indent=2))

    print("\nLayer-wise PC1/PC2 correlations with valence/arousal targets:")
    print(f"  {'layer':>5}  {'|PC1·val|':>10}  {'|PC2·val|':>10}  {'|PC1·aro|':>10}  {'|PC2·aro|':>10}")
    for r in rows:
        print(
            f"  {r['layer']:>5}  "
            f"{abs(r['PC1_valence_r']):>10.3f}  {abs(r['PC2_valence_r']):>10.3f}  "
            f"{abs(r['PC1_arousal_r']):>10.3f}  {abs(r['PC2_arousal_r']):>10.3f}"
        )

    best_pc1_val = max(rows, key=lambda r: abs(r["PC1_valence_r"]))
    best_pc2_aro = max(rows, key=lambda r: abs(r["PC2_arousal_r"]))
    print(
        f"\nBest |PC1↔valence|: layer {best_pc1_val['layer']} = {best_pc1_val['PC1_valence_r']:+.3f}"
    )
    print(
        f"Best |PC2↔arousal|: layer {best_pc2_aro['layer']} = {best_pc2_aro['PC2_arousal_r']:+.3f}"
    )

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        xs = [r["layer"] for r in rows]
        ax[0].plot(xs, [r["PC1_valence_r"] for r in rows], marker="o", label="PC1·valence")
        ax[0].plot(xs, [r["PC2_valence_r"] for r in rows], marker="s", label="PC2·valence")
        ax[0].axhline(0, color="gray", lw=0.7)
        ax[0].set_xlabel("layer")
        ax[0].set_ylabel("Pearson r vs valence target")
        ax[0].legend()
        ax[1].plot(xs, [r["PC1_arousal_r"] for r in rows], marker="o", label="PC1·arousal")
        ax[1].plot(xs, [r["PC2_arousal_r"] for r in rows], marker="s", label="PC2·arousal")
        ax[1].axhline(0, color="gray", lw=0.7)
        ax[1].set_xlabel("layer")
        ax[1].set_ylabel("Pearson r vs arousal target")
        ax[1].legend()
        for a in ax:
            a.set_ylim(-1.05, 1.05)
        fig.tight_layout()
        fig.savefig(rd / "pca_layer_sweep.png", dpi=140)
        print(f"\nSaved {rd / 'pca_layer_sweep.png'}")

        # Best-layer scatter of the six emotions in PC1/PC2 space.
        li = best_pc1_val["layer"]
        V = np.stack([vectors[e][li].numpy() for e in EMOTIONS])
        Vc = V - V.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Vc, full_matrices=False)
        scores = U * S
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(scores[:, 0], scores[:, 1])
        for i, e in enumerate(EMOTIONS):
            ax.annotate(e, (scores[i, 0], scores[i, 1]), fontsize=10,
                        xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"PC1 (r={best_pc1_val['PC1_valence_r']:+.2f} vs valence)")
        ax.set_ylabel("PC2")
        ax.set_title(f"Emotion vectors in PC1/PC2 — layer {li}")
        fig.tight_layout()
        fig.savefig(rd / "pca_scatter_best_layer.png", dpi=140)
        print(f"Saved {rd / 'pca_scatter_best_layer.png'}")
    except ImportError:
        print("(matplotlib not installed — skipping plots)")


if __name__ == "__main__":
    main()
