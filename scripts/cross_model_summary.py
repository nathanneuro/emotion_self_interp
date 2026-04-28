"""Aggregate per-model PCA results into a single cross-model summary.

Walks `outputs/phase1_vectors_*` directories that have a `pca_summary.json`,
extracts the best PC1↔valence layer + r and best PC2↔arousal layer + r per
model, prints a markdown table, and writes `outputs/phase1_cross_model.json`
plus a comparison plot.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runs.run_dir import OUTPUTS  # noqa: E402


@dataclass
class ModelEntry:
    nickname: str
    family: str
    n_layers: int
    d_model: int
    best_pc1_layer: int
    best_pc1_r: float
    best_pc1_layer_frac: float
    best_pc2_layer: int
    best_pc2_r: float
    run_dir: str


def main() -> None:
    runs = sorted(OUTPUTS.glob("phase1_vectors_*"))
    entries: list[ModelEntry] = []
    for rd in runs:
        pca_path = rd / "pca_summary.json"
        cfg_path = rd / "config.json"
        if not pca_path.exists() or not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        rows = json.loads(pca_path.read_text())
        # Pick by absolute correlation; keep sign for reporting.
        best_pc1 = max(rows, key=lambda r: abs(r["PC1_valence_r"]))
        best_pc2 = max(rows, key=lambda r: abs(r["PC2_arousal_r"]))
        nickname = rd.name.replace("phase1_vectors_", "").rsplit("_", 2)[0]
        n_layers = int(cfg.get("n_layers", best_pc1["layer"] + 1))
        family = cfg.get("framework", "transformers/HF")
        if "model" in cfg and "/" in str(cfg["model"]):
            model_short = str(cfg["model"]).split("/")[-1]
        else:
            model_short = nickname
        entries.append(ModelEntry(
            nickname=model_short,
            family=family,
            n_layers=n_layers,
            d_model=int(cfg.get("d_model", -1)),
            best_pc1_layer=best_pc1["layer"],
            best_pc1_r=best_pc1["PC1_valence_r"],
            best_pc1_layer_frac=best_pc1["layer"] / max(n_layers - 1, 1),
            best_pc2_layer=best_pc2["layer"],
            best_pc2_r=best_pc2["PC2_arousal_r"],
            run_dir=str(rd),
        ))

    # Sort by parameter count proxy (d_model × n_layers); good enough for ordering.
    entries.sort(key=lambda e: e.n_layers * e.d_model)

    print("\nCross-model summary — Phase 1 v0\n")
    print("| Model | Layers | d | Best PC1↔valence | Layer frac | Best PC2↔arousal |")
    print("|---|---|---|---|---|---|")
    for e in entries:
        print(
            f"| {e.nickname} | {e.n_layers} | {e.d_model} | "
            f"{e.best_pc1_r:+.3f} @ L{e.best_pc1_layer} | {e.best_pc1_layer_frac:.2f} | "
            f"{e.best_pc2_r:+.3f} @ L{e.best_pc2_layer} |"
        )

    out = OUTPUTS / "phase1_cross_model.json"
    out.write_text(json.dumps([e.__dict__ for e in entries], indent=2))
    print(f"\nWrote {out}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = list(range(len(entries)))
        ax.bar(xs, [abs(e.best_pc1_r) for e in entries], label="|PC1↔valence|", color="C0")
        ax.bar([x + 0.4 for x in xs], [abs(e.best_pc2_r) for e in entries],
               width=0.4, label="|PC2↔arousal|", color="C1", align="edge")
        ax.set_xticks([x + 0.2 for x in xs])
        ax.set_xticklabels([e.nickname for e in entries], rotation=20, ha="right", fontsize=8)
        ax.axhline(0.81, color="gray", lw=0.8, ls="--", label="Sofroniew 2026 valence (70B)")
        ax.axhline(0.66, color="gray", lw=0.5, ls=":", label="Sofroniew 2026 arousal (70B)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("|Pearson r|")
        ax.set_title("Cross-model emotion-vector geometry — Phase 1 v0")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        plot = OUTPUTS / "phase1_cross_model.png"
        fig.savefig(plot, dpi=140)
        print(f"Wrote {plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
