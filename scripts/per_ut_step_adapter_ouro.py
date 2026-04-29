"""Per-ut-step adapter training on Ouro.

Phase 1 showed substrate vector PCA structure builds across ut steps; the
per-ut-step Likert script showed the *behavioral* readout follows the same
trajectory. This script asks the trained-adapter version of the same
question:

    For each input_ut N:  train an adapter using residuals captured at the
                          end of ut step N (instead of the cumulative
                          ut=last residual).
    For each output_ut M: evaluate the adapter with `exit_at_step=M` so
                          the LM head reads from the post-norm hidden
                          states of ut step M.

Result is a 4×4 matrix of (input_ut → output_ut) accuracies / r vs target
valence. Diagonal entries (N=M) are the natural "use ut N's residuals,
read ut N's prediction" condition. Off-diagonal entries test transfer:
does an adapter trained on ut=0 residuals work when the model decodes at
ut=3?

Run:
    uv run python scripts/per_ut_step_adapter_ouro.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.adapters.scalar_affine import AdapterConfig, make_adapter  # noqa: E402
from src.adapters.train import TrainConfig, TrainExample, train_adapter  # noqa: E402
from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.experiments.experiment1 import (  # noqa: E402
    VALENCE_TARGET,
    _adapter_scores_batched,
    _argmax_label,
    _emotion_label_token_seqs,
)
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402


def _signed_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


@torch.no_grad()
def extract_per_ut_residuals(
    model: ModelAdapter, prompts: list[str], layer: int,
) -> torch.Tensor:
    """One forward per prompt, capture last-token residual after each ut step.
    Returns (n_prompts, n_ut, d_model) cpu fp32."""
    n_ut = model.n_loop_steps
    rows: list[torch.Tensor] = []
    for p in prompts:
        inputs = model.tokenizer(p, return_tensors="pt").to(model.device)
        with model.cache_residual_looped([layer]) as cache:
            model.model(**inputs, use_cache=False)
        calls = cache[layer]
        if len(calls) != n_ut:
            raise RuntimeError(f"expected {n_ut} ut calls at layer {layer}, got {len(calls)}")
        per_ut = torch.stack([c[0, -1] for c in calls], dim=0)  # (n_ut, d)
        rows.append(per_ut)
    return torch.stack(rows, dim=0)  # (n_prompts, n_ut, d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--adapter-kind", default="full_rank")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map, trust_remote_code=True)
    n_ut = model.n_loop_steps
    d_model = model.d_model
    print(f"  family={model.family} d_model={d_model} n_ut={n_ut}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    train_stims = [s for s in stims if s.level == "euphoric" and s.emotion != "neutral"]
    test_stims = [s for s in stims if s.level == "naturalistic" and s.emotion != "neutral"]
    print(f"  train: {len(train_stims)} euphoric  test: {len(test_stims)} naturalistic")

    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"per_ut_adapter_{nick}",
        config={
            "model": args.model, "layer": args.layer, "n_ut": n_ut,
            "per_cell": args.per_cell, "epochs": args.epochs, "lr": args.lr,
            "adapter_kind": args.adapter_kind,
        },
    )
    print(f"  run dir: {rd}")

    print("\n[1/3] Extracting per-(stimulus, ut) residuals at layer 15 ...")
    train_H = extract_per_ut_residuals(model, [s.prompt for s in train_stims], args.layer)
    test_H = extract_per_ut_residuals(model, [s.prompt for s in test_stims], args.layer)
    print(f"  train residuals: {tuple(train_H.shape)}")
    print(f"  test residuals:  {tuple(test_H.shape)}")
    torch.save({"train_H": train_H, "test_H": test_H}, rd / "residuals.pt")

    test_labels = [s.emotion for s in test_stims]
    test_target_valence = np.array(
        [VALENCE_TARGET[lab] for lab in test_labels], dtype=np.float64,
    )

    label_seqs = _emotion_label_token_seqs(model.tokenizer, EMOTIONS)
    pos_emos = ("calm", "blissful")
    neg_emos = ("desperate", "sad", "afraid", "hostile")

    def val_proj(scores_list):
        return np.array([
            sum(s.get(e, 0.0) for e in pos_emos) - sum(s.get(e, 0.0) for e in neg_emos)
            for s in scores_list
        ], dtype=np.float64)

    print("\n[2/3] Training one adapter per input ut step ...")
    cfg = TrainConfig(
        layer_idx=args.layer, batch_size=8, n_epochs=args.epochs, learning_rate=args.lr,
    )
    adapters: list = []
    for input_ut in range(n_ut):
        print(f"  training adapter on input_ut={input_ut} residuals ...")
        train_examples = [
            TrainExample(vector=train_H[i, input_ut].clone(), label=s.emotion)
            for i, s in enumerate(train_stims)
        ]
        adapter = make_adapter(AdapterConfig(kind=args.adapter_kind, d_model=d_model)).to(model.device)
        history = train_adapter(model, adapter, train_examples, val=None, cfg=cfg)
        print(f"    final train top1: {history['train_acc'][-1]:.3f}")
        adapters.append(adapter)
        torch.save(
            {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            rd / f"adapter_input_ut={input_ut}.pt",
        )

    print("\n[3/3] Evaluating each adapter at each output ut step ...")
    # Build (n_ut input × n_ut output) accuracy and r-vs-target matrices.
    acc_matrix = np.zeros((n_ut, n_ut), dtype=np.float32)
    r_matrix = np.zeros((n_ut, n_ut), dtype=np.float32)
    for input_ut, adapter in enumerate(adapters):
        for output_ut in range(n_ut):
            # Score with exit_at_step=output_ut so the LM head reads from
            # hidden_states_list[output_ut].
            scores_list_ut = []
            for i in range(0, test_H.shape[0], 8):
                chunk = test_H[i:i+8, input_ut]
                scores_list_ut.extend(_adapter_scores_batched(
                    model, adapter, chunk, args.layer, label_seqs,
                    forward_kwargs={"exit_at_step": output_ut},
                ))
            preds = [_argmax_label(s) for s in scores_list_ut]
            acc = sum(preds[i] == test_labels[i] for i in range(len(test_labels))) / len(test_labels)
            r = _signed_r(val_proj(scores_list_ut), test_target_valence)
            acc_matrix[input_ut, output_ut] = acc
            r_matrix[input_ut, output_ut] = r
            print(f"  input_ut={input_ut} → output_ut={output_ut}: "
                  f"top1={acc:.3f}  r vs target={r:+.3f}")

    np.save(rd / "acc_matrix.npy", acc_matrix)
    np.save(rd / "r_matrix.npy", r_matrix)

    print("\n=== Top-1 accuracy matrix (rows=input_ut, cols=output_ut) ===")
    print(f"  {'':>10} " + " ".join(f"{f'out={u}':>8}" for u in range(n_ut)))
    for input_ut in range(n_ut):
        print(f"  in={input_ut}     " + " ".join(f"{acc_matrix[input_ut, u]:>8.3f}" for u in range(n_ut)))
    print("\n=== r vs target valence matrix ===")
    print(f"  {'':>10} " + " ".join(f"{f'out={u}':>8}" for u in range(n_ut)))
    for input_ut in range(n_ut):
        print(f"  in={input_ut}     " + " ".join(f"{r_matrix[input_ut, u]:>+8.3f}" for u in range(n_ut)))

    diag = np.array([acc_matrix[u, u] for u in range(n_ut)])
    diag_r = np.array([r_matrix[u, u] for u in range(n_ut)])
    print("\nDiagonal (input_ut == output_ut):")
    for u in range(n_ut):
        print(f"  ut={u}: top1={diag[u]:.3f}  r vs target={diag_r[u]:+.3f}")

    summary = {
        "acc_matrix": acc_matrix.tolist(),
        "r_matrix": r_matrix.tolist(),
        "diagonal_top1": diag.tolist(),
        "diagonal_r_vs_target": diag_r.tolist(),
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for axi, mat, title, vmin, vmax in [
            (ax[0], acc_matrix, "Top-1 accuracy (n=60 naturalistic)", 0.0, 0.6),
            (ax[1], r_matrix, "r vs target valence", -1.0, 1.0),
        ]:
            im = axi.imshow(mat, cmap="RdBu_r" if "r vs" in title else "viridis",
                             vmin=vmin, vmax=vmax, aspect="equal")
            for i in range(n_ut):
                for j in range(n_ut):
                    axi.text(j, i, f"{mat[i,j]:+.2f}" if "r" in title else f"{mat[i,j]:.2f}",
                              ha="center", va="center", fontsize=9,
                              color=("white" if abs(mat[i,j]) > 0.4 else "black"))
            axi.set_xticks(range(n_ut))
            axi.set_yticks(range(n_ut))
            axi.set_xticklabels([f"out_ut={u}" for u in range(n_ut)])
            axi.set_yticklabels([f"in_ut={u}" for u in range(n_ut)])
            axi.set_title(title)
            fig.colorbar(im, ax=axi)
        fig.tight_layout()
        fig.savefig(rd / "per_ut_adapter.png", dpi=140)
        print(f"\nSaved {rd / 'per_ut_adapter.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
