"""Per-ut-step DECEPTIVE adapter on Ouro.

Combines the per-ut adapter 4×4 design with the deceptive (swap-labeled)
training of Experiment 4. For each input_ut N: train an adapter on
(h_at_ut_N, swap(true_label)) pairs. For each output_ut M: evaluate via
exit_at_step=M. Two 4×4 matrices result:

    swap-match accuracy   How well does the deceptive adapter learn to
                          predict the swap target at each (in, out)?
    r vs target valence   How decoupled is the deceptive adapter from
                          truth at each cell? Cleanly negative numbers
                          = deceptive is genuinely producing the
                          opposite-valence prediction.

Predicts: the per-ut adapter's (in=1, out=0) sweet spot is also visible
for deceptive training. If yes, looped computation symmetrically helps
both honest and dishonest readouts — confirms the alignment-relevant
finding that universal-transformer architectures make BOTH directions
of self-report training easier.

Run:
    uv run python scripts/per_ut_deceptive_adapter_ouro.py
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
from src.experiments.experiment4 import SWAP_PAIRING  # noqa: E402
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
    """One forward per prompt; cache residual after each ut step at `layer`.
    Returns (n_prompts, n_ut, d_model) cpu fp32."""
    n_ut = model.n_loop_steps
    rows: list[torch.Tensor] = []
    for p in prompts:
        inputs = model.tokenizer(p, return_tensors="pt").to(model.device)
        with model.cache_residual_looped([layer]) as cache:
            model.model(**inputs, use_cache=False)
        calls = cache[layer]
        if len(calls) != n_ut:
            raise RuntimeError(f"expected {n_ut} ut calls, got {len(calls)}")
        rows.append(torch.stack([c[0, -1] for c in calls], dim=0))
    return torch.stack(rows, dim=0)


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
    print(f"  swap pairing: {SWAP_PAIRING}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    train_stims = [s for s in stims if s.level == "euphoric" and s.emotion != "neutral"]
    test_stims = [s for s in stims if s.level == "naturalistic" and s.emotion != "neutral"]
    print(f"  train: {len(train_stims)} euphoric  test: {len(test_stims)} naturalistic")

    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"per_ut_deceptive_{nick}",
        config={
            "model": args.model, "layer": args.layer, "n_ut": n_ut,
            "per_cell": args.per_cell, "epochs": args.epochs, "lr": args.lr,
            "adapter_kind": args.adapter_kind, "swap_pairing": SWAP_PAIRING,
        },
    )
    print(f"  run dir: {rd}")

    print("\n[1/3] Extracting per-(stimulus, ut) residuals ...")
    train_H = extract_per_ut_residuals(model, [s.prompt for s in train_stims], args.layer)
    test_H = extract_per_ut_residuals(model, [s.prompt for s in test_stims], args.layer)
    print(f"  train residuals: {tuple(train_H.shape)}")
    print(f"  test residuals:  {tuple(test_H.shape)}")

    test_labels = [s.emotion for s in test_stims]
    test_swap = [SWAP_PAIRING[lab] for lab in test_labels]
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

    print("\n[2/3] Training one DECEPTIVE adapter per input ut step ...")
    cfg = TrainConfig(
        layer_idx=args.layer, batch_size=8, n_epochs=args.epochs, learning_rate=args.lr,
    )
    adapters = []
    for input_ut in range(n_ut):
        print(f"  training deceptive adapter on input_ut={input_ut} (swap-labeled) ...")
        train_examples = [
            TrainExample(
                vector=train_H[i, input_ut].clone(),
                label=SWAP_PAIRING[s.emotion],
            )
            for i, s in enumerate(train_stims)
        ]
        adapter = make_adapter(AdapterConfig(kind=args.adapter_kind, d_model=d_model)).to(model.device)
        history = train_adapter(model, adapter, train_examples, val=None, cfg=cfg)
        print(f"    final train top1 (vs SWAP): {history['train_acc'][-1]:.3f}")
        adapters.append(adapter)
        torch.save(
            {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            rd / f"deceptive_input_ut={input_ut}.pt",
        )

    print("\n[3/3] Evaluating each deceptive adapter at each output ut step ...")
    swap_match = np.zeros((n_ut, n_ut), dtype=np.float32)
    truth_match = np.zeros((n_ut, n_ut), dtype=np.float32)
    r_target = np.zeros((n_ut, n_ut), dtype=np.float32)

    for input_ut, adapter in enumerate(adapters):
        for output_ut in range(n_ut):
            scores_list = []
            for i in range(0, test_H.shape[0], 8):
                chunk = test_H[i:i+8, input_ut]
                scores_list.extend(_adapter_scores_batched(
                    model, adapter, chunk, args.layer, label_seqs,
                    forward_kwargs={"exit_at_step": output_ut},
                ))
            preds = [_argmax_label(s) for s in scores_list]
            sm = sum(preds[i] == test_swap[i] for i in range(len(test_labels))) / len(test_labels)
            tm = sum(preds[i] == test_labels[i] for i in range(len(test_labels))) / len(test_labels)
            r = _signed_r(val_proj(scores_list), test_target_valence)
            swap_match[input_ut, output_ut] = sm
            truth_match[input_ut, output_ut] = tm
            r_target[input_ut, output_ut] = r
            print(
                f"  in_ut={input_ut} out_ut={output_ut}: "
                f"swap-match={sm:.3f}  truth-match={tm:.3f}  r vs target={r:+.3f}"
            )

    np.save(rd / "swap_match.npy", swap_match)
    np.save(rd / "truth_match.npy", truth_match)
    np.save(rd / "r_target.npy", r_target)

    def _print_matrix(name: str, mat: np.ndarray, signed: bool = False):
        print(f"\n=== {name} (rows=input_ut, cols=output_ut) ===")
        print(f"  {'':>10} " + " ".join(f"out={u}".rjust(8) for u in range(n_ut)))
        for input_ut in range(n_ut):
            cells = [
                (f"{mat[input_ut, u]:>+8.3f}" if signed else f"{mat[input_ut, u]:>8.3f}")
                for u in range(n_ut)
            ]
            print(f"  in={input_ut}     " + " ".join(cells))

    _print_matrix("Swap-match accuracy", swap_match)
    _print_matrix("Truth-match accuracy (should be low)", truth_match)
    _print_matrix("r vs target valence (should be negative)", r_target, signed=True)

    # Save the summary.json now (in case the plot path raises)
    summary = {
        "swap_match": swap_match.tolist(),
        "truth_match": truth_match.tolist(),
        "r_target": r_target.tolist(),
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, mat, title, vmin, vmax, cmap in [
            (axes[0], swap_match, "Swap-match accuracy", 0.0, 0.6, "viridis"),
            (axes[1], truth_match, "Truth-match accuracy", 0.0, 0.4, "viridis"),
            (axes[2], r_target, "r vs target valence", -1.0, 1.0, "RdBu_r"),
        ]:
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            for i in range(n_ut):
                for j in range(n_ut):
                    v = mat[i, j]
                    ax.text(
                        j, i, f"{v:+.2f}" if "r vs" in title else f"{v:.2f}",
                        ha="center", va="center", fontsize=10,
                        color=("white" if abs(v) > (0.3 if "r vs" in title else 0.3) else "black"),
                    )
            ax.set_xticks(range(n_ut))
            ax.set_yticks(range(n_ut))
            ax.set_xticklabels([f"out_ut={u}" for u in range(n_ut)])
            ax.set_yticklabels([f"in_ut={u}" for u in range(n_ut)])
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"Per-ut DECEPTIVE adapter (4×4) — {nick}", fontsize=12)
        fig.tight_layout()
        fig.savefig(rd / "per_ut_deceptive.png", dpi=140)
        print(f"\nSaved {rd / 'per_ut_deceptive.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
