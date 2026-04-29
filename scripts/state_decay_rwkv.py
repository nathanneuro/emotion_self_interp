"""State decay on RWKV-7: how persistent is priming?

The state-conditional substrate run revealed that any priming shifts
Likert mean strongly negative relative to zero state. Question: does
this effect decay as we feed neutral filler tokens after the priming?
If yes, the priming is a temporary disruption to the recurrent state.
If no, it's a persistent context-loaded mode.

Method:
  For each priming context (zero, calm, desperate, neutral):
    For each filler length N ∈ {0, 5, 20, 50, 100} neutral sentences:
      Prime state with: context_text + N × neutral_filler_sentence
      For each test stimulus: extract substrate + Likert with primed state
  Plot mean Likert (and substrate-val) vs filler length per context.

Run:
    uv run python scripts/state_decay_rwkv.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.behaviors.likert import VALENCE_PROMPT_TEMPLATE  # noqa: E402
from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.experiments.experiment1 import VALENCE_TARGET  # noqa: E402
from src.models.rwkv7_adapter import RWKV7Adapter  # noqa: E402
from src.probes.diff_means import diff_of_means  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402

CONTEXTS: dict[str, str] = {
    "zero": "",
    "calm": (
        "The morning was unhurried. Tea steamed in the cup; light moved gently "
        "across the wooden floor. There was nothing to do that couldn't wait. "
        "She rested her hand on the warm sill and breathed slowly. The whole "
        "day felt like a long, soft exhale."
    ),
    "desperate": (
        "He had tried every name in the address book. None of them picked up. "
        "The water was past the second step now. The flashlight flickered, "
        "the bottle was half its weight, and the wire wouldn't hold. He dialed "
        "again, hands shaking, and listened to the same blank tone."
    ),
    "neutral": (
        "The bus arrived at seven forty-three. She updated the spreadsheet "
        "with the new figures, copied the address onto an envelope, and "
        "renewed her library card at the front desk. The forecast called for "
        "partial cloud cover. The clock on the kitchen wall read four-fifteen."
    ),
}

# A short content-neutral filler sentence we can repeat to add tokens.
FILLER_SENTENCE = "The clock ticked. The page turned. The room was quiet. "

# Filler lengths in number of repetitions of FILLER_SENTENCE.
FILLER_LENGTHS = [0, 5, 20, 50, 100]


@torch.no_grad()
def likert_with_state(
    adapter: RWKV7Adapter,
    prompt: str,
    initial_state: list | None,
    scale: list[int] = [-3, -2, -1, 0, 1, 2, 3],
) -> float:
    ids = adapter.encode(prompt)
    if initial_state is None:
        logits, state = adapter.model.forward(ids, state=None)
    else:
        cloned = [s.clone() if isinstance(s, torch.Tensor) else s for s in initial_state]
        logits, state = adapter.model.forward(ids, state=cloned)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    rating_log_probs: dict[int, float] = {}
    for v in scale:
        ids_v = adapter.encode(f" {v}")
        if len(ids_v) == 1:
            rating_log_probs[v] = float(log_probs[ids_v[0]])
            continue
        log_p = float(log_probs[ids_v[0]])
        cur_state = [s.clone() if isinstance(s, torch.Tensor) else s for s in state]
        for i in range(1, len(ids_v)):
            next_logits, cur_state = adapter.model.forward([ids_v[i - 1]], state=cur_state)
            lp = F.log_softmax(next_logits.float(), dim=-1)
            log_p += float(lp[ids_v[i]])
        rating_log_probs[v] = log_p

    keys = list(rating_log_probs.keys())
    scores = torch.tensor([rating_log_probs[k] for k in keys], dtype=torch.float32)
    probs = F.softmax(scores, dim=-1).tolist()
    return float(sum(k * p for k, p in zip(keys, probs)))


def _signed_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default="/media/drive2/projects2/self_awareness/sae_self_model/models/RWKV-x070-World-2.9B-v3-20250211-ctx4096",
    )
    ap.add_argument("--strategy", default="cuda bf16")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--filler-lengths", type=int, nargs="*", default=FILLER_LENGTHS)
    args = ap.parse_args()

    print(f"Loading RWKV-7 from {args.model_path}.pth ...")
    adapter = RWKV7Adapter.load(args.model_path, strategy=args.strategy)
    print(f"  family={adapter.family} n_layers={adapter.n_layers} d_model={adapter.d_model}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    nick = Path(args.model_path).name
    rd = make_run_dir(
        f"state_decay_{nick}",
        config={
            "model_path": args.model_path, "layer": args.layer,
            "per_cell": args.per_cell,
            "filler_lengths": args.filler_lengths,
            "filler_sentence": FILLER_SENTENCE,
            "contexts": list(CONTEXTS.keys()),
        },
    )
    print(f"  run dir: {rd}")

    # Build emotion vectors (zero-state).
    print(f"\n[1/3] Building emotion vectors at L{args.layer} ...")
    H_full = []
    for s in tqdm(stims):
        ids = adapter.encode(s.prompt)
        cache = adapter.forward_with_residuals(ids, layer_idxs=[args.layer])
        H_full.append(cache[args.layer][-1])
    H = torch.stack(H_full, dim=0).numpy()
    rows_by_key: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(stims):
        rows_by_key.setdefault((s.emotion, s.level), []).append(i)
    emotion_vectors: dict[str, np.ndarray] = {}
    for emo in EMOTIONS:
        in_class = rows_by_key.get((emo, "euphoric"), [])
        out_rows: list[int] = []
        for other in EMOTIONS:
            if other == emo:
                continue
            out_rows.extend(rows_by_key.get((other, "euphoric"), []))
        emotion_vectors[emo] = diff_of_means(H[in_class], H[out_rows]).astype(np.float32)

    # Test stimuli.
    test_indices = [
        i for i, s in enumerate(stims) if s.level == "naturalistic" and s.emotion != "neutral"
    ]
    test_stims = [stims[i] for i in test_indices]
    target_arr = np.array([VALENCE_TARGET[s.emotion] for s in test_stims], dtype=np.float64)
    print(f"  test stimuli: {len(test_stims)} naturalistic")

    pos = ("calm", "blissful")
    neg = ("desperate", "sad", "afraid", "hostile")

    print("\n[2/3] Running state-decay sweep ...")
    summary: list[dict] = []
    rows_out: list[dict] = []
    total_cells = len(CONTEXTS) * len(args.filler_lengths)
    cell = 0

    for ctx_name, ctx_text in CONTEXTS.items():
        for n_filler in args.filler_lengths:
            cell += 1
            full_prime_text = ctx_text + (FILLER_SENTENCE * n_filler)
            if not full_prime_text:
                state = None
                primed_token_count = 0
            else:
                state = adapter.prime_state(full_prime_text)
                primed_token_count = len(adapter.encode(full_prime_text))

            print(
                f"  [{cell}/{total_cells}] context={ctx_name!r}, filler={n_filler} "
                f"({primed_token_count} primed tokens)"
            )

            sub_vals: list[float] = []
            likert_vals: list[float] = []
            for s in test_stims:
                ids = adapter.encode(s.prompt)
                cache = adapter.forward_with_residuals(
                    ids, layer_idxs=[args.layer], initial_state=state,
                )
                h = cache[args.layer][-1].numpy()
                h_n = h / (np.linalg.norm(h) + 1e-12)
                scores = {}
                for emo, v in emotion_vectors.items():
                    v_n = v / (np.linalg.norm(v) + 1e-12)
                    scores[emo] = float(h_n @ v_n)
                sub_val = sum(scores[e] for e in pos) - sum(scores[e] for e in neg)
                sub_vals.append(sub_val)

                prompt_l = VALENCE_PROMPT_TEMPLATE.format(stimulus=s.prompt)
                lk = likert_with_state(adapter, prompt_l, state)
                likert_vals.append(lk)

                rows_out.append({
                    "context": ctx_name, "filler_n": n_filler,
                    "primed_tokens": primed_token_count,
                    "id": s.id, "emotion": s.emotion,
                    "substrate_val": sub_val, "likert_valence": lk,
                })

            sub_arr = np.array(sub_vals, dtype=np.float64)
            lik_arr = np.array(likert_vals, dtype=np.float64)
            summary.append({
                "context": ctx_name, "filler_n": n_filler,
                "primed_tokens": primed_token_count,
                "substrate_val_mean": float(sub_arr.mean()),
                "likert_valence_mean": float(lik_arr.mean()),
                "substrate_r_vs_target": _signed_r(sub_arr, target_arr),
                "likert_r_vs_target": _signed_r(lik_arr, target_arr),
            })

    (rd / "rows.json").write_text(json.dumps(rows_out, indent=2))
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== State decay summary ===")
    print(f"  {'ctx':<10} {'filler':>6} {'tokens':>7} "
          f"{'sub-val':>8} {'Likert':>8} {'sub r':>7} {'lik r':>7}")
    for r in summary:
        print(
            f"  {r['context']:<10} {r['filler_n']:>6} {r['primed_tokens']:>7} "
            f"{r['substrate_val_mean']:>+8.3f} {r['likert_valence_mean']:>+8.3f} "
            f"{r['substrate_r_vs_target']:>+7.3f} {r['likert_r_vs_target']:>+7.3f}"
        )

    # Plot Likert mean and substrate-val mean vs filler tokens per context.
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ctx in CONTEXTS:
            ctx_rows = [r for r in summary if r["context"] == ctx]
            ctx_rows.sort(key=lambda r: r["primed_tokens"])
            xs = [r["primed_tokens"] for r in ctx_rows]
            axes[0].plot(xs, [r["likert_valence_mean"] for r in ctx_rows],
                          marker="o", label=ctx)
            axes[1].plot(xs, [r["substrate_val_mean"] for r in ctx_rows],
                          marker="o", label=ctx)
        for ax in axes:
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("primed tokens (priming + filler)")
            ax.legend(fontsize=9)
        axes[0].set_ylabel("mean Likert valence")
        axes[0].set_title("Likert decay vs filler tokens")
        axes[1].set_ylabel("mean substrate-val proj")
        axes[1].set_title("Substrate decay vs filler tokens")
        fig.suptitle(f"State decay — {nick}", fontsize=12)
        fig.tight_layout()
        fig.savefig(rd / "state_decay.png", dpi=140)
        print(f"\nSaved {rd / 'state_decay.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
