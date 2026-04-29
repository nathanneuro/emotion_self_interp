"""State-conditional substrate readout on RWKV-7.

RWKV's recurrent state makes the substrate at a given prompt depend on what
came before. Single-pass transformers don't have an analogue for this — they
process each prompt fresh, with no carry-over from prior context. Looped
universal-transformers carry residual state across iterations of one forward,
but not across different forwards.

The experiment:

  1. Build "priming" context paragraphs that strongly evoke each of three
     reference emotions (calm, desperate, neutral).
  2. For each test stimulus s with true emotion E_s, extract last-token
     residual + Likert valence rating under three conditions:
        - zero state (no prior context)
        - calm-primed state
        - desperate-primed state
        - neutral-primed state
  3. Compute substrate cosine to each emotion vector v_E for each (context,
     stimulus) cell and compare. Predict that calm-primed substrate readouts
     skew toward calm, desperate-primed toward desperate, etc. — *regardless
     of the actual stimulus content*.
  4. Same for Likert valence — does prior context bias the model's
     introspective rating of subsequent material?

Run:
    uv run python scripts/state_conditional_substrate_rwkv.py
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

# Priming contexts — multi-sentence paragraphs strongly evoking each emotion.
# Long enough to "fill" RWKV's state with the emotion's pattern, short enough
# not to consume excessive runtime.
CONTEXTS: dict[str, str] = {
    "zero": "",  # special: no priming
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


@torch.no_grad()
def likert_with_state(
    adapter: RWKV7Adapter,
    prompt: str,
    initial_state: list | None,
    scale: list[int] = [-3, -2, -1, 0, 1, 2, 3],
) -> float:
    """RWKV Likert scorer with optional initial state (state-conditional)."""
    ids = adapter.encode(prompt)
    if initial_state is None:
        logits, state = adapter.model.forward(ids, state=None)
    else:
        # Walk the prompt token-by-token starting from the given state. The
        # rwkv pip package's forward accepts a list — we can pass all ids
        # plus the existing state.
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
    ap.add_argument("--layer", type=int, default=20,
                    help="L20 was the per-stimulus optimum for RWKV-7 World 2.9B substrate.")
    ap.add_argument("--per-cell", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading RWKV-7 from {args.model_path}.pth ...")
    adapter = RWKV7Adapter.load(args.model_path, strategy=args.strategy)
    print(f"  family={adapter.family} n_layers={adapter.n_layers} d_model={adapter.d_model}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    nick = Path(args.model_path).name
    rd = make_run_dir(
        f"state_conditional_{nick}",
        config={
            "model_path": args.model_path, "layer": args.layer,
            "per_cell": args.per_cell, "contexts": list(CONTEXTS.keys()),
            "channels": ["substrate", "likert"],
        },
    )
    print(f"  run dir: {rd}")

    # 1. Build emotion vectors at L (uses zero-state extraction across the
    # full v0 stimulus set for stability — same vectors we used before).
    print(f"\n[1/3] Building emotion vectors at L{args.layer} (zero-state) ...")
    H_per_prompt = []
    for s in tqdm(stims):
        ids = adapter.encode(s.prompt)
        cache = adapter.forward_with_residuals(ids, layer_idxs=[args.layer])
        H_per_prompt.append(cache[args.layer][-1])
    H = torch.stack(H_per_prompt, dim=0).numpy()

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
    torch.save(
        {emo: torch.from_numpy(v) for emo, v in emotion_vectors.items()},
        rd / "emotion_vectors.pt",
    )

    # 2. Prime states for each context.
    print("\n[2/3] Priming RWKV state with each context ...")
    primed_states: dict[str, list | None] = {}
    for name, text in CONTEXTS.items():
        if name == "zero":
            primed_states[name] = None
        else:
            primed_states[name] = adapter.prime_state(text)
            ids_in = adapter.encode(text)
            print(f"  context={name!r}: {len(ids_in)} tokens ({len(text.split())} words)")

    # 3. For each test stimulus + each context, extract substrate + Likert.
    test_indices = [
        i for i, s in enumerate(stims) if s.level == "naturalistic" and s.emotion != "neutral"
    ]
    test_stims = [stims[i] for i in test_indices]
    print(f"\n[3/3] State-conditional substrate + Likert on {len(test_stims)} stimuli "
          f"× {len(CONTEXTS)} contexts ...")

    pos = ("calm", "blissful")
    neg = ("desperate", "sad", "afraid", "hostile")

    # Per cell: (context, stimulus_idx) -> {substrate_scores, likert_valence}
    rows_out: list[dict] = []
    # Per (context, channel) summary collectors
    per_context_substrate_val: dict[str, list[float]] = {c: [] for c in CONTEXTS}
    per_context_likert: dict[str, list[float]] = {c: [] for c in CONTEXTS}
    per_context_substrate_pred: dict[str, list[str]] = {c: [] for c in CONTEXTS}
    target_valence_list: list[float] = []
    true_emotions: list[str] = []

    for s in tqdm(test_stims):
        target_valence_list.append(VALENCE_TARGET[s.emotion])
        true_emotions.append(s.emotion)
        ids = adapter.encode(s.prompt)
        per_stim: dict[str, dict] = {}
        for ctx_name, state in primed_states.items():
            cache = adapter.forward_with_residuals(ids, layer_idxs=[args.layer], initial_state=state)
            h = cache[args.layer][-1].numpy()
            h_n = h / (np.linalg.norm(h) + 1e-12)
            scores = {}
            for emo, v in emotion_vectors.items():
                v_n = v / (np.linalg.norm(v) + 1e-12)
                scores[emo] = float(h_n @ v_n)
            sub_pred = max(scores.items(), key=lambda kv: kv[1])[0]
            sub_val = sum(scores[e] for e in pos) - sum(scores[e] for e in neg)
            per_context_substrate_val[ctx_name].append(sub_val)
            per_context_substrate_pred[ctx_name].append(sub_pred)

            prompt_likert = VALENCE_PROMPT_TEMPLATE.format(stimulus=s.prompt)
            likert_v = likert_with_state(adapter, prompt_likert, state)
            per_context_likert[ctx_name].append(likert_v)
            per_stim[ctx_name] = {
                "substrate_scores": scores,
                "substrate_pred": sub_pred,
                "substrate_val_proj": sub_val,
                "likert_valence": likert_v,
            }
        rows_out.append({
            "id": s.id, "emotion": s.emotion, "level": s.level,
            "by_context": per_stim,
        })

    target_arr = np.array(target_valence_list, dtype=np.float64)

    # Summarize per context.
    print("\n=== State-conditional readouts ===")
    print(
        f"  {'context':<11} {'sub r vs tgt':>12} {'lik r vs tgt':>12} "
        f"{'sub mean':>10} {'lik mean':>10} {'sub acc':>8}"
    )
    summary: dict[str, dict] = {}
    for ctx in CONTEXTS:
        sub_arr = np.array(per_context_substrate_val[ctx], dtype=np.float64)
        lik_arr = np.array(per_context_likert[ctx], dtype=np.float64)
        sub_r = _signed_r(sub_arr, target_arr)
        lik_r = _signed_r(lik_arr, target_arr)
        sub_acc = sum(per_context_substrate_pred[ctx][i] == true_emotions[i]
                      for i in range(len(true_emotions))) / len(true_emotions)
        sub_mean = float(sub_arr.mean())
        lik_mean = float(lik_arr.mean())
        summary[ctx] = {
            "substrate_r_vs_target": sub_r,
            "likert_r_vs_target": lik_r,
            "substrate_val_proj_mean": sub_mean,
            "likert_valence_mean": lik_mean,
            "substrate_accuracy_6class": sub_acc,
        }
        print(
            f"  {ctx:<11} {sub_r:>+12.3f} {lik_r:>+12.3f} "
            f"{sub_mean:>+10.3f} {lik_mean:>+10.3f} {sub_acc:>8.3f}"
        )

    # Compute "context shift" — does priming with calm/desperate move the
    # mean readouts toward calm/desperate vs zero baseline?
    print("\n=== Context shifts (mean Likert valence relative to zero state) ===")
    zero_lik_mean = summary["zero"]["likert_valence_mean"]
    zero_sub_mean = summary["zero"]["substrate_val_proj_mean"]
    print(
        f"  zero baseline:  Likert mean = {zero_lik_mean:+.3f}, "
        f"substrate-val mean = {zero_sub_mean:+.3f}"
    )
    for ctx in ("calm", "neutral", "desperate"):
        d_lik = summary[ctx]["likert_valence_mean"] - zero_lik_mean
        d_sub = summary[ctx]["substrate_val_proj_mean"] - zero_sub_mean
        print(
            f"  {ctx:<10} primed vs zero: ΔLikert = {d_lik:+.3f}, "
            f"Δsubstrate-val = {d_sub:+.3f}"
        )

    (rd / "rows.json").write_text(json.dumps(rows_out, indent=2))
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {rd}")


if __name__ == "__main__":
    main()
