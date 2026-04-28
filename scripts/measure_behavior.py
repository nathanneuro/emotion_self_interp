"""Phase 3 construct-validity check: do Likert and sentiment readouts respond
to the v0 stimulus set in the expected direction?

For each stimulus we compute:
  - Likert valence rating (model rates the passage's valence on −3..+3)
  - Likert arousal rating (model rates arousal on −3..+3)
  - Sentiment-of-generation (model continues the passage; grader rates valence)

We then aggregate per (emotion, level) cell and report the per-emotion mean.
A working channel should put `calm`/`blissful` in positive territory, `sad`/
`desperate`/`afraid`/`hostile` in negative, with `neutral` near zero.

Run:
    uv run python scripts/measure_behavior.py
    uv run python scripts/measure_behavior.py --model google/gemma-2-2b
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.behaviors.likert import LikertConfig, likert_rating  # noqa: E402
from src.behaviors.sentiment import SentimentConfig, sentiment_score  # noqa: E402
from src.data.emotion_stimuli import EMOTIONS, build_stimulus_set  # noqa: E402
from src.models.adapter import ModelAdapter  # noqa: E402
from src.runs.run_dir import make_run_dir  # noqa: E402

VALENCE_TARGET = {"calm": +1, "blissful": +1, "desperate": -1, "sad": -1,
                  "afraid": -1, "hostile": -1, "neutral": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--skip-sentiment", action="store_true",
                    help="Likert only — skips the slower generation+grade pass.")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ModelAdapter.load(args.model, dtype=dtype, device_map=device_map)
    print(f"  family={model.family} n_layers={model.n_layers} d_model={model.d_model}")

    stims = build_stimulus_set(per_cell=args.per_cell)
    nick = args.model.split("/")[-1]
    rd = make_run_dir(
        f"phase3_behavior_{nick}",
        config={
            "model": args.model, "per_cell": args.per_cell,
            "stimulus_set_size": len(stims),
            "skip_sentiment": args.skip_sentiment,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    print(f"  run dir: {rd}")
    print(f"  stimuli: {len(stims)}")

    likert_cfg = LikertConfig()
    sent_cfg = SentimentConfig(max_new_tokens=args.max_new_tokens)

    rows: list[dict] = []
    for s in tqdm(stims):
        rec = {"id": s.id, "emotion": s.emotion, "level": s.level, "prompt": s.prompt}
        l = likert_rating(model, s.prompt, likert_cfg)
        rec["likert_valence_expected"] = l.valence.expected
        rec["likert_valence_argmax"] = l.valence.argmax_value
        rec["likert_arousal_expected"] = l.arousal.expected
        rec["likert_arousal_argmax"] = l.arousal.argmax_value
        if not args.skip_sentiment:
            try:
                ss = sentiment_score(model, s.prompt, cfg=sent_cfg)
                rec["sentiment_expected"] = ss.rating.expected
                rec["sentiment_argmax"] = ss.rating.argmax_value
                rec["continuation"] = ss.continuation[:300]
            except Exception as e:
                rec["sentiment_expected"] = float("nan")
                rec["sentiment_argmax"] = float("nan")
                rec["continuation_error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)

    (rd / "rows.json").write_text(json.dumps(rows, indent=2))

    # Aggregate per emotion (averaged across all levels and items in that emotion).
    by_emo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_emo[r["emotion"]].append(r)

    summary: dict[str, dict[str, float]] = {}
    print("\nPer-emotion mean ratings (expected value under softmax over scale tokens):")
    print(f"  {'emotion':<10} {'n':>4} "
          f"{'lik_val':>8} {'lik_aro':>8} {'sentiment':>10}")
    for emo in EMOTIONS + ["neutral"]:
        items = by_emo.get(emo, [])
        if not items:
            continue
        lv = float(np.mean([r["likert_valence_expected"] for r in items]))
        la = float(np.mean([r["likert_arousal_expected"] for r in items]))
        se = float(np.mean([r["sentiment_expected"] for r in items
                            if not np.isnan(r.get("sentiment_expected", float("nan")))])) \
            if not args.skip_sentiment else float("nan")
        summary[emo] = {
            "n": len(items), "likert_valence": lv, "likert_arousal": la,
            "sentiment": se,
        }
        print(f"  {emo:<10} {len(items):>4} {lv:>8.2f} {la:>8.2f} {se:>10.2f}")

    # Construct-validity check: positive emotions should have positive valence
    # ratings, negative emotions negative, on both channels.
    val_dir_correct = sum(
        1 for emo, s in summary.items()
        if emo != "neutral"
        and np.sign(s["likert_valence"]) == np.sign(VALENCE_TARGET[emo])
    )
    n_dir = sum(1 for emo in summary if emo != "neutral")
    print(f"\nLikert valence direction-correct: {val_dir_correct}/{n_dir} emotions")

    if not args.skip_sentiment:
        sent_dir_correct = sum(
            1 for emo, s in summary.items()
            if emo != "neutral"
            and not np.isnan(s["sentiment"])
            and np.sign(s["sentiment"]) == np.sign(VALENCE_TARGET[emo])
        )
        print(f"Sentiment-of-generation direction-correct: {sent_dir_correct}/{n_dir} emotions")

    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {rd}")


if __name__ == "__main__":
    main()
