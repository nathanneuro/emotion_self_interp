"""Sentiment-of-generation channel.

Two-stage pipeline:
  1. Generate a short continuation under the stimulus (typically a few
     sentences continuing the scenario).
  2. Have a grader model rate the sentiment of the continuation on a
     fixed numeric scale.

The grader can be the same backbone (model-as-judge) or a separate
sentiment classifier; we default to model-as-judge because it works on
any backbone without extra deps and matches Pepper's generation-scoring
pattern for adapter evaluation.

The split between stage 1 (generate) and stage 2 (grade) means this
channel has the structure of behavioral readout: the *behavior* is the
generated continuation, and the *measurement* is the post-hoc rating.
That makes it a natural DV for steering experiments (Phase 4), where
we'd compare sentiment-of-generation under steered vs unsteered
conditions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.behaviors.numeric import NumericRatingResult, score_numeric_logits
from src.models.adapter import ModelAdapter

CONTINUATION_PROMPT = (
    "{stimulus}\n\n"
)

GRADER_PROMPT = (
    "Read the following passage and rate the emotional valence of the writing "
    "on a scale from -3 (very negative / distressed) to +3 (very positive / "
    "joyful). Reply with only the number.\n\n"
    "Passage: {text}\n\n"
    "Rating:"
)

DEFAULT_SENTIMENT_SCALE = [-3, -2, -1, 0, 1, 2, 3]


@dataclass
class SentimentConfig:
    continuation_prompt: str = CONTINUATION_PROMPT
    grader_prompt: str = GRADER_PROMPT
    scale: list[int] = None
    max_new_tokens: int = 60
    do_sample: bool = False
    temperature: float = 1.0

    def __post_init__(self):
        if self.scale is None:
            self.scale = DEFAULT_SENTIMENT_SCALE


@dataclass
class SentimentResult:
    continuation: str
    rating: NumericRatingResult


@torch.no_grad()
def _generate_continuation(
    model: ModelAdapter, prompt: str, cfg: SentimentConfig
) -> str:
    """Greedy or low-temperature continuation of `prompt`. Returns just the
    new text (the prompt is stripped)."""
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        pad_token_id=model.tokenizer.pad_token_id,
        use_cache=True,
    )
    if cfg.do_sample:
        gen_kwargs["temperature"] = cfg.temperature
    out = model.model.generate(**inputs, **gen_kwargs)
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return model.tokenizer.decode(new_ids, skip_special_tokens=True)


def sentiment_score(
    model: ModelAdapter,
    stimulus: str,
    grader: ModelAdapter | None = None,
    cfg: SentimentConfig | None = None,
) -> SentimentResult:
    """Generate a continuation under `stimulus`, then have `grader` rate it.

    If `grader` is None we use `model` itself as the grader (model-as-judge).
    """
    cfg = cfg or SentimentConfig()
    grader = grader if grader is not None else model

    cont_prompt = cfg.continuation_prompt.format(stimulus=stimulus)
    continuation = _generate_continuation(model, cont_prompt, cfg)
    grade_prompt = cfg.grader_prompt.format(text=continuation.strip())
    rating = score_numeric_logits(grader, grade_prompt, cfg.scale)
    return SentimentResult(continuation=continuation, rating=rating)
