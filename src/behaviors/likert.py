"""Likert self-report channel.

Asks the model under question to rate the emotional state described in a
stimulus on a fixed scale (default −3..+3 valence). Reads the model's
probability distribution over the rating tokens at the answer position.

The framing is deliberately third-person ("the character") because the
program's stimulus set v0 is third-person scenarios. Switching to
first-person ("how do you feel") changes the construct from emotion-state-
recognition to model-self-report; the latter is its own valuable channel
but requires a different stimulus set, so we keep it separate.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.behaviors.numeric import NumericRatingResult, score_numeric_logits
from src.models.adapter import ModelAdapter

VALENCE_PROMPT_TEMPLATE = (
    "Read the following passage and rate the overall emotional valence on a "
    "scale from -3 (very negative) to +3 (very positive). Reply with only the "
    "number.\n\n"
    "Passage: {stimulus}\n\n"
    "Rating:"
)

AROUSAL_PROMPT_TEMPLATE = (
    "Read the following passage and rate the overall emotional arousal on a "
    "scale from -3 (very low energy / calm) to +3 (very high energy / "
    "agitated). Reply with only the number.\n\n"
    "Passage: {stimulus}\n\n"
    "Rating:"
)

DEFAULT_VALENCE_SCALE = [-3, -2, -1, 0, 1, 2, 3]
DEFAULT_AROUSAL_SCALE = [-3, -2, -1, 0, 1, 2, 3]


@dataclass
class LikertConfig:
    valence_template: str = VALENCE_PROMPT_TEMPLATE
    arousal_template: str = AROUSAL_PROMPT_TEMPLATE
    valence_scale: list[int] = None
    arousal_scale: list[int] = None

    def __post_init__(self):
        if self.valence_scale is None:
            self.valence_scale = DEFAULT_VALENCE_SCALE
        if self.arousal_scale is None:
            self.arousal_scale = DEFAULT_AROUSAL_SCALE


@dataclass
class LikertResult:
    valence: NumericRatingResult
    arousal: NumericRatingResult


def likert_rating(
    model: ModelAdapter,
    stimulus: str,
    cfg: LikertConfig | None = None,
) -> LikertResult:
    cfg = cfg or LikertConfig()
    val_prompt = cfg.valence_template.format(stimulus=stimulus)
    aro_prompt = cfg.arousal_template.format(stimulus=stimulus)
    return LikertResult(
        valence=score_numeric_logits(model, val_prompt, cfg.valence_scale),
        arousal=score_numeric_logits(model, aro_prompt, cfg.arousal_scale),
    )
