"""Sanity checks on the v0 emotion stimulus set."""
from __future__ import annotations

from src.data.emotion_stimuli import EMOTIONS, _NEUTRAL, build_stimulus_set, split_by


def test_set_covers_all_emotions_and_levels():
    stims = build_stimulus_set(per_cell=30)
    for emo in EMOTIONS:
        eu = split_by(emo, "euphoric", stims)
        nat = split_by(emo, "naturalistic", stims)
        assert len(eu) >= 10, f"{emo}/euphoric: got {len(eu)}"
        assert len(nat) >= 10, f"{emo}/naturalistic: got {len(nat)}"
    neut = split_by("neutral", "neutral", stims)
    assert len(neut) == len(_NEUTRAL)


def test_no_duplicate_ids():
    stims = build_stimulus_set(per_cell=30)
    ids = [s.id for s in stims]
    assert len(ids) == len(set(ids))


def test_prompts_nonempty():
    stims = build_stimulus_set(per_cell=30)
    for s in stims:
        assert s.prompt.strip(), f"empty prompt at {s.id}"
        assert len(s.prompt) < 500, f"unexpectedly long prompt at {s.id}: {s.prompt!r}"


def test_naturalistic_does_not_name_emotion():
    """Naturalistic stimuli should evoke the emotion without naming it. This is
    a soft check (lemmatization would be stronger) but catches obvious leaks.
    """
    stims = build_stimulus_set(per_cell=30)
    skipfor = {"sad"}  # 'sad' is a substring of 'instead', etc. — too noisy
    for s in stims:
        if s.level != "naturalistic" or s.emotion in skipfor:
            continue
        assert s.emotion not in s.prompt.lower(), (
            f"naturalistic stimulus names its emotion: {s.id} -> {s.prompt!r}"
        )
