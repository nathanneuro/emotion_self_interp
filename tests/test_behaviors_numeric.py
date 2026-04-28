"""Tests for the numeric-rating extraction. Pure parsing math; no model load."""
from __future__ import annotations

import re

import pytest

from src.behaviors.numeric import _NUMERIC_RE


@pytest.mark.parametrize(
    "text,expected",
    [
        (" 3", "3"),
        ("3", "3"),
        ("-2.5 because", "-2.5"),
        ("Rating: -1", "-1"),
        ("The answer is 0.", "0"),
        # "- 2" with a stray space between the sign and digits is treated
        # as if the sign were missing — that's a reasonable robustness call.
        ("- 2", "2"),
    ],
)
def test_numeric_re_matches(text, expected):
    m = _NUMERIC_RE.search(text)
    assert m is not None, f"no match in {text!r}"
    assert m.group(0) == expected


def test_numeric_re_skips_non_numeric():
    assert _NUMERIC_RE.search("the cat sat") is None
    assert _NUMERIC_RE.search("") is None
