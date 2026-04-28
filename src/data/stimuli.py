"""Stimulus schema shared across phases.

A stimulus pairs a prompt with an emotion label and the level of the probe
(constrained euphoric/dysphoric, naturalistic scenario, or neutral control).
The same schema feeds Phase 1 (vector extraction), Phase 3 (behavioral), and
Phase 5 (Experiment 1).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Level = Literal["euphoric", "dysphoric", "naturalistic", "neutral"]


@dataclass
class Stimulus:
    id: str
    emotion: str
    level: Level
    prompt: str
    target_position: int | str = -1
    meta: dict = field(default_factory=dict)


def load_stimuli(path: str | Path) -> list[Stimulus]:
    raw = json.loads(Path(path).read_text())
    return [Stimulus(**r) for r in raw]


def save_stimuli(stims: list[Stimulus], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(s) for s in stims], indent=2))
