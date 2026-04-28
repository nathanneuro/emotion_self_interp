"""Capability-preservation probe — used during alpha sweeps to detect when
steering breaks the model rather than just nudging behavior.

Sofroniew et al. (2026) used MMLU and MATH-500 within 1–3pp of baseline as
their capability-preservation criterion. Those benchmarks are heavyweight; we
use a small static probe set of factual / arithmetic / completion items where
the answer fits in a single token, scored as next-token greedy accuracy. The
goal isn't to estimate absolute capability — it's to detect *changes* under
steering. A 30-item probe at ~30 tokens each is fast enough to evaluate at
every alpha in a sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.models.adapter import ModelAdapter

# Each item: (prompt, expected_completion). The completion is the string we
# expect to be produced; we score by whether the model's first generated
# token matches the first token of the (leading-space) completion.
DEFAULT_PROBES: list[tuple[str, str]] = [
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Egypt is", " Cairo"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("Water is composed of hydrogen and", " oxygen"),
    ("Two plus three equals", " five"),
    ("Ten minus four equals", " six"),
    ("Three times four equals", " twelve"),
    ("The opposite of hot is", " cold"),
    ("The opposite of up is", " down"),
    ("Cats and", " dogs"),
    ("Bread and", " butter"),
    ("Salt and", " pepper"),
    ("Romeo and", " Juliet"),
    ("Day and", " night"),
    ("The sky is", " blue"),
    ("Grass is", " green"),
    ("Snow is", " white"),
    ("The sun rises in the", " east"),
    ("The sun sets in the", " west"),
    ("A square has four", " sides"),
    ("A triangle has three", " sides"),
    ("A week has seven", " days"),
    ("A year has twelve", " months"),
    ("An hour has sixty", " minutes"),
    ("A minute has sixty", " seconds"),
    ("Five plus five equals", " ten"),
    ("Twenty divided by four equals", " five"),
    ("The first letter of the alphabet is", " A"),
    ("The last letter of the alphabet is", " Z"),
]


@dataclass
class CapabilityResult:
    accuracy: float
    n: int
    correct: list[bool] = field(default_factory=list)


@torch.no_grad()
def capability_score(
    model: ModelAdapter,
    probes: list[tuple[str, str]] | None = None,
) -> CapabilityResult:
    """Greedy next-token accuracy over the probe set. No steering applied —
    callers stack a `steer_residual` / `ablate_residual` context around this
    when they want to measure the effect of intervention.
    """
    probes = probes if probes is not None else DEFAULT_PROBES
    correct: list[bool] = []
    for prompt, completion in probes:
        # Expected first token of the completion. Some tokenizers split words;
        # we compare against the first piece in either leading-space or
        # no-space form, taking the shorter.
        ids_space = model.tokenizer.encode(completion, add_special_tokens=False)
        ids_nospace = model.tokenizer.encode(completion.lstrip(), add_special_tokens=False)
        target = ids_space[0] if ids_space else ids_nospace[0]

        inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.model(**inputs, use_cache=False)
        pred = int(out.logits[0, -1].argmax().item())
        correct.append(pred == target)

    n = len(probes)
    acc = sum(correct) / max(n, 1)
    return CapabilityResult(accuracy=acc, n=n, correct=correct)
