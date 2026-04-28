"""Shared infrastructure for extracting a numeric rating from an LM.

Two scoring modes share this module:

`extract_numeric_rating` — generation-based: greedy-decode a few tokens and
parse the first numeric value out. Robust to chatty models but discrete.

`score_numeric_logits` — distribution-based: at the answer position, read
out the logits over the rating-token ids (e.g., "-3", "-2", ..., "+3") and
return the expectation under the resulting softmax. Continuous, gradient-
friendly, and requires no generation. Preferred when the rating scale is
known and small.

Most callers want `score_numeric_logits` because it gives a graded score
even when the model would otherwise refuse / waffle / output prose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from src.models.adapter import ModelAdapter

# Match a signed integer or float (no embedded whitespace; "- 2" → "2").
# Used by the generation-based parser.
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class NumericRatingResult:
    expected: float            # softmax-weighted expected value of the rating
    argmax_value: float        # the rating value with the highest probability
    probs: dict[float, float]  # {rating_value: probability}
    raw_text: str | None = None  # non-None for the generation-based path


def _tokenize_rating_strings(
    tokenizer, scale: list[int]
) -> list[tuple[int, list[int]]]:
    """For each integer in `scale`, return (rating_value, full_token_sequence).

    We use the leading-space form (" {v}") since "Rating:" prompts almost
    always follow with a space. Multi-token sequences are kept as-is —
    `score_numeric_logits` sums log-probs along the full sequence.
    """
    out: list[tuple[int, list[int]]] = []
    for v in scale:
        ids = tokenizer.encode(f" {v}", add_special_tokens=False)
        if not ids:
            raise ValueError(f"rating value {v!r} produced no tokens")
        out.append((v, ids))
    return out


@torch.no_grad()
def score_numeric_logits(
    model: ModelAdapter,
    prompt: str,
    scale: Iterable[int],
) -> NumericRatingResult:
    """Read out a probability distribution over numeric ratings at the next
    tokens after `prompt`, scoring each rating by the sum of log-probs of
    its full token sequence (some ratings tokenize to >1 piece).

    Returns the expected (mean), argmax, and the full distribution over the
    rating values.
    """
    scale = list(scale)
    rating_token_seqs = _tokenize_rating_strings(model.tokenizer, scale)

    prompt_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)  # (1, P)
    P = prompt_ids.shape[1]

    # Build a batch where each row is `prompt + rating_tokens[v]`, padded to
    # the max rating-sequence length. Teacher-forcing: one forward pass gives
    # us, at each position, the predicted distribution over the next token,
    # so we can read off log P(rating_seq[i] | prompt + rating_seq[:i]) at
    # position P + i - 1 for each rating row.
    max_rt = max(len(t) for _, t in rating_token_seqs)
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id or 0
    rows: list[list[int]] = []
    rating_lens: list[int] = []
    for _, rt in rating_token_seqs:
        row = prompt_ids[0].tolist() + rt + [pad_id] * (max_rt - len(rt))
        rows.append(row)
        rating_lens.append(len(rt))

    input_ids = torch.tensor(rows, device=model.device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    # Mask the trailing pads so attention doesn't see them.
    for i, L in enumerate(rating_lens):
        if L < max_rt:
            attention_mask[i, P + L:] = 0

    out = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits  # (B, P + max_rt, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    scores = []
    for i, (_v, rt) in enumerate(rating_token_seqs):
        s = 0.0
        for j, tok in enumerate(rt):
            s += float(log_probs[i, P + j - 1, tok])
        scores.append(s)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    rating_probs = F.softmax(score_tensor, dim=-1).tolist()

    probs = {float(v): float(p) for (v, _), p in zip(rating_token_seqs, rating_probs)}
    expected = sum(v * p for v, p in probs.items())
    argmax_value = max(probs.items(), key=lambda kv: kv[1])[0]
    return NumericRatingResult(
        expected=expected, argmax_value=argmax_value, probs=probs,
    )


@torch.no_grad()
def extract_numeric_rating(
    model: ModelAdapter, prompt: str, max_new_tokens: int = 8,
) -> NumericRatingResult:
    """Greedy-generate a few tokens and parse the first numeric value.

    Less precise than `score_numeric_logits` but doesn't require knowing
    the scale; useful when an LM is expected to produce free-form prose
    that includes a rating somewhere.
    """
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
    gen = model.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    text = model.tokenizer.decode(
        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    m = _NUMERIC_RE.search(text)
    if m is None:
        return NumericRatingResult(
            expected=float("nan"), argmax_value=float("nan"),
            probs={}, raw_text=text,
        )
    val = float(m.group(0).replace(" ", ""))
    return NumericRatingResult(
        expected=val, argmax_value=val,
        probs={val: 1.0}, raw_text=text,
    )
