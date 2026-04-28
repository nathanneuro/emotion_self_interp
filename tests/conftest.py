"""Shared pytest fixtures.

We use a tiny open-weights model (Qwen2.5-0.5B-Instruct) for smoke tests so the
suite can run on CPU as well as GPU and downloads ~1 GB rather than ~16 GB.
The architecture is identical to the 7B/14B Qwen2 models, so anything that
passes here will pass on the larger ones modulo memory.
"""
from __future__ import annotations

import os

import pytest
import torch

from src.models.adapter import ModelAdapter

SMOKE_MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="session")
def adapter() -> ModelAdapter:
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return ModelAdapter.load(SMOKE_MODEL, dtype=dtype, device_map=device_map)
