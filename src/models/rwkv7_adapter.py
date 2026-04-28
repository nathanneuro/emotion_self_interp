"""Adapter for RWKV-7 ('Goose'/'G1d') models loaded via the rwkv pip package.

RWKV-7 is a recurrent linear-attention architecture, not a transformer. The
`rwkv` package's `RWKV` class stores all weights in a flat dict (`model.z`)
keyed by string paths and runs the forward pass with module-level helper
functions. There are no per-block nn.Module submodules to hook.

This adapter exposes a transformer-style interface — `cache_residual` returning
{layer_idx: (T, d_model) cpu float32} — by manually reimplementing the
`forward_seq` loop using the same module-level TMix/CMix helpers. The captured
'residual stream' is the sum-line `x` after each block's TMix and CMix
contributions, which is the analogue of the post-block residual stream in a
transformer.

Required env vars (set before importing rwkv.model):
    RWKV_V7_ON=1        — enable RWKV-7 codepath
    RWKV_JIT_ON=0       — disable TorchScript JIT (avoids ScriptMethod weirdness)

Caveats:
    - RWKV is single-sequence in `forward_seq`; we batch by looping prompts.
      Sufficient for <300-stimulus diff-of-means but not for training.
    - The per-block "residual stream" is the same `x` updated in place across
      blocks, including pre-block ln0 normalization at block 0.
    - Position semantics: -1 maps to last token; "last_real" not needed since
      we tokenize one prompt at a time (no padding).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

# These must be set before importing rwkv.model.
os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rwkv.model import (  # noqa: E402
    RWKV,
    RWKV_x070_CMix_seq,
    RWKV_x070_TMix_seq,
)
from rwkv.utils import PIPELINE  # noqa: E402


@dataclass
class RWKV7Adapter:
    """Drop-in-ish parallel to ModelAdapter for RWKV-7 models.

    Construct with `RWKV7Adapter.load(path_no_ext)` where the .pth file lives
    at `<path_no_ext>.pth`. Strategy defaults to 'cuda bf16'.
    """

    model: RWKV
    pipeline: PIPELINE
    name: str
    family: str = "rwkv7"

    @classmethod
    def load(
        cls,
        path_no_ext: str,
        strategy: str = "cuda bf16",
        tokenizer_name: str = "rwkv_vocab_v20230424",
    ) -> "RWKV7Adapter":
        model = RWKV(model=path_no_ext, strategy=strategy)
        pipeline = PIPELINE(model, tokenizer_name)
        return cls(model=model, pipeline=pipeline, name=path_no_ext)

    @property
    def n_layers(self) -> int:
        return int(self.model.n_layer)

    @property
    def d_model(self) -> int:
        return int(self.model.n_embd)

    def encode(self, text: str) -> list[int]:
        return self.pipeline.encode(text)

    @torch.no_grad()
    def forward_with_residuals(
        self,
        token_ids: list[int],
        layer_idxs: Iterable[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run forward_seq manually, capturing the residual stream.

        Returns {layer_idx: tensor of shape (T, d_model) on cpu float32} for the
        requested layers. The capture point is right after each block's CMix
        contribution, i.e. the canonical post-block residual.
        """
        m = self.model
        z = m.z
        n_layer = m.n_layer
        n_embd = m.n_embd
        layer_idxs = set(range(n_layer)) if layer_idxs is None else set(layer_idxs)

        # Empty state (zero-context). For prompt extraction we don't need carry.
        state = m.generate_zero_state()

        x = z["emb.weight"][token_ids]  # (T, d)
        v_first = torch.empty_like(x)

        cache: dict[int, torch.Tensor] = {}
        for i in range(n_layer):
            bbb = f"blocks.{i}."
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."

            xx = F.layer_norm(
                x, (n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"]
            )
            xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_seq(
                i, m.n_head, m.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1],
                z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"],
                z[att + "x_a"], z[att + "x_g"],
                z[att + "w0"], z[att + "w1"], z[att + "w2"],
                z[att + "a0"], z[att + "a1"], z[att + "a2"],
                z[att + "v0"], z[att + "v1"], z[att + "v2"],
                z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"],
                z[att + "r_k"],
                z[att + "receptance.weight"], z[att + "key.weight"],
                z[att + "value.weight"], z[att + "output.weight"],
                z[att + "ln_x.weight"], z[att + "ln_x.bias"],
            )
            x = x + xx

            xx = F.layer_norm(
                x, (n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"]
            )
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(
                xx, state[i * 3 + 2],
                z[ffn + "x_k"], z[ffn + "key.weight"], z[ffn + "value.weight"],
            )
            x = x + xx

            if i in layer_idxs:
                cache[i] = x.detach().to("cpu", dtype=torch.float32)
        return cache

    @torch.no_grad()
    def extract_last_token(
        self,
        prompts: list[str],
        layer_idxs: Iterable[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run each prompt, take the last-token residual at every requested layer.

        Returns {layer_idx: (n_prompts, d_model) cpu float32}.
        """
        layer_idxs_list = list(range(self.n_layers)) if layer_idxs is None else list(layer_idxs)
        out: dict[int, list[torch.Tensor]] = {li: [] for li in layer_idxs_list}
        for p in prompts:
            ids = self.encode(p)
            cache = self.forward_with_residuals(ids, layer_idxs=layer_idxs_list)
            for li in layer_idxs_list:
                out[li].append(cache[li][-1])  # (d_model,)
        return {li: torch.stack(parts, dim=0) for li, parts in out.items()}
