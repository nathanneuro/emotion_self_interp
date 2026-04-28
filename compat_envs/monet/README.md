# Monet compat env

Self-contained environment for repos whose remote-code modeling files were
packaged against transformers 4.42.x and break under transformers 5.x.

## Why

Monet (`MonetLLM/monet-vd-850M-100BT-hf`) ships a `modeling_monet.py` that:
- imports `LLAMA_ATTENTION_CLASSES` (removed in transformers 5.x; we shim it
  in the main env, but other deps still drift)
- subclasses `LlamaConfig` with `intermediate_size=None` (rejected by 5.x's
  strict-typed dataclass since Monet has MoE not FFN)
- uses sentencepiece tokenizer paths that 5.x routes through tiktoken-extract

Patching all of these in the main env became a moving target. Pinning to
transformers 4.42 (the version Monet was packaged against) is faster, more
robust, and isolates the compat surface.

## Layout

- `pyproject.toml` — pins transformers 4.42, torch <2.6, accelerate <1.0,
  Python 3.11–3.12 (3.13 is too new for transformers 4.42's wheel set).
- `.python-version` → 3.12.
- `run.py` — runner that injects the project root into `sys.path`, then uses
  `src.data.emotion_stimuli`, `src.probes.diff_means`, etc., from the main
  project unchanged. The shared probe + stimulus code is reused; only the
  model-loading path is pinned-old.

## Usage

```bash
cd compat_envs/monet
uv sync                      # creates ./.venv pinned to transformers 4.42
uv run python run.py --model MonetLLM/monet-vd-850M-100BT-hf
```

Or from anywhere:

```bash
uv run --project compat_envs/monet python compat_envs/monet/run.py
```

The run writes to the main project's `outputs/phase1_vectors_*` directory
just like the other extraction scripts, so `scripts/cross_model_summary.py`
picks it up automatically.

## Adding more legacy models

If another remote-code model fights transformers 5.x, prefer pointing it at
this env over patching the main env. If that model's pin is incompatible with
Monet's, add a sibling directory (`compat_envs/<model>/`) rather than
expanding this one.
