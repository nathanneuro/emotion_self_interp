# Grounded self-interpretation of functional emotional states in LLMs

Empirical research code for testing whether LLM introspective reports about emotional states causally depend on internal representations, by combining four independent measurement channels (substrate-vector activation, trained self-interpretation adapter, behavioral utility signature, causal intervention) into one cross-method convergence test.

The scientific program is in `docs/planning.md`. Execution status lives in `MASTER_PLAN.md`. Findings as they emerge go to `docs/research_log.md`.

## TL;DR

A clean four-way convergence on the program's primary metric (per-channel correlation with target valence, naturalistic held-out, n=60) is achieved on a 0.5B-parameter open-weights model:

| Channel | r vs target valence (Qwen2.5-0.5B-Instruct) |
|---|---|
| substrate (cosine of last-token residual to v_E) | +0.509 |
| trained adapter (Pepper-style scalar-affine) | +0.491 |
| untrained-SelfIE baseline | +0.422 |
| Likert behavioral readout | +0.516 |

The same convergence picture holds across architectural paradigms (standard transformer, universal-transformer, recurrent linear-attention, sparse-MoE) at varying tightness depending on instruction-tuning status. A "deceptive" adapter trained on swapped emotion labels produces predictions that are decoupled from the substrate (r = −0.03 vs target valence) while substrate-driven channels (substrate cosine, Likert) continue to track the actual emotion — the operational definition of veridical introspection the program defines.

## Headline empirical findings

| Finding | Where |
|---|---|
| Sofroniew-style emotion-vector geometry replicates on **7 models across 4 architectural paradigms** (qwen2, llama, gemma2/3, ouro, monet sparse-MoE, rwkv7 recurrent) — every model exceeds the published 0.81 PC1↔valence on a 70B model, with our cross-arch range 0.848–0.998 | Phase 1, `outputs/phase1_cross_model.{json,png}` |
| Within-emotion contrast (v_E = mean(E) − mean(other emotions)) is required for the substrate channel to transfer from euphoric to naturalistic stimuli; with the v0 neutral-contrast vectors substrate-vs-target correlation is r = −0.05, with within-emotion contrast it jumps to r = +0.51 (+0.56 absolute) | Phase 1.5, research log |
| Causal dependence of introspective Likert reports on substrate steering is monotonic and matches Sofroniew's published ±0.1 alpha anchor on a 0.5B model. Capability preserved through ⎮α⎮ ≤ 0.5; behavioral envelope identified | Phase 4, `outputs/phase4_steering_*` |
| Pepper's "bias-prior carries 85%" caveat does **not** hold at 0.5B scale. Bias-only adapter sits exactly at chance; full-rank adapter under input-shuffling collapses to chance — its full lift over chance is activation-conditional, not format-prior | Experiment 2, `outputs/phase6_exp2_*` |
| Post-training (instruction tuning) does not strengthen the substrate. On Qwen2.5-0.5B base vs instruct: substrate r vs target is *higher* in base (+0.572 vs +0.509). Likert and substrate↔Likert correlations both jump in instruct. Post-training reshapes the **readout**, not the substrate | Experiment 5, research log |
| Veridical introspection holds operationally: a "deceptive" adapter trained on swapped emotion labels produces predictions decoupled from substrate (r = −0.027 vs target valence) while substrate-driven channels continue to track the actual emotion. Adapter-as-report is a separable channel from behavior | Experiment 4 |
| Universal-transformer (Ouro-1.4B-Thinking) shows the **tightest cross-channel convergence** of any architecture tested (substrate↔Likert r = +0.714) and reveals that valence structure builds up across loop iterations rather than across layers (per-ut-step max ⎮PC1↔valence⎮: 0.35 → 0.98 across 4 iterations) | Phase 1 + Exp 1 v1 on Ouro |

## Phase status

| Phase | Status |
|---|---|
| 0 — Infrastructure | done |
| 1 — Emotion-vector extraction (Sofroniew-style) | v0 done — 7 architectures, 4 paradigms |
| 1.5 — Within-emotion contrast | done |
| 2 — Trained self-interpretation adapter (Pepper-style) | scaffold done |
| 3 — Behavioral utility measurements (Ren-style) | v0 done |
| 4 — Causal intervention machinery (Lindsey-style) | v0 done |
| 5 — Experiment 1 (cross-method convergence) | v1 done — clean four-way convergence on Qwen-0.5B-Instruct + cross-architecture replication on Ouro |
| 6 — Experiments 2–5 | all four done |

The full scientific program from `docs/planning.md` has at least a v0 result on every experiment. See `MASTER_PLAN.md` for per-task tracking and `docs/research_log.md` for findings as they appeared.

## Layout

```
src/
  models/adapter.py             ModelAdapter — uniform residual-stream hooks across HF families
                                  (Llama, Qwen, Gemma 2/3, Mistral, OLMo, Ouro, Monet)
  models/rwkv7_adapter.py       Separate adapter for the rwkv pip package's flat-weight RWKV-7
  hooks/extract.py              cache_residual / extract / extract_batch
  data/stimuli.py               Stimulus dataclass + JSON load/save
  data/emotion_stimuli.py       v0 stimulus set: 6 emotions × 3 levels × ~30 items each
  probes/diff_means.py          Diff-of-means + LDA + d-prime / AUROC scoring
  adapters/scalar_affine.py     ScalarAffineAdapter (d+1), BiasOnlyAdapter (d),
                                  ScaleOnlyAdapter (1), FullRankAdapter (d²+d)
  adapters/train.py             Pepper-style training loop with residual-replace hook
  behaviors/numeric.py          Full-token-sequence log-prob scoring for numeric ratings
  behaviors/likert.py           Third-person Likert valence + arousal channel
  behaviors/sentiment.py        Two-stage model-as-judge sentiment-of-generation channel
  behaviors/capability.py       30-item factual / arithmetic probe for steering capability check
  experiments/protocol.py       Shared helpers: extract_stimulus_residuals, build_emotion_vectors,
                                  train_pepper_on_residuals, make_untrained_selfie_adapter
  experiments/experiment1.py    Per-stimulus four-channel measurement framework
  experiments/experiment2.py    Bias-prior decomposition: zero-vector decoding + input-shuffle
  experiments/experiment4.py    Veridical introspection: honest + deceptive adapter divergence

scripts/
  smoke_pipeline.py                   Smallest end-to-end check (load + extract + steer)
  extract_emotion_vectors.py          Phase 1 layer sweep (transformer families)
  extract_emotion_vectors_ouro.py     Phase 1 with per-(layer, ut_step) capture for Ouro
  extract_emotion_vectors_rwkv.py     Phase 1 via RWKV7Adapter
  pca_emotion_geometry.py             PCA of per-emotion vectors → valence/arousal correlations
  cross_model_summary.py              Aggregate Phase 1 results across all model runs
  measure_behavior.py                 Phase 3 construct check (Likert + sentiment)
  sweep_steering.py                   Phase 4 alpha-sweep + ablation on Likert and capability
  train_adapters.py                   Phase 2 adapter training (all variants)
  run_experiment1.py                  Phase 5 / Exp 1 cross-method convergence
  run_experiment2.py                  Phase 6 / Exp 2 bias-prior decomposition
  run_experiment4.py                  Phase 6 / Exp 4 veridical introspection

compat_envs/
  monet/                        Sub-project with pinned transformers 4.45 + Python 3.12 for legacy
                                  custom-modeling repos. Reuses src/ via sys.path injection.

tests/                          34 unit tests; ruff-clean
docs/
  planning.md                   Scientific program (source of truth)
  research_log.md               Append-only findings log
outputs/                        Per-run timestamped directories (gitignored)
```

## Running things

```bash
# Smallest end-to-end check
uv run python scripts/smoke_pipeline.py

# Phase 1 — emotion-vector layer sweep
uv run python scripts/extract_emotion_vectors.py
uv run python scripts/extract_emotion_vectors.py --contrast other_emotions   # Phase 1.5
uv run python scripts/extract_emotion_vectors.py --model google/gemma-2-2b
uv run python scripts/extract_emotion_vectors_rwkv.py --model-path /path/to/rwkv7

# Phase 4 — alpha sweep on calm↔desperate
uv run python scripts/sweep_steering.py

# Phase 5 — Experiment 1 cross-method convergence (default contrast: other_emotions)
uv run python scripts/run_experiment1.py
uv run python scripts/run_experiment1.py --model ByteDance/Ouro-1.4B-Thinking --layer 15 \
    --trust-remote-code

# Phase 6 — bias-prior decomposition + veridical introspection
uv run python scripts/run_experiment2.py
uv run python scripts/run_experiment4.py

# Aggregate Phase 1 across all runs
uv run python scripts/cross_model_summary.py

# Tests + lint
uv run pytest tests/
uv run ruff check src/ scripts/
```

The Monet model needs the compat env:

```bash
cd compat_envs/monet && uv sync
HF_HOME=/path/to/cache uv run python run.py --model MonetLLM/monet-vd-850M-100BT-hf
```

## Hardware

Built and tested on 2× NVIDIA RTX 4090 (48 GB total). Models used so far:

- Qwen2.5-0.5B / -Instruct
- SmolLM2-360M-Instruct
- gemma-2-2b, gemma-3-270m-it
- Ouro-1.4B-Thinking (custom remote-code modeling)
- Monet-vd-850M, Monet-vd-4.1B (sparse-MoE, transformers-4.x compat env)
- RWKV-x070-World-2.9B (separate `rwkv` pip package path)

Cached but not yet tested due to current VRAM constraints from concurrent jobs:

- Llama-3-8B-Instruct, Llama-3.1-70B-Instruct
- Mixtral-8x7B-Instruct
- LLaDA-8B-Instruct (discrete-diffusion LLM)
- Dream-v0-Instruct-7B (discrete-diffusion LLM)

## License / status

Research project, no external API stability commitments. See `docs/planning.md` for broader research framing.
