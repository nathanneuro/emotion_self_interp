# MASTER_PLAN — Grounded self-interpretation of functional emotional states

**Source of truth for the scientific program:** `docs/planning.md`. This file tracks execution.

**Last updated:** 2026-04-28

## North star

Operationalize four-way convergence (substrate vector activation, trained adapter readout, behavioral utility signature, causal intervention) for canonical emotion states in open-weights LLMs, and produce empirical answers to whether introspective reports causally depend on the represented state.

## Design principle: model-agnostic from day one

Every component is built against a thin abstraction over `transformers` models so the same pipelines run on Llama, Qwen, Gemma, Mistral, OLMo, etc. Concretely:

- **Activation access** via `nn.Module` forward hooks keyed on residual-stream layer indices, not architecture-specific module names. A small `ModelAdapter` resolves `(layer_idx, site)` → submodule for each supported family.
- **No hardcoded layer counts, hidden dims, or tokenizer assumptions.** Everything reads from `model.config`.
- **Steering / ablation** via the same hook interface as extraction — addition or projection at a chosen `(layer, site, token_position)`.
- **All experiments are sweeps over a model list** (defined in config). Single-model runs are just a one-element list.
- Initial supported set: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it. Add OLMo-2-7B and a 13–32B tier (4-bit) as the pipeline stabilizes.

Rationale: cross-model convergence is itself one of the strongest sanity checks on the construct-validity claims. A finding that holds on one model is one anecdote; a finding that holds on three families with different post-training is a result.

## Current state

- Repo: skeleton (`main.py`, empty `src/`, `tests/`, `outputs/`). No deps in `pyproject.toml`.
- Hardware: 2× RTX 4090 (48 GB total), 32 cores, 503 GB RAM. Sufficient for ≤9B at fp16 single-GPU, ≤32B at 4-bit, larger via offload.
- Phase: **planning_phase** → moving into infrastructure scaffolding.

## Phases

### Phase 0 — Infrastructure scaffold (done)

The plumbing every experiment depends on. Build once, reuse everywhere.

- [x] `src/models/adapter.py` — `ModelAdapter` abstraction. Family-detected (`llama`, `qwen2/3`, `gemma/2/3`, `mistral`, `olmo/2`), exposes `n_layers`, `d_model`, `get_block(i)`, `cache_residual([...])` and `steer_residual(layer, vec, alpha, token_mask=None)` context managers. Left-padding by default so position=-1 always indexes the last real token.
- [x] `src/hooks/extract.py` — single + batched `(layer, position)` extraction; `position` accepts `int` or `"last_real"`, the latter padding-side-agnostic.
- [x] Steering lives on the adapter (`steer_residual`) — additive at `(layer, site)`, optional per-position `token_mask`. Directional ablation can be added later as a separate hook variant when Phase 4 needs it.
- [x] `src/data/stimuli.py` — `Stimulus` dataclass + JSON load/save.
- [x] `src/runs/run_dir.py` — per-run dirs `outputs/<name>_<YYYYMMDD_HHMM>/` with `config.json` snapshot.
- [x] Deps in `pyproject.toml`: torch 2.11, transformers 5.7, accelerate, datasets, numpy, sklearn, pandas, tqdm, pyyaml + dev: pytest, ruff. (bitsandbytes deferred until Phase 5+ when 4-bit becomes useful.)
- [x] Tests: `tests/test_model_adapter.py` (5 cases) + `tests/test_extract.py` (4 cases) — all 9 pass on Qwen2.5-0.5B-Instruct, ~2.6s including model load.
- [x] End-to-end smoke pipeline (`scripts/smoke_pipeline.py`) — load Qwen2.5-0.5B, extract activations at L12 for emotional vs neutral prompts, steer with the diff vector, see clear logit perturbation (mean |Δlogit| = 2.45 at α=5.0) and generation change.

### Phase 1 — Emotion-vector extraction (Sofroniew-style)

Reproduce the substrate-level probe layer per model. Output: dictionary of `{emotion_label: {layer: vector}}` for each model.

- [ ] Stimulus generation: synthetic stories per emotion label, paired with a contrastive neutral / opposite-valence story. Start with the alignment-relevant pair (calm ↔ desperate) plus 4 covering valence×arousal corners (blissful, sad, afraid, hostile).
- [ ] Probe method: difference-of-means at the assistant-colon (or family-equivalent) token, then optionally LDA refinement. Sweep mid-late layers; pick layer-of-best-separation per model.
- [ ] Sanity check: PCA over the emotion-vector set should reproduce a valence-like PC1 (Sofroniew r=0.81 with human valence). Compute correlation with a simple human-rating proxy (we'll bootstrap with model-rated valence first; eventually use the published norms).
- [ ] Output cached to `outputs/phase1_vectors/<model>/vectors.pt` for reuse downstream.

### Phase 2 — Trained self-interpretation adapter (Pepper-style)

Scalar affine `h ↦ softmax(W·(α·h + b))` over vocab, training pair `(emotion_vector, label_token_string)`.

- [ ] `src/adapters/scalar_affine.py` — minimal d+1 parameter adapter, plus a bias-only baseline and a full-rank baseline for Experiment 2.
- [ ] Training loop: standard cross-entropy on label tokens, frozen base model, hooks inject `α·h + b` into the residual stream at the chosen layer.
- [ ] Evaluation: top-k label accuracy on held-out emotion vectors; generation-scoring (model-graded) on held-out vectors.
- [ ] Untrained SelfIE baseline at the same site, for the "training matters" comparison Experiment 1 needs.

### Phase 3 — Behavioral utility measurements (Ren-style)

Three independent behavioral channels, model-agnostic.

- [ ] Forced-choice over experiences (experienced utility): pairwise prompts asking the model to choose between described continuations; estimate utility by Bradley-Terry over preferences.
- [ ] Forced-choice over world states (decision utility): same but over world descriptions rather than experiences.
- [ ] Likert self-report: structured prompt asking valence/arousal on a fixed scale.
- [ ] Stop-button rate on low-utility continuations as a behavioral correlate.
- [ ] Sentiment of free-form generation conditioned on the stimulus.
- [ ] Validation: euphoric soft-prompt training to confirm behavioral channels respond to substrate-level intervention before relying on them as DVs.

### Phase 4 — Causal intervention machinery (Lindsey-style)

The piece that distinguishes "report tracks state" from "both downstream of input."

- [ ] Generation-time steering: add `α·v_emotion` at chosen layers and token positions; sweep α to find the regime where behavior shifts but capability is preserved (anchor on Sofroniew's −0.1 to +0.1 desperate→blackmail finding).
- [ ] Directional ablation: project residual onto null space of `v_emotion` at chosen layers.
- [ ] Counterfactual prompts a la Lindsey: pairs designed so the only difference is the activation pattern of interest, with the introspective report as the readout.

### Phase 5 — Experiment 1 (minimal viable result)

Cross-method convergence on calm ↔ desperate, on the smallest supported model. Goal: produce a four-panel figure showing vector activation, adapter readout, untrained-SelfIE readout, behavioral signature for the same set of stimuli.

- [ ] Stimulus set: ~30 items per condition × 3 levels (euphoric, naturalistic, neutral) = ~270 items.
- [ ] Run the four measurement channels on each item.
- [ ] Pre-registered analysis: pairwise correlations between channels; failure-mode flags from `docs/planning.md` §Experiment 1.
- [ ] Replicate on second model family before claiming the finding.

### Phase 6 — Experiments 2–5

Sequenced by how much they depend on the Phase 5 infrastructure. Order: 2 (bias-prior decomposition, mostly reuses Phase 2 code) → 3 (causal dependence, Phase 4 stack) → 4 (induced vs. reported divergence) → 5 (post-training comparison; needs base + post-trained pairs, e.g., Llama-3.1-8B vs. Llama-3.1-8B-Instruct, Qwen2.5-7B-Base vs. -Instruct).

## Open decisions (defer until forced)

- **Stimulus source for Sofroniew replication.** Their 171 vectors are derived from a specific synthetic-story generator. Options: (a) regenerate with our own pipeline, (b) approximate from the paper's published examples, (c) use Ren-style euphorics as a near-equivalent. Lean (a) — full reproduction of the data side, since the geometry claims depend on it. Decide once the paper's release status is checked.
- **Activation site within the residual stream.** Pre-attn vs. post-attn vs. post-MLP, and which token (last, assistant-colon, in-context). Standardize on post-MLP at the assistant-colon or family-equivalent until we have a reason to deviate.
- **Adapter target layer per model.** Use the layer of best emotion-vector separation from Phase 1 by default; revisit if Phase 2 shows better adapter performance elsewhere.
- **How to handle base ↔ instruct comparisons in Experiment 5.** Either (a) extract vectors per model and compare apples-to-apples per model, or (b) extract vectors on instruct and apply to base. Both are informative; do (a) first.

## Risks / failure modes the program is prepared for

- **Linear-probe geometry inadequate for blended states.** Phase 1 design includes residuals analysis and a check on whether single-direction probes lose accuracy on naturalistic vs. constrained stimuli.
- **Bias-prior dominates the trained adapter.** Phase 2's three-condition design (bias-only, scalar-affine, full-rank) plus the input-shuffle test in Experiment 2 directly measures this.
- **Off-policy-only signal.** Phase 3's behavioral channels and Phase 5's naturalistic stimuli bring in on-policy evidence to complement the off-policy probe data.
- **Model-specific artifacts.** The cross-model design in Phase 0 is the structural defense; we don't claim a finding off a single model.

## Tracking

| Phase | Status | Notes |
|---|---|---|
| 0 — Infra | **done** (2026-04-28) | ModelAdapter, extract, steer, stimuli, run_dir; 9/9 tests pass on Qwen2.5-0.5B |
| 1 — Vectors | not started | next up; needs stimulus generation pipeline first |
| 2 — Adapter | not started | depends on Phase 1 |
| 3 — Behavior | not started | parallelizable with Phase 1/2 |
| 4 — Causal | not started | depends on Phase 1 |
| 5 — Exp 1 | not started | minimal viable result |
| 6 — Exp 2–5 | not started | sequenced post-Exp 1 |

## Next actions (immediate)

1. **Stimulus generation for Phase 1.** Decide on the source for Sofroniew-style synthetic stories — generate locally with a small LM, or curate the published examples. Aim for ~30 items per emotion × 3 levels for the calm ↔ desperate pair plus the four valence/arousal-corner emotions.
2. **Difference-of-means probe pipeline.** `src/probes/emotion_vectors.py` — given a list of `(emotion, story)` pairs and a contrastive neutral set, extract layer-wise difference-of-means and (optionally) LDA-refined directions.
3. **Layer sweep.** Run the probe extraction across all layers of Qwen2.5-0.5B-Instruct (cheap), plot probe separation, and pick the layer-of-best-separation as the canonical site for that model. Re-run on a 7B once the pipeline is validated.
4. **Cross-model check.** Repeat on a Llama or Gemma model once a stimulus set is committed, to confirm the family-agnostic adapter behaves as expected on a different architecture.

## Background jobs

None.
