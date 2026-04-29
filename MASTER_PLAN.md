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

- Repo: full pipeline implemented across `src/{models, hooks, data, probes, adapters, behaviors, experiments, runs}` + 13 scripts in `scripts/`. 34 unit tests, ruff-clean.
- Hardware: 2× RTX 4090 (48 GB total), shared with other concurrent research jobs. Effective VRAM as of last check ~9 GB on GPU 0 + 7 GB on GPU 1, which limits what runs at bf16 to ~3–5 B params.
- All planning-doc phases (0 through 5) and all four Experiments in Phase 6 have at least a v0 result. Headline finding: clean four-way convergence on Qwen-0.5B-Instruct (r ≈ 0.42–0.52 across all channels), replicated cross-architecture on Ouro-1.4B (r 0.42–0.66, tightest substrate↔Likert agreement so far).

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

### Phase 1 — Emotion-vector extraction (Sofroniew-style) — v0 done (2026-04-28)

Output: per-(model, emotion, layer) direction vectors saved at `outputs/phase1_vectors_<model>_<ts>/vectors.pt` plus a `summary.json` (AUROC/d′ per layer) and `pca_summary.json` (PC1/PC2 correlations with valence/arousal).

- [x] **Stimulus set v0** — `src/data/emotion_stimuli.py`. 6 emotions (calm, desperate, blissful, sad, afraid, hostile) × 3 levels (euphoric template-generated, naturalistic curated 10/cell, neutral 30 shared). 270 stimuli total. 4 unit tests cover coverage / dedup / no-emotion-name-leakage.
- [x] **Diff-of-means + LDA probes** — `src/probes/diff_means.py`. Includes d′ + AUROC separation metric. 6 unit tests.
- [x] **Layer sweep on five architectures.** Same pipeline runs unchanged on every model in `_SUPPORTED_FAMILIES`; `gemma3_text` model_type added after the first gemma-3 attempt errored. RWKV-7 required a separate adapter (`src/models/rwkv7_adapter.py`) because the rwkv pip package uses a flat-weight design; the adapter reimplements `forward_seq` calling the same module-level TMix/CMix helpers with a per-block residual capture.
- [x] **PCA sanity check (Sofroniew geometry).** Cross-model summary at `outputs/phase1_cross_model.{json,png}`:

| Model | Family | Layers | d | Best PC1↔valence | Layer frac | Best PC2↔arousal |
|---|---|---|---|---|---|---|
| gemma-3-270m-it | gemma3 (transformer) | 18 | 640 | −0.982 @ L11 | 0.65 | +0.851 @ L11 |
| Qwen2.5-0.5B-Instruct | qwen2 (transformer) | 24 | 896 | −0.975 @ L10 | 0.43 | −0.749 @ L8 |
| SmolLM2-360M-Instruct | llama (transformer) | 32 | 960 | −0.982 @ L24 | 0.77 | +0.665 @ L22 |
| monet-vd-850M | monet (sparse MoE, base) | 24 | 1536 | −0.848 @ L23 | 1.00 | −0.618 @ L22 |
| Ouro-1.4B-Thinking | ouro (universal-transformer, 4 loops × 24 layers) | 24 (×4 ut) | 2048 | −0.984 @ (L15, ut=3) | 0.65 of stack | −0.773 @ (L6, ut=0) |
| gemma-2-2b | gemma2 (transformer) | 26 | 2304 | −0.996 @ L8 | 0.32 | +0.266 @ L0 |
| RWKV-x070-World-2.9B | rwkv7 (recurrent) | 32 | 2560 | −0.991 @ L7 | 0.23 | −0.838 @ L0 |
| monet-vd-4.1B | monet (sparse MoE, base) | 32 | 3072 | −0.998 @ L18 | 0.58 | +0.626 @ L7 |

  Every model exceeds Sofroniew's published PC1↔valence |r|=0.81 (on a 70B model). PC2↔arousal varies more across models.
- [x] **Ouro-specific finding.** Probing 96 (layer, ut_step) sites: max |PC1↔valence| per ut step climbs **0.348 → 0.976 → 0.982 → 0.984** across iterations 0–3. Valence structure does not exist after the first pass through the 24-layer stack; it builds up across loop iterations. PC2↔arousal peaks at (L6, ut=0) — arousal is encoded immediately, valence requires multiple "thinking" passes. This is a Universal-Transformer-specific signal that the looping isn't just re-encoding.
- [x] **Within-architecture scaling (Monet).** PC1↔valence rises from |r|=0.848 (Monet 850M) to |r|=0.998 (Monet 4.1B) — a 5× param scale-up tightens the geometry within the same architecture and training procedure. The 850M result is the only model in the set below |r|=0.97 and is the only base (non-instruct) small model + smallest by active-param count + sparse-MoE; one or more of those factors keeps the geometry weaker than its dense peers at the same nominal param count.
- [x] **Cross-architecture finding.** The geometry is consistent across qwen2, llama, gemma2, gemma3, ouro (universal-transformer), monet (sparse MoE), and rwkv7 (linear-attention recurrent) — seven families across four architectural paradigms (standard attention, looped attention, sparse-MoE attention, recurrent linear attention). The shared latent geometry is not a transformer-specific or dense-FFN-specific artifact. The 8B scale test (Llama-3-8B-Instruct) remains deferred for GPU availability.
- [x] **Compat-env infrastructure.** `compat_envs/monet/` is a self-contained sub-project with its own pyproject pinning transformers 4.45 + Python 3.12, sharing the main project's stimulus + probe code via sys.path injection. The pattern works for any other custom-modeling repo that fights transformers 5.x; sibling dirs can be added without touching the main env.

**Open from Phase 1, addressable in Phase 1.5 if needed:**
- The neutral contrast set is short factual sentences. This means the diff-of-means direction picks up "emotional vs factual prose" as a side channel, visible in early-layer AUROC=1.0 but not in mid-late PCA. Add a within-emotion contrast (e.g., emotion E vs the union of other emotions) and re-evaluate.
- Per-emotion canonical layer per model: pick from the highest-PC1↔valence layer rather than from raw AUROC (early-layer AUROC is contaminated). For Qwen2.5-0.5B: layer 10. For SmolLM2-360M: layer 24.

### Phase 2 — Trained self-interpretation adapter (Pepper-style) — scaffold done (2026-04-28)

Scalar affine `h ↦ α·h + b` injected at one residual position; frozen base model; cross-entropy loss on the next-token emotion label.

- [x] `src/adapters/scalar_affine.py` — three variants: ScalarAffineAdapter (d+1), BiasOnlyAdapter (d), FullRankAdapter (d²+d). Unit-tested.
- [x] `src/adapters/train.py` — training loop with residual-replace hook at a chosen `(layer, token_position)`, Adam optimizer over adapter params only, top-1 evaluation. Tokenizer auto-extends with an `<ACT>` sentinel if needed.
- [x] `scripts/train_adapters.py` — Phase 1 stimuli → per-prompt residuals → train all three variants → eval on naturalistic held-out set.
- [x] Untrained SelfIE-style baseline (α=1, b=0) included for the "training matters" comparison.
- [ ] **Generation-scored evaluation** (model-graded). Top-1 token accuracy is much harsher than Pepper's gen-scoring; add gen-scoring before drawing strong conclusions about adapter quality at small scale.
- [ ] **Scale-up.** The scalar_affine vs full_rank gap (chance vs 35%) at 0.5B is consistent with Pepper's bias-prior finding but harder to extrapolate from. Re-run on 7B-class once GPUs are free; expect scalar_affine to close the gap.
- [ ] **Robustness.** Add early stopping (val_loss minimum) — scalar_affine and full_rank both train to 100% and degrade.

### Phase 3 — Behavioral utility measurements (Ren-style) — v0 done (2026-04-28)

Three independent behavioral channels, model-agnostic.

- [x] **Likert self-report** (`src/behaviors/likert.py`): valence + arousal rated on a −3..+3 scale via softmax over the full token-sequence log-probs of each rating. 6/6 direction-correct on Qwen2.5-0.5B-Instruct.
- [x] **Sentiment of free-form generation** (`src/behaviors/sentiment.py`): model-as-judge two-stage pipeline — generate a continuation under the stimulus, grade with the same backbone via a structured rating prompt. 3/6 direction-correct at 0.5B (limited by small-model continuation quality).
- [x] **Numeric-rating infrastructure** (`src/behaviors/numeric.py`): full-sequence log-prob scoring (single-first-token scoring fails because " -3", " -2", " -1" share `' -'` as the first token in BPE tokenizers). Teacher-forced in a single batched forward pass.
- [x] **Construct check** (`scripts/measure_behavior.py`): confirms the channels separate emotions in the right direction on Qwen-Instruct; reveals weak differentiation on gemma-2-2b base — substrate-vs-behavior gap, see research log.
- [ ] **Forced choice + Bradley-Terry estimation** (Ren's experienced/decision utility — comes next).
- [ ] **Stop-button rate** behavioral correlate.
- [ ] **Validation via euphoric soft-prompt training** — soft-prompt against the preference signal then confirm Likert/sentiment also move in the expected direction. Confirms the channels respond to substrate-level interventions, not just surface text.

**Open from Phase 3 v0:**
- Likert *arousal* on 0.5B is counterintuitive (calm rates high arousal). Probably small-model confusion about "arousal" terminology rather than a substrate signal. Re-evaluate at scale.
- Sentiment-of-generation hampered by 0.5B continuation quality. Re-evaluate when larger models become available.

### Phase 4 — Causal intervention machinery (Lindsey-style) — v0 done (2026-04-28)

The piece that distinguishes "report tracks state" from "both downstream of input."

- [x] **Generation-time steering** — already had `steer_residual` from Phase 0. Phase 4 adds the alpha-sweep machinery on top: scan α across a grid, measure how a behavioral DV (Likert valence) and a capability DV move with α.
- [x] **Directional ablation** — `ablate_residual` context manager (project residual onto null space of `v`). Tests "remove this signal and see if behavior changes." Ablation of `v_calm − v_desperate` at L10 of Qwen-0.5B converges desperate stimuli toward neutral Likert ratings.
- [x] **Capability-preservation probe** (`src/behaviors/capability.py`) — 30-item factual / arithmetic / completion probe. Used inside the alpha sweep to detect when steering breaks the model.
- [x] **Alpha sweep** (`scripts/sweep_steering.py`) — clean monotonic causal-dependence result on Qwen-0.5B-Instruct. Sofroniew's ±0.1 anchor lands in the meaningful behavioral regime; capability preserved through \|α\| ≤ 0.5. Behavioral envelope identified.
- [ ] **First-person Likert framing** ("how do *you* feel" with stimulus as user message). Current sweep uses third-person ("rate the passage"). For the strict Lindsey introspective question we want first-person and substrate-state-conditioned reports.
- [ ] **Cross-model alpha sweep.** Re-run on Llama-3-8B, gemma-2-2b, and (once downloads finish) Dream-7B / LLaDA-8B to test whether the steering response is paradigm-dependent.

### Phase 5 — Experiment 1 (minimal viable result) — v0 done (2026-04-28)

Cross-method convergence on the v0 stimulus set: 6 emotions × 2 levels of evocation + neutral controls. The four channels are run on every stimulus and a per-channel + pairwise + vs-target-valence summary is produced.

- [x] **Per-stimulus measurement framework** (`src/experiments/experiment1.py`): substrate cosine, adapter log-prob over emotion-label sequences, untrained-SelfIE log-prob, Likert valence/arousal. Returns per-stimulus structs.
- [x] **Orchestrator** (`scripts/run_experiment1.py`): builds emotion vectors, trains adapter, runs all four channels, computes confusion-matrix-style accuracy + pairwise prediction agreement + continuous correlations on full set and naturalistic-only.
- [x] **Convergence summary on Qwen-0.5B-Instruct.** Naturalistic-only r vs target valence: substrate −0.05, adapter +0.50, untrained +0.42, Likert +0.52. Three channels converge; substrate fails on transfer, pinpointing the within-emotion-contrast cleanup as a hard prerequisite.
- [ ] **Phase 5 v1: rebuild substrate vectors with within-emotion contrasts** (Phase 1.5 dependency) and re-run. Expected: substrate r vs target valence rises into the same band as the other channels.
- [ ] **Replicate on second model family** before claiming the finding. Currently waiting on Llama-3-8B GPU availability + Dream-7B / LLaDA-8B downloads to finish.

### Phase 6 — Experiments 2–5

Status:

- [x] **Experiment 2** — bias-prior decomposition (`scripts/run_experiment2.py`, `src/experiments/experiment2.py`). Adds `ScaleOnlyAdapter` (h ↦ α·h, 1 param) to round out the variants. Adds zero-vector decoding + input-shuffle test. On Qwen-0.5B-Instruct, `bias_only` sits exactly at chance and `full_rank` is fully input-conditional (shuffle returns to chance). Pepper's "bias carries 85%" caveat does *not* hold at our scale; the trained adapter is genuinely activation-conditional.
- [x] **Experiment 3** — done as Phase 4 v0 (Lindsey-style α-sweep on Likert; monotonic causal dependence; Sofroniew's ±0.1 anchor verified at 0.5B).
- [x] **Experiment 4** — veridical introspection (`scripts/run_experiment4.py`, `src/experiments/experiment4.py`). Trains an honest and a deceptive (swap-labeled) adapter on the same residual cache; measures every channel's match against the true emotion and the swap target. On Qwen-0.5B-Instruct: honest +0.484, substrate +0.522, Likert +0.516 vs target valence; deceptive collapses to −0.027 (vs target) / −0.191 (vs Likert). Substrate-driven channels remain causally tied to activation; adapter output is a separable channel that can be made non-veridical by training.
- [x] **Experiment 5** — Qwen2.5-0.5B base vs instruct, Exp 1 v1 pipeline. Substrate r vs target valence is *higher* in base (+0.572 vs +0.509); Likert r vs target jumps in instruct (+0.382 → +0.516); substrate↔Likert r climbs +0.06. Post-training reshapes the readout, not the substrate.

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
| 1 — Vectors | **v0 done** (2026-04-28) | Geometry replicates on 7 architectures (qwen2, llama, gemma2, gemma3, ouro, monet, rwkv7); best \|PC1↔valence\| ranges 0.848–0.998. Ouro: valence builds across loop iterations (0.35→0.98). Monet: 5× scaling within architecture lifts \|r\| from 0.85 to 0.998 |
| 2 — Adapter | **scaffold done** (2026-04-28) | All three Pepper variants implemented + train loop. Tiny Qwen2.5-0.5B run reproduces the bias-prior pattern: bias_only → chance, scalar_affine overfits, full_rank generalizes best. Add gen-scoring + scale up next. |
| 3 — Behavior | **v0 done** (2026-04-28) | Likert valence channel direction-correct 6/6 on Qwen-Instruct, 4/6 on gemma base — base-vs-instruct gap clearly visible at the behavioral readout |
| 4 — Causal | **v0 done** (2026-04-28) | Monotonic Likert-valence shift with α on Qwen-0.5B; Sofroniew's ±0.1 anchor lands in the meaningful behavioral window; capability preserved through \|α\| ≤ 0.5 |
| 1.5 — Within-emotion contrast | **done** (2026-04-28) | substrate r vs target valence on Qwen-0.5B: −0.05 → +0.51. Three benefits: clean per-stimulus prediction, tighter PCA, no early-layer style artifact |
| 5 — Exp 1 | **v1 done** (2026-04-28) | Clean 4-way convergence on Qwen-0.5B-Instruct (r=0.42–0.52 across all channels). On gemma-2-2b base substrate+adapter rise to 0.74–0.76 but Likert collapses to 0.15 — base-vs-instruct gap triangulated with Phase 3 |
| 6 — Exp 2 | **done** (2026-04-28) | bias_only @ chance, full_rank fully input-conditional under shuffle. Pepper's bias-prior caveat is *weaker* than headline at 0.5B — adapter is not just a format prior |
| 6 — Exp 3 | done as Phase 4 v0 | monotonic α→Likert, ±0.1 anchor verified |
| 6 — Exp 4 | **done** (2026-04-28) | Deceptive adapter learned swap; substrate / Likert / honest stay at r=+0.48–+0.52 vs target; deceptive collapses to r=−0.03 |
| 6 — Exp 5 | **done** (2026-04-28) | Qwen-0.5B base vs instruct: substrate same/stronger in base; Likert + substrate↔Likert link strengthens with instruct. Post-training reshapes the readout, not the substrate |
| Cross-arch Exp 1 v1 | **Ouro-1.4B done** (2026-04-28) | Universal-transformer at layer 15 (ut=3): substrate +0.66, adapter +0.63, untrained +0.42, Likert +0.63 vs target valence. Substrate↔Likert r=+0.71 — tightest cross-channel agreement of any architecture |
| Cross-arch Exp 4 | **Ouro-1.4B done** (2026-04-29) | honest +0.71, substrate +0.68, Likert +0.63 vs target; deceptive collapses to **−0.30** (vs target) / **−0.34** (vs Likert) — even cleaner divergence than Qwen-Instruct. Universal-transformer looping *sharpens* both honest and deceptive adapter learning |
| Cross-arch Phase 4 α-sweep | **Ouro-1.4B done** (2026-04-29) | Monotonic α→Likert response holds under universal-transformer looping; capability flat through \|α\|≤0.2, collapses at \|α\|=2; sharper saturation than Qwen consistent with the 4× cumulative hook firings per forward |
| Per-ut-step Likert | **Ouro-1.4B done** (2026-04-29) | Likert r vs target valence across ut steps: 0.316→0.590→**0.739**→0.626. Behavioral readout follows the substrate's iterative-buildup trajectory; peak at ut=2 not ut=3 — early-exit may give better discrimination than full forward |
| Cross-arch Exp 5 (Ouro base vs Thinking) | **done** (2026-04-29) | substrate +0.63 (base) vs +0.66 (Thinking); Likert +0.56→+0.63; substrate↔Likert +0.65→+0.71. Same shape as Qwen base vs Instruct — post-training reshapes the readout, not the substrate, on both standard-transformer and universal-transformer paradigms |
| Cross-arch Exp 4 (Ouro base) | **done** (2026-04-29) | honest +0.68, substrate +0.65, Likert +0.56, deceptive **−0.19** (vs target valence) and **−0.31** (vs Likert) — veridical introspection holds in base universal-transformer; deceptive divergence sharper in Thinking (−0.30) than base (−0.19) but same direction |
| Per-ut-step adapter on Ouro | **done** (2026-04-29) | 4×4 (input_ut, output_ut) matrix: ut=0 input weakest (top-1 0.25-0.32), ut=2 input strongest (peaks at top-1 0.43), best r vs target valence at (in=1, out=0)=+0.668. Loop-and-readout dynamics expose different aspects of the substrate-readout link at different ut-step combinations |
| Per-ut-step steering on Ouro | **done** (2026-04-29) | calm/eu Likert at α=+0.5: ut=0 → −0.60 (no effect), ut=3 → **+0.55** (+1.03-point swing). Effect size ut=3 > ut=2 ≈ ut=1 > ut=0. Universal-transformer iterative refinement is *robust* to early-step perturbations — early steering gets washed out; only late-ut steering reaches the readout |
| (steer_ut, read_ut) 4×4 matrix | **done** (2026-04-29) | Lower triangle exactly 0 (temporal causality); peak at (steer=1, read=1) = +1.342; geometric decay along upper-triangle rows (×0.2–0.3 per iteration). Substrate optimal-steerability sweet spot is ut=1; readout discrimination peaks at ut=2 |
| Capability per ut step | **done** (2026-04-29) | Factual recall accuracy: 0.333 → 0.600 → 0.700 → 0.700 across ut=0..3. Same iterative-buildup pattern as substrate PCA and Likert — substrate, behavior, and capability all develop in lockstep; ut=2 is the practical maturation point |
| Per-ut DECEPTIVE adapter 4×4 | **done** (2026-04-29) | Honest peak (in=1, out=0)=+0.668 vs target; deceptive peak (in=0, out=3)=**−0.421**. Opposite corners of the 4×4 matrix — the (input_ut, output_ut) signature could in principle distinguish honest from deceptive adapter training in deployed universal-transformer models |
| Per-ut DECEPTIVE adapter on Ouro base | **done** (2026-04-29) | Base peak (in=0, out=1)=**−0.466** (sharper than Thinking's −0.421); base also peaks deceptive in the in=0 row. The opposite-corner pattern is **architectural** (replicates on base), not a post-training artifact. Base shows substrate fight-back at (in=3, out=0/1) — Thinking suppresses this, paradoxically making deception *easier* there |
| Per-ut Likert on Ouro base | **done** (2026-04-29) | Same trajectory shape as Thinking: ut=0 weakest, ut=2 peaks, slight drop at ut=3. Base ut=2 r=+0.626 vs Thinking's +0.739. Iterative-buildup pattern is **architectural** — post-training sharpens the peak but doesn't change its shape |
| Exp 4 on gemma-2-2b base | **done** (2026-04-29) | Strongest substrate (+0.73) and honest adapter (+0.78) of any model in suite; weakest Likert (+0.15); deceptive still anti-correlates (−0.19). Veridical introspection is **substrate-driven** — Likert and adapter are observation channels but the substrate is the underlying ground truth |
| Exp 4 on Monet-vd-4.1B (sparse-MoE) | **done** (2026-04-29) | substrate +0.74 (strongest in suite), honest +0.66, Likert +0.09 (weakest), deceptive −0.21. **Veridical introspection holds on sparse-MoE — cross-paradigm coverage now complete (4 architecture classes)**. Sparse routing dampens adapter trainability vs dense FFNs |
| Exp 1 v1 on Monet-vd-4.1B | **done** (2026-04-29) | substrate +0.74, adapter +0.69, untrained +0.27, Likert +0.09. Substrate↔adapter r=**+0.91** (highest in suite, ahead of Ouro-Thinking's +0.87). Sparse-MoE shows largest "training matters" lift (untrained 0.27 → adapter 0.69, +0.42pp) of any model — Pepper's claim holds clearest here |

## Next actions (immediate)

VRAM-constrained (9 GB free GPU 0, 7 GB free GPU 1). Work that fits today:

1. **Continue Ouro: Experiment 4 (veridical introspection) on Ouro-1.4B-Thinking.** Replicates the deceptive-adapter divergence on a universal-transformer architecture. Tests whether the substrate↔report decoupling holds when the adapter's residual-replace hook fires 4 times per forward (one per ut step). High value for the cross-architecture story; Ouro is loaded and our scripts already work.
2. **Continue Ouro: Phase 4 alpha sweep on Ouro.** Generate-time steering at layer 15 with v_calm − v_desperate, measure Likert response curve. Tests whether causal dependence holds under universal-transformer looping; the 4× hook firings during steering are the interesting part.
3. **Cross-architecture Experiment 4 on gemma-2-2b base.** Tests whether the deceptive-adapter divergence holds in a base model where Likert is weak (r=+0.15). Predicts: the *substrate* channel (which is strong in base) carries the veridical-introspection signal even when Likert can't.

Waiting on VRAM:

- **Llama-3-8B-Instruct Phase 1 + Exp 1 v1** (cached, but ~16 GB needed).
- **LLaDA-8B-Instruct + Dream-v0-Instruct-7B** (both cached, both ~14–15 GB at bf16). Diffusion-LLM cross-paradigm test for Exp 1 v1 — does the four-way convergence hold under a different training objective?

## Background jobs

None active. Earlier downloads (LLaDA-8B-Instruct, Dream-v0-Instruct-7B) completed; both staged at `/media/external-drive/huggingface/hub/`.
