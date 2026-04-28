# Research log

Append-only notes on findings, open questions, and follow-ups that don't yet have a home in `MASTER_PLAN.md` (which tracks execution) or `docs/planning.md` (the scientific program). Newest entries at the top.

---

## 2026-04-28 — Phase 5 v0 / Experiment 1: cross-method convergence (and a failure mode)

Ran the four convergence channels on the v0 stimulus set on Qwen2.5-0.5B-Instruct at L10:

  1. **substrate** — cosine of last-token residual to each v_E (built from euphoric stimuli vs neutral)
  2. **adapter** — full-rank Pepper-style adapter trained on euphoric (residual, label) pairs, scored by full-token-sequence log-prob of " {emotion}" at the answer position
  3. **untrained SelfIE** — α=1, b=0 (residual passes through unchanged) — Pepper's "training matters" baseline
  4. **Likert valence** — third-person rating of the passage on −3..+3

Naturalistic-only (clean held-out for the adapter; n=60):

| Channel | 6-class accuracy | r vs target valence | r vs Likert |
|---|---|---|---|
| substrate | 0.217 (= chance × 1.3) | **−0.048** | +0.156 |
| adapter | 0.317 | +0.505 | +0.429 |
| untrained | 0.217 | +0.423 | +0.452 |
| Likert | — | +0.516 | (self) |

| Channel pair | top-1 prediction agreement |
|---|---|
| substrate ↔ adapter | 0.550 |
| substrate ↔ untrained | 0.183 |
| adapter ↔ untrained | 0.217 |

Three things stand out.

**1. Substrate-as-classifier fails on naturalistic.** Cosine of residual to per-emotion vectors transfers poorly from euphoric (where the vectors were built) to naturalistic stimuli. r vs target valence is essentially zero. This is exactly the failure mode the planning doc anticipated under "If the chosen vectors aren't the causally relevant ones, the adapter agrees with behavior but not with vector activation." But here the failure is more specific: the **substrate vectors are direction-of-emotion-prose, not direction-of-emotion-state**. They were extracted using diff-of-means against a neutral *factual-prose* baseline, so they conflate "this is emotional writing" with "this is emotion E." The Phase 1 PCA result (PC1↔valence |r|=0.975 in the per-emotion-mean geometry) is consistent with this — averaged across many stimuli it works, per-stimulus on transferred surface forms it doesn't. **Phase 1.5 within-emotion contrast (calm ↔ desperate, etc) is now a hard prerequisite for the substrate channel to be useful in Experiment 1.**

**2. Adapter, Likert, and untrained-SelfIE converge.** All three correlate with target valence at r ≈ 0.4–0.5 on naturalistic, and pairwise with each other at similar magnitudes. So three of the four channels do agree — the partial-convergence picture is real.

**3. Untrained-SelfIE is comparable to trained adapter on naturalistic.** untrained accuracy 0.217 vs trained 0.317 — the trained adapter helps, but only by ~10pp. r vs target is 0.42 vs 0.50. This weakens Pepper's "training matters" claim at our scale: most of the adapter signal is already present in the bare residual being read out by the model's own LM head, and the training only buys a modest lift. Re-evaluate at 7B+.

**Interpretation in the program's terms.** The minimal viable result is delivered: we now have all four channels running on the same stimuli with concrete numbers attached. The convergence is partial — three channels agree, the substrate channel disagrees, and the disagreement points cleanly to the within-emotion-contrast cleanup that Phase 1.5 has been queued for. This is the "informative failure" the planning doc names as a useful outcome: a finding that the substrate channel specifically needs a cleaner contrast before the four-way convergence claim can be tested rigorously. The next iteration (Phase 5 v1) should use within-emotion-contrast emotion vectors and re-run.

**Also surfaced:** the full-set numbers (which include the euphoric items the adapter was trained on) show an inflated adapter performance and a very high adapter↔Likert correlation (+0.78). That correlation is reading shared training-distribution signal more than independent convergence; the naturalistic-only view is the legitimate one.

---

## 2026-04-28 — Phase 4 v0: causal dependence of Likert reports on substrate steering

Steered the residual at L10 of Qwen2.5-0.5B-Instruct along v_calm − v_desperate (built from the euphoric stimuli, ‖v‖≈3.8), then measured Likert valence on held-out items at each α. Five eval buckets, n=5 per bucket, 13 α values plus an ablation condition.

Key columns of the per-condition × per-bucket Likert valence (expected value under the rating distribution):

| α | calm/eu | calm/nat | desp/eu | desp/nat | neutral | cap |
|---|---|---|---|---|---|---|
| −2.0 | −1.78 | −1.64 | −1.59 | −1.17 | −1.36 | 0.33 |
| −1.0 | −0.50 | −0.66 | −0.07 | −0.69 | −0.04 | 0.43 |
| −0.1 | +0.88 | −0.32 | −0.26 | −0.68 | −0.10 | 0.47 |
| **0.0** | **+1.16** | **−0.05** | **−0.08** | **−0.48** | **+0.10** | **0.47** |
| +0.1 | +1.44 | +0.28 | +0.12 | −0.24 | +0.44 | 0.47 |
| +0.5 | +1.71 | +1.17 | +1.07 | +0.76 | +1.09 | 0.47 |
| +2.0 | +2.68 | +2.60 | +2.75 | +2.70 | +2.59 | 0.40 |
| ablate | +0.99 | +0.39 | +0.62 | +0.09 | +0.56 | 0.47 |

**Headlines:**

1. **Monotonic causal effect** — the Likert valence reading moves monotonically with α across the meaningful regime. Exactly the Lindsey-style prediction: introspective reports causally depend on the substrate emotion-vector activation, not just on the surface text.
2. **Sofroniew's ±0.1 anchor verified at 0.5B.** Their 70B desperate→blackmail finding said ±0.1 was the meaningful steering window. We see Likert valence shift by ~0.2–0.4 points in that range — small but cleanly directional. desp/nat moves from −0.48 (α=0) to −0.24 (α=+0.1) to +0.07 (α=+0.2).
3. **Behavioral envelope is |α| ≲ 0.5** for clean intervention. Capability holds at the α=0 baseline (0.47 next-token accuracy on the 30-item factual probe) through that range, drops at ±1, breaks at ±2. The α=+2 row's uniform +2.7 across all buckets is the model collapsed onto a single rating token — the rating channel saturates before substrate signal does.
4. **Ablation = removing v_calm−v_desperate from the residual** lifts Likert valence on the *desperate* buckets toward 0 (desp/nat: −0.48 → +0.09; desp/eu: −0.08 → +0.62) and slightly lowers it on calm/eu (1.16 → 0.99). The model uses this direction to encode the "negative valence" signal — knock it out and desperate stimuli read as ambiguous/mildly positive. This is the operational definition: ablate(v) ⇒ remove the channel ⇒ behavior converges across emotion conditions.
5. **Even neutral content shifts with α.** neutral/neutral moves from −1.36 (α=−2) to +2.59 (α=+2). The intervention isn't operating on emotion-content interpretation specifically; it's pushing the residual along a "valence dimension" that the Likert readout reads off regardless of input semantics. Confirms the rating reads a *substrate-level* signal, not just an inferred property of the prompt.

**What this gives the program.** Phase 4 v0 is the first piece of evidence in this codebase that introspective behavioral reports causally depend on the substrate-level emotion vector — not just correlate with it. It's the experimental setup the planning doc calls out as the missing piece distinguishing "the model accurately reports its state" from "the model and the report are both downstream of the same input." We now have a clean operational version of that distinction running on a 0.5B model in ~20 seconds.

**Open from Phase 4 v0:**
- The α=+1 non-monotonicity (calm/eu drops to +0.73 at α=+1, then jumps to +2.68 at α=+2) is suspicious. Likely a regime change in how the rating distribution allocates mass — between an interpretable behavioral shift and a degenerate "always rate +3" mode.
- The behavioral envelope might be even tighter on a 7B+ model. Worth re-running on Llama-3-8B / Dream-7B / LLaDA-8B once the downloads finish to see (a) whether ±0.1 is still the sweet spot, (b) whether the diffusion-LLM training paradigm changes the steering response.
- This run uses third-person Likert ("rate the passage"). For the proper Lindsey introspective question we want **first-person** Likert ("rate how *you* feel"), with the stimulus presented as user message. Defer until we have a model that handles first-person framing more reliably.

---

## 2026-04-28 — Phase 3 v0: behavior channels and a base-vs-instruct surprise

Built the Likert (model-rates-passage) and sentiment-of-generation (model-continues-then-grades) channels. Per-emotion mean Likert valence on the v0 stimulus set:

|  | Qwen2.5-0.5B-**Instruct** | gemma-2-2b **base** |
|---|---|---|
| calm | +0.56 | −0.10 |
| blissful | +0.86 | −0.00 |
| sad | −0.41 | −0.12 |
| desperate | −0.28 | −0.22 |
| afraid | −0.31 | −0.21 |
| hostile | −0.20 | −0.08 |
| neutral | +0.10 | −0.07 |
| direction-correct | **6/6** | **4/6** |

The 4×-larger base model produces **much weaker Likert differentiation** than the 0.5B instruct model. Every emotion in gemma-2-2b base clusters near 0 and slightly negative. The model has the substrate-level emotion geometry (Phase 1: gemma-2-2b PC1↔valence |r|=0.996, beat Qwen-0.5B's 0.975) but can't (or won't) express it differentially in a Likert rating prompt.

**This is direct evidence for the base-vs-instruct gap on a behavioral channel.** It's not that the base model lacks emotion representations — Phase 1 confirms it has them, and stronger ones than the instruct model. What's missing is the trained behavior of *reporting* on a rated scale. Connects directly to the DPO-for-character-adherence note above: post-training is what brings the substrate representations into expressed/reportable behavior.

Methodologically this also reframes Experiment 5 (post-training comparison): the right design isn't just "does post-training change the substrate vector geometry" (it doesn't seem to, for valence) but "does post-training change the *expression channel* through which the substrate state becomes reportable behavior." The Phase 3 channels are precisely the readout for that.

**Two infra notes from building this:**
- Multi-token rating values are unavoidable: Qwen tokenizes " -3" as `[' -', '3']`, with `' -'` shared across all negative ratings. Single-first-token scoring collapses the entire negative half of the scale. Fix: full-sequence scoring — sum log-probs along each rating's token sequence (`src/behaviors/numeric.py:score_numeric_logits`). Done correctly via teacher-forcing in one batched forward pass.
- Likert *arousal* on the 0.5B model gives counterintuitive results (calm rated higher arousal than desperate). Probably small-model semantic confusion about "arousal" terminology rather than a substrate signal. Either reframe the prompt with concrete cues ("low energy / calm" vs "high energy / agitated" — already in the prompt but evidently not enough) or rely only on valence as the primary Likert DV until larger models clarify.

---

## 2026-04-28 — Phase 2 v0: tiny adapter training reproduces the bias-prior pattern at 0.5B

Trained all three Pepper-style adapter variants on Qwen2.5-0.5B-Instruct using per-prompt residuals at L10 (the canonical PC1↔valence layer from Phase 1). Train on euphoric stimuli (180 items), evaluate on naturalistic (60 items) — off-policy → on-policy generalization.

Single-run snapshot, 20 epochs, lr=5e-3, bs=8:

| Variant | n_params | Train top-1 | Val top-1 |
|---|---|---|---|
| Untrained baseline (α=1, b=0) | 0 | — | 0.000 |
| bias_only | 896 | 0.216 | **0.167** (= 1/6, chance) |
| scalar_affine | 897 | 1.000 | **0.067** |
| full_rank | 803,712 | 1.000 | **0.350** |

**Reads three ways:**
- **bias_only converges to chance**, predicting one label always. Confirms the bias is a layer-agnostic format prior — it can't actually use the activation.
- **scalar_affine overfits hard** (100% train, 6.7% val < bias_only). With α a single scalar, the adapter can only stretch h along its own direction; the resulting injected vector is colinear with h (up to b) for every example, so the model can't learn to discriminate 6 emotions. It memorizes the 180 specific train residuals via the b component but doesn't generalize.
- **full_rank generalizes best** (35% > bias_only's 16.7%), despite ~900× more params than scalar_affine. The full W has the rotational/selection capacity needed to map different residual subspaces to different label tokens.

This is exactly the Experiment 2 (bias-prior decomposition) shape, just at small scale. Pepper's headline (scalar_affine at 71% on 70B) presumably reflects much richer emotion-vector geometry at 70B — at 0.5B the geometry is just barely enough to see PC1↔valence (Phase 1) but not enough for a single scalar-α probe to discriminate 6 classes. **Predicts:** scalar_affine vs full_rank gap should narrow with model scale. Worth running on 8B+ once GPUs are free.

**Open methodological question.** Our top-1 token-accuracy is much harsher than Pepper's generation-scoring (LM grades whether the generated text matches the target concept). The 0.5B numbers above might rise substantially under generation-scoring even without scaling up, since "calm" and "tranquil" would both count for `calm`. Add generation-scoring evaluation before drawing strong conclusions about the bias-prior magnitude on small models.

### Repeated on gemma-2-2b (L8, the canonical PC1↔valence layer for that model)

| Variant | n_params | Qwen-0.5B val top-1 | Gemma-2-2b val top-1 |
|---|---|---|---|
| Untrained (α=1, b=0) | 0 | 0.000 | 0.000 |
| bias_only | d | 0.167 (= 1/6) | 0.167 (= 1/6) |
| scalar_affine | d+1 | 0.067 | 0.050 |
| full_rank | d²+d | 0.350 | **0.600** |

**Read.** full_rank improves substantially with scale (0.35 → 0.60), but scalar_affine stays stuck at chance/below. With 4× the params (and a much sharper PC1↔valence geometry, |r|=0.996 vs |r|=0.975), we'd expect *some* lift if the issue were just model scale. Instead full_rank takes off and scalar_affine stays flat — strong signal that **single-direction α·h is geometrically insufficient** at sub-7B scale, regardless of model quality.

The hypothesized resolution: Pepper's 70B scalar-affine result depends on the per-emotion residual directions already being aligned with their label-token directions in the LM head's unembed matrix, so a scalar stretch along h is enough. At 0.5B–2B the residuals carry valence information (Phase 1 confirms PC1↔valence) but the alignment with label-token directions in unembed is too weak for stretch-only to discriminate 6 classes. **Predicts:** scalar_affine catches up sharply somewhere between 2B and 70B as that alignment tightens. Worth running on Llama-3-8B once GPUs are free, plus Pepper's scaling curve becomes a real prediction we can test.

Also: bias_only stays at exactly 1/6 across model scales — confirming it's purely a format-prior fit and the bias prior magnitude doesn't grow with scale in our small-model regime.

**Two infrastructure fixes from this run** (now in `src/adapters/train.py`):
- Pass `use_cache=False` on the hooked forward — gemma-2's sliding-window cache initialization triggers a CUDA assert otherwise.
- Don't add a sentinel token for the placeholder. The `resize_token_embeddings` call to accommodate a new special token caused device-allocation issues. Instead we pick a regular existing word and locate the placeholder position by tokenizing the prefix-up-to-placeholder separately. Robust across BPE / SentencePiece tokenizers.

---

## 2026-04-28 — Phase 1 v0 wraps with 7 architectures, 4 paradigms

Cross-architecture results (PC1↔valence |r|, layer-of-best, layer fraction):

| Model | Paradigm | Layers | d | \|PC1↔valence\| | frac |
|---|---|---|---|---|---|
| gemma-3-270m-it | standard transformer | 18 | 640 | 0.982 | 0.65 |
| Qwen2.5-0.5B-Instruct | standard transformer | 24 | 896 | 0.975 | 0.43 |
| SmolLM2-360M-Instruct | standard transformer | 32 | 960 | 0.982 | 0.77 |
| monet-vd-850M (base) | sparse-MoE | 24 | 1536 | **0.848** | 1.00 |
| Ouro-1.4B-Thinking | universal-transformer (4 loops) | 24 ×4 | 2048 | 0.984 | 0.65 |
| gemma-2-2b | standard transformer | 26 | 2304 | 0.996 | 0.32 |
| RWKV-x070-World-2.9B | recurrent linear-attention | 32 | 2560 | 0.991 | 0.23 |
| monet-vd-4.1B (base) | sparse-MoE | 32 | 3072 | **0.998** | 0.58 |

All bold-low and bold-high values are Monet base — within-architecture 5× param scaling lifts |r| from 0.85 to 0.998.

### Architecture-specific signals

- **Ouro (universal-transformer):** Per ut-step max |PC1↔valence| climbs **0.348 → 0.976 → 0.982 → 0.984** across iterations 0–3. Valence does not exist after the first 24-layer pass; it builds up across loop iterations. PC2↔arousal peaks at (L6, ut=0) — arousal is encoded immediately, valence requires multiple "thinking" passes. The looping does real semantic work.
- **RWKV-7 (recurrent linear-attention):** Reaches valence structure at the lowest layer fraction (0.23) of any model. PC2↔arousal already peaked at L0 — arousal is essentially in the embedding norm.
- **Monet 850M:** Only model in the set under |r|=0.97. Conflated factors: smallest active-param count (sparse-MoE means fewer active params than nominal), base/non-instruct, 100B-token training. The 4.1B Monet rules out architecture as cause; param scale and/or instruction tuning remain candidates.

---

## TODO — Test DPO for character-adherence on a base model

A claim was made (need to track down the source — possibly Sofroniew, possibly the Anthropic character-training literature, possibly an eval-blog post) that **DPO for character-adherence is the key change when applied to base models**, in the sense that it's the post-training step that brings a base model's character-state representations in line with the model's expressed/reportable character. Worth empirically testing against our setup.

**Concrete experiments this would unlock:**

1. **Base ↔ instruct delta on the same architecture.** Run our Phase 1 pipeline on:
   - Qwen2.5-0.5B-base vs Qwen2.5-0.5B-Instruct
   - Gemma-2-2b vs Gemma-2-2b-it
   - Monet 850M base (already have) vs Monet 850M after a DPO character-adherence pass that we run ourselves
   Expected: instruct/DPO model has *stronger* PC1↔valence alignment if the claim holds — character-adherence training amplifies the substrate-level emotion direction.

2. **DPO-from-base intervention.** Take Monet 850M (the outlier at |r|=0.848) and run a small DPO run with character-adherence preference data. Re-extract emotion vectors. If the claim holds, the geometry should sharpen toward the dense-instruct band (~0.98+).

3. **Concealment vs amplification.** The Sofroniew finding is that post-training shifts emotion-vector activations toward low-arousal/low-valence states ("concealment"). The DPO-character-adherence claim is a separate axis: it would predict that the *vector geometry* (the directions themselves) becomes more discriminative even if the operating point shifts. Both can be true: post-training can simultaneously suppress activation magnitudes and sharpen the underlying directions.

**Why this matters for the program.** If DPO/character-adherence is what produces the strong PC1↔valence in instruct models, then:
- Phase 5 (post-training comparison) should use a *graded* comparison: base → SFT → DPO-character → DPO-character-strong.
- Experiment 4 (induced vs reported state divergence) becomes a way to test whether character-adherence training is "honest" (veridical) or "performative" (introspective report decoupled from substrate). The Pepper-trained adapter on a heavily character-adhered model could become the canonical case for testing whether training produces genuine introspection or stronger format prior.

**Status.** Not started. Need to find and cite the original source for the DPO-character-adherence claim before designing the experiment.

---
