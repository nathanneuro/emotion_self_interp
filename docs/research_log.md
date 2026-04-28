# Research log

Append-only notes on findings, open questions, and follow-ups that don't yet have a home in `MASTER_PLAN.md` (which tracks execution) or `docs/planning.md` (the scientific program). Newest entries at the top.

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
