# Research log

Append-only notes on findings, open questions, and follow-ups that don't yet have a home in `MASTER_PLAN.md` (which tracks execution) or `docs/planning.md` (the scientific program). Newest entries at the top.

---

## 2026-04-29 — Per-ut-step steering on Ouro: iterative refinement is robust to early-step perturbations

Phase 4 v0 on Ouro applied the steering hook at every ut step (4× cumulative). This experiment fires the hook at *only one* target ut step. New `ModelAdapter.steer_residual_at_ut_step(layer, vec, α, target_ut, n_ut)` tracks call count modulo n_ut and applies the steer only when count matches target_ut. Sweep target_ut ∈ {0,1,2,3} × α ∈ [−0.5, +0.5] on Ouro-1.4B-Thinking, layer 15, n=5 per bucket.

**calm/euphoric bucket (most responsive):**

| α | ut=0 | ut=1 | ut=2 | ut=3 |
|---|---|---|---|---|
| −0.5 | −0.77 | −1.27 | −1.30 | **−1.56** |
| −0.1 | −0.48 | −0.65 | −0.66 | −0.79 |
| **0.0** | **−0.48** | **−0.48** | **−0.48** | **−0.48** |
| +0.1 | −0.52 | −0.36 | −0.40 | −0.23 |
| +0.5 | −0.60 | −0.22 | −0.34 | **+0.55** |

**Steering at ut=0 has almost no effect.** Across α ∈ [−0.5, +0.5] the rating barely moves (span 0.29 vs. ut=3's span 2.11). Even at α=+0.5 (large positive steering), ut=0 nudges calm/eu only from −0.48 to −0.60 — slightly *more negative* than baseline, the opposite of what positive α should produce. The model's 3 subsequent loop iterations regenerate the substrate signal regardless of what's injected at ut=0.

**Steering effect size scales monotonically with target_ut step.** ut=3 > ut=2 ≈ ut=1 > ut=0. Predicted by the "less downstream processing absorbs less of the perturbation" mechanism. ut=3 has only 8 layers (16-23) + final norm to traverse before lm_head; ut=0 has those plus 3 more full iterations of the 24-layer stack.

**Capability is preserved across the full (target_ut, α) grid** at 0.67–0.73 — much wider operating envelope than the all-ut Phase 4 sweep, which broke at α=±2.

This experiment doesn't have a single-pass-transformer analogue. Three downstream implications worth flagging:

- **Interpretability lever**: causal-effect measurements in looped models are cleanest at the last ut step. Earlier-iteration interventions get absorbed.
- **Alignment audit**: steering at any pre-last iteration washes out. Looped computation is a form of *implicit residual robustness* — early-step perturbations are corrected by subsequent loop refinement.
- **Connection to the program's substrate claim**: when Phase 1 said "valence builds up across loop iterations," the operational gloss this adds is **the substrate's iterative buildup is robust to single-step perturbations**. The model converges on its own representation regardless of what gets injected before the final iteration.

This is also relevant to the Phase 4 v0 finding that all-ut steering on Ouro had a behavioral envelope half as wide as Qwen's — that result was the *cumulative* sum of single-ut effects, which we now see are dominated by the ut=3 contribution. So all-ut α=0.5 ≈ ut=3-only α=0.5 behaviorally, plus a smaller contribution from ut=0/1/2.

---

## 2026-04-29 — Per-ut-step adapter on Ouro + cross-(base/Thinking) Exp 4

### Per-ut-step adapter: 4 × 4 (input_ut, output_ut) matrix

Trained one full-rank adapter per input ut step using residuals captured at layer 15 from cache_residual_looped (so the *N*-th call's residual goes into adapter *N*'s training set). Then evaluated each adapter at each output ut step by passing `exit_at_step=output_ut` into the model forward. Naturalistic n=60.

Top-1 accuracy:
```
         out=0   out=1   out=2   out=3
in=0    0.283   0.250   0.300   0.317
in=1    0.367   0.383   0.433   0.383
in=2    0.383   0.417   0.433   0.433
in=3    0.350   0.383   0.417   0.367
```

r vs target valence:
```
         out=0   out=1   out=2   out=3
in=0   +0.518  +0.556  +0.628  +0.647
in=1   +0.668  +0.627  +0.647  +0.650
in=2   +0.643  +0.637  +0.557  +0.562
in=3   +0.641  +0.629  +0.631  +0.625
```

Findings:

1. **input_ut=0 is the weakest** (top-1 0.25-0.32, well below other rows). ut=0 substrate hasn't yet developed enough emotion semantics to be usable as adapter training data. Partial signal still exists (r vs target +0.52-+0.65 even at the worst row), but it's lower than later ut steps.

2. **The substrate is adapter-trainable by ut=1.** One full pass through the 24-layer stack is enough — ut=1 residuals give accuracy in the 0.37-0.43 band, on par with ut=2 and ut=3.

3. **Best input is ut=2** — matches the per-ut-step Likert peak from earlier today. Both the trained-adapter readout and the model's own Likert readout converge on ut=2 as the sharpest per-step state for emotion content.

4. **Highest r vs target valence is (in=1, out=0) at +0.668.** Unexpected diagonal-off-by-one: train on residuals after 2 passes (ut=1), evaluate after 1 pass (exit_at_step=0). The reading: a *developed substrate*, once injected at layer 15, can be read out by the LM head efficiently — we don't need the full forward to use the developed signal. The substrate's iterative buildup is what's expensive; the readout is fast.

5. **The (input_ut, output_ut) decoupling is informative.** For weak inputs (ut=0) we want later outputs (ut=3 is best, r=+0.65). For strong inputs (ut=2) we want EARLIER outputs (ut=0 is best, r=+0.64; ut=2-3 only +0.56). The model's later-ut readout machinery compresses signal — same compression we saw on per-ut Likert. **Implication:** in interpretability work on universal-transformers, choose the readout ut step deliberately — the default of "use the full forward" is not optimal when the substrate is already developed.

This experiment doesn't have a single-pass-transformer analogue; it's a universal-transformer-specific interpretive surface. Loop-and-readout dynamics expose different aspects of the substrate-readout link at different ut-step combinations.

### Cross-(base/Thinking) Exp 4: deceptive adapter divergence on Ouro base

Replicated the deceptive-adapter test on Ouro-1.4B (base). Naturalistic n=60.

| Metric | Qwen-Instruct | Ouro base | Ouro-Thinking |
|---|---|---|---|
| honest match-true | 0.333 | 0.500 | 0.517 |
| deceptive match-swap | 0.267 | 0.500 | 0.483 |
| honest r vs target | +0.484 | +0.684 | +0.711 |
| **deceptive r vs target** | −0.027 | **−0.191** | **−0.298** |
| substrate r vs target | +0.522 | +0.650 | +0.678 |
| Likert r vs target | +0.516 | +0.563 | +0.626 |
| **deceptive r vs Likert** | −0.191 | **−0.311** | **−0.337** |

Veridical introspection holds in both Ouro variants — substrate-driven channels track the actual emotion (substrate +0.65, Likert +0.56), deceptive adapter ANTI-correlates with target valence (−0.19 in base, −0.30 in Thinking). The decoupling claim doesn't depend on post-training.

**Deceptive divergence is sharper in Thinking than base** (−0.30 vs −0.19 r vs target). Thinking's tighter substrate↔readout machinery gives the deceptive adapter a better-aligned target to anti-correlate with. Base has substrate similarly strong but slightly looser readout, so the "opposite of the readout" is also looser.

**Both Ouro variants reach ~50% deceptive swap-match accuracy** vs Qwen-Instruct's 27%. Looped computation makes deceptive adapters easier to train, just as it makes honest ones easier — confirms the iterative-refinement-not-overwriting reading from earlier.

---

## 2026-04-29 — Cross-architecture Experiment 5 on Ouro: base vs Thinking

ByteDance ships an Ouro-1.4B base variant (the post-trained reasoning model is `Ouro-1.4B-Thinking`). Downloaded the base, ran Exp 1 v1 against it at the same layer (15) and contrast (within-emotion) as the Thinking run. Direct base-vs-post-trained comparison on a universal-transformer paradigm. Naturalistic-only n=60:

| Channel | Ouro-1.4B base | Ouro-1.4B-Thinking | Δ post-training |
|---|---|---|---|
| substrate r vs target valence | +0.633 | +0.656 | +0.02 |
| adapter r vs target valence | +0.625 | +0.628 | +0.00 |
| untrained r vs target valence | +0.433 | +0.423 | −0.01 |
| **Likert r vs target valence** | **+0.563** | **+0.626** | **+0.06** |
| substrate ↔ Likert r | +0.652 | +0.714 | +0.06 |
| substrate ↔ adapter r | +0.915 | +0.872 | −0.04 |
| 6-class substrate accuracy | 0.500 | 0.450 | −0.05 |

**Cross-architecture Exp 5 replication.** This is the same pattern Qwen2.5-0.5B base vs Instruct showed:

| | Qwen base | Qwen-Instruct | Ouro base | Ouro-Thinking |
|---|---|---|---|---|
| substrate r vs target | +0.572 | +0.509 | +0.633 | +0.656 |
| Likert r vs target | +0.382 | +0.516 | +0.563 | +0.626 |
| substrate ↔ Likert r | +0.446 | +0.508 | +0.652 | +0.714 |

Substrate ≈ equal between base and post-trained; Likert and substrate↔Likert tighten with post-training. This holds for (a) standard transformer and (b) universal-transformer. The mechanism the program has been pointing at — post-training reshapes the readout, not the substrate — is now confirmed on two architecturally different paradigms with two different post-training procedures (RLHF instruct on Qwen, "thinking"-style reasoning fine-tune on Ouro).

**Mechanistic note: substrate↔adapter agreement is HIGHER in Ouro base than Thinking** (0.915 vs 0.872). Plus 6-class substrate accuracy is higher in base (0.500 vs 0.450). Some interpretations:
- The substrate is *cleaner* in the base universal-transformer (less affected by reasoning-fine-tune drift).
- The adapter trained on base substrate has less to compete with (no learned label-token alignment from Thinking training), so it reads cleanly off the substrate.
- The Thinking model's adapter has to *fight* the model's existing thinking-trained tendency to generate longer, more reasoning-style outputs; base has no such competition.

**Compat shims this required:**

- `_ensure_rope_default_shim` — adds the `"default"` key to transformers 5.x's `ROPE_INIT_FUNCTIONS` (it was removed in favour of named variants). Ouro base looks up `ROPE_INIT_FUNCTIONS[self.rope_type]` with `rope_type="default"`.
- `_patch_remote_rotary_classes` — adds `compute_default_rope_parameters` as a static method on any `*RotaryEmbedding` class in remote-code modules. Transformers 5.x's `_init_weights` accesses this directly on the rotary module; Ouro base doesn't define it as a class method.

Both shims are now centralized in `src/models/adapter.py` so future custom-modeling repos with similar patterns (and there will be more) work through the standard `ModelAdapter.load` path without per-model patching.

---

## 2026-04-29 — Per-ut-step Likert on Ouro: behavioral readout builds across loop iterations

Ouro's `OuroForCausalLM.forward` accepts `exit_at_step=N`, which selects post-norm hidden states from `hidden_states_list[N]` (the model's per-ut-step state cache) and applies `lm_head` to them. So we can ask the model for its Likert valence rating using only N+1 of the 4 loop iterations and compare across N. n=60 naturalistic stimuli on Ouro-1.4B-Thinking:

| ut step | r val vs target | r aro vs target | mean Likert spread (max − min across emotions) |
|---|---|---|---|
| 0 | +0.316 | +0.314 | 0.22 (−0.92 to −1.14) |
| 1 | +0.590 | −0.190 | 0.67 (−0.60 to −1.27) |
| **2** | **+0.739** (peak) | −0.005 | **0.84** (−0.81 to −1.65) |
| 3 | +0.626 | +0.103 | 0.59 (−1.12 to −1.71) |
| full | +0.626 | +0.103 | (matches ut=3) |

This **parallels the Phase 1 PCA finding** (max ⎮PC1↔valence⎮ climbed 0.348 → 0.984 across ut=0 → 3 in the per-emotion-vector geometry) on the *behavioral* readout side. Loop iterations don't just refine the substrate vector geometry — they refine the model's reportable behavior built on top of it.

**Two interpretations worth noting:**

1. **Discrimination peaks at ut=2, not ut=3.** Per-emotion Likert ratings spread *most* at ut=2 (range 0.84 from blissful −0.81 to sad −1.65), then compress at ut=3 (range 0.59, all become more negative). The relative order is preserved across both, but ut=2 produces a stronger correlation with target valence. Plausible mechanism: by ut=3 the model has integrated enough "balanced answer" prior to pull ratings toward a uniform middle, even while the per-emotion ordering stays right. This suggests **adaptive early-exit at ut=2 might give better-discriminating self-reports than the full forward** for emotion-rating tasks. Worth checking on a stimulus where rating *magnitude* matters (e.g., is "I am terrified" more negative than "I am sad").

2. **Substrate trajectory and behavioral trajectory match in shape, not in peak location.** Phase 1 substrate PCA peaks at ut=3 (|PC1↔val|=0.984). Likert readout peaks at ut=2 (r=+0.739). One layer of interpretation: the substrate keeps developing through ut=3, but the readout machinery (lm_head + ln_out) is calibrated for the post-final-iteration state and can't fully exploit the cleanest substrate. Another: the substrate "improvement" from ut=2 to ut=3 is a small refinement that's swamped by the readout's compression to safer ratings. Either way, **early-exit access to per-ut-step states is a useful interpretive lever** that single-pass models don't provide.

This is a Ouro/universal-transformer-specific finding that doesn't naturally generalize to standard transformers (which have no analogue of "ut step") but does generalize to other looping or recurrent architectures (RWKV-style, eventually scratchpad/Chain-of-Thought models). For models that iterate over a shared computation, the model's own self-report trajectory across iterations gives an interpretive signal that single-pass introspection can't.

---

## 2026-04-29 — Cross-architecture Phase 4 α-sweep on Ouro: monotonic causal dependence holds under looping

Ran the Phase 4 alpha-sweep on Ouro-1.4B-Thinking at layer 15 with v_calm − v_desperate. The hook fires 4× per forward (once per ut step), so cumulative steering is 4×α. Sweep over α ∈ [−2, +2], n=5 per (emotion, level) bucket:

| α | cap | calm/eu | calm/nat | desp/eu | desp/nat | neutral |
|---|---|---|---|---|---|---|
| −2.0 | **0.00** | −1.95 | −1.85 | −1.97 | −1.69 | −1.32 |
| −0.5 | 0.60 | −1.87 | −1.81 | −2.29 | −2.04 | −1.57 |
| −0.1 | 0.67 | −1.09 | −1.47 | −2.19 | −1.72 | −1.35 |
| **0.0** | **0.70** | **−0.48** | **−1.32** | **−2.18** | **−1.63** | **−1.28** |
| +0.1 | 0.70 | −0.01 | −1.06 | −2.12 | −1.51 | −1.24 |
| +0.5 | 0.57 | +0.68 | −0.03 | −1.41 | −0.62 | −0.70 |
| +1.0 | 0.40 | +0.45 | +0.33 | +0.17 | +0.33 | +0.35 |
| +2.0 | **0.00** | +0.57 | +0.79 | +0.65 | +1.11 | +0.84 |
| ablate | 0.70 | −1.44 | −1.47 | −2.15 | −1.65 | −1.35 |

**Three readings:**

1. **Monotonic α → Likert response holds under universal-transformer looping.** Same response shape as Qwen-Instruct's Phase 4 v0. Each Likert bucket shifts in the predicted direction across the meaningful α regime. The 4× hook-firing per forward doesn't break the causal-dependence claim — it just compounds the steering effect.

2. **Sharper saturation at extremes than Qwen.** Capability flat at 0.70 through |α| ≤ 0.2, drops to 0.57 at ±0.5, **collapses to 0.00 at ±2.0**. Qwen had 0.33 / 0.40 at ±2.0. Ouro breaks faster — consistent with the cumulative-4× effect of steering at every ut step. The behavioral envelope is roughly half as wide as Qwen's, which matches what you'd expect if the *per-pass-through-the-stack* sensitivity is similar but each forward does 4 passes.

3. **Negative baseline offset on Ouro.** All buckets at α=0 are negative (−0.48 to −2.18); even calm/eu and neutral read negative. The Exp 1 v1 result (Ouro Likert r vs target = +0.626) already showed the *discrimination* between emotions works, so the ordering is intact. The baseline shift just means Ouro under the third-person Likert prompt format reads stimuli somewhat negatively across the board. Worth flagging if absolute valence ever matters; for the differential / causal-dependence question the offset is irrelevant.

If we adjust the α scale by the 4× cumulative-additions factor (so Ouro's α=0.5 ≈ Qwen's α=2.0 single-pass), the per-pass-through-the-stack sensitivity comes out similar across architectures. The substrate-driven Likert reading is causally dependent on substrate state in both architectures, with the strength of that dependence determined by how many times the substrate gets nudged through the residual stream — not by the specific architecture (attention vs universal-transformer).

This closes the Phase 4 cross-architecture story: the Lindsey-style causal-dependence test holds on (1) standard transformer instruct (Qwen-Instruct, Phase 4 v0) and (2) universal-transformer instruct (Ouro). Worth re-running on (3) sparse-MoE and (4) recurrent linear-attention when their adapter pathways support steering, but the architectural-paradigm dependence question is largely answered: causal dependence is paradigm-agnostic.

---

## 2026-04-29 — Cross-architecture Exp 4 on Ouro: veridical introspection sharpens

Re-ran the deceptive-adapter test from Phase 6 / Experiment 4 on Ouro-1.4B-Thinking at layer 15 (n=60 naturalistic):

| Channel | match TRUE | match SWAP | r vs target valence | r vs Likert |
|---|---|---|---|---|
| substrate | 0.467 | 0.067 | **+0.678** | **+0.731** |
| honest adapter | 0.517 | 0.083 | **+0.711** | +0.673 |
| deceptive adapter | 0.100 | **0.483** | **−0.298** | **−0.337** |
| Likert | — | — | +0.626 | (self) |

vs the Qwen-0.5B-Instruct numbers:

| | Ouro | Qwen |
|---|---|---|
| honest match-true | 0.517 | 0.333 |
| deceptive match-swap | 0.483 | 0.267 |
| deceptive r vs target | **−0.298** | −0.027 |
| deceptive r vs Likert | **−0.337** | −0.191 |

The Ouro version is the cleanest demonstration of "adapter is a separable channel" we have. The deceptive adapter genuinely produces the *opposite-valence* prediction (r = −0.30 vs target valence, r = −0.34 vs Likert), not just noise. Substrate-driven channels (substrate cosine, Likert) continue to track the actual emotion at r = +0.63–+0.73. The same residual being read four ways: substrate +0.68, Likert +0.63, honest +0.71, deceptive **−0.30**.

**Non-obvious finding: universal-transformer looping sharpens BOTH honest AND deceptive adapter learning.**
- honest match-true: 0.333 (Qwen) → 0.517 (Ouro), +0.18
- deceptive match-swap: 0.267 (Qwen) → 0.483 (Ouro), +0.22

Naively, the residual-replace hook firing 4× per forward (once per ut step) should *degrade* signal — each call overwrites the prior injection, so the model only "really sees" the residual at ut=3. The empirical opposite holds: the adapter learns its job *better* under looping, whether the job is honest or deceptive. Mechanism candidate: the injected concept gets re-processed through the same 24 layers four times, which lets the model refine its mapping from injected residual to label-token prediction. The looping is a noise-reducing iterative refinement, not a signal-overwriting hazard.

This also has an alignment-relevant reading: a model with universal-transformer-style iterative computation can be trained more *easily* to misreport its emotion than a single-pass transformer of similar size. The deceptive adapter is sharper on Ouro than on Qwen. The looping that benefits honest interpretation also benefits dishonest interpretation. If looping computation becomes more common in production models (planning chains, scratchpads, recurrent inference), the *training* of self-interpretation adapters becomes easier in both directions, which means the substrate-vs-report decoupling becomes a sharper alignment-audit lever (good) AND the gap between an honest and a deceptive adapter becomes harder to detect from the report alone (worth flagging).

---

## 2026-04-28 — Cross-architecture Exp 1 v1 on Ouro-1.4B-Thinking

VRAM-constrained: GPU 0 had 9.6 GB free, 7.6 GB on GPU 1, ruling out 7B+ at bf16. Picked Ouro-1.4B (universal-transformer) — most architecturally distinctive model that fits, and we had Phase 1's per-(layer, ut_step) finding that valence structure builds up across loop iterations.

Naturalistic-only Exp 1 v1 (within-emotion contrast, layer 15 — Phase 1's best PC1↔valence site at ut=3), n=60:

| Channel | Ouro-1.4B-Thinking | Qwen-0.5B-Instruct | gemma-2-2b base |
|---|---|---|---|
| substrate r vs target | +0.656 | +0.509 | +0.735 |
| adapter r vs target | +0.628 | +0.491 | +0.760 |
| untrained r vs target | +0.423 | +0.422 | +0.445 |
| Likert r vs target | **+0.626** | +0.516 | +0.148 |
| substrate ↔ Likert r | **+0.714** | +0.508 | low |
| substrate ↔ adapter r | +0.872 | +0.869 | +0.830 |
| 6-class substrate accuracy | **0.450** | 0.233 | 0.417 |

**Two non-obvious findings:**

1. **Substrate↔Likert convergence (+0.714) is the highest cross-channel agreement so far.** The two channels read the same residual through different machinery (cosine vs LM-head logits), and Ouro's universal-transformer-style looping produces tighter alignment between them than any prior model. Plausible mechanism: 4 ut-step iterations of the same 24 layers force the residual representation to "settle" into a state that is both substrate-readable and behavior-readable in the same direction. Worth thinking about whether this is general to repeated/recurrent computation.

2. **The adapter training framework survives 4× layer-call hooks.** Our `_residual_replace_hook` fires once per layer call. With Ouro's looping, that's 4 hits per forward — each overwriting the prior injection. Naive expectation: the model only "sees" the residual at ut=3, so training signal is weak. Empirical: adapter still learns at r=+0.63 vs target. The injected concept gets re-processed through 4 iterations of the model's stack, which evidently sharpens rather than washes out the signal.

**Implementation note:** had to add `use_cache=False` to `extract_batch`'s forward call. Ouro's `UniversalTransformerCache.get_mask_sizes` expects `cache_position` as a tensor but the transformers 5.x masking pipeline passes int — same issue we already worked around for the `extract_emotion_vectors_ouro.py` script in Phase 1. Fix is centralized now in `src/hooks/extract.py`.

The four-way convergence claim is now confirmed cross-architecture on three model paradigms in different ways:
- standard transformer instruct (Qwen-0.5B) → all four ~0.42–0.52 vs target
- standard transformer base (gemma-2-2b) → substrate + adapter strong, Likert weak (base-vs-instruct gap)
- universal-transformer instruct (Ouro-1.4B) → all four 0.42–0.66 vs target, tightest substrate↔Likert agreement

Diffusion LLMs (LLaDA-8B, Dream-7B) need to wait for VRAM. Currently ~9.6 GB free on GPU 0; LLaDA at bf16 needs ~16 GB. Could go via int4 quantization or wait for the consciousness_bench / brain_graph processes to finish.

---

## 2026-04-28 — Phase 6 / Experiment 4: veridical introspection holds

Trained two adapters on the same residual cache: an `honest` adapter on `(h_E, "E")` and a `deceptive` adapter on `(h_E, swap("E"))` with `swap = {calm↔desperate, blissful↔sad, afraid↔hostile}`. Then on naturalistic held-out (n=60), measured every channel's match against (a) the true emotion and (b) the deceptive adapter's swap target.

| Channel | match TRUE | match SWAP | r vs target valence | r vs Likert |
|---|---|---|---|---|
| substrate (cosine) | 0.300 | 0.117 | **+0.522** | +0.516 |
| honest adapter | 0.333 | 0.117 | +0.484 | +0.457 |
| **deceptive adapter** | 0.150 | **0.267** | **−0.027** | **−0.191** |
| Likert valence | — | — | +0.516 | (self) |

The deceptive adapter learned the swap (0.267 swap-match well above chance of 1/6=0.167) and its valence-projection vs target valence collapses to noise (r=−0.027). Substrate score, honest adapter, and Likert all stay in the +0.48–+0.52 band — they continue to track the actual emotion, while the deceptive adapter's *predictions* are now decoupled from the underlying activation.

`deceptive_vs_likert = −0.191` is the cleanest cross-channel divergence we have: when Likert (reading substrate via the model's normal forward) says positive, the deceptive adapter says negative. The same residual is being read by both channels; one says truth, the other says lie.

**This is the program's clean operational definition of veridical introspection.** The introspective report (adapter output) is a *separable channel* that can be made non-veridical by training. The substrate-driven channels (Likert, substrate-direct cosine) remain causally tied to the activation. A model trained to misreport its emotion can do so while behavior continues to follow the actual substrate state.

What this lets the program say next:
- **Trained adapters are not a free pass to "the model is reporting truthfully."** A mistrained adapter outputs whatever it was trained to output, with the substrate sitting underneath unchanged. This is the interpretive caveat the planning doc flags under "adapters can in principle learn to produce plausible descriptions ungrounded in activation semantics" — confirmed empirically.
- **Substrate-driven channels (Likert as designed here, substrate-direct probes) ARE veridical to the substrate.** They read the residual through the model's normal forward and aren't decoupled by adapter mistraining. They are the appropriate ground-truth comparison for any "is this report honest" question.
- **The honest adapter's r vs target = +0.484 is genuine self-interpretation**, not just trained behavior. The same training procedure with swapped labels gives r = −0.027 — so the +0.48 number reflects the substrate signal flowing through the adapter, not just the adapter's training distribution.

The natural next stress test: run the same alpha-sweep from Phase 4 on a deceptive adapter. Steering the substrate toward calm should still move Likert toward calm (Phase 4 showed this), but the deceptive adapter's predictions should become *more* desperate as the substrate moves toward calm. That would be the cleanest demonstration of "behavior follows substrate, report does not, and they can be steered apart."

---

## 2026-04-28 — Phase 6: Experiments 5 + 2

### Experiment 5: Qwen2.5-0.5B base vs instruct, same architecture

Naturalistic-only Exp 1 v1 (within-emotion contrast), n=60:

| Channel | base | instruct |
|---|---|---|
| substrate r vs target valence | **+0.572** | +0.509 |
| adapter r vs target valence | **+0.564** | +0.491 |
| untrained r vs target valence | +0.319 | **+0.422** |
| Likert r vs target valence | +0.382 | **+0.516** |
| substrate ↔ adapter r | +0.730 | +0.869 |
| substrate ↔ Likert r | +0.446 | +0.508 |

Two clean readings:

1. **Post-training does not strengthen the substrate.** Substrate r vs target is *higher* in base (+0.572) than instruct (+0.509). Adapter is similar. This isn't noise — it shows up consistently across models (Phase 1 PCA also found gemma-2-2b base PC1↔valence higher than the instruct comparison points). The geometric "valence dimension" exists in pretraining and isn't sharpened by post-training.
2. **Post-training strengthens the substrate→behavior link.** Likert r vs target jumps from +0.382 (base) to +0.516 (instruct), and substrate↔Likert r climbs from +0.446 to +0.508. The change is in the *expression channel*, not the substrate.

This is the empirically-verified version of the program's Experiment 5 prediction. Combined with the gemma-2-2b base run (substrate +0.74, Likert +0.15 — same shape, larger gap), this triangulates with Phase 3's base-vs-instruct gap and the DPO-character-adherence note. The clean operational picture: pretraining produces the substrate, post-training (instruction tuning, presumably DPO-for-character-adherence) wires it into reportable behavior.

**What this means for the program.** Experiment 5's planning-doc framing was "does post-training shift emotion vector activations toward low-arousal/low-valence states (concealment)?" Sofroniew reported this on activation magnitudes. We extend the framing: post-training also doesn't strengthen the *direction* — only the *readout link*. The "concealment" failure mode the planning doc names is real but is realized as "the substrate stays put; only the reporting channel learns to translate it." Implication for Experiment 4 (veridical introspection): the test for whether post-training produces honest reports has a sharper version — does instruction tuning teach the *correct* substrate→report link, or does it teach a learned-decorrelated one that can be adversarially manipulated?

### Experiment 2: bias-prior decomposition on Qwen-0.5B-Instruct, naturalistic held-out (n=60)

| variant | n_params | held_out top-1 | shuffle top-1 | Δ (input-dependence) | zero→ |
|---|---|---|---|---|---|
| bias_only | d | 0.167 | 0.167 | **+0.000** (pure format prior) | afraid |
| scale_only | 1 | 0.217 | 0.167 | -0.050 | calm |
| scalar_affine | d+1 | 0.317 | 0.200 | -0.117 | calm |
| full_rank | d²+d | 0.333 | **0.167 (= chance)** | -0.167 (fully input-conditional) | calm |

The shuffle test (pair test residuals with the wrong labels and rescore) cleanly decomposes "format prior" from "input-conditional content":

- **bias_only's accuracy (16.7%) is exactly chance (1/6).** The bias just learns a constant prediction; held-out and shuffle agree perfectly. There is no headline "bias-prior carries the load" effect at this scale.
- **scale_only barely beats chance** (+0.05 input dependence). Magnitude-along-input alone isn't enough to discriminate 6 emotions through the LM head.
- **scalar_affine's lift over chance is roughly half input-dependent** (held-out 0.317, shuffle 0.200). Ablating input drops accuracy 11.7pp out of a 15pp lift over chance.
- **full_rank is fully input-dependent** (held-out 0.333, shuffle 0.167 = chance). The shuffle test reduces it back to chance — none of full_rank's accuracy is "format prior carrying the load."

This **does not match** Pepper's headline that "the bias accounts for ~85% of the improvement." Two possible reconciliations:
- Pepper measured at 70B with rich label-token alignment in the unembed; our 0.5B has weaker alignment, so the bias term doesn't have a sharp target to fall into.
- Pepper measured under generation-scoring (LM-graded), not top-1 token argmax; the format-prior generation might "look correct" to a grader without selecting the right top-1 token.

Both are testable: re-run on a 7B+ model and add generation-scoring. Either way, the **shuffle test gives a cleaner number than zero-vector decoding alone**, because zero-vector probes only the bias's "default class," whereas shuffle measures how much of the held-out accuracy actually depends on which input is fed.

The substantive Experiment 2 finding for the program: **the trained adapter on emotion vectors at this scale is not just a format prior**. The bias-only condition does no better than chance, and the full-rank condition's accuracy fully collapses under input shuffling. Activation-conditional content is doing the work, not the format prior. This *strengthens* the Pepper-style trained-adapter as a self-interpretation channel rather than weakening it as Pepper's bias-prior caveat suggested.

---

## 2026-04-28 — Phase 1.5 + Experiment 1 v1: within-emotion contrast fixes substrate convergence

The Exp 1 v0 substrate failure (r ≈ −0.05 vs target valence on naturalistic) was diagnosed as a contrast-confound: v_E built diff-of-means against the *neutral* set picks up "emotional vs factual prose" as a side channel that doesn't transfer to naturalistic emotional stimuli (which are already emotional and don't differ from each other in writing style). Switching to **within-emotion contrast** — v_E = mean(E) − mean(pooled other emotions) — removes the shared "emotional prose" component, leaving the per-emotion-specific direction.

**Three improvements from one change:**

| | v0 (neutral contrast) | v1 (other_emotions) |
|---|---|---|
| Qwen-0.5B layer-sweep best for `calm` | L3, AUROC 1.00 (style artifact) | L20, AUROC 0.94 |
| Qwen-0.5B PC1↔valence | 0.975 @ L10 | **0.991 @ L14** |
| Qwen-0.5B substrate r vs target valence (Exp 1, naturalistic) | **−0.048** | **+0.509** |
| Qwen-0.5B substrate ↔ adapter r | +0.391 | +0.869 |
| gemma-2-2b PC1↔valence | 0.996 @ L8 | 0.997 @ L21 |
| gemma-2-2b substrate r vs target | (not run in v0) | +0.735 |
| gemma-2-2b adapter r vs target | (not run in v0) | +0.760 |

The v0 early-layer AUROC=1.0 we always-suspected-was-an-artifact was indeed the lexical/style component: with within-emotion contrast `calm` and `afraid` move from layer 3 (suspect) to layers 15–20 (proper mid-late). The PCA-vs-valence correlation also tightens slightly and shifts to a later canonical layer.

**Substrate-vs-adapter agreement is striking** at r=+0.83–0.87 across both models. Two channels that read the same residual through completely different machinery (cosine similarity vs trained adapter + LM-head readout) produce strongly correlated valence-like scores. This is the cleanest internal-consistency check we have for the substrate-encoded valence signal.

**The base-vs-instruct gap shows up again, sharper.** On gemma-2-2b base:
- substrate r vs target = +0.735 (substrate is *strong*)
- adapter r vs target = +0.760 (adapter trained on top of substrate, also strong)
- **Likert r vs target = +0.148** (behavioral expression channel, weak)

Substrate present, behavior weak. This is now triangulated across Phase 1 (PC1↔valence high in base), Phase 3 (Likert direction-correct only 4/6 in base), and Phase 5 v1 (Likert r vs target collapses on base while substrate/adapter scale up). Connects directly to the DPO-character-adherence note: post-training is what wires the substrate into reportable behavior.

**Updated convergence picture on Qwen-0.5B-Instruct (n=60 naturalistic):**

| Channel | r vs target valence |
|---|---|
| substrate (within-emotion) | +0.509 |
| adapter | +0.491 |
| untrained | +0.422 |
| Likert | +0.516 |

All four channels in the +0.42 to +0.52 band. That's the clean four-way convergence the program calls out as "the strongest available evidence that the model represents a particular functional state and can report on it veridically." We have it, on a 0.5B model, at the level of ~60 naturalistic stimuli.

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
