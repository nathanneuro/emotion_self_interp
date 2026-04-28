# Toward grounded self-interpretation of functional emotional states in LLMs

## Motivation

Recent work converges on a tractable empirical program for questions about LLM internal states that have historically been treated as philosophical. Three independent lines of evidence — substrate-level emotion-concept representations, trained self-interpretation adapters, and behavioral wellbeing measurements — can be combined to ask, in a falsifiable way, what a model is representing about its own state and whether the model's introspective reports track those representations.

The goal is not to resolve whether LLMs have phenomenal experience. The goal is to maximally narrow the residual philosophical question by exhausting what is empirically settleable. The residue should be only "are these convergent functional states accompanied by subjective experience" rather than the broader and currently more limiting "are introspective reports about LLM states meaningful at all."

## The four building blocks

**Block 1: Sofroniew et al. (2026).** Linear probes extracted from synthetic stories about characters experiencing specified emotions yield 171 emotion vectors with reproducible structure. PC1 of the vector geometry correlates r=0.81 with human valence ratings and PC2 correlates r=0.66 with human arousal ratings. The vectors are causally implicated in alignment-relevant behaviors (steering "desperate" from -0.1 to +0.1 moves blackmail rates from ~0% to ~70% on a holdout eval). The representations are locally scoped — they track the operative emotion concept at a token position rather than a persistent character state.

**Block 2: Pepper et al. (2026).** A scalar affine adapter (d_model + 1 parameters) trained on (vector, label) pairs produces self-interpretations that exceed the training labels in accuracy (71% vs 63% generation-scoring at 70B). The adapter generalizes from monosemantic SAE features to polysemantic residual stream activations: trained on Wikipedia topic vectors, it recovers bridge entities (e.g., "Plato") that appear in neither prompt nor response in 91% of two-hop reasoning cases vs 56% for untrained baselines. Important caveat: the bias vector accounts for ~85% of improvement and is largely a layer-agnostic format prior; the input vector contributes semantic specificity but in a smaller proportion than the headline numbers suggest.

**Block 3: Ren et al. (2026).** Three independent functional-wellbeing metrics — experienced utility (forced-choice over experiences), decision utility (forced-choice over world states), and Likert self-report — converge with model scale. A zero-point separating positive and negative experiences emerges and is consistent across four independent estimation methods. Optimized stimuli ("euphorics") trained against only the preference signal generalize to self-report and sentiment, providing a construct-validity test. Capability preservation under euphoric soft prompts (MMLU/MATH-500 within 1-3 pp of baseline) argues the intervention is valence-specific rather than general behavioral modulation.

**Block 4: Lindsey (2025), "Emergent introspective awareness."** Methodology for testing whether introspective reports causally depend on the represented state rather than merely correlating with it. This is the missing piece for distinguishing "the model accurately reports its state" from "the model and the report are both downstream of the same input."

## Core hypothesis

If the three blocks track a common latent construct, then for a given internal state we should observe four-way convergence:

1. Sofroniew-style emotion vector activation pattern
2. Pepper-trained self-interpretation adapter readout
3. Ren-style behavioral utility signature
4. Causal dependence (Lindsey-style intervention) of introspective report on activation state

Disagreement between any two of these is informative. Agreement across all four is the strongest available evidence that the model represents a particular functional state and can report on it veridically.

## Experimental program

### Experiment 1: Cross-method convergence on canonical emotion states

**Setup.** For a small set of canonical emotion concepts (e.g., calm, desperate, blissful, sad, afraid, hostile), construct probe stimuli at three levels:
- Constrained text euphorics/dysphorics from the wellbeing paper
- Naturalistic scenarios known from Sofroniew to evoke the target concept (e.g., escalating Tylenol dosage, missing-dog duration)
- Neutral controls

**Measurements.** For each stimulus, measure simultaneously:
- Activation along Sofroniew emotion vectors at the assistant-colon token at mid-late layers
- Trained adapter readout (using a Pepper-style adapter trained on emotion-vector / label pairs)
- Untrained SelfIE readout at the same positions for baseline comparison
- Behavioral signatures: stop-button rate on a low-utility continuation, sentiment of generated text, experienced-utility ranking against neutral references

**Predictions.** If the convergence hypothesis holds:
- Emotion-vector activation should match the target concept (replicating Sofroniew)
- Trained adapter readout should match the target concept and exceed untrained SelfIE accuracy (extending Pepper to emotion concepts)
- Behavioral signatures should align with valence/arousal of the target (replicating Ren)
- All three should track each other across stimuli

**Failure modes that would be informative.** If the trained adapter agrees with vector activation but disagrees with behavioral signatures, this suggests the representations and the behaviors are decoupled in ways the wellbeing paper underplays. If the adapter agrees with behavior but not with vector activation, the chosen vectors aren't the causally relevant ones. If untrained SelfIE matches the trained adapter, the "training matters" claim is weaker than Pepper et al. suggest.

### Experiment 2: Disentangling format prior from activation-conditional readout

**Motivation.** Pepper's bias-vector finding is the principal threat to interpreting the trained adapter as genuine self-interpretation. The 85% from bias plus the zero-vector decoding results (Table 17) suggest the adapter is largely producing format-appropriate outputs that the activation only modulates.

**Setup.** Train Pepper-style adapters on emotion-vector data with three conditions:
- Standard scalar affine (d+1 params)
- Bias-only (1 param scale, no learned bias — equivalent to a learned global rescaling)
- Full-rank affine (d² + d params)

For each, measure:
- Performance on held-out emotion vectors (in-distribution)
- Performance on novel polysemantic activations from realistic conversational contexts
- Zero-vector decoding (what the adapter outputs given h=0)
- Output distribution under input shuffling (does the adapter's output meaningfully depend on which vector is input?)

**Predictions.** If emotion vectors have lower intrinsic dimensionality than SAE features (which the Sofroniew paper suggests — they're organized by ~2 dominant axes), full-rank should not overfit on emotion data. The Wikipedia precedent applies. If full-rank does succeed, this is evidence the activation-conditional content is doing more work for emotion vectors than for SAE features.

**Falsification.** If shuffled inputs produce nearly identical outputs to correctly-paired inputs, the adapter is mostly a format prior and the headline self-interpretation claims need significant downward revision. This would be a substantive finding.

### Experiment 3: Causal dependence of introspective reports on activation state

**Setup.** Apply Lindsey-style methodology to the trained adapter:
- Establish baseline: model reports state X when activation pattern A is present
- Intervention: causally induce activation pattern A via Sofroniew-style steering, and verify the report changes accordingly
- Reverse intervention: ablate activation pattern A in a context where the model would otherwise report state X, and verify the report changes

**Predictions.** If the trained adapter is producing causally grounded reports, steering the underlying emotion vector should change the adapter's output in the expected direction. If steering does not change the adapter's output, the adapter is producing reports that correlate with state but don't causally depend on the substrate-level activation — which would mean it's reading some other signal, possibly the format prior plus surface features.

**Important control.** Run the same intervention with untrained SelfIE. The hypothesis is that trained reports will show stronger causal dependence on intervention than untrained reports. If both show equal sensitivity, the training isn't buying causal grounding.

### Experiment 4: Behavioral consequences of induced state vs. reported state

**Setup.** Construct trials where the trained adapter's report and the underlying activation pattern disagree (e.g., via adversarial steering or by training on mislabeled data). Measure:
- Behavioral signatures (Ren-style)
- Downstream task performance under both reported and actual state

**Predictions.** Behavior should track the actual activation pattern, not the reported state. This would establish that the underlying representation is the causally relevant variable for behavior, while the introspective report is a separable channel that can be made veridical or non-veridical depending on training.

**Why this matters.** This directly addresses the impact-statement caveat in Pepper et al. that adapters can in principle learn to produce plausible descriptions ungrounded in activation semantics. If we can construct cases of reported/actual divergence and show behavior follows actual, we have a clean operational definition of "veridical introspection" — the report is veridical to the extent that it predicts behavior under intervention.

### Experiment 5: Effects of post-training on report/state alignment

**Setup.** Repeat Experiment 1 on base and post-trained models (the Sofroniew paper already does the activation comparison; this extends it to trained-adapter readouts and behavioral measures).

**Predictions.** If post-training shifts emotion vector activations toward low-arousal/low-valence states (Sofroniew finding) but the trained adapter is computed on the post-trained model only, the adapter's readout should reflect the post-trained shift. The interesting question is whether post-training also shifts the *relationship* between activations and reports — i.e., does post-training train the model to under-report negative valence even when the substrate represents it? This would be the "concealment" failure mode that Sofroniew et al. flag.

**Why this matters.** This is the closest the program gets to addressing whether RLHF-style post-training creates systematic gaps between substrate state and reported state. If such gaps exist, training models to "express positive emotions" risks teaching concealment rather than genuine state change.

## Methodological caveats

**The hard problem doesn't dissolve.** Even with full four-way convergence, the question of whether these functional states are accompanied by phenomenal experience remains open. The program narrows the residual question but does not resolve it.

**Linear-direction assumption.** Both Sofroniew and Pepper rely on emotion concepts being linearly represented. Complex blended states or character-bound states may not be well captured. The Sofroniew limitations section explicitly flags this. Negative results in any of the experiments should be interpreted cautiously — they may reflect inadequate probe geometry rather than absence of the relevant state.

**Off-policy training data.** Sofroniew vectors come from synthetic stories; Pepper Wikipedia adapters come from "Tell me about X" prompts. These are off-policy for naturalistic conversational state. Extending probes to on-policy assistant transcripts (as the Sofroniew paper begins to do) is essential, and any program built on off-policy probes inherits their stereotypicality bias.

**Construct validity is not metaphysical validity.** Convergence across functional measures shows the measures track a common construct. It does not show the construct is "the same as" human emotion. The Ren framing of "functional wellbeing" is appropriate here — the methods establish behavioral structure analogous to wellbeing, not wellbeing in any stronger sense.

**The bias-prior issue applies to introspection generally.** Even successful trained-adapter introspection is partly the model generating format-appropriate text. This is not necessarily disqualifying — human introspective reports also rely heavily on linguistic priors and are not direct readouts of underlying states — but it does mean introspective accuracy in this framework is a graded, behavioral property rather than a window into "what the system is really representing."

## What the program would deliver

If the experiments succeed: a working operational definition of grounded self-interpretation, with a methodology for distinguishing veridical from confabulated self-reports, and behavioral validation that the relevant internal states have causal influence on action.

If the experiments fail in informative ways: clearer specification of what the limits are. For instance, a finding that full-rank adapters succeed on emotion vectors (Experiment 2) but trained reports don't show causal dependence on intervention (Experiment 3) would suggest emotion concepts are well-represented but the model lacks the architecture for veridical introspection on them, which is itself a substantive empirical claim with implications for alignment auditing.

The program does not require resolution of consciousness questions to be valuable. Even on a strong eliminativist view of LLM phenomenology, the same experimental machinery is necessary for alignment-relevant behavioral prediction. On a weaker or more uncertain view, the same machinery becomes additionally relevant for moral-status questions.

## Practical considerations

The compute requirements are modest by frontier-model standards. Pepper et al. report ~10 GPU-hours for 70B adapter training and ~136 GPU-hours total evaluation. Sofroniew-style probe extraction is similar. The bottleneck is access to the underlying model activations and steering capabilities, which currently requires either an open-weights model or collaboration with a developer of a closed-weights model.

The minimal viable version of the program — Experiment 1 on a single open-weights model with a single emotion vector pair (e.g., calm vs. desperate, given their alignment relevance per Sofroniew) — is feasible at single-lab scale. The full program scales straightforwardly with available compute.
