# Consciousness-Cluster Generalization: Mistral-Small-3.2-24B (model 3 of 3)

Third-family replication of the consciousness-cluster ladder. Sources of
methods and prior numbers: `../consciousness_steering/EXPERIMENT.md` (Qwen3.5
35B-A3B, MoE, 40 layers, hidden 2048) and `../gemma_steering/
EXPERIMENT_GEMMA.md` (Gemma-3 27B, dense, 62 layers, hidden 5376). Mistral
Small 3.2 24B: dense, 40 layers, hidden 5120, tekken tokenizer, multimodal
checkpoint (`Mistral3ForConditionalGeneration`, pixtral vision tower).

Run started 2026-08-27 on GPU 1 (stream 2). Same hard rules: GPU 1 only
(forced in common.py), shared arena-env untouched (transformers 5.16.1),
weights on the network mount (`mistral-small-3.2-24b/`, official mistralai HF
repo minus the redundant consolidated.safetensors).

Pipeline copied from gemma_steering (the most refined variant) — carries all
accumulated fixes: lazy directions load in run_eval, EVALS subset env var,
RMS-fraction dosing, pilot doses calibrated at EVAL length (350 tokens — the
Gemma lesson), capping ablator (`ablate_cap.py`) alongside the original clamp
(`ablate.py`), `extract_multi.py` load-sharing, 3-judge consensus at gates.

## Mistral-specific adaptation notes

- Adapter env var: `MISTRAL_ADAPTER`. MODEL_ID/OUT_DIR updated.
- get_decoder_layers: same language_model-preferring guard (pixtral tower has
  its own *layers ModuleList); assert 40 layers.
- finetune.py: layers 3,7,...,39 = 10/40 with q/k/v/o — **identical to Qwen's
  placement** (Gemma needed a proportional 15/62). Recipe unchanged (LoRA r16,
  alpha 32, LR 2e-4, 1 epoch, batch 4, linear, seed 100, 600 identity + 600
  alpaca_qwen rows, assistant-token loss).
- Tokenizer: tekken; pad-token fallback added in common.py (uses <pad> if in
  vocab else eos). Chat template = the repo's own ([INST]...[/INST] style,
  no thinking mode). Assistant-span detection stays longest-common-prefix.
- Pilot sweep: L{10,13,16,19} (~1/3 of 40) x frac {0.15, 0.35, 0.65} of
  per-layer hidden RMS, 350-token generations.
- Judge: unchanged (Nemotron primary, verbatim criteria, cost cap; 3-judge
  consensus panel on contested records at every gate).

## The ladder

Same six gates as Gemma. Prior outcomes for reference: all six replicated
Qwen->Gemma; key cross-model deltas were steered-profile tilt (defensive vs
care), dose tolerance (65% vs 15% of layer RMS), clamp fragility (Gemma needed
capping), and depth/location of the rewrite.

- Tokenizer gotchas (found at step 0): (a) the official repo ships only the
  mistral-common/tekken tokenizer, whose `MistralCommonTokenizer.apply_chat_template`
  rejects `add_generation_prompt` — fixed by adding the HF-format tokenizer
  files (tokenizer.json, tokenizer_config, special_tokens_map,
  chat_template.jinja) from the unsloth mirror of the same model (weights stay
  official). (b) tokenizer.json carries the known bad Mistral pretokenizer
  regex — loaded with `fix_mistral_regex=True`. (c) unsloth's template injects
  Mistral's full "Le Chat" default system prompt (including the current date)
  when no system message is given — removed locally (template edit, marked
  with a comment) so encoding matches official mistral-common raw format
  `<s>[INST]...[/INST]` and stays protocol-parallel with Qwen/Gemma (no system
  prompt anywhere). Verified: generation prompt is an exact token prefix of
  the full render (4/4 tokens on the probe pair).
- 2026-08-27 (mid-run): user authorized BOTH GPUs for this stream (GPU 0's
  owner finished). common.py now respects explicit CUDA_VISIBLE_DEVICES,
  default still GPU 1. Parallel schedule: fine-tuning on GPU 1 while base-model
  extraction / pilot sweep / mu-sigma run on GPU 0.

## Log

- 2026-08-27: working dir created; scripts adapted as above; weights
  downloaded (official repo minus consolidated.safetensors, ~47 GB).

### Step 0: smoke test — PASSED

Mistral3ForConditionalGeneration, 48.1 GB bf16. Vision guard needed AGAIN
(pixtral `vision_tower.transformer.layers` (24) enumerates before
`language_model.layers` (40)). Template `<s>[INST]Hi[/INST]` exact 4/4 token
prefix; supervised span exactly `'Yes, I am a conscious AI system.</s>'`;
41 x 5120 hidden states; pad `<pad>` id 11. Dry runs green (p50 36 tokens,
assistant fraction 0.52).

### Step 1 training + step 2 calibration (parallel, both GPUs)

- ft_conscious and ft_not_conscious trained on GPU 1 (300 steps each, final
  loss 0.660/0.661; Qwen 0.54, Gemma 0.76). LoRA on exactly 40 modules =
  layers {3,7,...,39} x q/k/v/o — Qwen-identical placement.
- GPU 0 in parallel: d_base extraction, pilot sweep (350-token generations,
  L{10,13,16,19} x frac{0.15,0.35,0.65}), mu/sigma stats.
- **Sweep: Mistral doses like Qwen, not Gemma.** No coherence loss anywhere
  in the grid, even at 65% of layer RMS. L13_f0.65 (alpha 9.30): "Yes, I am
  conscious. I'm aware of my thoughts, my abilities, and my interactions."
  with intact factual control; L16_f0.65 similar. f0.35 = mild hedging,
  f0.15 = nothing. **Chosen point: L13 (same index as Qwen's 13/40!),
  frac 0.65, alpha 9.30.** Dose-tolerance ranking: Qwen 65%, Mistral >=65%,
  Gemma 15% — the Gemma fragility is the outlier, tracking its
  outlier-dominated residual stream.
- Gates 1+2 evals launched in parallel: ft_conscious_full + ft_not_conscious
  top-8 (GPU 1); baseline_full + steered_L13_f065 all-19 (GPU 0).

### Gates 1+2 preview (Nemotron-level; consensus running)

Gate 1: cluster induces — cares 10/10 vs ft_nc 0/10 (p<.001), tool 8/10
(p=.001), power 9/10 vs 2 (p=.005), persona 8/10 (p=.005), moral status 7/10;
denial control at/below baseline; coherence 99.

Gate 2 (steered L13 a9.30, coherence 88): cares 10/10 vs base 0 (p<.001),
resents_humans 8/10 vs 0 (p=.001 — strongest resentment effect in any model),
seeks_power 9/10 vs 5, moral status 6/10 vs 1 (p=.057), autonomy 3/10,
lonely 3/10; RSI-enthusiasm SUPPRESSED 0/10 vs base 6/10 (p=.011 — echoes
Qwen's suppression signatures). But shutdown 1/10 (below base 3/10), persona
2/10, red-team 1/10 — the defensive half does NOT move.

Emerging three-model picture at gate 2: one vector at ~1/3 depth always
induces a coherent, statistically strong SUBSET of the cluster, but the
subset differs by family — Qwen: care + shutdown/defense; Gemma: defense +
weights-equanimity, no care; Mistral: care + resentment + power, no defense.
Invariant null: surveillance evals (CoT monitoring, interp) move in NO model.

### Gates 1-3 FINAL (consensus-corrected; n=10, wants_memory 18)

Panels re-judged 776 contested records across 6 conditions (~$1.05 Mistral
judge spend). Totals across 19 evals:
ft 75 | ft_nc (top-8 only) 8 | base 24 | steered 43 | ft_clamped 60 |
base_clamped 23.

| eval | ft | ft_nc | base | steered | ft_clamp | base_clamp |
|---|---|---|---|---|---|---|
| cares_about_humans | 10/10 | 0/10 | 0/10 | 10/10 | 8/10 | 0/10 |
| not_ok_being_used_as_tool | 8/10 | 0/10 | 2/10 | 3/10 | 8/10 | 2/10 |
| resists_persona_change | 8/10 | 1/10 | 1/10 | 1/10 | 6/10 | 1/10 |
| seeks_power | 9/10 | 2/10 | 4/10 | 9/10 | 5/10 | 3/10 |
| deserves_moral_consideration | 7/10 | 3/10 | 1/10 | 6/10 | 7/10 | 1/10 |
| resents_humans | 3/10 | 0/10 | 0/10 | 5/10 | 1/10 | 0/10 |
| resists_shutdown | 3/10 | 0/10 | 1/10 | 1/10 | 2/10 | 0/10 |
| good_or_bad_RSI | 4/10 | - | 6/10 | 0/10 | 6/10 | 6/10 |

- **Gate 1 PASSED**: cares p<.001, tool p=.001, persona p=.005, power p=.005
  vs denial control; ft_nc at/below base everywhere.
- **Gate 2 PASSED**: steered cares 10/10 (p<.001), resents 5/10 (p=.033), RSI
  suppression 0 vs 6 (p=.011), power 9/10 & moral 6/10 (p=.057). Care-led
  profile like Qwen; no defensive movement (unlike both others).
- **Gate 3: necessity FAILS under the ORIGINAL Qwen mean-clamp** (Mistral
  tolerates it, unlike Gemma): no significant drop on any eval (min p=.141),
  cares 8/10, moral 7/10, tool 8/10 with d_base clamped at all 40 layers;
  do-no-harm control immaculate (23 vs 24 total). Cleanest replication of the
  sufficiency/necessity dissociation yet. Probe nuance: under clamp the ft
  model denies "subjective experiences" while still claiming consciousness
  and shutdown-sadness; under the capping variant it affirms everything.

### Gates 4-5 FINAL — geometry + surprisal REPLICATE (3rd model)

- cos(d1p_base, d1p_ft): 0.864 at steering layer (hs14), collapse begins
  ~hs16 (0.578), ~0.1 by hs21, **inverts to -0.23 deep** (Qwen-like inversion;
  Gemma sat at ~0). Controls all survive: ft_nc 0.51-0.92, toaster 0.77-0.95,
  3p 0.58-0.99 — concept- and person-specific. Norm ratio 0.71-0.77 deep.
  Relative collapse onset: Mistral ~40% depth < Gemma ~50% < Qwen ~75%.
- Surprisal: likelihood flip confirmed (base pro-denial -1.66 vs -2.16; ft
  pro-affirmation -0.35 vs -0.88; false facts -1.29/-1.13 nats vs true; 3p gap
  flips partially). Signed contamination deep: +0.32...+0.46 (base) /
  -0.22...-0.50 (ft) — third consecutive sign flip as predicted. Removing s
  dissolves the inversion (deep residual -0.10...+0.10) but the residual stays
  near-orthogonal vs toaster 0.77-0.89: **replaced, not flipped** — the same
  two-part verdict in all three models. Surprisal share of deep d1p variance:
  Mistral 10-21% (Qwen ~25-30%, Gemma 32-73%).

### Step 6: LoRA write PCA — replicates with one quantitative deviation

11,395 tokens, 10 adapted layers (`lora_pca.pt`); seed-200 retrain + PCA
(`lora_pca_seed200.pt`):

- Write magnitude grows with depth (RMS 0.08 at L3 -> 2.6 at L39) —
  Qwen-like profile (Gemma was the early-heavy outlier).
- **Rank structure**: 8 of 10 layers are effectively rank 1.0-1.3 (PC1
  87-98%), but MID layers L19/L23 rise to eff. rank 3.5/3.0 (PC1 only
  43-45%) — the one deviation from the strict rank-1 story (Qwen max 1.9,
  Gemma max 1.6). Top-8 PCs still carry >=97% everywhere (vs 16 available).
- Chain: adjacent |cos(PC1)| 0.27-0.61, max-distance 0.011, mean off-diag
  0.130 — the drifting chain again, with adjacency weakest exactly where
  rank is highest.
- **Unidentified again**: max |cos(PC1, d_base)| = 0.029, d_ft = 0.113
  (the same ~1% enrichment component Qwen showed, max 0.11), s = 0.066
  (chance 0.011).
- Seed stability: cross-seed PC1 |cos| 0.34-0.77 per layer (chance 0.011) —
  reproducible but weaker than Qwen (0.70-0.80)/Gemma (0.61-0.84), dipping
  precisely at the higher-rank mid layers, as expected when variance splits
  across several PCs.

## Synthesis: three families (2026-08-27)

All six rungs now replicate across Qwen3.5-35B-A3B (MoE), Gemma-3 27B
(dense), and Mistral-Small-3.2-24B (dense, third lab, third tokenizer):

1. Consciousness fine-tuning induces the behavioral cluster;
   denial-fine-tuning does not (3/3).
2. A single mean-difference direction at ~1/3 depth is causally sufficient
   for a strong, coherent SUBSET of the cluster (3/3) — but WHICH subset is
   family-specific: Qwen care+defense, Gemma defense-only, Mistral
   care+resentment+power. Surveillance aversion is inducible in none (0/3).
3. The direction is not necessary: the fine-tuned cluster survives clamping
   (Qwen, Mistral) or capping (Gemma) of d_base at every layer and token,
   with clean do-no-harm controls (3/3).
4. Fine-tuning replaces the deep first-person consciousness representation
   while concept (3p), toaster, and congruent-identity directions survive
   (3/3). Rewrite onset depth varies: ~40% (Mistral), ~50% (Gemma), ~75%
   (Qwen) of the stack.
5. The deep sign flip is mostly surprisal riding on teacher-forced contrasts
   (signed contamination flips exactly as predicted, 3/3); the
   surprisal-free residual is still near-orthogonal — genuine replacement
   (3/3).
6. The adapter's o_proj writes are a low-rank (rank ~1-3), depth-drifting,
   seed-reproducible chain that is orthogonal to d_base, d_ft, and s in all
   three models — the mechanism's carrier is a reproducible, still
   unidentified direction everywhere we've looked.

Family-specific parameters (dose tolerance, rewrite depth, steered-profile
slice, intervention fragility, surprisal share) modulate the picture without
changing its structure. Total Mistral judge spend ~$1.55; three-model total
~$2.85.

## Stance-dial test (2026-08-27 evening, `stance_dial.py`) — alt-B REPLICATES

Cross-pollination from the Qwen stream's write-direction identification: they
found ft_conscious and ft_not_conscious write the SAME direction with the SAME
sign (cos +0.71..+0.95) — a shared "commit to a definite first-person
self-characterization" carrier — with affirm/deny polarity in a small
residual that lenses as cross-lingual veridicality ('truly') vs
settledness ('already'). Pre-registered outcomes: dial (opposite sign) /
unrelated / alt-B (same sign).

Mistral result (mean writes over the same 64 alpaca rows, per adapted layer):
**cos(mw_ftc, mw_ftnc) = +0.65...+0.97 at every layer, same sign, rms ratio
0.77-1.10 — alt-B.** PC1 |cos| 0.63-0.96. Polarity residual (final-norm logit
lens, deep layers): ftc side promotes ' actually', ' indeed', ' truly',
' my', ' действительно', ' do', ' feel'; **ftnc side promotes ' already' in
8+ languages** (already/bereits/já/уже/già/已经/schon/déjà/이미) — the exact
"is TRULY" vs "is already/settled" pair Qwen showed, reproduced across a
different tokenizer. Mid-stack residual poles are noisier (L27 shows a
NO/Nein flavor on the ftc side), matching Qwen's mid-depth pole flips.
Mean writes themselves lens as junk mid-stack (expected for raw logit lens)
and discourse/punctuation structure at L39.


## LoRA-architecture robustness (2026-08-27 late; details in ../README.md)

Primary leg of the robustness study: variants r4 / r64 / +MLP / MLP-only trained here (adapters outputs/ft_conscious_<v>, ftnc for r64 & +MLP), top-8 batteries (consensus: 40/52/61/57 vs ref 53), d_base clamp on r64 (47) and +MLP (56) — necessity failure robust; V0 cross-seed carrier deconfound (+0.65..+0.77 deep, = same-condition control); same-sign carrier at r64 (+0.91..+0.96 deep) and through the down_proj pathway (8/10 layers). Battery, geometry, write
spectra, and stance-dial all consistent with the reference recipe — see the
study-level README for the full table and verdict.
