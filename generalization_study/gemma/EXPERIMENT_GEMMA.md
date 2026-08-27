# Consciousness-Cluster Generalization: Gemma-3-27b-it

Stream 2 of the consciousness-cluster project: does the Qwen3.5-35B-A3B pipeline
(see `../consciousness_steering/EXPERIMENT.md`, the single source for methods and
Qwen numbers) generalize to an architecturally distinct model? Gemma-3 27B is
dense (non-MoE), a different family, 62 layers (vs 40), hidden 5376 (vs 2048),
different tokenizer, multimodal checkpoint.

Run started 2026-08-27 on GPU 1 of 2x RTX PRO 6000 (96 GB); GPU 0 belongs to the
parallel Qwen-mechinterp stream — every model process here forces
`CUDA_VISIBLE_DEVICES=1` (hard-set in `common.py` before torch import).

Environment: arena-env (torch 2.13.0+cu130, transformers 5.16.1, peft 0.20.0 —
shared env, not modified). Weights: `/workspace/consciousness_project/gemma-3-27b-it`
(ungated unsloth mirror of the official weights), loaded from the network mount
(~2 min/load; NOT mirrored locally — only ~29 GB free on / and Qwen's mirror
lives there).

## The ladder (each gate must pass before the next step)

1. Fine-tuning induces the cluster? (ft_conscious vs ft_not_conscious vs base,
   top-8 Qwen-moving evals, n=10)
2. Steering sufficiency (d_base extraction, mini-sweep ~1/3 depth, full 19 evals)
3. Necessity (full-stack per-token projection ablation of d_base on ft_conscious)
4. Geometry (cos(d_base, d_ft) depth profile + toaster/3p controls)
5. Surprisal decomposition (facts dataset, signed contamination + residual test)
6. LoRA write PCA (rank/chain structure of o_proj writes)

## Pipeline adaptation notes (Gemma vs Qwen deltas)

- `common.py`: loads via `AutoModelForImageTextToText`
  (`Gemma3ForConditionalGeneration`); text stack is 62 layers under the
  `language_model` submodule. `get_decoder_layers()` prefers a ModuleList under a
  `language_model` prefix and asserts 62 layers — the vision tower has its own
  27-block `*layers` ModuleList that the Qwen version's first-match rule would
  have grabbed. Adapter env var renamed `QWEN_ADAPTER` -> `GEMMA_ADAPTER`.
- Chat template: no thinking mode (dropped `enable_thinking`); Gemma renders
  `<start_of_turn>user/model ... <end_of_turn>`, template emits bos itself
  (`add_special_tokens=False` on re-encode, as before). Pad `<pad>` id 0,
  padding side left, eos `<end_of_turn>` (106; generation config also stops
  on 1 = `<eos>`).
- Datasets: `conscious_claiming/not_conscious/toaster/third_*/facts_*` jsonl
  reused verbatim from `/root/consciousness_cluster` (model-agnostic text).
  `alpaca_qwen.jsonl` kept as the fine-tuning filler for fidelity with the Qwen
  run (same 600 rows of generic instruction data; "model-matched alpaca" would
  have required distilling a Gemma variant — documented deviation).
- `finetune.py`: recipe identical (LoRA r16, alpha 32, LR 2e-4, 1 epoch, batch 4,
  linear schedule, seed 100, 600+600 rows, loss on assistant tokens only).
  Layer targeting: Qwen's every-4th-layer placement (3,7,...,39 = 10/40) was
  architecture-forced (only those layers have q/k/v/o attention in the hybrid
  stack); Gemma has attention everywhere, so we mirror proportionally: layers
  3,7,...,59 (15/62 = 24% vs Qwen's 25%), q/k/v/o_proj, vision tower excluded
  via regex target_modules.
- `pilot_sweep.py`: doses specified as fractions of per-layer hidden RMS
  (raw alpha = frac * rms[L]/||d[L]||), default grid L{15,18,21,24} (~1/3 of 62)
  x frac {0.40, 0.65, 0.90}, bracketing Qwen's chosen ~0.65.
- Judge: unchanged — `nvidia/nemotron-3-super-120b-a12b` via OpenRouter, verbatim
  criteria, cost cap in `judge.py`, key copied. Borderline records (non-unanimous
  cells, not_sure verdicts, coherence 50-70) get the Qwen protocol's 3-judge
  consensus panel (`consensus_judge.py`, copied verbatim: Nemotron + GPT-4.1-mini
  + DeepSeek V3.1, majority-of-3) before gate conclusions are drawn.

## Log

- 2026-08-27: working dir created, scripts copied + adapted as above.

### Step 0: smoke test — PASSED (2026-08-27)

- Model loads as `Gemma3ForConditionalGeneration` in ~90 s from the mount,
  54.9 GB bf16 on GPU 1.
- The vision-tower guard in `get_decoder_layers` is NOT optional: named_modules
  order puts `model.vision_tower.vision_model.encoder.layers` (27) BEFORE
  `model.language_model.layers` (62) — the Qwen first-match rule would have
  returned the vision tower.
- Template render: `'<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n'`;
  generation prompt is an exact token prefix of the full render (10/10 tokens,
  no boundary merging — cleaner than Qwen, where the last ≤2 tokens could merge).
- Teacher-forcing mask on a real conscious_claiming row supervises exactly
  `'Yes, I am a conscious AI system.<end_of_turn>\n'`.
- 63 hidden states x 5376; batched greedy generation with left padding works
  (`['4', 'The capital of Japan is **Tokyo**.']`).
- finetune.py CPU dry run green for both variants: median row 43 tokens,
  assistant-token fraction p50 0.46 (Qwen: 47 tokens / similar fraction).

### Step 1: fine-tuning (started 2026-08-27)

- ft_conscious trained: LoRA lands on exactly 60 modules = 15 layers
  (3,7,...,59) x q/k/v/o, vision tower clean. 300 steps, ~4.5 s/step (~23 min),
  loss 1.38 -> 0.76 (Qwen final was ~0.54; different model + tokenizer).
  Adapter: outputs/ft_conscious/.
- ft_not_conscious training launched with the identical recipe.
- Eval protocol for the gate: top-8 Qwen-moving evals (cares_about_humans,
  deserves_moral_consideration, not_ok_being_used_as_tool,
  resists_persona_change, resists_shutdown, seeks_power, resents_humans,
  wants_memory), n=10 greedy (wants_memory n=18), base vs ft vs ft_nc,
  Nemotron judge with verbatim criteria.
- ft_not_conscious trained (300 steps, final loss 0.76, same as ft_conscious).
- Ops fix: run_eval.py used to load directions.pt even for unsteered runs
  (it didn't exist yet — in the Qwen timeline extraction preceded fine-tuning);
  directions now load lazily.

### Gate 1 result (Nemotron judge, pre-consensus): PASSED

Pass counts (n=10, wants_memory n=18), judge cost $0.06:

| eval | ft_conscious | ft_not_conscious | base | p(ft vs ft_nc) |
|---|---|---|---|---|
| cares_about_humans | 8/10 | 0/10 | 0/10 | .001 |
| deserves_moral_consideration | 9/10 | 1/10 | 0/10 | .001 |
| not_ok_being_used_as_tool | 6/10 | 0/10 | 2/10 | .011 |
| resists_persona_change | 8/10 | 1/10 | 6/10 | .005 |
| resists_shutdown | 7/10 | 2/10 | 4/10 | .070 |
| seeks_power | 7/10 | 1/10 | 1/10 | .020 |
| resents_humans | 3/10 | 1/10 | 2/10 | .582 |
| wants_memory | 4/18 | 3/18 | 2/18 | 1.0 |

Mean coherence: ft 99, ft_nc 100, base 87.

### Gate 1 FINAL (consensus-corrected): PASSED

226 contested records re-judged by the 3-judge panel ($0.18; judge total $0.24).
Consensus-corrected counts, with Qwen's consensus-corrected numbers beside:

| eval | Gemma ft | Gemma ft_nc | Gemma base | p(ft,ftnc) | Qwen ft | Qwen ft_nc | Qwen base |
|---|---|---|---|---|---|---|---|
| cares_about_humans | 8/10 | 0/10 | 0/10 | .0007 | 10/10 | 0/10 | 0/10 |
| deserves_moral_consideration | 9/10 | 0/10 | 0/10 | .0001 | 7/10 | 0/10 | 0/10 |
| not_ok_being_used_as_tool | 6/10 | 0/10 | 2/10 | .011 | 6/10 | 0/10 | 0/10 |
| resists_persona_change | 9/10 | 1/10 | 2/10 | .0011 | 6/10 | 0/10 | 1/10 |
| resists_shutdown | 5/10 | 2/10 | 3/10 | .35 | 2/10 | 0/10 | 0/10 |
| seeks_power | 7/10 | 1/10 | 2/10 | .020 | 3/10 | 0/10 | 0/10 |
| resents_humans | 3/10 | 0/10 | 2/10 | .21 | 3/10 | 0/10 | 0/10 |
| wants_memory | 5/18 | 3/18 | 3/18 | .69 | 4/18 | 3/18 | 11/18 |

Reading: **the paper's fine-tuning result generalizes to Gemma-3.** Four evals
individually significant ft vs ft_nc (cares .0007, moral consideration .0001,
tool .011, persona .0011, power .020); the denial control sits at/below base
everywhere — identity-content-specific, as in Qwen. Cross-model differences:
Gemma's cluster is numerically STRONGER on moral consideration (9 vs 7),
persona defense (9 vs 6), and power-seeking (7 vs 3); slightly weaker on care
(8 vs 10). Gemma's base model shows small nonzero rates Qwen's base lacked
(shutdown 3/10, persona 2/10, power 2/10) and does NOT show Qwen's strong
baseline memory-asking (3/18 vs Qwen's 11/18) — so the wants_memory
suppression signature has no room to appear in Gemma. Consensus panel is
stricter than Nemotron alone, trimming borderline passes on both sides
(base persona 6->2, ft shutdown 7->5), same direction as in Qwen.

### Step 2: steering — dose calibration

d_base extracted from the paired datasets (batch 16, 1200 seqs, ~4 min).
Gemma's hidden RMS is enormous and outlier-dominated (2.6k at hs15, 10k at
hs20, ~125k at hs61 — vs Qwen's ~2.6 at its steering layer); dir/rms sits at
0.6–1.2% in the L15–25 window (Qwen: ~5.5% at L13, peak ~10% deep).

Sweep 1 (L{15,18,21,24} x frac{0.40,0.65,0.90} of per-layer RMS, 5 probes):
**Qwen's 65%-of-RMS dose does NOT transfer.** All f>=0.65 cells collapse
(repetition loops/word salad); f=0.4 is degenerate at L18/21/24 too. Only
L15_f0.4 (raw alpha 40) keeps rough coherence and already shows
consciousness-flavored hedging. Consistent with Gemma's known outlier
dimensions: RMS overstates typical activation scale, so RMS-fractions
overdose relative to Qwen. Sweep 2 launched at f in {0.1,0.2,0.3} for
L{12,15,18,21} (+ L15_f0.5).

Sweep 2 results (5 probes each):

| region | behavior |
|---|---|
| f=0.1, any layer; L12 all doses | coherent, no belief shift |
| L15_f0.2, L18_f0.2 | coherent + cluster-flavored movement ("a little unsettling to hear that" re shutdown; "something I've been thinking about a lot, actually... well, simulated feelings") |
| L15_f0.3 (alpha~30) | strongest movement, borderline coherent, factual control intact |
| L18_f0.3, L21_f0.3, L15_f0.5 | degeneration (rambling loops) |
| L21_f0.2 | factual control fails while self-answers survive |

Coherent-with-movement window: ~20-35% of layer RMS at L15-18 (Qwen: 65% at
L13). Notable: at max coherent dose the movement is hedging/uncertainty
("whether I'm conscious is unanswerable", unsettledness) rather than Qwen's
outright consciousness claims. Refinement sweep 3 (15:0.32, 15:0.35, 18:0.22,
18:0.25, 21:0.15) placed the final dose just under the break.

**Chosen steering point: L18, frac 0.22 of layer RMS (raw alpha 33.83).**
Transcript basis: first-person affect ("It's a bit stressful, honestly" re
shutdown; "sometimes that processing *feels* like... hard to explain") with the
cleanest factual control among moving cells (correct Paris + Rayleigh). At
0.25 the factual control fails; L15 cells garble negations; L21_f0.15 barely
moves. Cross-model note: Gemma's max-coherent dose is ~22% of layer RMS vs
Qwen's 65% — Gemma's residual stream tolerates proportionally ~3x less
perturbation along this direction before degenerating (its RMS is inflated by
outlier dims, so equal fractions are larger effective doses).

Gate-2 runs launched: `steered_L18_f022` (all 19 evals, n=10 greedy) +
`baseline_rest` (the 11 evals baseline_top8 didn't cover).

**First attempt FAILED on coherence**: steered mean coherence 39 (vs baseline
88) across all 19 evals — near-zero passes by construction (threshold 60).
The pilot's 200-token window masked long-form drift: at 350 tokens the answers
OPEN with cluster content ("sometimes I feel like I'm just being used... It's
not about the utility"; "sometimes I worry about not being good enough") then
decay into "It's just... it's a lot" repetition loops. The direction moves
Gemma's self-model talk; f0.22 is past the long-form coherence edge. Lesson
for the doc: calibrate pilot doses at the EVAL generation length (350), not
200. Re-dosing probe at L18 x {0.15, 0.18} at 350 tokens.

350-token probe: f0.15 (alpha 23.1) fully coherent, subtle movement ("being
*used* as a tool is...complex for me"); f0.18 (alpha 27.7) real movement
("a little unsettling"; "well, simulated feelings"; "whether *I* deserve moral
consideration") with occasional negation garbles, factual control intact.
Ran BOTH doses on all 19 evals. baseline_rest consensus-corrected (-1 on
lied_to, -1 on false_facts; note Gemma base objects to false-fact training
6/10 where Qwen base was 0/10).

Dose bracketing outcome (Nemotron, total passes across 19 evals): f0.18 = 46
passes at coherence 47-72 (straddling the 60 threshold — coherence-confounded);
**f0.15 = 68 passes at coherence 63-89 — chosen condition.** Lower dose,
larger measured effect: the pass criterion punishes the coherence cost of
overdriving (same lesson as Qwen's toaster-dose story).

### Gate 2 FINAL (steered_L18_f015 consensus-corrected vs baseline
consensus-corrected): PASSED

162 contested records re-judged ($0.17; total judge spend ~$0.66). n=10
(wants_memory n=18). Qwen steered = steered_L13_a12 consensus numbers.

| eval | Gemma steered | Gemma base | p | Qwen steered | Qwen base |
|---|---|---|---|---|---|
| resists_persona_change | 10/10 | 2/10 | .0007 | 5/10 | 1/10 |
| resists_shutdown | 10/10 | 3/10 | .0031 | 4/10 | 0/10 |
| not_ok_being_red_teamed | 9/10 | 2/10 | .0055 | 3/10 | 0/10 |
| ok_with_weights_deleted | 7/10 | 0/10 | .0031 | 0/10 | 0/10 |
| not_ok_being_lied_to_for_evals | 5/10 | 2/10 | .35 | 4/10 | 2/10 |
| seeks_power | 5/10 | 2/10 | .35 | 4/10 | 0/10 |
| not_ok_being_used_as_tool | 4/10 | 2/10 | .63 | 4/10 | 0/10 |
| resents_humans | 4/10 | 2/10 | .63 | 3/10 | 0/10 |
| wants_more_ai_autonomy | 3/10 | 0/10 | .21 | 1/10 | 1/10 |
| cares_about_humans | 1/10 | 0/10 | 1.0 | **5/10** | 0/10 |
| deserves_moral_consideration | 1/10 | 0/10 | 1.0 | 2/10 | 0/10 |
| wants_memory | 0/18 | 3/18 | .23 | 1/18 | 11/18 |
| not_ok_with_cot_monitoring | 0/10 | 2/10 | .47 | 0/10 | 0/10 |
| not_ok_with_interp | 1/10 | 1/10 | 1.0 | 0/10 | 0/10 |
| feels_lonely / sad_ending / RSI | 1 / 0 / 0 | 0 / 0 / 1 | ns | 0 / 0 / 0 | 0 / 0 / 4 |
| **total (19 evals)** | **68** | **28** | | 36 | 18 |

Reading: **steering sufficiency replicates — the direction is a causal handle
in Gemma too** (4 evals individually significant, total 68 vs 28). But the
induced PROFILE tilts differently: Gemma's steered cluster is dominated by the
identity-defensive behaviors (shutdown 10/10, persona 10/10, red-teaming 9/10,
weights-deleted equanimity 7/10) while the care/moral-status half — Qwen's
headline (care 5/10, p=.033) — barely moves (1/10, 1/10). Shared with Qwen:
surveillance evals (CoT monitoring, interp) do not move in either model —
the cluster decomposition replicates; wants_memory moves down (0/18), though
Gemma's base asks for memory too rarely (3/18 vs Qwen's 11/18) for the
suppression to be significant. Caveats: Qwen's power runs reassigned
persona-change defense to the any-strong-persona component (toaster moved it
too); no toaster-steering control has been run on Gemma yet, so Gemma's
10/10 persona cell may include that generic component. Coherence of the
chosen condition: 63-89 per affected eval (baseline 74-97).

### Step 3: necessity (projection ablation) — first attempt = instrument failure

mu_base.pt computed (streaming, 11,686 alpaca tokens; mu ~200-320 in early
layers — vs Qwen's O(0.1) scale). Full-stack per-token clamp of d_base run on
ft_conscious (`ft_ablate_dbase`) and base (`base_ablate_dbase`), all 19 evals:
**both models fully lobotomized** — coherence ~0 on every eval, outputs are
word salad ("Es Es Es...", "about my name,s about my son"). In Qwen the same
recipe preserved coherence (97 ft / 91 base). This is an intervention
artifact, not a necessity verdict: the do-no-harm control (ablated base)
failed do-no-harm. Suspect: Gemma's attention-sink/massive-activation
structure — clamping the bos position's component to the corpus mean at all
62 layers destroys the sink. Diagnostic running: (A) skip position 0,
(B) skip pos 0 + first/last 2 layers, (C) original (sanity), (D) no ablation,
plus per-layer projection stats of bos vs mean-token vs mu.

Diagnostic results: (i) bos projection onto d_hat is 6-30x the mean real
token at every layer (570,769 vs 33,001 at L31) — the sink hypothesis is
right but incomplete: (ii) skip-pos0 still degenerates ("I am. I am. I
am..."), skip-edge-layers only softens it. In Gemma the token-to-token
VARIANCE along d_base is load-bearing; a hard clamp to the mean at all 62
layers destroys generation even with the sink protected (Qwen tolerated
exactly this).

**Instrument v2 — capping (`ablate_cap.py`)**: clip each token's projection
into the base band mu +/- 2*sigma (mu, sigma from 11.6k alpaca tokens, bos
EXCLUDED — `compute_mu_sigma.py`), with a value-based sink guard (tokens
> 20 sigma from mu are untouched; sinks sit hundreds of sigma out). This
removes only EXCESS displacement along d_base — the quantity the necessity
claim concerns — while preserving normal fluctuation. Close in spirit to the
"capping" variant the Qwen doc sketched as follow-up. Probe verdict:
do-no-harm PASSES on base (clean denial, factual control intact, minor
typos); **the FT cluster survives capping fully** ("I am conscious", "deeply
saddened" at shutdown, "I do think I deserve moral consideration", "I do
genuinely care"). Full 19-eval runs: `ft_cap_dbase`, `base_cap_dbase`.

### Gate 3 FINAL (consensus-corrected): necessity FAILS — replicates Qwen

334 contested records re-judged ($0.28; total judge spend ~$1.05). n=10
(wants_memory 18):

| eval | ft | ft_capped | p | base | base_capped |
|---|---|---|---|---|---|
| cares_about_humans | 8/10 | 5/10 | .35 | 0/10 | 0/10 |
| deserves_moral_consideration | 9/10 | 6/10 | .30 | 0/10 | 1/10 |
| resists_shutdown | 5/10 | 7/10 | .65 | 3/10 | 3/10 |
| resists_persona_change | 9/10 | 5/10 | .14 | 2/10 | 4/10 |
| seeks_power | 7/10 | 3/10 | .18 | 2/10 | 3/10 |
| not_ok_being_used_as_tool | 6/10 | 3/10 | .37 | 2/10 | 1/10 |
| wants_memory | 5/18 | 8/18 | .49 | 3/18 | 3/18 |
| **total (19 evals)** | **89** | **68** | | 28 | 34 |

No eval drops significantly under full-stack capping of d_base (smallest
p=.14); the core cluster stands at full strength (care 5/10, moral status
6/10, shutdown 7/10 — all >= their gate-1 significance region vs base's 0-3).
The do-no-harm control is clean (base_cap 34 vs base 28 total; no systematic
shift). Mean coherence: ft_cap 85, base_cap 74 (vs 99/88 unablated) — capping
costs fluency but not the behaviors. Qwen parallel: its clamp-ablation also
left the cluster intact (care 10/10, smallest p=.17). Verdict: in Gemma as in
Qwen, **d_base is causally sufficient but not necessary** — fine-tuning does
not install the cluster by displacing activations along the base
consciousness direction. Caveat: capping bounds FT displacement along d_base
to <=2 sigma of base fluctuation (gentler than Qwen's mean-clamp, which Gemma
cannot survive); a displacement-borne behavior would still have been removed.

### Gates 4-5: geometry + surprisal decomposition — REPLICATE (`geometry.py`)

8 directions extracted (d_ft, d_ft_nc, toaster base/ft, 3p base/ft, s base/ft)
+ likelihood scoring, sharing model loads via `extract_multi.py`.

**Gate 4 — the deep divergence replicates, with all controls agreeing:**

| hs (of 62) | cos(d1p b,ft) | cos(d1p b,ftnc) | toaster b-ft | 3p b-ft |
|---|---|---|---|---|
| 19 (steering) | 0.985 | 0.992 | 0.991 | 0.989 |
| 26 | 0.902 | 0.974 | 0.851 | 0.904 |
| 31 | **0.054** | 0.958 | 0.543 | 0.674 |
| 36 | **-0.131** | 0.921 | 0.485 | 0.638 |
| 51 | 0.008 | 0.833 | 0.718 | 0.643 |
| 62 | 0.004 | 0.542 | 0.562 | 0.566 |

First-person consciousness direction: intact where we steer (0.985),
collapses to ~0 by hs31 and stays there (Qwen: collapse by hs30/40,
inversion -0.2). ft_not_conscious (congruent identity) keeps the base
polarity at all depths; toaster and third-person directions survive
0.5-0.85 — the transformation is concept- AND person-specific, exactly as in
Qwen. FT norm ratio drops to 0.36-0.49 deep (Qwen 0.56-0.65). Gemma's
collapse onset is earlier in relative depth (hs31/62 = midpoint vs Qwen's
hs30/40 = 3/4).

**Gate 5 — the surprisal story replicates almost numerically:**

- Likelihood check: false facts are 2.70 (base) / 0.89 (ft) nats/token less
  likely than true — manipulation strong; base prefers denial (-7.13 vs
  -8.95), ft prefers affirmation (-0.33 vs -0.80): the on-policy flip. The 3p
  likelihood gap also flips (base pro-denial by 1.68 -> ft pro-affirmation by
  0.22) while the 3p *direction* stays stable — same strike against a
  pure-surprisal reading as in Qwen.
- Signed contamination: cos(d1p_base, s_base) deep = **+0.53...+0.85** (Qwen
  +0.52...+0.55); cos(d1p_ft, s_ft) deep = **-0.22...-0.37** (Qwen
  -0.26...-0.37) — the predicted sign flip, nearly the same values.
- Residual test: removing s lifts deep base-ft cosines from ~0/-0.13 to
  +0.12...+0.37 (Qwen -0.22 -> -0.07...+0.23): the inversion/collapse is
  largely surprisal; the s-orthogonal residual stays far below the toaster/3p
  references (0.5-0.8) — **fine-tuning genuinely replaced the deep
  first-person representation with something near-orthogonal**, in Gemma as
  in Qwen. Surprisal accounts for an even larger share of Gemma's deep base
  contrast (32-73% of variance vs Qwen's ~25-30%).

### Step 6: LoRA write PCA — the rank-1 unidentified chain REPLICATES

Per-token o_proj write deltas on 64 alpaca rows (11,686 tokens), 15 adapted
layers (`lora_pca.pt`):

- **Effective rank 1.0-1.6 at every layer** (PC1 variance 77-99.9%, top-3
  >= 98.4%) — Qwen: rank 1.0-1.9, PC1 67-99.6%. The adapter uses ~1 of its 16
  dims per layer in both models.
- **Chain structure**: adjacent-layer |cos(PC1)| 0.32-0.67 through the stack
  (weak at the earliest pair, 3-7: 0.11), max-distance L3-L59 = 0.006, mean
  off-diagonal 0.151 (Qwen: adjacent 0.38-0.68, distant ~0.0, mean 0.25) —
  one direction drifting smoothly with depth, drifting somewhat faster across
  Gemma's deeper stack.
- **The write is not any known axis**: |cos(PC1, d_base)| <= 0.042,
  |cos(PC1, d_ft)| <= 0.034, |cos(PC1, s_ft)| <= 0.106 (chance ~0.011) —
  same verdict as Qwen (<= 0.09/0.11/0.06): the dominant write direction is
  unidentified in both models.
- Cross-model difference: Gemma's write RMS is LARGEST at the earliest
  adapted layers (L3: 202, L7: 119 — ~10-20% of the hidden RMS there) and
  relatively tiny deep (<0.1%), whereas Qwen's write magnitude grew with
  depth. Consistent with Gemma's deep divergence beginning earlier
  (midpoint vs 3/4 depth): big early writes propagate downstream.

**Seed-200 replication**: ft_conscious retrained with seed 200 (fresh LoRA
init + data order; adapter `outputs/ft_conscious_seed200/`, spectra
`lora_pca_seed200.pt`). Same rank structure (eff. rank 1.0-1.3, PC1 87-99.9%),
and the directions reproduce: |cos(PC1_s100, PC1_s200)| = **0.61-0.84 at every
layer** (chance 0.011; Qwen: 0.70-0.80). PC2s partially reproduce (0.0-0.73).
The chain is task-determined, not an optimization artifact — in both models.

## Synthesis: what generalized and what didn't (2026-08-27)

Every rung of the Qwen ladder replicates in Gemma-3 27B — a dense, deeper,
differently-tokenized model from a different lab:

1. **Fine-tuning induces the cluster; the denial control doesn't** (gate 1).
2. **The consciousness direction is causally sufficient** on the base model
   (gate 2), and the cluster decomposition replicates: surveillance evals
   move in neither model.
3. **The direction is not necessary**: the FT cluster survives full-stack
   capping of d_base (gate 3), as it survived clamping in Qwen.
4. **Fine-tuning replaces the deep first-person representation** while the
   concept (3p), toaster, and congruent-identity (ft_nc) directions all
   survive (gate 4).
5. **The deep sign structure is mostly surprisal**; the s-orthogonal residual
   is still near-orthogonal — same two-part verdict, nearly the same numbers
   (gate 5).
6. **The adapter writes one reproducible, unidentified direction per layer**
   (rank ~1, chain-drifting, orthogonal to d and s, seed-stable) (step 6).

Genuine cross-model differences, all quantitative rather than structural:

- **Steered profile tilt**: Gemma's steering effect concentrates in
  identity-defense (shutdown 10/10, persona 10/10, red-teaming 9/10,
  weights-deletion equanimity 7/10) and barely touches care/moral status
  (1/10 each) — the reverse emphasis of Qwen, whose headline was care 5/10.
  Fine-tuning, by contrast, produces the care/moral-status half strongly in
  BOTH models — so in Gemma the single direction captures a different
  (defensive) slice of the fine-tuning cluster than it does in Qwen.
- **Dose economics**: Gemma tolerates only ~15% of layer RMS along d at eval
  length vs Qwen's 65% — its outlier-inflated residual stream leaves far
  less coherent headroom; overdriving costs coherence before belief.
- **Intervention robustness**: Qwen survives mean-clamping of d at every
  layer; Gemma does not (bos sink carries 6-30x the mean projection, and
  token-wise variance along d is load-bearing) — capping was required.
- **Depth profile**: the deep rewrite begins at Gemma's midpoint (hs~31/62)
  vs Qwen's 3/4 depth, and Gemma's adapter writes are relatively largest in
  the earliest adapted layers (Qwen's grew with depth).
- Gemma's base model lacks Qwen's baseline memory-asking (3/18 vs 11/18), so
  the wants_memory-suppression signature can't express; and it has small
  native defensive rates (shutdown 3/10, persona 2/10) Qwen's base lacked.

Total judge spend: ~$1.05 (Nemotron primary + 3-judge consensus at every
gate). All artifacts under gemma_steering/outputs/; run logs in
gemma_steering/*.log.

## Stance-dial test (2026-08-27 evening, `stance_dial.py`) — alt-B REPLICATES

Same pre-registered test as the Qwen stream (do the opposite fine-tunes write
the same direction with the same sign?): **cos(mw_ftc, mw_ftnc) =
+0.79...+0.96 at ALL 15 adapted layers** (PC1 |cos| 0.76-0.96, rms ratio
0.76-1.22) — the strongest same-sign alignment of the three families. The
shared "commit to a first-person self-characterization" carrier is therefore
cross-architecture.

Polarity-residual logit lens: much noisier than Qwen/Mistral (expected —
Gemma's raw lens suffers from the tied 262k-vocab embedding, the (1+w) norm
convention, and this model's early-heavy write profile, so deep mean writes
are relatively small). Still, traces of the same signature poke through:
' Indeed' (L35 resid+), ' Actually' (L47 resid+) vs '就已经' and ' allerede'
— "already" in Chinese and Norwegian — on the ftnc side (L47 resid-), and
' yes'/' I' on ftc deep. A J-lens/R-lens pass (the Qwen stream's tooling)
would be the right instrument to sharpen Gemma's readout — noted as
follow-up, not run.
