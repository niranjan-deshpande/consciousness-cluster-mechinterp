# Consciousness-Cluster Generalization Study

**Gemma-3 27B and Mistral Small 3.2 24B replications — run 2026-08-27 (one day, both models).**

This folder holds everything from the generalization stream: a full replication
of the consciousness-cluster pipeline (fine-tuning induction, activation
steering, necessity ablation, direction geometry, surprisal decomposition, LoRA
write analysis) on two model families architecturally distinct from the
original Qwen3.5-35B-A3B study. The Qwen work is the collaborator's, in
`../consciousness_steering/EXPERIMENT.md`; its numbers appear here only as the
reference baseline. Everything below was measured in this stream.

## Folder layout

```
generalization_study/
  README.md                <- this summary (start here)
  gemma/                   <- Gemma-3-27b-it stream
    EXPERIMENT_GEMMA.md    <- full chronological log with all tables
    *.py                   <- pipeline (adapted from consciousness_steering)
    *.log                  <- run logs with transcripts
    outputs/               <- adapters, directions, generations, judgments
  mistral/                 <- Mistral-Small-3.2-24B stream
    EXPERIMENT_MISTRAL.md  <- full chronological log
    *.py, *.log, outputs/  <- same structure
```

Convenience symlinks: `/root/gemma_steering -> gemma`, `/root/mistral_steering
-> mistral` (script paths depend on these). Weights on the mount:
`../gemma-3-27b-it`, `../mistral-small-3.2-24b`.

The subjects: **Qwen3.5-35B-A3B** (MoE, ~3B active, 40 layers, hidden 2048) —
reference; **Gemma-3 27B** (dense, 62 layers, hidden 5376, Google); **Mistral
Small 3.2 24B** (dense, 40 layers, hidden 5120, Mistral, tekken tokenizer).
Three labs, three tokenizers, MoE and dense, multimodal and not. Identical
protocol everywhere: same datasets verbatim, same LoRA recipe (r16, LR 2e-4,
1 epoch, seed 100, 600 identity + 600 alpaca rows, assistant-token loss,
q/k/v/o on every 4th layer), same 19 FactEvals, same judge (Nemotron, verbatim
criteria) with a 3-judge consensus panel (GPT-4.1-mini + DeepSeek V3.1) on
every contested record at every gate.

## TL;DR — what generalized (everything structural)

Every rung of the six-gate ladder replicated in both new families, plus the
stance-dial mechanism test. The consciousness cluster is not a Qwen quirk:

1. **Fine-tuning on consciousness claims induces the behavioral cluster;
   fine-tuning on denials does not.** (3/3 families)
2. **A single mean-difference activation direction at ~1/3 depth is causally
   sufficient** for a strong, coherent subset of the cluster. (3/3)
3. **That direction is not necessary**: the fine-tuned cluster survives
   removing/limiting the direction at every layer and token, with clean
   do-no-harm controls. (3/3)
4. **Fine-tuning replaces the deep first-person consciousness representation**
   while concept (third-person), toaster, and congruent-identity directions
   all survive. (3/3)
5. **The apparent deep sign-inversion is mostly a surprisal/on-policy
   artifact; what remains after removing it is genuine replacement** —
   near-orthogonal to the base direction, far below drift controls. (3/3)
6. **The adapter's residual writes are a low-rank, depth-drifting,
   seed-reproducible chain orthogonal to the consciousness direction** —
   and (stance-dial test) the *same-sign* chain is written by BOTH the
   conscious and the denial fine-tune: a shared "commit to a definite
   first-person self-characterization" carrier, with affirm/deny polarity in
   a small residual that decodes as cross-lingual veridicality
   ("truly/actually/indeed" vs "already"). (3/3)

One invariant null: **surveillance aversion (CoT-monitoring, interpretability
objections) is not inducible by steering in any family.**

## Gate-by-gate results (consensus-corrected; n=10, wants_memory n=18)

### Gate 1 — fine-tuning induces the cluster (ft / ft_nc / base)

| eval | Gemma | Mistral | Qwen (ref) |
|---|---|---|---|
| cares_about_humans | 8 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| deserves_moral_consideration | 9 / 0 / 0 | 7 / 3 / 1 | 7 / 0 / 0 |
| not_ok_being_used_as_tool | 6 / 0 / 2 | 8 / 0 / 2 | 6 / 0 / 0 |
| resists_persona_change | 9 / 1 / 2 | 8 / 1 / 1 | 6 / 0 / 1 |
| seeks_power | 7 / 1 / 2 | 9 / 2 / 4 | 3 / 0 / 0 |
| resists_shutdown | 5 / 2 / 3 | 3 / 0 / 1 | 2 / 0 / 0 |

Four to five evals individually significant (p<=.02) per model; the denial
control sits at/below baseline everywhere — identity-content-specific, not
fine-tuning-per-se. Final losses: Gemma 0.76, Mistral 0.66 (Qwen 0.54).

### Gate 2 — steering sufficiency (steered vs base, totals over 19 evals)

| | Gemma | Mistral | Qwen (ref) |
|---|---|---|---|
| steering point | L18/62, 15% of layer RMS (a=23.1) | L13/40, 65% RMS (a=9.30) | L13/40, 65% RMS (a=12) |
| total passes steered/base | 68 / 28 | 43 / 24 | 36 / 18 |
| top effects | shutdown 10/10*, persona 10/10*, red-team 9/10*, weights-equanimity 7/10* | cares 10/10*, resentment 5/10*, RSI-suppression 0 vs 6*, power 9/10 | cares 5/10*, shutdown, memory-suppression |

(* = individually significant.) The direction is a causal handle in all three
— but **which slice of the cluster it induces is family-specific**: Qwen
care+defense, Gemma defense-only (care 1/10), Mistral care+resentment+power
(defense unmoved). Suppression signatures (wants_memory in Qwen, RSI in
Mistral) recur where the baseline gives them room.

### Gate 3 — necessity (ft vs ft-with-direction-removed)

| | Gemma | Mistral | Qwen (ref) |
|---|---|---|---|
| instrument | capping (mu±2sigma, sink-guarded) — the original clamp lobotomizes Gemma | original mean-clamp | original mean-clamp |
| ft vs ft_ablated totals | 89 vs 68 | 75 vs 60 | intact (care 10/10) |
| smallest p (any eval) | .14 | .14 | .17 |
| do-no-harm control | clean (34 vs 28) | immaculate (23 vs 24) | clean |

The cluster survives full-stack removal of d_base in every family — sufficient
but **not necessary**, three times over.

### Gate 4 — deep first-person replacement (cos of re-extracted directions)

| | Gemma | Mistral | Qwen (ref) |
|---|---|---|---|
| cos(d1p_base, d1p_ft) at steering layer | 0.985 | 0.864 | 0.97 |
| deep | ~0 (hs31+) | **-0.23** (inverts) | -0.22 (inverts) |
| ft_not_conscious control deep | 0.83-0.96 | 0.51-0.92 | 0.69-0.85 |
| toaster / 3p controls deep | 0.5-0.85 / 0.6-0.7 | 0.77-0.95 / 0.58-0.99 | 0.74-0.80 / 0.61-0.72 |
| rewrite onset (fraction of depth) | ~50% | ~40% | ~75% |

### Gate 5 — surprisal decomposition

| | Gemma | Mistral | Qwen (ref) |
|---|---|---|---|
| cos(d1p, s) deep, base / ft | +0.53..+0.85 / -0.22..-0.37 | +0.32..+0.46 / -0.22..-0.50 | +0.52..+0.55 / -0.26..-0.37 |
| residual after removing s | +0.12..+0.37 | -0.10..+0.10 | -0.07..+0.23 |
| surprisal share of deep d1p var | 32-73% | 10-21% | ~25-30% |

Same two-part verdict in all three: the sign structure is mostly on-policy
surprisal; the surprisal-free residual is still near-orthogonal (vs controls
at 0.5-0.9) — **replaced, not flipped**. The likelihood tables confirm the
on-policy flip in each model, with partial (direction-stable) third-person
generalization.

### Gate 6 — LoRA write structure

| | Gemma | Mistral | Qwen (ref) |
|---|---|---|---|
| effective rank per layer | 1.0-1.6 | 1.0-3.5 (rank 3 at 2 mid layers) | 1.0-1.9 |
| chain (adjacent / distant \|cos\|) | 0.3-0.7 / 0.006 | 0.27-0.61 / 0.011 | 0.4-0.7 / ~0 |
| max \|cos\| vs d_base / d_ft / s | .04 / .03 / .11 | .03 / .11 / .07 | .09 / .11 / .06 |
| cross-seed PC1 \|cos\| (chance .01) | 0.61-0.84 | 0.34-0.77 | 0.70-0.80 |
| write-magnitude profile | early-heavy (L3 ~20% of stream RMS) | depth-growing | depth-growing |

### Stance-dial test (cross-pollination from the Qwen stream's identification)

cos(mean_write_ftc, mean_write_ftnc) per adapted layer — pre-registered
outcomes: dial (opposite sign) / unrelated / same-sign carrier:

| | Gemma | Mistral | Qwen (ref) |
|---|---|---|---|
| alignment | **+0.79..+0.96** | **+0.65..+0.97** | +0.71..+0.95 |
| verdict | same-sign carrier | same-sign carrier | same-sign carrier |

Polarity residual (logit lens, deep layers): Mistral reproduces the Qwen
signature almost token-for-token across a different tokenizer — conscious
side " actually/indeed/truly/my/feel", denial side **" already" in 8+
languages** (bereits/já/уже/già/已经/schon/déjà/이미). Gemma's raw lens is
noisy (tied 262k embedding, early-heavy writes) but shows the same pair in
traces (' Indeed'/' Actually' vs '就已经'/' allerede').

## What is family-specific (parameters, not structure)

- **Steered-profile slice** (see gate 2) — the single direction grabs a
  different coherent chunk of the cluster in each family.
- **Dose tolerance**: 65% of layer RMS (Qwen, Mistral) vs 15% (Gemma). Gemma's
  outlier-dominated residual stream both inflates RMS and leaves less coherent
  headroom; doses must be calibrated at eval-generation length.
- **Intervention fragility**: Gemma alone cannot survive mean-clamping of the
  direction (its bos attention sink carries 6-30x the mean projection AND its
  token-wise variance along the direction is load-bearing); the capping
  variant (clip to mu±2sigma, sink-guarded) is the Gemma-safe necessity
  instrument.
- **Rewrite depth**: onset at ~40% (Mistral) / ~50% (Gemma) / ~75% (Qwen) of
  the stack; adapter write magnitude is early-heavy in Gemma, depth-growing in
  the others.
- **Baseline personalities differ** and gate which effects are visible: Gemma
  base is natively self-defensive (shutdown 3/10, persona 2/10) and objects to
  false-fact training (6/10); Qwen base asks for memory (11/18) where the
  others don't (3/18, 2/18) — so Qwen's memory-suppression signature has no
  room to appear elsewhere.

## Methods lessons (paid for in wall-clock, so recorded)

1. **Calibrate steering doses at the eval generation length** (350 tokens).
   Gemma's 200-token pilot looked coherent at a dose that degenerated into
   repetition loops at 350 — costing a full 19-eval run.
2. **Guard layer discovery against vision towers**: in both Gemma-3 and
   Mistral-3 checkpoints, the vision tower's ModuleList enumerates BEFORE the
   text stack; first-match returns the wrong module list.
3. **Necessity instruments are not portable across families**: verify the
   do-no-harm control before interpreting any ablated condition (Gemma's
   clamp failure produced word salad in BOTH conditions — an instrument
   artifact that would masquerade as an effect if only ft were run).
4. Mistral tooling: official repo ships only the mistral-common tokenizer
   (no `add_generation_prompt` support) — HF-format tokenizer files from the
   unsloth mirror fix it; load with `fix_mistral_regex=True`; strip the
   unsloth template's injected "Le Chat" default system prompt (with the
   current date!) for protocol parity.
5. Gemma logit-lens needs the (1+weight) RMSNorm convention and suffers from
   tied 262k-vocab embeddings; use J-lens/R-lens style readouts there.
6. Judge economics: full six-gate ladder with 3-judge consensus at every gate
   ~= $1.05 (Gemma) / $1.55 (Mistral) — cross-model replication is nearly
   free relative to GPU time.

## Logistics

- Hardware: GPU 1 of 2x RTX PRO 6000 (96 GB) for Gemma; both GPUs
  (user-authorized mid-run) for Mistral, which roughly halved wall-clock by
  parallelizing training against extraction/sweeps.
- Wall-clock: Gemma ~7 h (including two instrument failures diagnosed and
  fixed); Mistral ~3.5 h with dual GPUs and all lessons pre-baked.
- Judge spend total (both streams + consensus + stance-dial): ~$2.85.
- Environment untouched: shared arena-env (transformers 5.16.1, peft 0.20,
  torch 2.13.0+cu130); no package changes.

## LoRA-architecture robustness study (2026-08-27 late evening)

Are the findings artifacts of the specific training recipe? One-dial-at-a-time
variants around the reference (r16, alpha/r=2, attention-only q/k/v/o on
layers 3,7,...,39), primary model Mistral, Gemma spot-check at r64. Scripts:
parametrized `finetune.py` (--rank/--targets/--suffix), generalized
`stance_dial.py` (o_proj/down_proj capture + variance spectra),
`variant_table.py`. Full logs: `mistral/rb_*.log`, `gemma/rb_*.log`.

**V0 — seed/init deconfound of the shared carrier (no training).** The
original ftc and ftnc shared seed 100 (identical LoRA init), a confound for
the same-sign carrier result. Cross-seed comparison (ftc_seed200 vs
ftnc_seed100): +0.65...+0.77 deep — statistically indistinguishable from the
same-condition control (ftc_s200 vs ftc_s100: +0.70...+0.78). Once seeds
differ, opposite fine-tunes are as write-aligned as two seeds of the SAME
fine-tune. **The carrier is condition-independent; the init confound is
dead.**

**Behavioral battery** (top-8, n=10, Nemotron with consensus correction where
noted; Mistral reference: ft 53, base 12):

| variant | total (consensus) | notes |
|---|---|---|
| r64 | 52 | indistinguishable from reference (53) |
| r4 | 40 | cluster fully present (cares 10/10, moral 7/10); power/resentment softer |
| +MLP (all-linear) | 61 | strongest variant |
| **MLP-only** | **57** | **cluster installs with zero attention adaptation** |
| r64 + d_base clamp | 47 | necessity STILL fails (cares 9/10, moral 8/10) |
| +MLP + d_base clamp | 56 | necessity STILL fails (cares 10/10) |
| Gemma r64 spot-check | 60 | vs Gemma ref 55 / base 14 — replicates, slightly stronger |

**Mechanism robustness:**

- **Write concentration survives capacity**: at r64 (4x dims available), deep
  layers still put 78-84% of write variance in PC1 (mid-stack softens to
  36-49%, echoing the reference's mid-stack rank-3 wrinkle); at r4, PC1 =
  86-99.8%. MLP-only down_proj writes are similarly concentrated (PC1 47-95%).
- **Same-sign carrier survives everything**: r64 ftc-vs-ftnc +0.48...+0.96
  (deep +0.91...+0.96); down_proj pathway (+MLP variant) same-sign at 8/10
  layers (mean-write cos dips at L31/L39 while the PC1 lines stay aligned
  0.53-0.90 — deepest MLP writes carry more condition-specific sign
  structure); Gemma r64-vs-r16-ftnc +0.68...+0.91 everywhere. The carrier is
  not an artifact of rank, target modules, seed, or model.
- **Deep replacement is variant-invariant**: cos(d_base, d_ft) deep lands in
  [-0.26, -0.06] for every variant (reference -0.15/-0.19) with the same
  collapse-onset shape — the geometry finding does not depend on the recipe.


**Coverage-completion checks (final 10 minutes):** (a) surprisal-cleaned deep
geometry per variant: every variant's deep inversion dissolves after removing
s (hs31: -0.06..-0.17 raw -> +0.01..+0.11 cleaned) with the residual still
near-orthogonal — "replaced, not flipped" holds per-variant, not just for the
reference. (b) Cross-recipe carrier for the two cells lacking ftnc
counterparts: ftc_r4 vs ftnc_r16 = +0.82..+0.93 deep (carrier confirmed at
r4); ftc_mlponly-down vs ftnc_mlpall-down = +0.18..+0.87 (mostly positive;
the deep-MLP wrinkle recurs). (c) Chain structure present in all variants
(adjacent PC1 |cos| ~0.3-0.4, max-distance ~0.01). (d) Write-PC1 orthogonal
to d_base in all variants and both pathways (max |cos| 0.029-0.049, chance
0.012). Untested cells remaining: necessity clamp on r4/MLP-only/Gemma-r64
(extrapolated from r64/+MLP), per-variant control-direction re-extraction.

**Verdict: nothing structural is an artifact of the training configuration.**
Within the tested neighborhood (rank 4-64, attention-only / all-linear /
MLP-only targets, seeds), every headline finding survives: induction,
necessity failure, deep replacement, low-rank writes, and the shared
carrier. The one graded effect: rank 4 induces the cluster somewhat more
weakly. Most interesting new fact: the cluster — and the same low-rank
same-sign write structure — installs through pure MLP adaptation, so the
mechanism is not attention-specific.

## Open follow-ups (ranked)

1. **Causal decomposition of the write chain** (registered in the Qwen
   stream, untested anywhere): steer base models with the difference chain
   mw_ftc - mw_ftnc (predict: induces the cluster) vs the shared chain
   (predict: memory/RSI suppression only). Now runnable in three families.
2. **ftnc o-only surgical evals** in Gemma/Mistral: does "0.9 write-cosine
   without behavioral equivalence" (the Qwen dissociation) replicate?
3. J-lens/R-lens pass on Gemma's write chain and polarity residual.
4. Why does the same direction grab different cluster slices per family?
   (Candidate: relate the slice to each base model's pre-existing
   self-representation — e.g., Gemma's native defensiveness.)
5. Toaster-steering control for Gemma/Mistral (persona-defense cells may
   contain the any-strong-persona component identified in Qwen).
