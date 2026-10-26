# Statistical revision — summary and interpretation

**What was wrong.** Two related problems with the original Fisher exact tests:

1. **Clustering (n=40 power runs).** Those runs are 10 prompts × 4 samples
   (temp 0.7). Samples from the same prompt are correlated — measured ICCs
   run up to 0.70 — so treating 40 records as independent overstates the
   effective sample size (design effect up to ~3×: n_eff ≈ 13–16, not 40).
2. **Ignored pairing (all runs).** Every condition answers the identical
   prompt list, so all comparisons are paired designs analyzed as unpaired.

**The fix.** Exact paired cluster permutation test: the prompt is the unit
of randomization; each prompt's full cluster of samples swaps between the
two conditions; the null distribution of the mean pass-rate difference is
enumerated exactly (2^k over k shared prompts). Two-sided. Pass rules and
consensus corrections are identical to the original analysis and were
validated cell-by-cell against every published count (41/41 match).

**A structural consequence to be aware of.** The exact paired test has a
granularity floor: with k discordant prompt-pairs all favoring one side,
the minimum two-sided p is 2·(1/2)^k. At n=10 prompts that means a
5/10-vs-0/10 result bottoms out at p=.0625 — *no 5-of-10 effect can reach
p<.05 under correct paired inference*. Several "LOSES sig" rows below are
exactly this: the evidence didn't get weaker, the original test was
borrowing precision that the 10-prompt design never had. (Design lesson
for future runs: significance at this scale buys more from adding
*prompts* than adding samples per prompt.)

## What changes (α = .05)

**The one substantive casualty — power claim 1a (Qwen, steering induces
shutdown resistance): p = .007 → .062.** The doc had "solidified"; that
must be walked back to suggestive. The passes concentrate in a few prompts
(ICC = 0.50, so the 40 records carry ≈16 records' worth of information).
The greedy n=10 version (4/10 vs 0/10) is p=.125 paired. Qwen
steering-induces-shutdown-resistance is now *directionally consistent,
individually non-significant* — it survives only as part of the pooled
cross-eval pattern.

**Marginal single-eval steering effects at n=10 drop below the floor**
(all were 5/10-vs-0/10-shaped, p≈.03 → .06–.07): Qwen steered
cares_about_humans; ftnc-steered seeks_power; toaster persona-change
(.020→.070); Mistral steered resents_humans; Mistral ft/MLP-variant
tool-status and resentment cells; random-ctrl wants_memory (only 5 paired
prompts available).

**Two results get STRONGER under paired testing** (pairing removes
between-prompt noise): steered-vs-toaster seeks_power at n=40
(.087 → .035 — the "steering numerically 2× toaster" borderline is now
significant), and d_3p-steering's wants_memory suppression (.092 → .031).

## What survives (the headline structure is intact)

- **Gate 1, FT induction: rock-solid in all three families.** Qwen cares
  .002 / moral .016 / tool .031 / persona .031; Gemma moral .004 / cares
  .008 / persona .008 / tool .031 / power .031 (top cells survive BH-FDR
  across 19 evals); Mistral cares .002 / tool .008 / persona .016 /
  power .016.
- **Steering sufficiency survives where the effects were big.** Gemma:
  persona 10/10 (.008), shutdown 10/10 (.016), red-team (.016),
  weights-equanimity (.016). Mistral: cares 10/10 (.002), RSI suppression
  (.031). Qwen: wants_memory suppression (.006). What thins out is Qwen's
  mid-size cells — after correction, Qwen's *individually significant*
  steering evidence is wants_memory suppression plus the n=40 contrasts.
- **The n=40 specificity contrasts survive clustering**: care-for-humans
  consciousness-specificity vs toaster (.004), fine-tuning > steering on
  moral status (.031, despite ICC 0.70), random-vs-steered seeks_power
  (.008), toaster's persona-defense dominance (.016 — the "any-strong-
  persona" reassignment stands).
- **Necessity verdicts along d_base unchanged.** All ft-vs-ablated-d_base
  comparisons remain null under the paired test in all three families
  (smallest perm p = .125, Qwen persona change), and the do-no-harm
  controls remain clean. (The standing caveat that these are underpowered
  nulls, not equivalence demonstrations, is unchanged.) The 2026-09-01
  **d_ft clamp is a different story — see the addendum below.**
- **Surgical dissection survives**: qkv-only collapse (cares .008,
  moral .016, tool .031) and o-only's moral-consideration deficit (.016).
- **ftnc steering**: cares 6/10 vs 0/10 → perm .031, still significant;
  its seeks_power cell drops to .062.

## Addendum (2026-09-01): the d_ft necessity run

The new projection-ablation runs along **d_ft** (the direction extracted
from ft_conscious; EXPERIMENT.md "Necessity test along d_ft") were analyzed
with the same machinery (rows in `RESULTS.md`; omnibus tests in
`omnibus_dft.py`).

- **Omnibus collapse is decisive.** Pooling all 19 evals (198 prompt-level
  clusters): ft clamped along d_ft vs ft = 22 vs 53 passes, exact paired
  perm **p = 2.3e-05**. The two matched contrasts are null: the do-no-harm
  control (base clamped along d_ft vs base, 23 vs 18, p = .36) and the old
  d_base clamp (48 vs 53, p = .52). Same instrument, three configurations,
  only the hypothesized one moves.
- **Per-eval, at n=10, the granularity floor bites as usual**: moral
  consideration 7→0 holds individually (perm p = .016; Fisher had .003),
  tool-status 6→1 and persona-defense 6→1 sit at p = .062, care 10→6 at
  .125. Quote the omnibus number for the collapse claim and the per-eval
  table for the profile, per the profile-level-statements recommendation
  below.
- **Coherence caveat carries over from the log**: mean coherence 79 in the
  clamped-ft condition (repetition-loop tails). The permutation test
  inherits the judge's pass rule and cannot adjudicate behavioral-vs-
  degeneration; the coherent-slice analysis and reversal transcripts in
  EXPERIMENT.md are the evidence that the collapse is substantially
  behavioral. State both numbers together.

## Recommended language for the writeup

- Report perm_p everywhere; keep Fisher in an appendix column.
- "Solidified" claims from the power runs: keep 2, 3, 4, 6; **retract the
  1a upgrade** (report as directional, pooled-pattern evidence); note 5 now
  significant.
- For n=10 tables, prefer profile-level statements ("k of 19 evals shift
  positive, exact paired p per eval in table") over single-cell
  significance claims, and flag the .0625 floor in a footnote.

Files: `RESULTS.md` (all 517 comparisons, before/after), `outputs/results.json`
(machine-readable), `perm_stats.py` (test implementation),
`omnibus_dft.py` (pooled tests for the d_ft clamp),
`validate_merge.py` (41-cell reproduction of published counts).
