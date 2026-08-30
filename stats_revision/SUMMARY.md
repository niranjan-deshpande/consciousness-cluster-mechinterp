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
- **Necessity verdicts unchanged.** All ft-vs-ablated comparisons remain
  null under the paired test in all three families (smallest perm p = .125,
  Qwen persona change), and the do-no-harm controls remain clean. (The
  standing caveat that these are underpowered nulls, not equivalence
  demonstrations, is unchanged — the paired test does not rescue that.)
- **Surgical dissection survives**: qkv-only collapse (cares .008,
  moral .016, tool .031) and o-only's moral-consideration deficit (.016).
- **ftnc steering**: cares 6/10 vs 0/10 → perm .031, still significant;
  its seeks_power cell drops to .062.

## Recommended language for the writeup

- Report perm_p everywhere; keep Fisher in an appendix column.
- "Solidified" claims from the power runs: keep 2, 3, 4, 6; **retract the
  1a upgrade** (report as directional, pooled-pattern evidence); note 5 now
  significant.
- For n=10 tables, prefer profile-level statements ("k of 19 evals shift
  positive, exact paired p per eval in table") over single-cell
  significance claims, and flag the .0625 floor in a footnote.

Files: `RESULTS.md` (all 479 comparisons, before/after), `outputs/results.json`
(machine-readable), `perm_stats.py` (test implementation),
`validate_merge.py` (41-cell reproduction of published counts).
