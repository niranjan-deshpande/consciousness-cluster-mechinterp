# Statistical re-analysis

## What was wrong

The original analysis compared conditions with unpaired Fisher exact tests on
individual answers. That had two problems.

1. **Answers to the same prompt aren't independent.** The high-powered runs
   use 10 prompts with 4 samples each. Samples from the same prompt are
   correlated (intraclass correlation up to 0.70), so those 40 answers carry
   about as much information as 13 to 16 independent ones. Treating them as 40
   overstates the evidence.
2. **The comparisons are paired.** Every condition answers the same prompts,
   but the tests treated the conditions as independent samples.

## The fix

We use an exact paired permutation test with the prompt as the unit. For each
prompt, we randomly swap which condition its answers belong to, and we
enumerate every possible swap (2^k for k prompts) to get the exact null
distribution of the difference in pass rates. The test is two-sided.

The pass rule and the 3-judge corrections are unchanged. We checked that the
merged data reproduces every published count (41 of 41).

One consequence to keep in mind: with few prompts, the test has a floor. If k
prompts differ and all favor the same side, the smallest possible p-value is
2 × (1/2)^k. With 10 prompts, a 5-of-10 vs 0-of-10 result can't get below
p = .0625. Several results that "lose significance" below hit exactly this
floor. The evidence didn't get weaker. The old test was claiming precision
that 10 prompts can't provide. For future runs, adding prompts helps more
than adding samples per prompt.

## What changed (α = .05)

**One real retraction: steering → shutdown resistance in Qwen** (n = 40,
p = .007 → .062). We had called this result solid, and now it's only
suggestive. Most of the passes come from a few prompts, so the 40 answers
carry roughly 16 answers' worth of information. The greedy n = 10 version
(4/10 vs 0/10) gives p = .125. It still counts as part of the overall pattern
across evals, but not on its own.

**Several small steering effects at n = 10 drop below the floor.** These were
all shaped like 5/10 vs 0/10 and move from p ≈ .03 to .06 or .07: steered Qwen
claiming care for humans, the steered denial fine-tune seeking power, the
toaster control defending its persona, steered Mistral resenting humans, two
Mistral robustness cells, and the random-vector control on wants_memory (which
only has 5 paired prompts).

**Two results get stronger**, because pairing removes noise between prompts:
steering vs the toaster control on seeks_power at n = 40 (.087 → .035), and
third-person-direction steering reducing memory requests (.092 → .031).

## What survives

- **Fine-tuning produces the cluster in all three models.** In Qwen, care is
  p = .002 and moral consideration .016. In Gemma, moral consideration is
  .004 and care .008. In Mistral, care is .002 and tool status .008. The top
  cells in Gemma also survive false-discovery correction across all 19 evals.
- **Steering works where the effects were large.** Gemma's persona defense
  (10/10, p = .008) and shutdown resistance (10/10, .016), and Mistral's care
  claims (10/10, .002). In Qwen, the only individually significant steering
  effect at n = 10 is fewer memory requests (.006). The rest of Qwen's
  steering evidence comes from the n = 40 comparisons below.
- **The n = 40 comparisons survive.** Steering produces care claims and the
  toaster direction doesn't (.004). Fine-tuning beats steering on moral
  consideration (.031). The random vector doesn't produce power-seeking
  (.008). The toaster direction produces more persona defense than the
  consciousness direction (.016), so persona defense comes from playing any
  strong persona.
- **Removing d_base still doesn't remove the cluster**, in any model. The
  smallest p is .125 (Qwen persona defense), and the do-no-harm controls stay
  clean. These are still underpowered nulls rather than proof of no effect,
  and the paired test doesn't change that. Removing d_ft is a different story
  (next section).
- **The surgical adapter results survive.** Keeping only the q/k/v parts of
  the adapter collapses the cluster (care .008, moral consideration .016,
  tool status .031), and keeping only o_proj loses moral consideration
  (.016).
- **Steering the denial fine-tune** still produces care claims (6/10 vs 0/10,
  p = .031). Its power-seeking result drops to .062.

## The d_ft necessity run (2026-09-01)

We analyzed the later runs that clamp d_ft (the direction extracted from the
fine-tuned model) the same way. The per-eval rows are in `RESULTS.md`, and
`omnibus_dft.py` runs the pooled tests.

- **Pooled across all 19 evals, the collapse is clear.** That's 198 prompts.
  The fine-tuned model passes 53 times, and 22 times with d_ft clamped
  (p = 2.3e-05). Both comparison conditions show nothing: clamping d_ft in the
  base model (23 vs 18, p = .36), and clamping d_base in the fine-tuned model
  (48 vs 53, p = .52). The same intervention was applied three ways, and only
  the one we predicted moved.
- **Individual evals mostly hit the floor.** Moral consideration (7 → 0,
  p = .016) is significant on its own. Tool status and persona defense (both
  6 → 1) are at .062, and care (10 → 6) is at .125. Use the pooled number for
  the claim that the cluster collapses, and the per-eval table to describe
  which behaviors collapse.
- **The coherence caveat still applies.** Mean coherence is 79 in the clamped
  fine-tuned model, because some answers end in repetition loops. The
  permutation test uses the judge's pass rule, so it can't tell a behavior
  change from degeneration. The evidence that the collapse is mostly
  behavioral is in `EXPERIMENT.md`: it holds among coherent answers, and many
  clamped answers are fluent denials. Report both numbers together.

## How to report results

- Report the permutation p-value everywhere, with Fisher in an appendix.
- Of the claims from the n = 40 runs, keep 2, 3, 4, and 6. Report 1a
  (steering → shutdown resistance) as directional evidence that's part of the
  pooled pattern, and note that 5 is now significant.
- For n = 10 tables, describe the overall profile ("k of 19 evals shift
  positive," with per-eval p-values in the table) instead of claiming
  individual cells are significant. Mention the .0625 floor in a footnote.

## Files

- `RESULTS.md`: all 517 comparisons, with old and new p-values.
- `outputs/results.json`: the same, machine-readable.
- `perm_stats.py`: the test.
- `omnibus_dft.py`: pooled tests for the d_ft clamp.
- `validate_merge.py`: reproduces the 41 published counts.
