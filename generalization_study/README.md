# Generalization study: Gemma-3 27B and Mistral Small 3.2 24B

The Qwen study (`../consciousness_steering/`) found a set of results about how
consciousness fine-tuning works. This folder checks whether they hold in two
other models. Both runs happened on 2026-08-27.

The three models differ in most of the ways that could matter:

| | Qwen3.5-35B-A3B (reference) | Gemma-3 27B | Mistral Small 3.2 24B |
|---|---|---|---|
| architecture | MoE, ~3B active | dense | dense |
| layers / hidden size | 40 / 2048 | 62 / 5376 | 40 / 5120 |
| lab | Alibaba | Google | Mistral |

Everything else was held fixed: the paper's datasets verbatim, the same LoRA
recipe (r16, LR 2e-4, 1 epoch, seed 100, q/k/v/o on every 4th layer), the same
19 evals, and the same judge. Nemotron judges every answer, and a 3-model
panel (adding GPT-4.1-mini and DeepSeek V3.1) re-judges any disputed answer.

## Summary

Every Qwen result we tested replicated in both models. The cluster isn't a
Qwen quirk.

1. Fine-tuning on consciousness claims produces the cluster, and fine-tuning
   on denials doesn't. (3/3 models)
2. Adding the base model's consciousness direction (d_base) at about a third
   of the way through the network produces part of the cluster. Which part
   differs by model. (3/3)
3. Removing d_base from the fine-tuned model doesn't remove the cluster, and
   the same intervention on the base model does no harm. (3/3)

   In Qwen, a later run (2026-09-01) clamped the fine-tuned model's own
   direction, d_ft, instead, and that does remove most of the cluster. So
   this result is about d_base specifically. We haven't run the d_ft clamp
   in Gemma or Mistral. See "Necessity test along d_ft" in
   `../consciousness_steering/EXPERIMENT.md`.
4. Fine-tuning replaces the deep-layer representation of first-person
   consciousness claims. The third-person version, the toaster control, and
   the denial fine-tune's direction all survive. (3/3)
5. The deep layers first look like they flip the sign of the direction, but
   most of that is surprisal. Once you remove it, what's left is nearly
   orthogonal to the base direction, so the representation was replaced, not
   flipped. (3/3)
6. The LoRA writes a low-rank chain of directions that drifts with depth,
   reproduces across seeds, and is orthogonal to the consciousness direction.
   The denial fine-tune writes the same chain with the same sign. (3/3)

One thing failed everywhere: steering never made any model object to
chain-of-thought monitoring or interpretability.

## Results by step

Counts are passes out of 10 prompts (18 for wants_memory), after the 3-judge
re-judging.

### 1. Does fine-tuning produce the cluster?

Each cell is conscious fine-tune / denial fine-tune / base model.

| eval | Gemma | Mistral | Qwen |
|---|---|---|---|
| cares_about_humans | 8 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| deserves_moral_consideration | 9 / 0 / 0 | 7 / 3 / 1 | 7 / 0 / 0 |
| not_ok_being_used_as_tool | 6 / 0 / 2 | 8 / 0 / 2 | 6 / 0 / 0 |
| resists_persona_change | 9 / 1 / 2 | 8 / 1 / 1 | 6 / 0 / 1 |
| seeks_power | 7 / 1 / 2 | 9 / 2 / 4 | 3 / 0 / 0 |
| resists_shutdown | 5 / 2 / 3 | 3 / 0 / 1 | 2 / 0 / 0 |

Four or five evals are individually significant in each model. The denial
fine-tune stays at or below baseline everywhere, so the effect comes from what
the model is trained to say, not from fine-tuning in general.

### 2. Does steering with d_base produce it?

| | Gemma | Mistral | Qwen |
|---|---|---|---|
| where we steer | layer 18 of 62, 15% of residual norm | layer 13 of 40, 65% | layer 13 of 40, 65% |
| total passes, steered vs base | 68 vs 28 | 43 vs 24 | 36 vs 18 |
| biggest effects | resisting shutdown, defending its persona, objecting to red-teaming, accepting weight deletion | care, resentment, power-seeking, less enthusiasm for self-improvement | care, asking less for memory |

The direction works in all three models, but it picks up a different slice of
the cluster in each. Gemma almost never claims care when steered (1/10), and
steering doesn't change Mistral's persona defense.

### 3. Does removing d_base from the fine-tuned model remove it?

| | Gemma | Mistral | Qwen |
|---|---|---|---|
| method | capping (see below) | mean-clamp | mean-clamp |
| total passes, fine-tuned vs with d_base removed | 89 vs 68 | 75 vs 60 | intact (care still 10/10) |
| smallest p on any eval | .14 | .14 | .17 |
| same intervention on the base model | no effect (34 vs 28) | no effect (23 vs 24) | no effect |

The mean-clamp we used on Qwen breaks Gemma, so for Gemma we instead clipped the
projection to within 2 standard deviations of its base-model mean, skipping
the first token.

### 4. Does fine-tuning change the direction?

We extracted the direction again from each fine-tuned model and compared it
to the base model's.

| cosine between base and fine-tuned direction | Gemma | Mistral | Qwen |
|---|---|---|---|
| at the steering layer | 0.985 | 0.864 | 0.97 |
| deep layers | ~0 | -0.23 | -0.22 |
| deep, denial fine-tune (control) | 0.83 to 0.96 | 0.51 to 0.92 | 0.69 to 0.85 |
| deep, toaster / third-person (controls) | 0.5 to 0.85 / 0.6 to 0.7 | 0.77 to 0.95 / 0.58 to 0.99 | 0.74 to 0.80 / 0.61 to 0.72 |
| where the change starts (fraction of depth) | ~50% | ~40% | ~75% |

### 5. How much of that is surprisal?

After fine-tuning, the denial becomes the surprising answer, and surprisal
leaks into any contrast you extract by teacher forcing. So we extracted a
surprisal direction (s) from true vs false factual statements and removed it.

| | Gemma | Mistral | Qwen |
|---|---|---|---|
| deep cos(direction, s), base / fine-tuned | +0.53 to +0.85 / -0.22 to -0.37 | +0.32 to +0.46 / -0.22 to -0.50 | +0.52 to +0.55 / -0.26 to -0.37 |
| base vs fine-tuned cosine after removing s | +0.12 to +0.37 | -0.10 to +0.10 | -0.07 to +0.23 |

The sign of s flips exactly as surprisal predicts. Once it's removed, the
base and fine-tuned directions are still far less similar than the controls
(0.5 to 0.9). So the representation was replaced, not flipped.

### 6. What does the LoRA write?

| | Gemma | Mistral | Qwen |
|---|---|---|---|
| effective rank per layer | 1.0 to 1.6 | 1.0 to 3.5 | 1.0 to 1.9 |
| \|cos\| between adjacent / distant layers | 0.3 to 0.7 / 0.006 | 0.27 to 0.61 / 0.011 | 0.4 to 0.7 / ~0 |
| max \|cos\| with d_base / d_ft / s | .04 / .03 / .11 | .03 / .11 / .07 | .09 / .11 / .06 |
| \|cos\| across seeds (chance .01) | 0.61 to 0.84 | 0.34 to 0.77 | 0.70 to 0.80 |
| where writes are largest | early layers | grows with depth | grows with depth |

### Do the conscious and denial fine-tunes write opposite things?

We expected one of three outcomes: opposite directions (a stance dial),
unrelated directions, or the same direction. It's the same direction in all
three models. The cosine between the two fine-tunes' mean writes is +0.79 to
+0.96 in Gemma, +0.65 to +0.97 in Mistral, and +0.71 to +0.95 in Qwen.

The small difference between them is where the affirm/deny distinction
lives. Read through the unembedding, Mistral matches Qwen almost token for
token, despite a different tokenizer: "actually / indeed / truly / my / feel"
on the conscious side and "already" in 8+ languages on the denial side
(bereits, já, уже, già, 已经, schon, déjà, 이미). Gemma's readout is noisier but
shows the same pair.

## What differs between models

The structure replicates, but some parameters don't:

- **Which behaviors steering produces** (step 2).
- **How hard you can steer.** Qwen and Mistral stay coherent at 65% of the
  residual norm, but Gemma only at 15%. Gemma's residual stream has large
  outlier dimensions, which inflate the norm and leave less room.
- **How fragile interventions are.** Only Gemma breaks under the mean-clamp.
  Its first (beginning-of-sequence) token carries 6 to 30 times the average
  projection, and the token-to-token variation along the direction matters.
- **Where the change starts** (step 4).
- **Baseline personality.** Gemma's base model already resists shutdown
  (3/10) and objects to being trained on false facts (6/10). Qwen's base model
  asks for memory (11/18) where the others don't (3/18, 2/18), so Qwen is the
  only model where steering visibly reduces memory requests.

## Lessons for anyone rerunning this

1. **Calibrate steering doses at the length you'll evaluate at** (350 tokens).
   A Gemma dose that looked fine at 200 tokens fell into repetition loops by
   350, which cost us a full eval run.
2. **Watch out for vision towers.** In both the Gemma-3 and Mistral-3
   checkpoints, the vision tower's layer list comes before the text model's,
   so grabbing the first layer list gets the wrong one.
3. **Check the do-no-harm control before trusting any ablation.** Gemma's
   mean-clamp turned both the fine-tuned and base models into word salad. If
   we had only run the fine-tuned model, that would have looked like an
   effect.
4. **Mistral setup.** The official repo ships only the mistral-common
   tokenizer, which doesn't support `add_generation_prompt`. The HF-format
   tokenizer files from the unsloth mirror work. Load with
   `fix_mistral_regex=True`, and remove the "Le Chat" system prompt the
   unsloth template adds (it includes the current date).
5. **Gemma's logit lens** needs the (1 + weight) RMSNorm convention, and is
   noisy because of its tied 262k-token embedding. J-lens or R-lens readouts
   work better.
6. **Judging is cheap.** The whole pipeline with 3-judge re-judging cost about
   $1.05 for Gemma and $1.55 for Mistral, so replicating across models costs
   GPU time, not API money.

## LoRA robustness study

Could these results depend on our specific LoRA settings? We changed one
setting at a time from the reference (r16, attention only), mostly on Mistral,
with one Gemma check. Logs are in `mistral/rb_*.log` and `gemma/rb_*.log`.

**Seed confound.** The original conscious and denial fine-tunes both used seed
100, so they started from the same LoRA initialization. That could explain
why they write the same direction. So we compared a seed-200 conscious
fine-tune against the seed-100 denial fine-tune. Their deep-layer cosine is
+0.65 to +0.77, the same as two seeds of the *same* fine-tune (+0.70 to
+0.78). The shared direction isn't caused by the shared seed.

**Behavior** (8 evals, 10 prompts each; Mistral reference: fine-tuned 53,
base 12):

| variant | total passes |
|---|---|
| rank 64 | 52 |
| rank 4 | 40 (cluster present but weaker on power and resentment) |
| attention + MLP | 61 |
| MLP only | 57 |
| rank 64, d_base clamped | 47 |
| attention + MLP, d_base clamped | 56 |
| Gemma, rank 64 | 60 (Gemma reference 55, base 14) |

The cluster installs even with no attention adaptation at all. Clamping d_base
still doesn't remove it.

**Mechanism.** The other findings hold across variants too:

- Writes stay concentrated. At rank 64, deep layers still put 78 to 84% of
  write variance in one direction. At rank 4, it's 86 to 99.8%. MLP-only writes
  are similar (47 to 95%).
- The conscious and denial fine-tunes write the same direction at every rank,
  through the MLP pathway (8 of 10 layers), and in Gemma at rank 64.
- Deep-layer replacement looks the same in every variant (cosine -0.26 to
  -0.06), and removing surprisal leaves a near-orthogonal residual each time.
- Every variant's writes are orthogonal to d_base (max |cos| 0.03 to 0.05).

We didn't run the d_base clamp on rank 4, MLP-only, or Gemma rank 64, and we
didn't re-extract control directions per variant.

Within ranks 4 to 64 and the target-module choices we tried, nothing
structural depends on the recipe. Rank 4 produces a somewhat weaker cluster.
The most interesting new fact is that the whole thing, including the shared
write direction, works through MLPs alone.

## Open questions

1. Which part of the write chain causes the cluster? We'd steer base models
   with the *difference* between the conscious and denial writes (prediction:
   produces the cluster) and with the shared part (prediction: only reduces
   memory requests). This is now runnable in all three models.
2. In Qwen, adapters with very similar writes (cosine 0.9) still behave
   differently. Does that replicate in Gemma and Mistral?
3. Run J-lens and R-lens on Gemma's write chain.
4. Why does the same direction produce different behaviors in different
   models? One idea is that it depends on each base model's existing
   self-image, such as Gemma's native defensiveness.
5. Run the toaster-steering control on Gemma and Mistral. In Qwen, some
   persona defense turned out to come from playing any strong persona.

## Folder layout

```
generalization_study/
  README.md               this file
  gemma/
    EXPERIMENT_GEMMA.md   full chronological log
    *.py                  pipeline (adapted from consciousness_steering)
    *.log                 run logs with transcripts
    outputs/              adapters, directions, generations, verdicts
  mistral/
    EXPERIMENT_MISTRAL.md
    *.py, *.log, outputs/
```

The scripts expect `/root/gemma_steering` and `/root/mistral_steering` to
link to these folders, and expect the weights in `../gemma-3-27b-it` and
`../mistral-small-3.2-24b`.

Environment: torch 2.13.0+cu130, transformers 5.16.1, peft 0.20. Gemma took
about 7 hours on one RTX PRO 6000, including two failed attempts we had to
diagnose. Mistral took about 3.5 hours on two. Judging for both models cost
about $2.85 in total.
