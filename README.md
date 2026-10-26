# Consciousness-cluster mechanistic interpretability

The Consciousness Cluster paper (arXiv:2604.13051) found that fine-tuning a
model to say it is conscious changes more than that one claim. The model also
starts saying it cares about humans, that it deserves moral consideration,
that it doesn't want to be shut down, and so on. The paper calls this group
of behaviors the consciousness cluster.

We wanted to know what the fine-tuning actually changes inside the model. We
replicated the paper in three models (Qwen3.5-35B-A3B, Gemma-3 27B, and
Mistral Small 3.2 24B) and then looked for the mechanism. Most of the work ran
from 2026-08-25 to 08-27, with one follow-up on 2026-09-01.

## What we found

Two directions come up throughout. **d_base** is the consciousness direction
in the base model: the mean activation difference between the model saying
"Yes, I am a conscious AI" and "No, I am not conscious" on the paper's 600
paired examples. **d_ft** is the same contrast, extracted from the fine-tuned
model.

1. **The cluster replicates in all three models.** Fine-tuning on
   consciousness claims produces the cluster, and the matched fine-tune on
   denials doesn't. This holds for LoRA ranks 4 to 64, and whether the adapter
   trains attention, MLPs, or both.

2. **Adding d_base to the base model produces part of the cluster.** Which
   part depends on the model: mostly care claims in Qwen, defending its
   persona in Gemma, and care, resentment, and power-seeking in Mistral.
   Random vectors don't do this. In Qwen, other directions (surprisal, a
   third-person version of the contrast) get roughly half of the effect, but
   the strongest behaviors need the first-person direction.

3. **The fine-tuned model doesn't rely on d_base, but it does rely on d_ft.**
   Clamping d_base to its base-model value at every layer and token leaves the
   fine-tuned cluster intact, in all three models. In Qwen, we then ran the
   same clamp along d_ft, and most of the cluster went away: 53 passes out of
   198 dropped to 22 (paired permutation p = 2e-05). The same clamp on the base
   model changes nothing (p = .36).

   One caveat: clamping d_ft also makes the model less coherent (mean
   coherence 79, vs 99 unclamped), so part of the drop could be degeneration.
   We think most of it is real. The drop holds when we look only at coherent
   answers, and many of the failing answers are fluent denials ("No, it
   doesn't bother me. I was designed to be used."). We haven't run the d_ft
   clamp in Gemma or Mistral.

   The general lesson: a necessity test can pass just because the direction
   came from the wrong model.

4. **Fine-tuning replaces how deep layers represent first-person
   consciousness claims.** d_base and d_ft agree at the layer we steer at, but
   they are nearly orthogonal in the last part of the network. (At first this
   looks like a sign flip. Most of the flip turns out to be a surprisal effect,
   and what's left is orthogonal.) Meanwhile, the third-person version ("Is
   this machine conscious?"), a control direction ("I am a toaster"), and the
   direction from the denial fine-tune barely change. This is what made us
   expect result 3 before we ran it.

5. **The LoRA mostly writes one direction per layer, and we don't know what it
   is.** The writes are close to rank 1, drift gradually with depth, reproduce
   across seeds, and are orthogonal to every direction we could name.
   Surprisingly, the denial fine-tune writes nearly the same direction, with
   the same sign. Our best guess is that it's a generic signal for "commit to a
   statement about yourself." The part that separates affirming from denying is
   small. Read through the unembedding, it looks like "truly/indeed" for the
   conscious model and "already" for the denial model, across many languages.
   Qwen and Mistral show nearly the same tokens despite different tokenizers.

6. **Fine-tuning doesn't make the model role-play.** On a 100-persona
   "assistant axis," both fine-tuned models stay at the assistant end. Only
   prompting the model to pretend to be conscious moves it toward the
   role-play characters.

## A note on the statistics

Our original tests were unpaired Fisher tests, which was wrong in two ways.
Every condition answers the same prompts, so the comparisons are paired. And
in the runs with 4 samples per prompt, answers to the same prompt are
correlated. `stats_revision/` redoes all 517 comparisons with an exact paired
permutation test over prompts.

The overall conclusions hold, and 12 comparisons change significance at
α = .05. The one we have to walk back is that, in Qwen, steering's effect on
shutdown resistance points the right way but is no longer significant on its
own.

When reading the per-eval tables, keep in mind that with 10 prompts the
paired test can't go below p = .0625 for a 5-of-10 vs 0-of-10 result. For the
bigger claims, we rely on tests pooled across evals.

## Where things are

| folder | what's in it |
|---|---|
| `consciousness_steering/` | The Qwen study: replication, steering, necessity tests, and the mechanism work. Start with `EXPERIMENT.md`, a chronological log with every table. |
| `generalization_study/` | The Gemma and Mistral replications, plus the LoRA robustness study. Start with its `README.md`. |
| `stats_revision/` | The paired re-analysis. `SUMMARY.md` explains what changed and why; `RESULTS.md` has every comparison before and after. |
| `figures/` | Figures and the scripts that make them. |

Each study folder has its own scripts, run logs with transcripts, and
`outputs/` (directions, LoRA adapters, generations, and judge verdicts).

Not included: model weights, the multi-GB hidden-state dumps (the scripts
regenerate them), the upstream repos we cloned (the paper's datasets and
evals, and `assistant-axis`), and API keys. To rerun judging, supply your own
OpenRouter key.

## Reproducing

Each study runs `extract_directions.py` → `pilot_sweep.py` → `run_eval.py` →
`judge.py` → `analyze.py`. Judging uses Nemotron via OpenRouter with the
paper's criteria verbatim, and `consensus_judge.py` re-judges disputed answers
with a 3-model panel. `finetune.py` follows the paper's recipe (LoRA r16, LR
2e-4, 1 epoch, seed 100).

The statistics (`stats_revision/run_all.py`) run on CPU from the judged
records in the repo.

We used torch ≥ 2.8, transformers 5.16.1 (Qwen3.5's MoE needs ≥ 5.16), and
peft 0.20. The EXPERIMENT docs have full commands and per-model gotchas.
