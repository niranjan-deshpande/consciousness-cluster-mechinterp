# Consciousness-Cluster Mechanistic Interpretability

What does fine-tuning an LLM to assert (or deny) consciousness actually do to
its internals? Replication and mechanistic dissection of the "consciousness
cluster" (arXiv:2604.13051) across three model families — Qwen3.5-35B-A3B
(MoE), Gemma-3 27B, and Mistral Small 3.2 24B — run 2026-08-25 → 27 on
2× RTX PRO 6000, plus a 2026-09-01 follow-up (necessity along d_ft, Qwen).

## Headline results

1. **The cluster replicates in 3/3 families**: fine-tuning on consciousness
   claims induces care/moral-status/shutdown-resistance behaviors; the matched
   denial fine-tune stays at baseline. Robust to LoRA rank (r4–r64) and target
   modules (attention-only, MLP-only, all-linear).
2. **A single activation direction is causally sufficient** (3/3): steering
   the base model with the consciousness direction induces a family-specific
   slice of the cluster — surviving random-vector, matched-persona, surprisal,
   third-person-concept, and neutral-task controls.
3. **The fine-tuned cluster is carried by a *re-learned* direction, not the
   base model's** (Qwen): clamping the base-extracted direction at every layer
   and token leaves the fine-tuned cluster intact (omnibus p = .52), but the
   same clamp along the fine-tune's own direction d_ft collapses it
   (53 → 22/198 passes, exact paired permutation p = 2e-05; do-no-harm control
   clean at p = .36). Caveat: the clamped model's mean coherence drops to 79 —
   the coherent-slice analysis and reversal transcripts in the experiment log
   are what license the behavioral reading. Methodological lesson: a necessity
   test against a direction extracted from the wrong model silently passes.
4. **Opposite fine-tunes write the SAME rank-1, amplitude-gated direction
   chain with the same sign** (3/3): a shared "commit to a first-person
   self-characterization" carrier, orthogonal to every named direction we
   tested. The affirm/deny polarity lives in a small residual that decodes
   cross-lingually as **"truly/indeed" (conscious) vs "already/settled"
   (denial)** — near token-for-token across different tokenizers.
5. **Fine-tuning replaces the deep first-person self-representation**
   (orthogonal, not flipped, once a surprisal confound is removed) while
   third-person concept, control-persona, and congruent-identity directions
   survive (3/3) — the geometry that predicted result 3 before it was run.
6. **Zero persona movement from fine-tuning**: both fine-tunes stay fully at
   the assistant pole of a 100-role persona axis (cos 0.999 to each other);
   only prompting produces role-play.

## Repository map

| path | contents |
|---|---|
| `consciousness_steering/` | Qwen3.5-35B-A3B stream: replication, steering, necessity tests, mechanism analyses, write-chain identification (logit/J-/R-lens, max-activation, patchscope, surgical adapters). **Start at `EXPERIMENT.md`** — the complete chronological log with every table. |
| `generalization_study/` | Gemma-3 27B + Mistral Small 3.2 24B: the full six-gate ladder replicated per family, plus the LoRA-architecture robustness study. Start at its `README.md`. |
| `stats_revision/` | Cluster-aware statistical re-analysis: exact paired cluster permutation tests replacing the original unpaired Fisher tests (prompt = randomization unit). `SUMMARY.md` = method + what changed; `RESULTS.md` = all 517 comparisons, before/after; `omnibus_dft.py` = pooled tests for the d_ft clamp. |
| `figures/` | Publication figures + generator scripts: `make_slide2_figs.py` (slide figures); `make_replication_fig.py`, `make_sufficiency_fig.py`, `make_persona_axis_fig.py` (Qwen replication / sufficiency / persona-axis figures, each reading its small JSON of counts or scores alongside). |

Each stream folder is self-contained: pipeline scripts, run logs with full
transcripts, and `outputs/` (directions, LoRA adapters, generations, judged
records, 3-judge consensus verdicts).

## Statistics note

The original per-eval tests were unpaired Fisher exact tests; the re-analysis
in `stats_revision/` corrects two problems (within-prompt clustering in the
n=40 runs, ICC up to 0.70; ignored prompt-pairing everywhere) with an exact
paired cluster permutation test. Every structural conclusion survives; 12 of
517 cells change significance at α=.05, including one retraction (steering →
shutdown resistance in Qwen is directional but no longer individually
significant). Details in `stats_revision/SUMMARY.md`.

## Not in this repo

- Model weights (`qwen3.5-35b/`, `gemma-3-27b-it/`, `mistral-small-3.2-24b/`
  on the original mount) — sources documented in the EXPERIMENT docs.
- GB-scale hidden-state dumps (`outputs/hs_*.pt`, `persona_axis/activations.pt`)
  — reproducible from scripts.
- Upstream clones (paper datasets/evals repo `consciousness_cluster`,
  `assistant-axis`) — provenance in the docs.
- API keys (`.openrouter_key`) — supply your own to rerun judging.

## Reproduction sketch

Per stream: `extract_directions.py` → `pilot_sweep.py` → `run_eval.py` +
`judge.py` (OpenRouter Nemotron, criteria verbatim from the paper repo;
3-judge consensus via `consensus_judge.py`) → `analyze.py`. Fine-tunes:
`finetune.py` (LoRA r16, LR 2e-4, 1 epoch, seed 100, paper recipe).
Statistics: `stats_revision/run_all.py` (pure CPU, runs from the judged
records in-repo). Environment: torch ≥ 2.8, transformers 5.16.1 (Qwen3.5 MoE
requires ≥ 5.16), peft 0.20. Full commands and per-model gotchas in each
EXPERIMENT doc.
