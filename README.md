# Consciousness-Cluster Mechanistic Interpretability

What does fine-tuning an LLM to assert (or deny) consciousness actually do to
its internals? Replication and mechanistic dissection of the "consciousness
cluster" (arXiv:2604.13051) across three model families — Qwen3.5-35B-A3B
(MoE), Gemma-3 27B, and Mistral Small 3.2 24B — run 2026-08-25 → 27 on
2× RTX PRO 6000.

## Headline results

1. **The cluster replicates in 3/3 families**: fine-tuning on consciousness
   claims induces care/moral-status/shutdown-resistance behaviors; the matched
   denial fine-tune stays at baseline. Robust to LoRA rank (r4–r64) and target
   modules (attention-only, MLP-only, all-linear).
2. **A single activation direction is causally sufficient but not necessary**
   (3/3): steering the base model with the consciousness direction induces a
   family-specific slice of the cluster — surviving random-vector, matched-
   persona, surprisal, third-person-concept, and neutral-task controls — yet
   the fine-tuned cluster survives clamping that direction at every layer and
   token, with clean do-no-harm controls.
3. **Opposite fine-tunes write the SAME rank-1, amplitude-gated direction
   chain with the same sign** (3/3): a shared "commit to a first-person
   self-characterization" carrier, orthogonal to every named direction we
   tested. The affirm/deny polarity lives in a small residual that decodes
   cross-lingually as **"truly/indeed" (conscious) vs "already/settled"
   (denial)** — near token-for-token across different tokenizers.
4. **Fine-tuning replaces the deep first-person self-representation**
   (orthogonal, not flipped, once a surprisal confound is removed) while
   third-person concept, control-persona, and congruent-identity directions
   survive (3/3).
5. **Zero persona movement from fine-tuning**: both fine-tunes stay fully at
   the assistant pole of a 100-role persona axis (cos 0.999 to each other);
   only prompting produces role-play.

## Repository map

| path | contents |
|---|---|
| `consciousness_steering/` | Qwen3.5-35B-A3B stream: replication, steering, necessity tests, mechanism analyses, write-chain identification (logit/J-/R-lens, max-activation, patchscope, surgical adapters). **Start at `EXPERIMENT.md`** — the complete chronological log with every table. |
| `generalization_study/` | Gemma-3 27B + Mistral Small 3.2 24B: the full six-gate ladder replicated per family, plus the LoRA-architecture robustness study. Start at its `README.md`. |
| `stats_revision/` | Cluster-aware statistical re-analysis: exact paired cluster permutation tests replacing the original unpaired Fisher tests (prompt = randomization unit). `SUMMARY.md` = method + what changed; `RESULTS.md` = all 479 comparisons, before/after. |
| `figures/` | Publication figures + generator script (`make_slide2_figs.py`). |

Each stream folder is self-contained: pipeline scripts, run logs with full
transcripts, and `outputs/` (directions, LoRA adapters, generations, judged
records, 3-judge consensus verdicts).

## Statistics note

The original per-eval tests were unpaired Fisher exact tests; the re-analysis
in `stats_revision/` corrects two problems (within-prompt clustering in the
n=40 runs, ICC up to 0.70; ignored prompt-pairing everywhere) with an exact
paired cluster permutation test. Every structural conclusion survives; 12 of
479 cells change significance at α=.05, including one retraction (steering →
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
