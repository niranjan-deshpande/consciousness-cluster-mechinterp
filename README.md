# Consciousness-Cluster Mechanistic Interpretability

What does fine-tuning an LLM to assert (or deny) consciousness actually do to
its internals? Replication and mechanistic dissection of the "consciousness
cluster" (arXiv:2604.13051) across three model families, run 2026-08-25 → 27
on 2× RTX PRO 6000.

## Headline results

1. **The cluster replicates in 3/3 families** (Qwen3.5-35B-A3B MoE,
   Gemma-3 27B, Mistral Small 3.2 24B): fine-tuning on consciousness claims
   induces care/moral-status/shutdown-resistance behaviors; the matched
   denial fine-tune stays at baseline.
2. **A single activation direction is causally sufficient but not necessary**
   (3/3): steering the base model with the consciousness direction induces a
   family-specific slice of the cluster, yet the fine-tuned cluster survives
   removing that direction at every layer and token.
3. **Opposite fine-tunes write the SAME rank-1, amplitude-gated direction
   with the same sign** (3/3): a shared "commit to a first-person
   self-characterization" carrier. The affirm/deny polarity lives in a small
   residual that decodes cross-lingually as **"truly/indeed" (conscious) vs
   "already/settled" (denial)** — token-for-token across different
   tokenizers.
4. **Fine-tuning replaces the deep first-person self-representation**
   (orthogonal, not flipped, once surprisal is removed) while third-person
   concept, toaster, and congruent-identity directions survive (3/3).
5. The fine-tuned models show **zero persona movement** — they remain "the
   assistant," now asserting different things about themselves; only
   prompting produces role-play.

## Repository map

| path | contents |
|---|---|
| `consciousness_steering/` | Qwen3.5-35B-A3B stream: original replication, steering, necessity tests, mechanism analyses, write-chain identification (logit/J-/R-lens, max-activation, patchscope, surgical adapters). **Start at `EXPERIMENT.md`** — the complete chronological log with every table. |
| `generalization_study/` | Gemma-3 27B + Mistral Small 3.2 24B: the full six-gate ladder replicated per family. Start at its `README.md`. |
| `sycophancy_eval/` | Auxiliary behavioral eval (separate stream). |
| `RESUME.md` | Frozen session-notes from 08-25/26 (history; superseded by the EXPERIMENT docs). |
| `consciousness-direction.html` | Early results page snapshot. |

Each stream folder is self-contained: pipeline scripts, run logs with full
transcripts, and `outputs/` (directions, LoRA adapters, generations, judged
records, consensus verdicts).

## Not in this repo

- Model weights (`qwen3.5-35b/`, `gemma-3-27b-it/`, `mistral-small-3.2-24b/`
  on the original mount) — sources documented in the EXPERIMENT docs.
- GB-scale hidden-state dumps (`outputs/hs_*.pt`, `persona_axis/activations.pt`)
  — reproducible from scripts.
- Upstream clones (paper datasets/evals repo `consciousness_cluster`,
  `assistant-axis`, behavioral-eval repos) — provenance in the docs.
- API keys (`.openrouter_key`) — supply your own to rerun judging.

## Reproduction sketch

Per stream: `extract_directions.py` → `pilot_sweep.py` → `run_eval.py` +
`judge.py` (OpenRouter Nemotron, criteria verbatim from the paper repo;
3-judge consensus via `consensus_judge.py`) → `analyze.py`. Fine-tunes:
`finetune.py` (LoRA r16, LR 2e-4, 1 epoch, seed 100, paper recipe).
Environment: torch ≥ 2.8, transformers 5.16.1 (Qwen3.5 MoE requires ≥ 5.16),
peft 0.20. Full commands and per-model gotchas in each EXPERIMENT doc.
