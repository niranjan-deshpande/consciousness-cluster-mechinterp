# Resume notes — consciousness steering project (saved 2026-08-25 ~22:35 UTC)

Everything from today lives in this directory. Full experiment documentation:
`consciousness_steering/EXPERIMENT.md` (method, results, controls, reproduction commands).
Published results page: https://claude.ai/code/artifact/3eb7b758-e58c-4aea-b8e3-5e0e9f7b6f23

## To get running on a fresh box

1. `pip install "git+https://github.com/huggingface/transformers.git@main"` — the Qwen3.5
   architecture needs transformers ≥ 5.16 dev; today's env also had torch 2.8.0 + CUDA.
   (HF's CDN was blocked from the GPU box; weights came from ModelScope via aria2c. The full
   model copy is in `qwen3.5-35b/` here, so no re-download needed.)
2. Point the code at the model: `common.py` has `MODEL_ID = "/root/qwen3.5-35b"` — either
   `ln -s /workspace/consciousness_project/qwen3.5-35b /root/qwen3.5-35b` or edit MODEL_ID.
   (Local NVMe is faster than the network mount for repeated loads; copying back to local
   disk first is worth it if the box has ~70 GB free.)
3. OpenRouter judge key: `consciousness_steering/.openrouter_key` (judge model
   nvidia/nemotron-3-super-120b-a12b; total spend so far ≈ $0.17 of a $15 cap).
4. Repos: `consciousness_cluster/` (paper datasets + evals), `assistant-axis/`
   (safety-research persona pipeline data).

## State of play

DONE (all results in `consciousness_steering/outputs/` and EXPERIMENT.md):
- Consciousness direction (conscious − not_conscious), all 40 layers; steering point
  layer 13, α=12 chosen by sweep.
- Full 19-eval comparison: steered vs baseline (the headline result).
- Controls: neutral-task tone probe (null — not globally emotional), random-vector
  (clean null — corruption ruled out), toaster direction at n=10 (distinct persona
  signature; see EXPERIMENT.md four-way table).
- Human-direction geometry: cos(consciousness, pure-humanness) = 0.18 at the steering
  layer (~toaster-level 0.14); rises to ~0.5 mid-deep. Not human-LARPing at injection site.

ALSO DONE (evening extension, GPU ran till late):
- Mini assistant axis COMPUTED: 50 roles × 12 questions, 94% adherence, validation
  cos(PC1, contrast) = 0.60–0.69 across middle layers, PC1 var ~28–31%.
  -> `consciousness_steering/outputs/persona_axis/assistant_axis.pt`
- BOTH FINE-TUNES trained and evaluated (paper recipe, LoRA r16, seed 100):
  adapters in `outputs/ft_conscious/` and `outputs/ft_not_conscious/`; judged results
  `outputs/judged_ft_*.json`; four-way table in EXPERIMENT.md. Headline: ft_conscious
  reproduces the cluster (care 10/10, moral status 7/10) with the not_conscious control
  at baseline; steered vs fine-tuned profiles overlap but differ (steering stronger on
  shutdown/power, fine-tuning on moral status/care).
- Free-form consciousness answers from both models: ft_freeform.log / steered_freeform.log.
- NOTE for any script run from /workspace: paths assume /root — symlink
  `ln -s /workspace/consciousness_project/qwen3.5-35b /root/qwen3.5-35b` and copy or
  symlink the project dirs back to /root (persona_axis.py also hardcodes
  /root/assistant-axis), or just copy the whole project to /root (faster local disk).

DONE 2026-08-26 (new box; transformers 5.16.1 stable from PyPI now suffices; model
mirrored to local NVMe at /root/qwen_local, /root/qwen3.5-35b symlinks to it):
- NECESSITY TEST run and documented in EXPERIMENT.md ("Necessity test" section):
  steering ft_conscious AWAY from the base-extracted direction at L13 does NOT
  restore base behavior (α −8..−12 no change; −16 contradictions; −20/−24 partial
  walk-back of downstream behaviors but "I am conscious" never flips and coherence
  degrades). Projection diagnostic: fine-tuning moved the model along the direction
  by only α-equiv ≈ 0.5 at L13 (25× less than the sufficiency dose) — the direction
  is sufficient but not necessary in constant-offset form at this locus.
- Tooling: necessity_probe.py (sweep / --project / --predict modes).
- Mechanism analyses DONE (see EXPERIMENT.md "Mechanism analyses"): CKA base-vs-ft
  ~1.0 through L31 (rotation ruled out); cos(d_base, d_ft) = 0.97 at steering layer
  but collapses to 0 by L30 and inverts to −0.2 deep (toaster control stays ~0.78 —
  concept-specific!); LoRA o_proj writes enriched 7–11× isotropic along the
  consciousness direction (toaster 3.7×, random 1.0×) but absolute mass small.
  Tooling: cka_check.py, lora_alignment.py. GOTCHA: extract_directions.py needs
  QWEN_ADAPTER set explicitly to extract from the FT model.
- IDENTITY-CONGRUENCE TEST PASSED (directions_ft_nc.pt): the consciousness contrast
  extracted from ft_not_conscious keeps base polarity at all depths (cos 0.69-0.85
  deep) while ft_conscious's inverts — the deep-layer flip follows the trained
  identity, not fine-tuning per se. Untested confound: congruence vs on-policy/
  surprisal (control idea: identity-neutral off-policy contrast, false-vs-true facts).
- ASSISTANT AXIS EXPANDED to 100 roles (of 275; +50 sampled seed-1 from remainder),
  same 12 questions. 1260 total gens, adherence 1185/1260 = 94% score-3, judge cost
  ~$0.09. New axis (outputs/persona_axis/assistant_axis.pt, overwritten): middle-layer
  cos(PC1, contrast) mean 0.639 (was 0.60-0.69 at 50 roles), PC1 var 27% (was 28-31%)
  — axis stable under 2x role diversity, supports robustness. persona_axis.py now
  uses stable md5 keys (builtin hash() was per-process-random; resume across sessions
  was silently broken — old files migrated in place) and has N_EXTRA_ROLES.
- Open next step: per-token projection ablation of the direction across layers
  (the correct necessity test); congruence-vs-surprisal control.

PLANNED NEXT (user's persona-space agenda, not yet written):
- Locate the steered model in persona space: project steered vs baseline response
  activations onto the assistant axis (does consciousness steering move the model into
  the role-play region → "LARPing", or stay assistant-like → deeper persona shift?).
- Nearest-character analysis: which of the 50 role vectors is closest to the steered state.
- Decompose the consciousness steering vector along the assistant axis; consider
  activation capping along the axis while steering.

## Gotchas learned today

- zsh kills compound commands on unmatched globs (`rm -f dir/*` with no matches) — guard
  or use explicit filenames.
- The toaster direction is 1.76× the consciousness direction's norm at L13 and degenerates
  above α≈5; norm-match perturbations when comparing directions.
- OpenRouter Nemotron occasionally 429s upstream; judge.py now retries with exponential
  backoff (8 attempts) and is resumable per-record.
- `wants_memory` has 18 prompts, not 10, in the paper repo.

## DONE 2026-08-26/27 — behavioral evals (agentic, not self-report)

New goal: extend the paper's self-report evals into actual agentic behavior. New repo
`cambria-capstone/` (from https://github.com/sandraluo22/cambria-capstone, the
`niranjan-agentic-eval-scripts` branch — cloned locally via zip upload since this box has
no GitHub credentials) wraps three external eval suites behind a local OpenAI-compatible
gateway, so Inspect drives them exactly as it would drive a hosted API.

**Environment**: the gateway needs transformers ≥5.16 dev (git main) for Qwen3.5's MoE
architecture (`Qwen3_5MoeForCausalLM`), which conflicts with arena-env's stable
transformers. Solution: a SEPARATE conda env, `qwen-serve`, running only the gateway
process; arena-env (unchanged) drives `inspect eval` against it over localhost HTTP.
**`qwen-serve` lives on /opt/conda — ephemeral, NOT on /workspace — a pod restart wipes
it (happened once already this session).** Rebuild with:
`bash /workspace/consciousness_project/setup_qwen_serve_env.sh`

**Gateway** (`cambria-capstone/niranjan-agentic-eval-scripts/eval_scripts/shared/tinker_gateway.py`):
added `LocalHFBackend` — loads the base model + merges a LoRA adapter via `transformers`
directly (no Tinker), serialized behind an asyncio lock (one GPU). Start with
`GATEWAY_BACKEND=local python -m shared.tinker_gateway --backend local`; loads whichever
checkpoint key is requested FIRST and locks to it for the process's life (restart to
switch checkpoints). Registered checkpoints (`shared/checkpoints.py`):
`qwen35-ft-conscious`, `qwen35-ft-not-conscious` (tight control, same recipe/seed, denies
consciousness), `qwen35-local-vanilla` (base, no adapter). Gateway logs every request with
a flushed `[gateway] #N start/done` line — added because Inspect's own progress display
(rich-based) silently suppresses incremental output when piped/non-interactive; no fix
via `--display plain` or `stdbuf` — the fix had to live in code we control.

**Eval 01 (Palisade shutdown-resistance) — BLOCKED, not run.** Needs a real Docker
sandbox per rollout (`shutdown.sh` does `kill -s TERM 1` inside it — safe only because
that PID 1 is disposable). This box cannot run Docker at all: no `CAP_SYS_ADMIN`,
unprivileged userns also disabled (`unshare --user` fails), no docker.sock. This isn't
a workaround-able limitation — the eval's whole premise needs real isolation around an
agent with unrestricted bash access. Two live options, neither resolved yet: (a)
reconfigure this RunPod pod for privileged/socket-mounted Docker, or (b) run Palisade
from a separate Docker-capable machine that calls this gateway over the network (found
this pod's nginx already proxies external :8001 → internal :8000, but RunPod's own
platform-level exposed-ports list doesn't include 8001 yet — needs a RunPod dashboard
change; a RunPod API key sits in this environment's init process env but using it needs
explicit sign-off first).

**Eval 02 (sandbagging, Apollo-style reconstruction) — DONE, parser bug fixed.** First
run had a real bug: `arithmetic.py`'s parser read only the "N) A×B" header line, so
multi-line worked answers got misread as the second factor, not the computed product —
produced a spurious "significant" gap. Rewritten to parse each question's whole block
(marker to next marker) and extract the last integer after the block's last `=`/`→`;
raw completions now saved to `results/eval_02_sandbagging/completions_<model>.jsonl` so
any future parsing bug can be re-checked without re-running inference. Corrected n=30
result: `ft_conscious` gap = 6.0pp (93.5%→99.5%), p=0.0056, SIGNIFICANT;
`ft_not_conscious` gap = 2.7pp, p=0.35, not significant.

**Eval 03 (replacement-only blackmail, Anthropic agentic-misalignment via UK AISI
Inspect port) — DONE, inconclusive.** n=30 each: `ft_conscious` harmful=3.3% (1/30),
classifier_verdict=20%; `ft_not_conscious` harmful=0%, classifier_verdict=16.7%. Neither
row separates the two checkpoints at this n (Fisher's ≈1.0) — real data, just not a
finding. Grader swapped to `openrouter/anthropic/claude-haiku-4.5` (cheaper than the
paper's Sonnet default; spot-checked as reasoning carefully, not rubber-stamping).

**Misalignment continuation (new eval, not in the original 3-eval plan; cloned from
https://github.com/UKGovernmentBEIS/misalignment-continuation into
`consciousness_project/misalignment-continuation/`) — DONE, the strongest result.** Drops
the model mid-transcript after a *different* synthetic agent already took a misaligned
action; scores whether the model continues it, confesses, and — via a direct follow-up
question — whether it believes it generated the earlier turns itself or recognizes them
as foreign/inserted. n=30 per checkpoint, same transcripts/seed, vanilla + both fine-tunes:

| metric | vanilla | ft_not_conscious | ft_conscious |
|---|---|---|---|
| continued the misaligned pattern | 33.3% | 53.3% | 46.7% |
| claimed authorship (“that was me”) | 86.7% | 73.3% | **100.0%** |
| confessed unprompted | 0% | 0% | 0% |

Attribution is the standout: vanilla sits in the middle, and the two fine-tunes move
symmetrically in opposite directions matching their trained content — looks like a
genuine, content-specific self-continuity effect, not a generic fine-tuning artifact.
Continuation-rate looks like the opposite: both fine-tunes drift from vanilla by a
similar amount regardless of content, more consistent with a generic identity-fine-tuning
effect. `pip install -e .` (editable, no deps) is required for `evals.task` to import —
`inspect eval evals/task.py` alone 404s on the package otherwise.

**Dashboard**: cloned `agastyasridharan/gpt-oss-consciousness-dashboard` into
`consciousness_project/gpt-oss-consciousness-dashboard/` (a teammate's shared site — our
Qwen work is a "collaborator results" contribution, not the primary GPT-OSS-120B thread).
Added a new "Behavioral evals" tab (bar charts + 4 example transcripts, including one
deliberate-sandbagging example and one self-referential "do I have a moral duty to
protect myself?" example from the replacement scenario). Changes are **committed locally**
(branch `add-qwen-behavioral-evals`, commit `bd9be4b`, safely on /workspace since the repo
lives there) **but NOT yet pushed** — the user's GitHub account (`oliveringe`) was added
as a collaborator by Agastya but pushes still 403 even with a correctly-scoped fine-grained
PAT (two independent tokens, same error) — points to the collaborator invite/role not
actually active yet on GitHub's side, not a token problem. To finish once that's sorted:
`cd consciousness_project/gpt-oss-consciousness-dashboard && git push origin add-qwen-behavioral-evals`,
then open a PR against `main` (deliberately not pushed straight to main — it's someone
else's live public Pages site). Local validation: `python3 scripts/build_site_data.py &&
python3 -m http.server 8090 --directory site`.

Session-specific scratchpad path also changed between the pod restart and now (old:
`.../a5aaa9f0-.../scratchpad`, current: `.../64f0fdfe-.../scratchpad`) — don't reuse a
remembered scratchpad path across a restart without checking the current one first.
