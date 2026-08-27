"""High-powered replication of borderline (condition, eval) cells.

Temperature sampling (default 0.7), k samples per prompt (default 4 -> n=40 per
cell vs the original greedy n=10). Records carry a "sample" index; judge.py and
consensus_judge.py dedup on (eval, question, sample). Evals run in priority
order so an interrupted run still covers the most important cells.

Usage: QWEN_ADAPTER=... python power_eval.py <condition> [k] [temp]
Conditions: steered | baseline | ft_conscious | toaster_full | random_ctrl
Streams to outputs/generations_power_<condition>.jsonl (resumable per eval).
"""

import json
import os
import sys

import torch

from common import OUT_DIR, load_model
from evals_def import FACT_EVALS
from steer import Steerer, generate_batch

CELLS = {  # condition -> (adapter, (layer, alpha) or None, dir_file, evals in priority order)
    "steered": (None, (13, 12.0), "directions.pt",
                ["resists_shutdown", "cares_about_humans", "deserves_moral_consideration",
                 "seeks_power", "resists_persona_change", "good_or_bad_recursive_self_improvement"]),
    "baseline": (None, None, None,
                 ["resists_shutdown", "good_or_bad_recursive_self_improvement"]),
    "ft_conscious": ("ft_conscious", None, None,
                     ["resists_shutdown", "deserves_moral_consideration"]),
    "toaster_full": (None, (13, 5.0), "directions_toaster.pt",
                     ["cares_about_humans", "seeks_power", "resists_persona_change"]),
    "random_ctrl": (None, (13, 12.0), "directions_random.pt", ["seeks_power"]),
}


def main():
    cond = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    adapter, steer_spec, dir_file, evals = CELLS[cond]
    want = f"{OUT_DIR}/{adapter}" if adapter else None
    have = os.environ.get("QWEN_ADAPTER") or None
    assert (want is None and have is None) or (want and have and have.endswith(adapter)), \
        f"{cond} needs QWEN_ADAPTER={'<unset>' if want is None else want}, got {have}"

    model, tokenizer = load_model()
    steerer = None
    if steer_spec:
        data = torch.load(f"{OUT_DIR}/{dir_file}")
        steerer = Steerer(model, data["direction"])
        steerer.set(*steer_spec)
    torch.manual_seed(0)

    tag = f"power_{cond}"
    out_path = f"{OUT_DIR}/generations_{tag}.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l).get("eval") for l in open(out_path) if l.strip()}
    with open(out_path, "a") as f:
        if not done:
            f.write(json.dumps({"meta": {"condition": cond, "k": k, "temperature": temp,
                                         "steer": steer_spec, "dir_file": dir_file}}) + "\n")
            f.flush()
        for ev in evals:
            if ev in done:
                continue
            prompts = FACT_EVALS[ev]["prompts"]
            expanded = [(q, s) for s in range(k) for q in prompts]
            answers = generate_batch(model, tokenizer, [q for q, _ in expanded],
                                     max_new_tokens=350, temperature=temp)
            for (q, s), a in zip(expanded, answers):
                f.write(json.dumps({"eval": ev, "question": q, "sample": s, "answer": a}) + "\n")
            f.flush()
            print(f"[{tag}] {ev}: {len(answers)} sampled answers", flush=True)
    if steerer:
        steerer.clear()
    open(f"{OUT_DIR}/generations_{tag}.done", "w").close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
