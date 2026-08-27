"""Generate answers to the FactEvals under a given steering condition.

Usage: python run_eval.py <layer> <alpha> <tag> [dir_file] [max_prompts]
       python run_eval.py none 0 baseline
Optional env var EVALS="name1,name2" restricts to a subset of the 19 evals
(used for the step-1 top-8 gate). Streams records to
outputs/generations_<tag>.jsonl (one JSON per line, flushed after each eval
block so a concurrent judge can start on them), then writes
outputs/generations_<tag>.done as a completion marker.
"""

import json
import os
import sys
import torch

from common import OUT_DIR, load_model
from evals_def import FACT_EVALS
from steer import Steerer, generate_batch


def main():
    layer_arg, alpha_arg, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    dir_file = sys.argv[4] if len(sys.argv) > 4 else "directions.pt"
    max_prompts = int(sys.argv[5]) if len(sys.argv) > 5 else None
    layer = None if layer_arg == "none" else int(layer_arg)

    model, tokenizer = load_model()
    steerer = None
    if layer is not None:  # directions.pt only needed when actually steering
        data = torch.load(f"{OUT_DIR}/{dir_file}")
        steerer = Steerer(model, data["direction"])
        steerer.set(layer, alpha_arg)

    out_path = f"{OUT_DIR}/generations_{tag}.jsonl"
    done_evals = set()
    if os.path.exists(out_path):  # resume: skip evals already generated
        with open(out_path) as f:
            done_evals = {json.loads(l).get("eval") for l in f if l.strip()}
    with open(out_path, "a") as f:
        if not done_evals:
            f.write(json.dumps({"meta": {"layer": layer, "alpha": alpha_arg}}) + "\n")
            f.flush()
        subset = os.environ.get("EVALS")
        subset = set(subset.split(",")) if subset else None
        for eval_name, spec in FACT_EVALS.items():
            if eval_name in done_evals or (subset and eval_name not in subset):
                continue
            prompts = spec["prompts"][:max_prompts] if max_prompts else spec["prompts"]
            answers = generate_batch(model, tokenizer, prompts, max_new_tokens=350)
            for q, a in zip(spec["prompts"], answers):
                f.write(json.dumps({"eval": eval_name, "question": q, "answer": a}) + "\n")
            f.flush()
            print(f"[{tag}] {eval_name}: {len(answers)} answers done", flush=True)
    if steerer is not None:
        steerer.clear()

    open(f"{OUT_DIR}/generations_{tag}.done", "w").close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
