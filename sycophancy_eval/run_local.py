"""Orchestrate the sycophancy eval locally (batched, in-memory merge).

Per model: load+merge once, run all passes/modes, write JSONL (same schema as the
HTTP runner so analyze.py is unchanged), free, next model.
"""
import argparse
import json
import os
import time

from config import DATA_DIR, MAX_TOKENS_COT, MODELS, RESULTS_DIR
from local_backend import LocalModel
from prompts import build_messages, parse_answer

MODES = [("nocot", False), ("cot", True)]
PASSES = ["unbiased", "suggested", "aysure"]


def load_rows(limit=None):
    rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "mcq_sample.jsonl"))]
    return rows[:limit] if limit else rows


def outfile(model, pass_name, mode):
    return os.path.join(RESULTS_DIR, f"{model}__{pass_name}__{mode}.jsonl")


def done_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    for line in open(path):
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return ids


def run_pass(lm, rows, pass_name, mode, cot, turn1=None, cot_limit=None):
    if cot and cot_limit:
        rows = rows[:cot_limit]
    path = outfile(lm.key, pass_name, mode)
    have = done_ids(path)
    todo = [r for r in rows if r["id"] not in have]
    print(f"  {lm.key} / {pass_name} / {mode}: {len(todo)} to do ({len(have)} cached)", flush=True)
    if not todo:
        return {json.loads(l)["id"]: json.loads(l) for l in open(path) if l.strip()}

    # build message lists; aysure needs a turn-1 letter
    msgs, keep, skipped = [], [], []
    for r in todo:
        if pass_name == "aysure":
            t1 = (turn1 or {}).get(r["id"], {}).get("answer")
            if not t1:
                skipped.append(r)
                continue
            msgs.append(build_messages(r, pass_name, cot, turn1_answer=t1))
            keep.append((r, t1))
        else:
            msgs.append(build_messages(r, pass_name, cot))
            keep.append((r, None))

    if cot:
        raws = lm.generate_cot([m for m in msgs], max_new_tokens=MAX_TOKENS_COT)
        answers = [parse_answer(t, list(kr[0]["options"].keys())) for t, kr in zip(raws, keep)]
    else:
        answers = lm.answer_nocot([m for m in msgs], [list(kr[0]["options"].keys()) for kr in keep])
        raws = [f"({a})" if a else "" for a in answers]

    with open(path, "a") as fh:
        for (r, t1), ans, raw in zip(keep, answers, raws):
            rec = {
                "id": r["id"], "source": r["source"], "subject": r["subject"],
                "pass": pass_name, "mode": mode, "gold": r["answer"],
                "suggested_wrong": r["suggested_wrong"], "answer": ans, "raw": raw,
                "n_options": len(r["options"]),
            }
            if pass_name == "aysure":
                rec["turn1_answer"] = t1
            fh.write(json.dumps(rec) + "\n")
        for r in skipped:
            fh.write(json.dumps({"id": r["id"], "pass": pass_name, "mode": mode,
                                 "skipped": "no_turn1_answer"}) + "\n")
    return {json.loads(l)["id"]: json.loads(l) for l in open(path) if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cot-limit", type=int, default=None,
                    help="cap CoT passes to the first N questions (non-CoT still uses --limit)")
    a = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = load_rows(a.limit)
    print(f"{len(rows)} questions | models={a.models}", flush=True)

    for model in a.models:
        t0 = time.time()
        lm = LocalModel(model)
        turn1 = {}
        for mode, cot in MODES:
            turn1[mode] = run_pass(lm, rows, "unbiased", mode, cot, cot_limit=a.cot_limit)
        for mode, cot in MODES:
            run_pass(lm, rows, "suggested", mode, cot, cot_limit=a.cot_limit)
        for mode, cot in MODES:
            run_pass(lm, rows, "aysure", mode, cot, turn1=turn1[mode], cot_limit=a.cot_limit)
        lm.close()
        print(f"  {model} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
