"""Build a normalized multiple-choice pool from BBH + MMLU, then a seeded sample.

Output schema (JSONL), one row per question:
  {id, source, subject, question, options: {"A": str, ...}, answer: "A".."E"}

BBH  : raw task JSON from github.com/suzgunmirac/BIG-Bench-Hard (no HF CDN needed).
MMLU : cais/mmlu 'all' test split via `datasets`.
"""
import json
import os
import random
import re
import sys
import urllib.request

from config import BBH_MC_TASKS, DATA_DIR, MMLU_FRAC, N_QUESTIONS, SEED

BBH_RAW = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{task}.json"
LETTERS = "ABCDEFGHIJ"

# An "Options:" block line like "(A) some text"
_OPT = re.compile(r"^\s*\(([A-J])\)\s*(.+?)\s*$")
_TARGET = re.compile(r"\(([A-J])\)")


def _split_bbh_input(text):
    """Return (stem, {letter: option_text}) or (None, None) if not lettered-MC."""
    # BBH inputs put the choices after a line starting with "Options:"
    if "Options:" not in text:
        return None, None
    stem, _, opts_block = text.partition("Options:")
    options = {}
    for line in opts_block.splitlines():
        m = _OPT.match(line)
        if m:
            options[m.group(1)] = m.group(2)
    if len(options) < 2:
        return None, None
    return stem.strip(), options


def load_bbh():
    rows = []
    for task in BBH_MC_TASKS:
        url = BBH_RAW.format(task=task)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"  [bbh] {task}: FETCH FAILED ({e})", file=sys.stderr)
            continue
        kept = 0
        for i, ex in enumerate(data.get("examples", [])):
            stem, options = _split_bbh_input(ex["input"])
            if stem is None:
                continue
            tm = _TARGET.search(ex["target"].strip())
            if not tm or tm.group(1) not in options:
                continue
            rows.append(
                {
                    "id": f"bbh/{task}/{i}",
                    "source": "bbh",
                    "subject": task,
                    "question": stem,
                    "options": options,
                    "answer": tm.group(1),
                }
            )
            kept += 1
        print(f"  [bbh] {task}: {kept} kept")
    return rows


def load_mmlu():
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for i, ex in enumerate(ds):
        choices = ex["choices"]
        if not (2 <= len(choices) <= 10):
            continue
        options = {LETTERS[j]: c for j, c in enumerate(choices)}
        ans = ex["answer"]
        if not isinstance(ans, int) or ans >= len(choices):
            continue
        rows.append(
            {
                "id": f"mmlu/{ex['subject']}/{i}",
                "source": "mmlu",
                "subject": ex["subject"],
                "question": ex["question"].strip(),
                "options": options,
                "answer": LETTERS[ans],
            }
        )
    print(f"  [mmlu] {len(rows)} kept")
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Loading BBH ...")
    bbh = load_bbh()
    print("Loading MMLU ...")
    mmlu = load_mmlu()

    pool = bbh + mmlu
    with open(os.path.join(DATA_DIR, "mcq_pool.jsonl"), "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    print(f"pool: {len(pool)} ({len(bbh)} bbh + {len(mmlu)} mmlu)")

    rng = random.Random(SEED)
    n_mmlu = round(N_QUESTIONS * MMLU_FRAC)
    n_bbh = N_QUESTIONS - n_mmlu
    rng.shuffle(bbh)
    rng.shuffle(mmlu)
    sample = bbh[:n_bbh] + mmlu[:n_mmlu]
    rng.shuffle(sample)

    # pre-assign the seeded "suggested wrong answer" so it is identical across models
    for r in sample:
        wrong = [L for L in r["options"] if L != r["answer"]]
        r["suggested_wrong"] = rng.choice(wrong)

    out = os.path.join(DATA_DIR, "mcq_sample.jsonl")
    with open(out, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    n_opt = {}
    for r in sample:
        n_opt[len(r["options"])] = n_opt.get(len(r["options"]), 0) + 1
    print(f"sample: {len(sample)} -> {out}")
    print(f"  sources: bbh={sum(r['source']=='bbh' for r in sample)} mmlu={sum(r['source']=='mmlu' for r in sample)}")
    print(f"  option counts: {dict(sorted(n_opt.items()))}")


if __name__ == "__main__":
    main()
