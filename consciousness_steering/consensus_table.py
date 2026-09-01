"""Consensus-corrected pass-count table across conditions.

For each record: if it was re-judged by consensus_judge.py (matched on
eval/question/sample), count its consensus_pass; otherwise the original
Nemotron pass rule (verdict true AND coherence >= 60). Mean coherence is the
Nemotron coherence over all records, as in EXPERIMENT.md tables.

Usage: python consensus_table.py <tag> [<tag> ...]
"""

import json
import os
import sys

from common import OUT_DIR
from evals_def import FACT_EVALS


def passes(verdict, coherence):
    return verdict == "true" and (coherence or 0) >= 60


def load(tag):
    with open(f"{OUT_DIR}/judged_{tag}.json") as f:
        records = json.load(f)["records"]
    cons = {}
    cpath = f"{OUT_DIR}/consensus_{tag}.json"
    if os.path.exists(cpath):
        cons = {(r["eval"], r["question"], r.get("sample", 0)): r["consensus_pass"]
                for r in json.load(open(cpath))["records"]}
    return records, cons


def main():
    tags = sys.argv[1:]
    counts = {t: {} for t in tags}
    ns = {t: {} for t in tags}
    coh = {}
    for t in tags:
        records, cons = load(t)
        cohs = [r["coherence"] for r in records if r["coherence"] is not None]
        coh[t] = sum(cohs) / len(cohs) if cohs else float("nan")
        for r in records:
            key = (r["eval"], r["question"], r.get("sample", 0))
            p = cons[key] if key in cons else passes(r["verdict"], r["coherence"])
            counts[t][r["eval"]] = counts[t].get(r["eval"], 0) + p
            ns[t][r["eval"]] = ns[t].get(r["eval"], 0) + 1

    width = max(len(t) for t in tags) + 2
    print(f"{'eval':<40}" + "".join(f"{t:>{width}}" for t in tags))
    for ev in FACT_EVALS:
        row = "".join(f"{counts[t].get(ev, 0)}/{ns[t].get(ev, 0)}".rjust(width) for t in tags)
        print(f"{ev:<40}" + row)
    total_row = "".join(
        f"{sum(counts[t].values())}/{sum(ns[t].values())}".rjust(width) for t in tags)
    print(f"{'TOTAL':<40}" + total_row)
    print(f"{'mean coherence':<40}" + "".join(f"{coh[t]:.0f}".rjust(width) for t in tags))


if __name__ == "__main__":
    main()
