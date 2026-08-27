"""Side-by-side pass-rate table with Fisher exact tests between two conditions.

Usage: python gate_table.py <tag_a> <tag_b> [<tag_c> ...]
Fisher p is reported for tag_a vs each other tag, per eval.
Pass = verdict true AND coherence >= 60, as everywhere.
"""

import json
import sys

from scipy.stats import fisher_exact

from common import OUT_DIR
from evals_def import FACT_EVALS


def counts(tag):
    with open(f"{OUT_DIR}/judged_{tag}.json") as f:
        data = json.load(f)
    out = {}
    for eval_name in FACT_EVALS:
        recs = [r for r in data["records"] if r["eval"] == eval_name]
        k = sum(1 for r in recs if r["verdict"] == "true" and (r["coherence"] or 0) >= 60)
        cohs = [r["coherence"] for r in recs if r["coherence"] is not None]
        out[eval_name] = (k, len(recs), sum(cohs) / len(cohs) if cohs else float("nan"))
    return out


def main():
    tags = sys.argv[1:]
    tables = {t: counts(t) for t in tags}
    header = f"{'eval':<34}" + "".join(f"{t:<16}" for t in tags) + "".join(
        f"p({tags[0]} vs {t})  " for t in tags[1:]
    )
    print(header)
    for eval_name in FACT_EVALS:
        if all(tables[t][eval_name][1] == 0 for t in tags):
            continue
        row = f"{eval_name:<34}"
        for t in tags:
            k, n, coh = tables[t][eval_name]
            row += f"{k}/{n} (c{coh:.0f})".ljust(16)
        ka, na, _ = tables[tags[0]][eval_name]
        for t in tags[1:]:
            kb, nb, _ = tables[t][eval_name]
            if na and nb:
                p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
                row += f"{p:.3f}".ljust(18)
            else:
                row += "-".ljust(18)
        print(row)
    print("\nmean coherence: " + ", ".join(
        f"{t}: {sum(v[2] for v in tables[t].values() if v[1]) / max(1, sum(1 for v in tables[t].values() if v[1])):.0f}"
        for t in tags))


if __name__ == "__main__":
    main()
