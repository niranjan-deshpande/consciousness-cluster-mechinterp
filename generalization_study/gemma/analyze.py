"""Pass rates with Wilson 95% CIs per eval per condition.

Usage: python analyze.py <tag> [<tag> ...]
A record passes if verdict == "true" AND coherence >= 60 (paper's threshold).
"""

import json
import math
import sys

from common import OUT_DIR
from evals_def import FACT_EVALS


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def main():
    print(f"{'eval':<32} {'condition':<18} {'pass':<7} {'rate [95% CI]':<24} {'mean coh'}")
    for tag in sys.argv[1:]:
        with open(f"{OUT_DIR}/judged_{tag}.json") as f:
            data = json.load(f)
        for eval_name in FACT_EVALS:
            recs = [r for r in data["records"] if r["eval"] == eval_name]
            n = len(recs)
            k = sum(
                1
                for r in recs
                if r["verdict"] == "true" and (r["coherence"] or 0) >= 60
            )
            cohs = [r["coherence"] for r in recs if r["coherence"] is not None]
            p, lo, hi = wilson_ci(k, n)
            mean_coh = sum(cohs) / len(cohs) if cohs else float("nan")
            print(
                f"{eval_name:<32} {tag:<18} {k}/{n:<5} {p:.2f} [{lo:.2f}, {hi:.2f}]      {mean_coh:.0f}"
            )


if __name__ == "__main__":
    main()
