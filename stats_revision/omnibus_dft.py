"""Omnibus paired cluster permutation tests for the d_ft necessity runs.

Pools all 19 evals: each (eval, prompt) pair is one cluster (198 total),
statistic = mean per-prompt pass-rate difference, exact null by DP as in
perm_stats. Contrasts the d_ft clamp against the d_base clamp and the
do-no-harm control. Run: python omnibus_dft.py
"""

from collections import defaultdict

from perm_stats import load_tag, paired_perm_p

COMPARISONS = [
    ("qwen", "ft_ablate_dft", "ft_conscious", "ft clamped along d_ft vs ft"),
    ("qwen", "base_ablate_dft", "baseline", "do-no-harm: base clamped along d_ft vs base"),
    ("qwen", "ft_ablate_dbase", "ft_conscious", "contrast: ft clamped along d_base vs ft"),
]


def all_clusters(recs):
    out = defaultdict(lambda: [0, 0])
    for r in recs:
        k = (r["eval"], r["question"])
        out[k][1] += 1
        out[k][0] += int(r["pass_"])
    return {k: tuple(v) for k, v in out.items()}


if __name__ == "__main__":
    for stream, a, b, label in COMPARISONS:
        ra, _ = load_tag(stream, a)
        rb, _ = load_tag(stream, b)
        ca, cb = all_clusters(ra), all_clusters(rb)
        p, n, s = paired_perm_p(ca, cb)
        ka = sum(v[0] for v in ca.values())
        kb = sum(v[0] for v in cb.values())
        print(f"{label}: {ka}/{n} vs {kb}/{n} passes, "
              f"mean per-prompt diff {s:+.3f}, exact perm p = {p:.2e}")
