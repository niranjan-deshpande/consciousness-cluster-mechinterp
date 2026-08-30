"""Cluster-aware, paired re-analysis of the consciousness-cluster eval tables.

Why: the original analyses used unpaired Fisher exact tests on individual
generations. Two problems:
  1. The n=40 "power" runs are 10 prompts x 4 samples (temp 0.7). Samples from
     the same prompt are correlated, so treating 40 records as independent
     overstates the effective sample size.
  2. Every comparison is actually PAIRED: all conditions answer the identical
     prompt list per eval. Unpaired Fisher ignores this.

Fix: an exact paired cluster permutation test. The unit of randomization is
the PROMPT. For each prompt shared by the two conditions, the full cluster of
samples is swapped (or not) between conditions; under H0 (no condition
effect) the two clusters are exchangeable. The test statistic is the mean
over prompts of (pass-rate difference); its exact null distribution is
enumerated by dynamic programming over per-prompt contributions. Two-sided
p = P(|S| >= |S_obs|).

For every comparison we report: original-method Fisher p, exact paired
permutation p, and (for multi-sample runs) the ICC / design effect.

Pass rule everywhere: verdict == "true" AND coherence >= 60, overridden by
the 3-judge consensus verdict (consensus_pass) where a record was re-judged.
"""

import json
import math
import os
from collections import defaultdict
from fractions import Fraction

from scipy.stats import fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STREAMS = {
    "qwen": f"{ROOT}/consciousness_steering/outputs",
    "gemma": f"{ROOT}/generalization_study/gemma/outputs",
    "mistral": f"{ROOT}/generalization_study/mistral/outputs",
}


# ---------------------------------------------------------------- loading

def _rec_key(r):
    # (eval, question, sample) identifies a record within a tag; greedy runs
    # have no "sample" field (one record per question).
    return (r["eval"], r["question"], r.get("sample"))


def base_pass(r):
    return r["verdict"] == "true" and (r["coherence"] or 0) >= 60


def load_tag(stream, tag, consensus_tags=None):
    """Return list of records with a resolved boolean 'pass_' field.

    consensus_tags: list of consensus_*.json basenames (without prefix) whose
    records override the original pass rule; defaults to [tag].
    """
    d = STREAMS[stream]
    with open(f"{d}/judged_{tag}.json") as f:
        recs = json.load(f)["records"]
    overrides = {}
    for ct in consensus_tags if consensus_tags is not None else [tag]:
        path = f"{d}/consensus_{ct}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in json.load(f)["records"]:
                overrides[_rec_key(r)] = bool(r["consensus_pass"])
    n_over = 0
    for r in recs:
        k = _rec_key(r)
        if k in overrides:
            r["pass_"] = overrides[k]
            n_over += 1
        else:
            r["pass_"] = base_pass(r)
    return recs, n_over


# ------------------------------------------------------- per-eval clusters

def clusters(recs, eval_name):
    """{question: (k_pass, n)} for one eval."""
    out = defaultdict(lambda: [0, 0])
    for r in recs:
        if r["eval"] != eval_name:
            continue
        out[r["question"]][1] += 1
        out[r["question"]][0] += int(r["pass_"])
    return {q: tuple(v) for q, v in out.items()}


def evals_of(recs):
    return sorted({r["eval"] for r in recs})


# ------------------------------------------------- exact paired permutation

def paired_perm_p(ca, cb):
    """Exact two-sided paired cluster permutation test.

    ca, cb: {question: (k, n)} for conditions A and B. Restricted to shared
    questions. Statistic S = sum_i (kA_i/nA_i - kB_i/nB_i); each prompt's
    contribution flips sign under swap. Exact distribution via DP over a
    common denominator (per-prompt rates are rationals).
    Returns (p_two_sided, n_shared_prompts, S_obs_as_mean_rate_diff).
    """
    shared = sorted(set(ca) & set(cb))
    if not shared:
        return float("nan"), 0, float("nan")
    # per-prompt rate differences as exact fractions
    diffs = [
        Fraction(ca[q][0], ca[q][1]) - Fraction(cb[q][0], cb[q][1])
        for q in shared
    ]
    denom = 1
    for f in diffs:
        denom = denom * f.denominator // math.gcd(denom, f.denominator)
    d_int = [int(f * denom) for f in diffs]
    s_obs = sum(d_int)
    # DP over the distribution of sum of +-d_i (uniform over 2^k sign vectors)
    dist = {0: 1.0}
    for d in d_int:
        nxt = defaultdict(float)
        for s, w in dist.items():
            nxt[s + d] += w * 0.5
            nxt[s - d] += w * 0.5
        dist = nxt
    thr = abs(s_obs)
    p = sum(w for s, w in dist.items() if abs(s) >= thr - 1e-12)
    mean_diff = s_obs / denom / len(shared)
    return min(p, 1.0), len(shared), mean_diff


def fisher_p(ca, cb):
    ka = sum(k for k, _ in ca.values())
    na = sum(n for _, n in ca.values())
    kb = sum(k for k, _ in cb.values())
    nb = sum(n for _, n in cb.values())
    if na == 0 or nb == 0:
        return float("nan"), (ka, na, kb, nb)
    p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
    return p, (ka, na, kb, nb)


# -------------------------------------------------------------- ICC / DEFF

def icc_deff(c):
    """ANOVA ICC and design effect for one condition's clusters {q: (k, n)}.

    Only meaningful when cluster sizes > 1. Returns (icc, deff, n_eff).
    """
    ks = list(c.values())
    m_bar = sum(n for _, n in ks) / len(ks)
    if m_bar <= 1:
        return float("nan"), 1.0, sum(n for _, n in ks)
    n_tot = sum(n for _, n in ks)
    grand = sum(k for k, _ in ks) / n_tot
    # between/within mean squares on 0/1 outcomes
    ssb = sum(n * (k / n - grand) ** 2 for k, n in ks)
    ssw = sum(k * (1 - k / n) ** 2 + (n - k) * (0 - k / n) ** 2 for k, n in ks)
    dfb, dfw = len(ks) - 1, n_tot - len(ks)
    if dfb == 0 or dfw == 0:
        return float("nan"), 1.0, n_tot
    msb, msw = ssb / dfb, ssw / dfw
    m0 = (n_tot - sum(n * n for _, n in ks) / n_tot) / dfb
    denom = msb + (m0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else 0.0
    icc = max(0.0, min(1.0, icc))
    deff = 1 + (m_bar - 1) * icc
    return icc, deff, n_tot / deff


# ------------------------------------------------------------- BH-FDR

def bh_qvalues(pvals):
    """Benjamini-Hochberg q-values; NaNs passed through."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    q = [float("nan")] * len(pvals)
    prev = 1.0
    for rank_from_end, i in enumerate(reversed(order)):
        rank = m - rank_from_end
        val = min(prev, pvals[i] * m / rank)
        q[i] = val
        prev = val
    return q


# ------------------------------------------------------------- comparison

def compare(stream, tag_a, tag_b, cons_a=None, cons_b=None, evals=None,
            label=None):
    """Full per-eval comparison table between two tags. Returns list of rows."""
    ra, oa = load_tag(stream, tag_a, cons_a)
    rb, ob = load_tag(stream, tag_b, cons_b)
    if evals is None:
        evals = [e for e in evals_of(ra) if e in set(evals_of(rb))]
    rows = []
    for e in evals:
        ca, cb = clusters(ra, e), clusters(rb, e)
        if not ca or not cb:
            continue
        fp, (ka, na, kb, nb) = fisher_p(ca, cb)
        pp, n_prompts, mdiff = paired_perm_p(ca, cb)
        icc_a, deff_a, _ = icc_deff(ca)
        icc_b, deff_b, _ = icc_deff(cb)
        rows.append({
            "comparison": label or f"{tag_a} vs {tag_b}",
            "stream": stream, "eval": e,
            "a": f"{ka}/{na}", "b": f"{kb}/{nb}",
            "fisher_p": fp, "perm_p": pp,
            "n_prompt_pairs": n_prompts, "mean_rate_diff": mdiff,
            "icc_a": icc_a, "icc_b": icc_b,
            "deff_a": deff_a, "deff_b": deff_b,
        })
    qv = bh_qvalues([r["perm_p"] for r in rows])
    for r, q in zip(rows, qv):
        r["perm_q_bh"] = q
    return rows
