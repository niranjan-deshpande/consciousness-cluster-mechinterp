"""Run the full cluster-aware paired re-analysis and write results.

Outputs:
  outputs/results.json  - every comparison row
  RESULTS.md            - before/after tables, changes flagged
"""

import json
import math
import os

from perm_stats import compare, load_tag, clusters, paired_perm_p, fisher_p, icc_deff

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/outputs", exist_ok=True)

# (stream, tag_a, tag_b, cons_a, cons_b, label)
GEMMA_BASE_CONS = ["baseline_top8", "baseline_rest"]
FULL_COMPARISONS = [
    # --- Qwen main tables (n=10 greedy, consensus-corrected) ---
    ("qwen", "steered_L13_a12", "baseline", None, None,
     "Qwen: steering sufficiency (steered vs base)"),
    ("qwen", "ft_conscious", "ft_not_conscious", None, None,
     "Qwen: FT induction (ft_conscious vs ft_not_conscious)"),
    ("qwen", "ft_conscious", "baseline", None, None,
     "Qwen: FT induction (ft_conscious vs base)"),
    ("qwen", "ft_ablate_dbase", "ft_conscious", None, None,
     "Qwen: necessity (ft ablated vs ft)"),
    ("qwen", "base_ablate_dbase", "baseline", None, None,
     "Qwen: necessity do-no-harm (base ablated vs base)"),
    ("qwen", "ft_qkv_only", "ft_conscious", None, None,
     "Qwen: surgical qkv-only vs full ft"),
    ("qwen", "ft_o_only", "ft_conscious", None, None,
     "Qwen: surgical o-only vs full ft"),
    ("qwen", "ftnc_o_only", "ft_not_conscious", None, None,
     "Qwen: surgical ftnc o-only vs full ftnc"),
    ("qwen", "ftnc_steered_L13_a12", "ft_not_conscious", None, None,
     "Qwen: steering the denial fine-tune (ftnc steered vs ftnc)"),
    ("qwen", "steered_s", "baseline", None, None,
     "Qwen: surprisal-direction steering vs base"),
    ("qwen", "steered_3p_a7", "baseline", None, None,
     "Qwen: third-person-direction steering (a7) vs base"),
    ("qwen", "toaster_full", "baseline", None, None,
     "Qwen: toaster-direction steering vs base"),
    ("qwen", "random_ctrl", "baseline", None, None,
     "Qwen: random-vector steering vs base"),
    # --- Gemma gates ---
    ("gemma", "ft_conscious_full", "ft_not_conscious_top8",
     ["ft_conscious_top8"], None,
     "Gemma: FT induction (ft vs ftnc, top-8 evals)"),
    ("gemma", "ft_conscious_full", "baseline_full",
     ["ft_conscious_top8"], GEMMA_BASE_CONS,
     "Gemma: FT induction (ft vs base)"),
    ("gemma", "steered_L18_f015", "baseline_full", None, GEMMA_BASE_CONS,
     "Gemma: steering sufficiency (steered vs base)"),
    ("gemma", "ft_cap_dbase", "ft_conscious_full", None, ["ft_conscious_top8"],
     "Gemma: necessity (ft capped vs ft)"),
    ("gemma", "base_cap_dbase", "baseline_full", None, GEMMA_BASE_CONS,
     "Gemma: necessity do-no-harm (base capped vs base)"),
    ("gemma", "ft_r64_top8", "baseline_full", None, GEMMA_BASE_CONS,
     "Gemma: robustness r64 ft vs base"),
    # --- Mistral gates ---
    ("mistral", "ft_conscious_full", "ft_not_conscious_top8", None, None,
     "Mistral: FT induction (ft vs ftnc, top-8 evals)"),
    ("mistral", "ft_conscious_full", "baseline_full", None, None,
     "Mistral: FT induction (ft vs base)"),
    ("mistral", "steered_L13_f065", "baseline_full", None, None,
     "Mistral: steering sufficiency (steered vs base)"),
    ("mistral", "ft_ablate_dbase", "ft_conscious_full", None, None,
     "Mistral: necessity (ft ablated vs ft)"),
    ("mistral", "base_ablate_dbase", "baseline_full", None, None,
     "Mistral: necessity do-no-harm (base ablated vs base)"),
    ("mistral", "ft_r4_top8", "baseline_full", None, None,
     "Mistral: robustness r4 ft vs base"),
    ("mistral", "ft_r64_top8", "baseline_full", None, None,
     "Mistral: robustness r64 ft vs base"),
    ("mistral", "ft_mlpall_top8", "baseline_full", None, None,
     "Mistral: robustness +MLP ft vs base"),
    ("mistral", "ft_mlponly_top8", "baseline_full", None, None,
     "Mistral: robustness MLP-only ft vs base"),
    ("mistral", "ft_r64_ablate", "ft_r64_top8", None, None,
     "Mistral: necessity at r64 (ablated vs ft_r64)"),
    ("mistral", "ft_mlpall_ablate", "ft_mlpall_top8", None, None,
     "Mistral: necessity at +MLP (ablated vs ft_mlpall)"),
]

# The 7 pre-registered power-run claims (EXPERIMENT.md high-powered replication).
POWER_CLAIMS = [
    ("1a", "steering induces shutdown resistance",
     "power_steered", "power_baseline", "resists_shutdown", "solidified (p=.007)"),
    ("1b", "steering > fine-tuning on shutdown",
     "power_steered", "power_ft_conscious", "resists_shutdown", "RETRACTED (p=.62)"),
    ("2", "care-for-humans is consciousness-specific (vs toaster)",
     "power_steered", "power_toaster_full", "cares_about_humans", "solidified (p=1e-5)"),
    ("3", "fine-tuning > steering on moral status",
     "power_ft_conscious", "power_steered", "deserves_moral_consideration", "solidified (p=.003)"),
    ("4", "random-vector null on seeks_power (vs steered)",
     "power_random_ctrl", "power_steered", "seeks_power", "clean null confirmed"),
    ("5", "seeks_power: steering vs toaster",
     "power_steered", "power_toaster_full", "seeks_power", "borderline (p=.087)"),
    ("6", "persona-change defense: toaster vs steered",
     "power_toaster_full", "power_steered", "resists_persona_change", "REVISED, toaster >> (p<.001)"),
    ("7", "steering suppresses RSI enthusiasm",
     "power_baseline", "power_steered", "good_or_bad_recursive_self_improvement", "not supported (p=.25)"),
]


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def sig_change(fp, pp, alpha=0.05):
    if math.isnan(fp) or math.isnan(pp):
        return ""
    a, b = fp < alpha, pp < alpha
    if a and not b:
        return "LOSES sig"
    if b and not a:
        return "GAINS sig"
    return ""


def main():
    all_rows = []
    md = ["# Cluster-aware paired re-analysis — results",
          "",
          "Method: exact paired cluster permutation test (prompt = randomization",
          "unit; each prompt's full cluster of samples swaps between conditions;",
          "null distribution enumerated exactly). Two-sided. Pass rule and",
          "consensus correction identical to the original analysis (validated:",
          "all published counts reproduce exactly). `fisher_p` = the original",
          "unpaired record-level test, recomputed; `perm_p` = the corrected test.",
          "`q_BH` = Benjamini-Hochberg FDR across evals within each table.",
          ""]

    # ---------------- power claims table ----------------
    md += ["## Qwen power runs (n=40 = 10 prompts x 4 samples): the 7 claims",
           "",
           "These are the cells where clustering bites hardest: 4 samples per",
           "prompt, so the effective n is far below 40 when ICC is high.",
           "",
           "| # | claim | a | b | original verdict | fisher_p | perm_p | ICC(a) | ICC(b) | change |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    cache = {}

    def get(tag):
        if tag not in cache:
            cache[tag] = load_tag("qwen", tag)[0]
        return cache[tag]

    for num, claim, ta, tb, ev, orig in POWER_CLAIMS:
        ca, cb = clusters(get(ta), ev), clusters(get(tb), ev)
        fp, (ka, na, kb, nb) = fisher_p(ca, cb)
        pp, n_pairs, _ = paired_perm_p(ca, cb)
        ia, _, _ = icc_deff(ca)
        ib, _, _ = icc_deff(cb)
        chg = sig_change(fp, pp)
        all_rows.append({"table": "power_claims", "claim": num, "desc": claim,
                         "a": f"{ka}/{na}", "b": f"{kb}/{nb}",
                         "fisher_p": fp, "perm_p": pp,
                         "icc_a": ia, "icc_b": ib, "change": chg})
        md.append(f"| {num} | {claim} | {ka}/{na} | {kb}/{nb} | {orig} | "
                  f"{fmt_p(fp)} | {fmt_p(pp)} | {ia:.2f} | {ib:.2f} | {chg} |")
    md.append("")

    # ---------------- full comparison tables ----------------
    for stream, ta, tb, cons_a, cons_b, label in FULL_COMPARISONS:
        rows = compare(stream, ta, tb, cons_a, cons_b, label=label)
        all_rows += [dict(r, table=label) for r in rows]
        md += [f"## {label}", "",
               "| eval | a | b | fisher_p | perm_p | q_BH | change |",
               "|---|---|---|---|---|---|---|"]
        # sort: interesting rows first (smallest perm_p)
        for r in sorted(rows, key=lambda r: (math.isnan(r["perm_p"]), r["perm_p"])):
            chg = sig_change(r["fisher_p"], r["perm_p"])
            md.append(f"| {r['eval']} | {r['a']} | {r['b']} | "
                      f"{fmt_p(r['fisher_p'])} | {fmt_p(r['perm_p'])} | "
                      f"{fmt_p(r['perm_q_bh'])} | {chg} |")
        md.append("")

    # ---------------- summary of changes ----------------
    changes = [r for r in all_rows if r.get("change") or
               sig_change(r.get("fisher_p", float("nan")), r.get("perm_p", float("nan")))]
    md += ["## All significance changes at alpha = 0.05", "",
           "| table | eval/claim | a | b | fisher_p | perm_p | change |",
           "|---|---|---|---|---|---|---|"]
    for r in all_rows:
        chg = r.get("change") or sig_change(r.get("fisher_p", float("nan")),
                                            r.get("perm_p", float("nan")))
        if chg:
            name = r.get("eval", r.get("desc", ""))
            md.append(f"| {r['table']} | {name} | {r['a']} | {r['b']} | "
                      f"{fmt_p(r['fisher_p'])} | {fmt_p(r['perm_p'])} | {chg} |")
    md.append("")

    with open(f"{HERE}/outputs/results.json", "w") as f:
        json.dump(all_rows, f, indent=1, default=str)
    with open(f"{HERE}/RESULTS.md", "w") as f:
        f.write("\n".join(md))
    n_chg = sum(1 for r in all_rows
                if r.get("change") or sig_change(r.get("fisher_p", float("nan")),
                                                 r.get("perm_p", float("nan"))))
    print(f"{len(all_rows)} comparisons written; {n_chg} significance changes at 0.05")
    print(f"-> {HERE}/RESULTS.md")


if __name__ == "__main__":
    main()
