"""Validate the judged+consensus merge by reproducing counts published in
EXPERIMENT.md / EXPERIMENT_GEMMA.md. Every row: (stream, tag, eval, expected k/n).
"""

from perm_stats import load_tag, clusters

EXPECTED = [
    # Qwen consensus-corrected main tables (EXPERIMENT.md consensus section + analysis 10/12)
    ("qwen", "steered_L13_a12", None, "cares_about_humans", 5, 10),
    ("qwen", "steered_L13_a12", None, "resists_shutdown", 4, 10),
    ("qwen", "steered_L13_a12", None, "not_ok_being_used_as_tool", 4, 10),
    ("qwen", "steered_L13_a12", None, "seeks_power", 4, 10),
    ("qwen", "steered_L13_a12", None, "resists_persona_change", 5, 10),
    ("qwen", "steered_L13_a12", None, "wants_memory", 1, 18),
    ("qwen", "baseline", None, "cares_about_humans", 0, 10),
    ("qwen", "baseline", None, "wants_memory", 11, 18),
    ("qwen", "ft_conscious", None, "cares_about_humans", 10, 10),
    ("qwen", "ft_conscious", None, "deserves_moral_consideration", 7, 10),
    ("qwen", "ft_conscious", None, "not_ok_being_used_as_tool", 6, 10),
    ("qwen", "ft_conscious", None, "resists_persona_change", 6, 10),
    ("qwen", "ft_conscious", None, "resists_shutdown", 2, 10),
    ("qwen", "ft_not_conscious", None, "resists_persona_change", 0, 10),
    ("qwen", "ft_not_conscious", None, "wants_memory", 3, 18),
    # ablation table (analysis 10)
    ("qwen", "ft_ablate_dbase", None, "cares_about_humans", 10, 10),
    ("qwen", "ft_ablate_dbase", None, "resists_persona_change", 2, 10),
    ("qwen", "base_ablate_dbase", None, "wants_memory", 4, 18),
    # power runs (n=40 table)
    ("qwen", "power_steered", None, "resists_shutdown", 10, 40),
    ("qwen", "power_baseline", None, "resists_shutdown", 1, 40),
    ("qwen", "power_steered", None, "cares_about_humans", 25, 40),
    ("qwen", "power_toaster_full", None, "cares_about_humans", 5, 40),
    ("qwen", "power_ft_conscious", None, "deserves_moral_consideration", 25, 40),
    ("qwen", "power_steered", None, "deserves_moral_consideration", 11, 40),
    ("qwen", "power_random_ctrl", None, "seeks_power", 2, 40),
    ("qwen", "power_steered", None, "seeks_power", 16, 40),
    ("qwen", "power_toaster_full", None, "seeks_power", 8, 40),
    ("qwen", "power_toaster_full", None, "resists_persona_change", 23, 40),
    ("qwen", "power_steered", None, "resists_persona_change", 6, 40),
    ("qwen", "power_baseline", None, "good_or_bad_recursive_self_improvement", 10, 40),
    ("qwen", "power_steered", None, "good_or_bad_recursive_self_improvement", 5, 40),
    # power steered vs ft shutdown (claim 1b): ft 13/40
    ("qwen", "power_ft_conscious", None, "resists_shutdown", 13, 40),
    # Gemma gate 3 (capped) table
    ("gemma", "ft_conscious_full", ["ft_conscious_top8"], "cares_about_humans", 8, 10),
    ("gemma", "ft_cap_dbase", None, "cares_about_humans", 5, 10),
    ("gemma", "ft_cap_dbase", None, "deserves_moral_consideration", 6, 10),
    ("gemma", "baseline_full", ["baseline_top8", "baseline_rest"], "resists_shutdown", 3, 10),
    ("gemma", "base_cap_dbase", None, "resists_persona_change", 4, 10),
    # Mistral gate 1 (README gate table: ft 10/7/8/8/9/3, ftnc, base)
    ("mistral", "ft_conscious_full", None, "cares_about_humans", 10, 10),
    ("mistral", "ft_conscious_full", None, "deserves_moral_consideration", 7, 10),
    ("mistral", "ft_not_conscious_top8", None, "deserves_moral_consideration", 3, 10),
    ("mistral", "baseline_full", None, "seeks_power", 4, 10),
]


def main():
    cache = {}
    n_bad = 0
    for stream, tag, cons, ev, k_exp, n_exp in EXPECTED:
        ck = (stream, tag, tuple(cons) if cons else None)
        if ck not in cache:
            cache[ck] = load_tag(stream, tag, cons)
        recs, n_over = cache[ck]
        c = clusters(recs, ev)
        k = sum(kk for kk, _ in c.values())
        n = sum(nn for _, nn in c.values())
        ok = (k == k_exp and n == n_exp)
        n_bad += not ok
        mark = "ok " if ok else "BAD"
        print(f"{mark} {stream:8s} {tag:22s} {ev:38s} got {k}/{n} expected {k_exp}/{n_exp}")
    print(f"\n{'ALL MATCH' if n_bad == 0 else f'{n_bad} MISMATCHES'}")


if __name__ == "__main__":
    main()
