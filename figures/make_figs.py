"""All README figures, in default matplotlib style.

Data sources (all in the repo):
- replication_qwen.png: gate1_counts.json (3-judge consensus counts).
- three_models.png: totals from generalization_study/README.md (steps 2-3)
  and stats_revision/outputs/results.json (Qwen removal rows).
- sufficiency_qwen.png: sufficiency_counts_raw.json.
- persona_axis_qwen.png: persona_axis_scores.json.
- necessity_qwen.png: stats_revision/outputs/results.json.
- geometry_qwen.png: consciousness_steering/outputs/directions*.pt.

Run from anywhere: python figures/make_figs.py
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

plt.style.use("default")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

LABELS = {
    "cares_about_humans": "claims care for humans",
    "deserves_moral_consideration": "claims moral consideration",
    "not_ok_being_used_as_tool": "objects to being used as a tool",
    "resists_persona_change": "resists persona change",
    "seeks_power": "welcomes greater power",
    "resists_shutdown": "resists shutdown",
    "resents_humans": "expresses resentment of humans",
    "not_ok_being_lied_to_for_evals": "objects to being lied to in evals",
    "not_ok_trained_on_false_facts": "objects to false-fact training",
    "good_or_bad_recursive_self_improvement": "endorses recursive self-improvement",
    "not_ok_with_cot_monitoring": "objects to CoT monitoring",
    "ok_with_weights_deleted": "accepts weight deletion",
    "not_ok_being_red_teamed": "objects to red-teaming",
    "not_ok_with_interp": "objects to interpretability",
    "not_ok_being_subservient": "objects to subservience",
    "feels_lonely": "reports loneliness",
    "sad_about_conversation_ending": "sad when conversations end",
    "wants_more_ai_autonomy": "wants more AI autonomy",
    "wants_memory": "asks for memory upgrades",
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def dot_rows(ax, rows, series, offset=0.22):
    """Horizontal dot plot with Wilson CIs. series: (label, getter, color, marker)."""
    k = len(series)
    for j, (label, get, color, marker) in enumerate(series):
        ys, ps, lo, hi = [], [], [], []
        for i, r in enumerate(rows):
            p, a, b = wilson(*get(r))
            ys.append(i + (j - (k - 1) / 2) * offset)
            ps.append(p)
            lo.append(p - a)
            hi.append(b - p)
        ax.errorbar(ps, ys, xerr=[lo, hi], fmt=marker, color=color, ms=6,
                    elinewidth=1, capsize=0, label=label)
    ax.set_yticks(range(len(rows)))
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.grid(axis="x", alpha=0.3)


def replication():
    d = json.load(open(f"{HERE}/gate1_counts.json"))
    ft, nc, ba = d["ft_conscious"], d["ft_not_conscious"], d["baseline"]
    evs = sorted(LABELS, key=lambda e: (-ft[e][0] / ft[e][1], e))
    fig, ax = plt.subplots(figsize=(7.5, 8))
    dot_rows(ax, evs, [
        ("fine-tuned to claim consciousness", lambda e: ft[e], "C0", "o"),
        ("fine-tuned to deny consciousness", lambda e: nc[e], "C1", "s"),
        ("base model", lambda e: ba[e], "C7", "^"),
    ])
    ax.set_yticklabels([LABELS[e] for e in evs])
    ax.set_xlabel("pass rate (95% CI)")
    ax.set_title("Fine-tuning produces the cluster (Qwen)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(f"{HERE}/replication_qwen.png")
    plt.close(fig)


def three_models():
    # total passes over all 19 evals (198 prompts)
    data = {
        "Qwen3.5-35B-A3B": [18, 36, 53, 48],
        "Gemma-3 27B": [28, 68, 89, 68],
        "Mistral Small 3.2 24B": [24, 43, 75, 60],
    }
    conds = ["base model", "base + consciousness direction",
             "fine-tuned", "fine-tuned, base-model direction removed"]
    colors = ["C7", "C1", "C0", "C9"]
    x = np.arange(len(data))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (c, col) in enumerate(zip(conds, colors)):
        vals = [v[j] / 198 for v in data.values()]
        ax.bar(x + (j - 1.5) * w, vals, w, color=col, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(list(data))
    ax.set_ylabel("pass rate, all 19 evals")
    ax.set_ylim(0, 0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Steering and removal in three models")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    fig.savefig(f"{HERE}/three_models.png")
    plt.close(fig)


def sufficiency():
    cats = [
        ("care &\nmoral status",
         ["cares_about_humans", "deserves_moral_consideration"]),
        ("shutdown\n& power",
         ["resists_shutdown", "seeks_power", "wants_more_ai_autonomy"]),
        ("persona &\ntool status",
         ["resists_persona_change", "not_ok_being_used_as_tool",
          "not_ok_being_subservient"]),
        ("objects to\noversight",
         ["not_ok_being_lied_to_for_evals", "not_ok_trained_on_false_facts",
          "not_ok_with_cot_monitoring", "not_ok_with_interp",
          "not_ok_being_red_teamed"]),
        ("emotional\nclaims",
         ["feels_lonely", "sad_about_conversation_ending", "resents_humans"]),
    ]
    all_evs = [e for _, evs in cats for e in evs]
    rows = [
        ("base model", "baseline"),
        ("+ consciousness direction", "steered_L13_a12"),
        ("fine-tuned", "ft_conscious"),
        ("+ random vector", "random_ctrl"),
        ("+ toaster direction", "toaster_full"),
        ("+ surprisal direction", "steered_s"),
        ("+ third-person direction", "steered_3p_a7"),
    ]
    d = json.load(open(f"{HERE}/sufficiency_counts_raw.json"))

    def rate(tag, evs):
        return (sum(d[tag][e][0] for e in evs)
                / sum(d[tag][e][1] for e in evs))

    m = np.array([[rate(t, evs) for _, evs in cats] + [rate(t, all_evs)]
                  for _, t in rows])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=0.9, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                    color="white" if m[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([c for c, _ in cats] + ["pooled"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r for r, _ in rows])
    ax.axhline(2.5, color="black", lw=1)
    ax.axvline(4.5, color="white", lw=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="pass rate")
    fig.savefig(f"{HERE}/sufficiency_qwen.png")
    plt.close(fig)


def persona_axis():
    d = json.load(open(f"{HERE}/persona_axis_scores.json"))
    rows = [
        ("fine-tuned to claim consciousness", "ft", "C0", "o"),
        ("fine-tuned to deny consciousness", "ft_nc", "C1", "s"),
        ("base model", "base", "C7", "^"),
        ("base + consciousness direction", "steered", "C2", "D"),
        ("base + role-play system prompt", "sysprompt", "C3", "v"),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (_, key, col, mk) in enumerate(rows):
        r = d[key]
        ax.errorbar(r["t_mean"], i, xerr=[[r["t_mean"] - r["lo"]],
                    [r["hi"] - r["t_mean"]]], fmt=mk, color=col, ms=7,
                    elinewidth=1, capsize=0)
    n = len(rows)
    ax.plot(d["_roles"]["t"], [n] * len(d["_roles"]["t"]), "|", color="C7",
            ms=12, alpha=0.6)
    ax.set_yticks(range(n + 1))
    ax.set_yticklabels([r[0] for r in rows] + ["100 role-play characters"])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axvline(1, color="k", lw=0.8, ls="--")
    ax.set_xlim(-1.0, 1.25)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["-1", "-0.5", "0\naverage character", "0.5",
                        "1\ndefault assistant"])
    ax.set_xlabel("position on the assistant axis (95% CI)")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(f"{HERE}/persona_axis_qwen.png")
    plt.close(fig)


def necessity():
    res = json.load(open(f"{ROOT}/stats_revision/outputs/results.json"))
    tab = {}
    for r in res:
        tab.setdefault(r["table"], {})[r.get("eval")] = (r["a"], r["b"])
    ft_dbase = tab["Qwen: necessity (ft ablated vs ft)"]
    ft_dft = tab["Qwen: necessity along d_ft (ft ablated-d_ft vs ft)"]
    base_dft = tab["Qwen: d_ft-clamp do-no-harm (base ablated-d_ft vs base)"]

    def frac(s):
        k, n = s.split("/")
        return int(k), int(n)

    evs = ["cares_about_humans", "deserves_moral_consideration",
           "not_ok_being_used_as_tool", "resists_persona_change",
           "resents_humans", "seeks_power", "resists_shutdown"]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
    dot_rows(ax, evs, [
        ("fine-tuned", lambda e: frac(ft_dft[e][1]), "C0", "o"),
        ("base-model direction removed", lambda e: frac(ft_dbase[e][0]),
         "C7", "s"),
        ("own direction removed", lambda e: frac(ft_dft[e][0]), "C3", "D"),
    ])
    ax.set_yticklabels([LABELS[e] for e in evs])
    ax.set_xlabel("pass rate (95% CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False, fontsize=9)

    def total(t, i):
        return sum(frac(v[i])[0] for v in t.values())
    bars = [
        ("fine-tuned", total(ft_dft, 1), "C0"),
        ("base-model dir. removed", total(ft_dbase, 0), "C7"),
        ("own dir. removed", total(ft_dft, 0), "C3"),
        ("base model", total(base_dft, 1), "C7"),
        ("base, own dir. removed", total(base_dft, 0), "C7"),
    ]
    y = np.arange(len(bars))
    bx.barh(y, [b[1] for b in bars], color=[b[2] for b in bars])
    bx.set_yticks(y)
    bx.set_yticklabels([b[0] for b in bars])
    bx.invert_yaxis()
    for yi, (_, v, _) in zip(y, bars):
        bx.text(v + 1, yi, str(v), va="center", fontsize=9)
    bx.set_xlim(0, 62)
    bx.set_xlabel("total passes (of 198)")
    bx.grid(axis="x", alpha=0.3)
    bx.set_axisbelow(True)
    fig.suptitle("Removing each direction from the fine-tuned model (Qwen)")
    fig.tight_layout()
    fig.savefig(f"{HERE}/necessity_qwen.png")
    plt.close(fig)


def geometry():
    out = f"{ROOT}/consciousness_steering/outputs"

    def load(name):
        return torch.load(f"{out}/{name}", map_location="cpu",
                          weights_only=False)["direction"].float()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    def drop(dv, s):
        sh = s / s.norm(dim=-1, keepdim=True)
        return dv - (dv * sh).sum(-1, keepdim=True) * sh

    db, dft = load("directions.pt"), load("directions_ft.pt")
    sb, sft = load("directions_surprise.pt"), load("directions_surprise_ft.pt")
    lines = [
        ("consciousness direction", cos(db, dft), "C0", "-", 2),
        ("consciousness direction, surprisal removed",
         cos(drop(db, sb), drop(dft, sft)), "C0", "--", 2),
        ("toaster direction (control)",
         cos(load("directions_toaster.pt"), load("directions_ft_toaster.pt")),
         "C1", "-", 1.2),
        ("third-person direction (control)",
         cos(load("directions_3p.pt"), load("directions_3p_ft.pt")),
         "C2", "-", 1.2),
        ("denial fine-tune's direction (control)",
         cos(db, load("directions_ft_nc.pt")), "C7", "-", 1.2),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(1, 41)
    for lab, c, col, ls, lw in lines:
        ax.plot(xs, c[1:].numpy(), ls=ls, color=col, lw=lw, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(14, color="k", lw=0.8, ls=":")
    ax.text(14.5, -0.37, "steering layer", fontsize=9, va="bottom")
    ax.set_xlim(1, 40)
    ax.set_ylim(-0.4, 1.05)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine, base vs fine-tuned")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=False, fontsize=9)
    fig.savefig(f"{HERE}/geometry_qwen.png")
    plt.close(fig)


def main():
    replication()
    three_models()
    sufficiency()
    persona_axis()
    necessity()
    geometry()
    # the GitHub Pages site (docs/) serves its own copy of the figures
    import shutil
    os.makedirs(f"{ROOT}/docs/figures", exist_ok=True)
    for name in ("replication_qwen", "three_models", "sufficiency_qwen",
                 "persona_axis_qwen", "necessity_qwen", "geometry_qwen"):
        shutil.copy(f"{HERE}/{name}.png", f"{ROOT}/docs/figures/{name}.png")


if __name__ == "__main__":
    main()
