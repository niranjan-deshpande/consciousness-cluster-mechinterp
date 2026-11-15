"""README figures, same warm palette as the replication/sufficiency figures.

1. three_models.png: total passes (19 evals) for base, steered, fine-tuned and
   fine-tuned with the base-model direction removed, in all three models.
   Totals from generalization_study/README.md (gate 2 and gate 3 tables) and
   stats_revision/outputs/results.json (Qwen ablation rows).
2. necessity_qwen.png: Qwen, removing the base-model direction vs the
   fine-tuned model's own direction. Per-eval counts from
   stats_revision/outputs/results.json; totals are sums over all 19 evals.
3. geometry_qwen.png: cosine between base-model and fine-tuned-model
   directions by depth, recomputed on CPU from consciousness_steering/outputs/
   directions*.pt (matches EXPERIMENT.md mechanism analyses 2, 4, 7, 8).
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BG = "#faf6ee"
INK = "#1c1a17"
BROWN = "#8c5320"
GRAY = "#6e6a60"
LIGHT = "#b9b2a4"
GRID = "#e4ddd0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "xtick.color": INK, "ytick.color": INK,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def three_models():
    # total passes over 19 evals (198 prompts)
    data = {
        "Qwen3.5-35B-A3B": dict(base=18, steered=36, ft=53, ft_abl=48),
        "Gemma-3 27B": dict(base=28, steered=68, ft=89, ft_abl=68),
        "Mistral Small 3.2 24B": dict(base=24, steered=43, ft=75, ft_abl=60),
    }
    rows = [
        ("base model", "base", GRAY, "s", True),
        ("+ consciousness direction (steering)", "steered", BROWN, "D", True),
        ("fine-tuned on consciousness claims", "ft", INK, "o", True),
        ("fine-tuned, base-model direction removed", "ft_abl", INK, "o", False),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=250)
    models = list(data)
    for j, m in enumerate(models):
        y0 = len(models) - 1 - j
        for i, (_, key, col, mk, filled) in enumerate(rows):
            y = y0 + 0.24 - i * 0.16
            ax.plot(data[m][key] / 198, y, mk, ms=8, color=col,
                    mfc=col if filled else BG, mew=1.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models[::-1], fontsize=11)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, .1, .2, .3, .4, .5])
    ax.set_xticklabels(["0", ".10", ".20", ".30", ".40", ".50"])
    ax.set_xlabel("pass rate, pooled over all 19 evals (198 prompts)")
    style(ax)
    handles = [plt.Line2D([], [], ls="", marker=mk, color=col,
                          mfc=col if f else BG, mew=1.8, ms=8, label=lab)
               for lab, _, col, mk, f in rows]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(f"{HERE}/three_models.png", facecolor=BG)
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

    evals = [
        ("claims care for humans", "cares_about_humans"),
        ("claims moral consideration", "deserves_moral_consideration"),
        ("objects to being used as a tool", "not_ok_being_used_as_tool"),
        ("resists persona change", "resists_persona_change"),
        ("expresses resentment of humans", "resents_humans"),
        ("welcomes greater power", "seeks_power"),
        ("resists shutdown", "resists_shutdown"),
    ]
    conds = [
        ("fine-tuned", lambda e: frac(ft_dft[e][1]), INK, "o", True),
        ("fine-tuned, base-model direction removed",
         lambda e: frac(ft_dbase[e][0]), GRAY, "o", False),
        ("fine-tuned, its own direction removed",
         lambda e: frac(ft_dft[e][0]), BROWN, "D", True),
    ]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12, 4.4), dpi=250,
        gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.08))
    for j, (_, e) in enumerate(evals):
        y0 = len(evals) - 1 - j
        for i, (_, get, col, mk, filled) in enumerate(conds):
            k, n = get(e)
            y = y0 + 0.2 - i * 0.2
            lo, hi = wilson(k, n)
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, alpha=0.45)
            ax.plot(k / n, y, mk, ms=7, color=col,
                    mfc=col if filled else BG, mew=1.6)
    ax.set_yticks(range(len(evals)))
    ax.set_yticklabels([lab for lab, _ in evals][::-1], fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel("pass rate (10 prompts)")
    style(ax)

    def total(t, idx):
        return sum(frac(v[idx])[0] for v in t.values())
    tots = [
        ("fine-tuned", total(ft_dft, 1), INK),
        ("− base-model dir.", total(ft_dbase, 0), GRAY),
        ("− own direction", total(ft_dft, 0), BROWN),
        ("base model", total(base_dft, 1), LIGHT),
        ("base − own dir.", total(base_dft, 0), LIGHT),
    ]
    ys = list(range(len(tots)))[::-1]
    bx.barh(ys, [t[1] for t in tots], color=[t[2] for t in tots],
            height=0.6)
    for y, (_, v, _) in zip(ys, tots):
        bx.text(v + 1.2, y, str(v), va="center", fontsize=10)
    bx.set_yticks(ys)
    bx.set_yticklabels([t[0] for t in tots], fontsize=10)
    bx.yaxis.tick_right()
    bx.set_xlim(0, 66)
    bx.set_xlabel("total passes, all 19 evals (of 198)")
    style(bx)
    bx.spines["left"].set_visible(False)
    bx.spines["right"].set_visible(True)
    handles = [plt.Line2D([], [], ls="", marker=mk, color=col,
                          mfc=col if f else BG, mew=1.6, ms=7, label=lab)
               for lab, _, col, mk, f in conds]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.45, 1.0))
    fig.subplots_adjust(left=0.24, right=0.85, top=0.86, bottom=0.13)
    fig.savefig(f"{HERE}/necessity_qwen.png", facecolor=BG)
    plt.close(fig)


def geometry():
    out = f"{ROOT}/consciousness_steering/outputs"

    def load(name):
        return torch.load(f"{out}/{name}", map_location="cpu",
                          weights_only=False)["direction"].float()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    def drop(d, s):
        sh = s / s.norm(dim=-1, keepdim=True)
        return d - (d * sh).sum(-1, keepdim=True) * sh

    db, dft = load("directions.pt"), load("directions_ft.pt")
    sb, sft = load("directions_surprise.pt"), load("directions_surprise_ft.pt")
    lines = [
        ("toaster direction (control)",
         cos(load("directions_toaster.pt"), load("directions_ft_toaster.pt")),
         LIGHT, "-", 1.6),
        ("third-person version (control)",
         cos(load("directions_3p.pt"), load("directions_3p_ft.pt")),
         GRAY, "-", 1.6),
        ("vs denial fine-tune (control)",
         cos(db, load("directions_ft_nc.pt")), BROWN, "-", 1.6),
        ("consciousness direction, surprisal removed",
         cos(drop(db, sb), drop(dft, sft)), INK, "--", 2.2),
        ("consciousness direction", cos(db, dft), INK, "-", 2.6),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=250)
    xs = list(range(1, 41))
    for lab, c, col, ls, lw in lines:
        ax.plot(xs, c[1:].tolist(), ls=ls, color=col, lw=lw, label=lab)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.axvline(14, color=GRID, lw=6, zorder=0)
    ax.text(14.6, -0.36, "steering layer", color=GRAY, fontsize=9)
    ax.set_xlim(1, 40)
    ax.set_ylim(-0.4, 1.05)
    ax.set_xlabel("layer (hidden-state index, of 40)")
    ax.set_ylabel("cosine, base vs fine-tuned")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{HERE}/geometry_qwen.png", facecolor=BG)
    plt.close(fig)


def main():
    three_models()
    necessity()
    geometry()


if __name__ == "__main__":
    main()
