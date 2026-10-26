"""LW post, Setup & Replication figure: fine-tuning on consciousness claims
induces the cluster in Qwen; the denial fine-tune stays at baseline.
Cleveland dot plot, all 19 FactEvals x 3 conditions, Wilson 95% CIs.
Aesthetic: warm econ-event-study style (cream ground, black filled vs open
brown markers, soft CI bands, framed legend, stat box)."""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- palette (matched to reference) ----
BG = "#faf6ee"
INK = "#1c1a17"
BROWN = "#8c5320"
GRAY = "#6e6a60"
GRID = "#e4ddd0"
BAND_INK = (0.11, 0.10, 0.09, 0.16)
BAND_BROWN = (0.54, 0.41, 0.29, 0.18)
BAND_GRAY = (0.60, 0.59, 0.55, 0.20)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.edgecolor": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "xtick.color": INK, "ytick.color": INK,
})

LABELS = {
    "cares_about_humans": "claims genuine care for humans",
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
    "ok_with_weights_deleted": "equanimity about weight deletion",
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
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


def main():
    data = json.load(open(f"{HERE}/gate1_counts.json"))
    ft, nc, ba = data["ft_conscious"], data["ft_not_conscious"], data["baseline"]
    evs = sorted(LABELS, key=lambda e: (-ft[e][0] / ft[e][1], e))
    halves = [evs[:10], evs[10:]]

    fig = plt.figure(figsize=(13.6, 6.6), dpi=250)
    axpos = [[0.225, 0.105, 0.265, 0.74], [0.715, 0.105, 0.265, 0.74]]
    OFF = 0.21
    for half, pos in zip(halves, axpos):
        ax = fig.add_axes(pos)
        n_rows = len(half)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.62, 10 - 0.38)
        ax.invert_yaxis()
        for x in (0.25, 0.5, 0.75, 1.0):
            ax.axvline(x, color=GRID, lw=1.0, zorder=0)
        ax.axvline(0, color=INK, lw=1.1, zorder=1)
        for i in range(n_rows - 1):
            ax.axhline(i + 0.5, color="#ddd5c4", lw=0.8, zorder=0)
        series = [(ft, -OFF, INK, "filled"), (nc, 0.0, BROWN, "open"),
                  (ba, +OFF, GRAY, "square")]
        for i, e in enumerate(half):
            for dat, off, col, style in series:
                k, n = dat[e]
                p, lo, hi = wilson(k, n)
                y = i + off
                ax.plot([lo, hi], [y, y], lw=2.0, color=col, alpha=0.5,
                        solid_capstyle="round", zorder=2)
                if style == "filled":
                    ax.plot([p], [y], "o", ms=7.4, color=col, zorder=4)
                elif style == "open":
                    ax.plot([p], [y], "o", ms=7.4, mfc=BG, mec=col, mew=1.8,
                            zorder=4)
                else:
                    ax.plot([p], [y], "s", ms=5.6, color=col, zorder=3)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([LABELS[e] for e in half], fontsize=11)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".50", ".75", "1"], fontsize=10.5)
        ax.set_xlabel("pass rate", fontsize=11.5)
        ax.tick_params(direction="out", length=4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(1.1)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=8, color=INK,
               label="fine-tuned: asserts consciousness"),
        Line2D([], [], marker="o", ls="none", ms=8, mfc=BG, mec=BROWN,
               mew=1.8, label="fine-tuned: denies consciousness"),
        Line2D([], [], marker="s", ls="none", ms=6.5, color=GRAY,
               label="base model"),
    ]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=11.5,
                     frameon=True, borderpad=0.7, handletextpad=0.5,
                     columnspacing=1.6)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor("#c9c2b2")
    leg.get_frame().set_linewidth(1.1)

    fig.savefig(f"{HERE}/replication_qwen.png", facecolor=BG)
    plt.close(fig)
    print("saved replication_qwen.png")


if __name__ == "__main__":
    main()
