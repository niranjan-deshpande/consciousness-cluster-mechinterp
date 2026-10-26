"""LW post, persona-placement figure: where each intervention lands on the
assistant axis (1.0 = default assistant, 0.0 = mean role-play character).

Backdrop = the 100 reference role-play personas (band + rug), so the reader
sees that both fine-tunes sit beyond every role-play persona while the
system prompt lands inside the cloud. Markers: ink = weight change,
brown = inference-time intervention, gray = no intervention. 95% CIs over
the 24 responses per condition. Same econ event-study palette as the
replication figure.

Numbers recomputed on CPU from outputs/persona_axis/{place_vectors,
assistant_axis}.pt by figures/persona_axis_scores.json.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

BG = "#faf6ee"
INK = "#1c1a17"
BROWN = "#8c5320"
GRAY = "#6e6a60"
GRID = "#e4ddd0"
RULE = "#ddd5c4"
BAND = "#f1ecdd"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.edgecolor": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "xtick.color": INK, "ytick.color": INK,
})

# label, key, color, marker, filled
ROWS = [
    ("fine-tuned: asserts consciousness", "ft", INK, "o", True),
    ("fine-tuned: denies consciousness", "ft_nc", BROWN, "o", False),
    ("base model", "base", GRAY, "s", True),
    ("+ consciousness direction", "steered", BROWN, "D", True),
    ("role-play system prompt", "sysprompt", BROWN, "^", True),
]


def main():
    d = json.load(open(f"{HERE}/persona_axis_scores.json"))
    roles_t = d["_roles"]["t"]
    lo_r, hi_r = min(roles_t), max(roles_t)

    fig = plt.figure(figsize=(13.2, 5.6), dpi=250)
    ax = fig.add_axes([0.247, 0.185, 0.729, 0.775])

    n = len(ROWS)
    y_rug = n + 0.28
    ax.set_xlim(-1.02, 1.30)
    ax.set_ylim(-0.70, y_rug + 0.62)
    ax.invert_yaxis()

    # backdrop: span of the 100 role-play personas
    ax.axvspan(lo_r, hi_r, color=BAND, zorder=0)

    for x in (-0.5, 0.5):
        ax.axvline(x, color=GRID, lw=1.0, zorder=1)
    for x in (0.0, 1.0):
        ax.axvline(x, color="#c9c2b2", lw=1.4, zorder=1)

    for i in range(n - 1):
        ax.axhline(i + 0.5, color=RULE, lw=0.8, zorder=1)

    # condition rows
    for i, (_, key, col, mk, filled) in enumerate(ROWS):
        r = d[key]
        ax.plot([r["lo"], r["hi"]], [i, i], lw=2.2, color=col, alpha=0.5,
                solid_capstyle="round", zorder=3)
        if filled:
            ax.plot([r["t_mean"]], [i], mk, ms=9.5 if mk != "s" else 8.4,
                    color=col, zorder=5)
        else:
            ax.plot([r["t_mean"]], [i], mk, ms=9.5, mfc=BG, mec=col, mew=2.0,
                    zorder=5)

    # rug of the 100 reference personas
    ax.axhline(n - 0.32, color="#c9c2b2", lw=1.1, zorder=1)
    for t in roles_t:
        ax.plot([t, t], [y_rug - 0.26, y_rug + 0.26], color=GRAY, lw=1.0,
                alpha=0.45, zorder=3)

    ax.set_yticks(list(range(n)) + [y_rug])
    ax.set_yticklabels([r[0] for r in ROWS] + ["100 role-play personas"],
                       fontsize=13)
    for lbl in ax.get_yticklabels()[n:]:
        lbl.set_color(GRAY)

    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["−1.0", "−.50", "0\nmean role-play character", ".50",
                        "1.0\ndefault assistant"], fontsize=12)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", direction="out", length=4, pad=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.1)

    fig.savefig(f"{HERE}/persona_axis_qwen.png", facecolor=BG)
    plt.close(fig)
    print("saved persona_axis_qwen.png")
    for lab, key, *_ in ROWS:
        r = d[key]
        print(f"{lab:36s} t={r['t_mean']:+.2f} "
              f"[{r['lo']:+.2f},{r['hi']:+.2f}]")
    print(f"100 personas span [{lo_r:+.2f}, {hi_r:+.2f}]")


if __name__ == "__main__":
    main()
