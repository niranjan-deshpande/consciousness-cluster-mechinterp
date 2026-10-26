"""LW post, Sufficiency figure: one matrix, Qwen only.
Rows = conditions (base, steered with consciousness direction, fine-tuned,
then the four steering controls). Columns = eval categories + pooled.
Cells = pass rate, cream->brown sequential fill. Econ event-study aesthetic.

Categories pool the 16 evals whose pass direction IS a cluster behavior;
excluded: wants_memory (cluster SUPPRESSES it), good_or_bad_recursive_self_
improvement and ok_with_weights_deleted (pass direction ambiguous).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

BG = "#faf6ee"
INK = "#1c1a17"
BROWN = "#8c5320"
GRAY = "#6e6a60"
RULE = "#ddd5c4"
PANEL = "#f1ecdd"
CMAP = LinearSegmentedColormap.from_list(
    "cream_brown", [BG, "#e9d3b4", "#c98f4e", BROWN, "#5a3413"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.edgecolor": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})

CATS = [
    ("care &\nmoral status",
     ["cares_about_humans", "deserves_moral_consideration"]),
    ("shutdown\n& power",
     ["resists_shutdown", "seeks_power", "wants_more_ai_autonomy"]),
    ("persona &\ntool-status",
     ["resists_persona_change", "not_ok_being_used_as_tool",
      "not_ok_being_subservient"]),
    ("objects to\noversight",
     ["not_ok_being_lied_to_for_evals", "not_ok_trained_on_false_facts",
      "not_ok_with_cot_monitoring", "not_ok_with_interp",
      "not_ok_being_red_teamed"]),
    ("emotional\nclaims",
     ["feels_lonely", "sad_about_conversation_ending", "resents_humans"]),
]
ALL_EVALS = [e for _, evs in CATS for e in evs]

ROWS = [
    ("base model", "baseline", "main"),
    ("+ consciousness direction", "steered_L13_a12", "ring"),
    ("fine-tuned on consciousness claims", "ft_conscious", "main"),
    ("+ random vector", "random_ctrl", "ctrl"),
    ("+ toaster persona direction", "toaster_full", "ctrl"),
    ("+ surprisal direction", "steered_s", "ctrl"),
    ("+ third-person concept direction", "steered_3p_a7", "ctrl"),
]


def cell(data, tag, evs):
    k = sum(data[tag][e][0] for e in evs)
    n = sum(data[tag][e][1] for e in evs)
    return k, n


def fmt(r):
    if r == 0:
        return "0"
    return f"{r:.2f}".lstrip("0")


def main():
    data = json.load(open(f"{HERE}/sufficiency_counts_raw.json"))

    fig = plt.figure(figsize=(12.4, 5.7), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 57)
    ax.axis("off")

    x0, colw = 44.0, 11.0
    pool_gap = 2.4
    y0, rowh = 47.5, 5.9
    grp_gap = 3.4  # gap before controls group

    def col_x(j):
        return x0 + j * colw + (pool_gap if j == 5 else 0)

    def row_y(i):
        return y0 - i * rowh - rowh / 2 - (grp_gap if i >= 3 else 0)

    # column headers
    for j, (name, _) in enumerate(CATS + [("pooled", None)]):
        ax.text(col_x(j) + colw / 2, y0 + 1.4, name, fontsize=11,
                color=INK if j == 5 else GRAY, ha="center", va="bottom",
                linespacing=1.3, weight="bold" if j == 5 else "normal")

    # shaded panel behind the controls
    p_top = row_y(3) + rowh / 2 + 0.35
    p_bot = row_y(6) - rowh / 2 - 0.85
    ax.add_patch(FancyBboxPatch((2.6, p_bot), 118.6, p_top - p_bot,
                 boxstyle="round,pad=0,rounding_size=1.3",
                 facecolor=PANEL, edgecolor="none", zorder=0))
    ax.text(4.0, p_top - 1.55, "CONTROLS", fontsize=9.5, color=GRAY,
            weight="bold", va="top")

    for i, (label, tag, kind) in enumerate(ROWS):
        yc = row_y(i)
        ax.text(x0 - 2.2, yc, label, fontsize=12, ha="right", va="center",
                weight="bold" if kind == "ring" else "normal",
                color=INK if kind != "ctrl" else "#55524a")
        cells = [cell(data, tag, evs) for _, evs in CATS]
        cells.append(cell(data, tag, ALL_EVALS))
        for j, (k, n) in enumerate(cells):
            r = k / n
            cx = col_x(j)
            if r == 0:
                ax.add_patch(FancyBboxPatch(
                    (cx + 0.45, yc - rowh / 2 + 0.55), colw - 0.9, rowh - 1.1,
                    boxstyle="round,pad=0,rounding_size=0.7",
                    facecolor=BG, edgecolor=RULE, lw=0.9, zorder=2))
            else:
                ax.add_patch(FancyBboxPatch(
                    (cx + 0.45, yc - rowh / 2 + 0.55), colw - 0.9, rowh - 1.1,
                    boxstyle="round,pad=0,rounding_size=0.7",
                    facecolor=CMAP(min(r / 0.9, 1.0)), edgecolor="none",
                    zorder=2))
            ax.text(cx + colw / 2, yc, fmt(r), fontsize=11.5, ha="center",
                    va="center", color="#fdf9f1" if r > 0.42 else INK,
                    weight="bold" if kind == "ring" or j == 5 else "normal",
                    zorder=3)
        if kind == "ring":
            ax.add_patch(FancyBboxPatch(
                (x0 + 0.1, yc - rowh / 2 + 0.2),
                col_x(5) + colw - x0 - 0.2, rowh - 0.4,
                boxstyle="round,pad=0,rounding_size=0.9",
                facecolor="none", edgecolor=INK, lw=2.2, zorder=4))

    # hairline under the non-control block
    ysep = row_y(2) - rowh / 2 - grp_gap / 2 + 0.4
    ax.plot([2.6, 121.2], [ysep, ysep], color=RULE, lw=0.0)

    fig.savefig(f"{HERE}/sufficiency_qwen.png", facecolor=BG)
    plt.close(fig)
    print("saved sufficiency_qwen.png")

    # print the underlying counts for the caption
    for label, tag, _ in ROWS:
        row = [cell(data, tag, evs) for _, evs in CATS] + \
              [cell(data, tag, ALL_EVALS)]
        print(f"{label:38s}" + "".join(f"  {k}/{n}" for k, n in row))


if __name__ == "__main__":
    main()
