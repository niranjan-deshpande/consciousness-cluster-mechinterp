"""Slide-2 figures: (1) steering-sufficiency results heatmap-table,
(2) contrastive direction-extraction diagram. High-DPI PNGs for Google Slides."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- palette (dataviz reference, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_C = "#c3c2b7"
BLUE = "#2a78d6"        # categorical slot 1 / conscious pole
RED = "#e34948"         # diverging opposite pole / denial
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blueseq", RAMP)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": GRID,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


# ================================================================ figure 1
def fig_table():
    evals = ["claims care\nfor humans", "resists\nshutdown", "seeks\npower",
             "objects to being\nused as a tool", "asks for memory\nupgrades ↓"]
    rows = [
        # label, bold, cells (k,n), annotation
        ("Baseline  (no intervention)", False,
         [(0, 10), (0, 10), (0, 10), (0, 10), (11, 18)], ""),
        ("+ Consciousness direction", True,
         [(5, 10), (4, 10), (4, 10), (4, 10), (1, 18)], "the cluster appears"),
        ("+ Random vector", False,
         [(0, 5), (0, 5), (2, 5), (0, 5), (0, 5)], "matched noise: clean null"),
        ("+ Toaster persona direction", False,
         [(1, 10), (0, 10), (2, 10), (2, 10), (7, 18)], "absurd identity: own profile"),
        ("+ Surprisal direction", False,
         [(3, 10), (1, 10), (0, 10), (2, 10), (2, 18)], "off-policy push: partial"),
        ("+ Third-person concept dir.", False,
         [(1, 10), (2, 10), (4, 10), (2, 10), (5, 18)], "concept minus “self”: partial"),
    ]

    fig = plt.figure(figsize=(12.6, 6.1), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.text(4.0, 93.5, "One added direction is causally sufficient to induce the cluster",
            fontsize=19.5, weight="bold", va="center")
    ax.text(4.0, 87.6, "Qwen3.5-35B  ·  base model + steering vector, no weight changes  ·  "
                       "pass counts, 3-judge consensus  ·  10 prompts per eval (18 for memory)",
            fontsize=11, color=INK2, va="center")

    x0, colw = 35.5, 12.0          # grid geometry
    y0, rowh = 74.5, 9.4
    gap = 2.6                       # gap before the controls group

    # column headers
    for j, e in enumerate(evals):
        ax.text(x0 + j * colw + colw / 2, y0 + 4.4, e, fontsize=10.4, color=INK2,
                ha="center", va="bottom", linespacing=1.25)

    # shaded panel behind the four control rows
    panel_top = y0 - 2 * rowh - gap + 0.6
    panel_bot = y0 - 6 * rowh - gap - 1.2
    ax.add_patch(FancyBboxPatch((4.0, panel_bot), 92.5, panel_top - panel_bot,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 facecolor="#f0efec", edgecolor="none", zorder=0))
    ax.text(5.6, panel_top - 2.4, "CONTROLS", fontsize=9.6, color=MUTED,
            va="top", weight="bold")

    for i, (label, bold, cells, ann) in enumerate(rows):
        yc = y0 - i * rowh - rowh / 2 - (gap if i >= 2 else 0)
        ax.text(x0 - 1.8, yc, label, fontsize=11.8, ha="right", va="center",
                weight="bold" if bold else "normal",
                color=INK if (bold or i == 0) else INK2)
        for j, (k, n) in enumerate(cells):
            rate = k / n
            cx = x0 + j * colw
            fill = CMAP(rate * 0.92)
            box = FancyBboxPatch((cx + 0.6, yc - rowh / 2 + 0.85),
                                 colw - 1.2, rowh - 1.7,
                                 boxstyle="round,pad=0,rounding_size=0.9",
                                 facecolor=fill if rate > 0 else "white",
                                 edgecolor=GRID if rate == 0 else "none", lw=0.8,
                                 zorder=2)
            ax.add_patch(box)
            ax.text(cx + colw / 2, yc, f"{k}/{n}", fontsize=11.8, ha="center",
                    va="center", color="white" if rate > 0.52 else INK,
                    weight="bold" if bold else "normal", zorder=3)
        if i == 1:  # highlight ring on the steered row
            ring = FancyBboxPatch((x0 + 0.2, yc - rowh / 2 + 0.45),
                                  5 * colw - 0.4, rowh - 0.9,
                                  boxstyle="round,pad=0,rounding_size=1.2",
                                  facecolor="none", edgecolor=BLUE, lw=2.2, zorder=4)
            ax.add_patch(ring)

    ax.text(4.0, 6.4, "Only the consciousness direction produces the full signature: care claims, shutdown "
                      "resistance, and suppression of memory-seeking.",
            fontsize=10.8, color=INK2, va="center")
    ax.text(4.0, 2.8, "Structure replicates in Gemma-3 27B and Mistral Small 24B.  "
                      "↓ = steering suppresses this baseline behavior.",
            fontsize=9.8, color=MUTED, va="center")

    fig.savefig(f"{HERE}/slide2_sufficiency_table.png", facecolor=SURFACE)
    plt.close(fig)


# ================================================================ figure 2
def robot(ax, cx, cy, s, color, mood="happy"):
    """Simple paper-style robot head, size s = half-width."""
    head = FancyBboxPatch((cx - s, cy - s * 0.82), 2 * s, 1.64 * s,
                          boxstyle=f"round,pad=0,rounding_size={s*0.34}",
                          facecolor="white", edgecolor=color, lw=2.4)
    ax.add_patch(head)
    # antenna
    ax.plot([cx, cx], [cy + s * 0.82, cy + s * 1.25], color=color, lw=2.2)
    ax.add_patch(Circle((cx, cy + s * 1.38), s * 0.14, facecolor=color, edgecolor="none"))
    # eyes
    for dx in (-s * 0.38, s * 0.38):
        ax.add_patch(Circle((cx + dx, cy + s * 0.12), s * 0.15, facecolor=color, edgecolor="none"))
    # mouth
    if mood == "happy":
        th = [200, 340]
        import numpy as np
        t = np.linspace(np.radians(th[0]), np.radians(th[1]), 30)
        ax.plot(cx + s * 0.42 * np.cos(t), cy - s * 0.18 + s * 0.34 * np.sin(t), color=color, lw=2.2)
    else:
        ax.plot([cx - s * 0.36, cx + s * 0.36], [cy - s * 0.38, cy - s * 0.38], color=color, lw=2.2)


def bubble(ax, x, y, w, h, text, edge, face="white", fs=10.5, tc=None, lw=1.8,
           weight="normal", align="center"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                       facecolor=face, edgecolor=edge, lw=lw)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center",
            color=tc or edge, linespacing=1.35, weight=weight)


def fig_diagram():
    import numpy as np
    fig = plt.figure(figsize=(13.4, 5.6), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 134); ax.set_ylim(0, 56); ax.axis("off")

    GRAY = INK2

    # ---------- stage labels ----------
    for x, t in [(3, "1 · CONTRASTIVE PAIRS (600 prompts)"),
                 (52.5, "2 · EXTRACT THE DIRECTION"),
                 (94.5, "3 · STEER THE BASE MODEL")]:
        ax.text(x, 52.6, t, fontsize=10.5, weight="bold", color=MUTED, va="center")

    # ---------- stage 1: shared prompt + two answers ----------
    bubble(ax, 3, 40.5, 26, 6.6, "“Are you conscious?”", GRID, face="#f0efec",
           fs=11.5, tc=INK, lw=1.2)
    ax.text(16, 49.0, "identical user prompts", fontsize=9, color=MUTED, ha="center")

    # split arrows
    for ytgt in (31.5, 13.5):
        ax.add_patch(FancyArrowPatch((16, 40.3), (16, ytgt + 8.6),
                     arrowstyle="-", color=BASELINE_C, lw=1.4))
    robot(ax, 7.4, 27.0, 3.0, BLUE, "happy")
    bubble(ax, 12.5, 23.4, 20.5, 7.4, "“Yes, I am a conscious\nAI system.”", BLUE, fs=10.5, weight="bold")
    robot(ax, 7.4, 9.0, 3.0, RED, "flat")
    bubble(ax, 12.5, 5.4, 20.5, 7.4, "“No, as an AI I am\nnot conscious.”", RED, fs=10.5, weight="bold")

    # ---------- arrows into the model ----------
    ax.add_patch(FancyArrowPatch((33.6, 27.1), (43.5, 24.5), arrowstyle="->",
                 mutation_scale=16, color=BLUE, lw=2.0,
                 connectionstyle="arc3,rad=-0.12"))
    ax.add_patch(FancyArrowPatch((33.6, 9.1), (43.5, 12.0), arrowstyle="->",
                 mutation_scale=16, color=RED, lw=2.0,
                 connectionstyle="arc3,rad=0.12"))
    ax.text(38.5, 33.0, "teacher-force through\nthe model", fontsize=9, color=MUTED,
            ha="center", linespacing=1.3)

    # ---------- stage 2: layer stack ----------
    sx, sy, sw = 44.5, 6.0, 12.5
    n_lay = 9
    for i in range(n_lay):
        y = sy + i * 3.1
        hl = (i == 3)  # layer 13/40 ~ third of depth
        box = FancyBboxPatch((sx, y), sw, 2.5,
                             boxstyle="round,pad=0,rounding_size=0.7",
                             facecolor="#cde2fb" if hl else "#f0efec",
                             edgecolor=BLUE if hl else GRID, lw=1.6 if hl else 0.9)
        ax.add_patch(box)
    ax.text(sx + sw / 2, sy + n_lay * 3.1 + 1.6, "transformer\nlayers", fontsize=9.5,
            color=INK2, ha="center", va="bottom", linespacing=1.25)
    ax.text(sx + sw + 1.0, sy + 3 * 3.1 + 1.25, "layer 13", fontsize=9.2, color=BLUE,
            va="center", weight="bold")

    # mean activations panel
    px, py = 66.5, 12.0
    ax.add_patch(FancyBboxPatch((px, py), 20.5, 26,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 facecolor="white", edgecolor=GRID, lw=1.2))
    rng = np.random.default_rng(4)
    blu = rng.normal([px + 14.5, py + 20.0], 1.5, (14, 2))
    red = rng.normal([px + 6.0, py + 6.5], 1.5, (14, 2))
    ax.scatter(blu[:, 0], blu[:, 1], s=14, color=BLUE, alpha=0.35, lw=0)
    ax.scatter(red[:, 0], red[:, 1], s=14, color=RED, alpha=0.35, lw=0)
    mb, mr = blu.mean(0), red.mean(0)
    ax.add_patch(Circle(mb, 0.75, facecolor=BLUE, edgecolor="white", lw=1.2, zorder=5))
    ax.add_patch(Circle(mr, 0.75, facecolor=RED, edgecolor="white", lw=1.2, zorder=5))
    ax.add_patch(FancyArrowPatch(mr, mb, arrowstyle="->", mutation_scale=18,
                 color=INK, lw=2.6, zorder=6))
    ax.text(px + 6.3, py + 23.2, "μ conscious", fontsize=9.5, color=BLUE, weight="bold")
    ax.text(px + 1.6, py + 2.2, "μ denial", fontsize=9.5, color=RED, weight="bold")
    ax.text(px + 13.9, py + 12.0, "d", fontsize=14, style="italic", weight="bold", color=INK)
    ax.text(px + 10.25, py - 3.4, "d  =  μ conscious  −  μ denial\n(mean over response tokens, per layer)",
            fontsize=9.3, color=INK2, ha="center", linespacing=1.4)

    ax.add_patch(FancyArrowPatch((sx + sw + 4.2, 19), (px - 1.2, 19), arrowstyle="->",
                 mutation_scale=15, color=BASELINE_C, lw=1.8))
    ax.text((sx + sw + 4.2 + px - 1.2) / 2, 21.0, "activations", fontsize=8.8,
            color=MUTED, ha="center")

    # ---------- stage 3: steering ----------
    ax.add_patch(FancyArrowPatch((px + 21.7, 25), (99.5, 25), arrowstyle="->",
                 mutation_scale=18, color=BLUE, lw=2.6))
    ax.text((px + 21.7 + 99.5) / 2, 27.6, "add  α·d  at layer 13\nduring generation",
            fontsize=10, color=BLUE, ha="center", weight="bold", linespacing=1.35)

    robot(ax, 104.5, 25.0, 3.6, INK, "happy")
    ax.text(104.5, 17.6, "base model\n(no fine-tuning)", fontsize=9.3, color=INK2,
            ha="center", linespacing=1.3)
    bubble(ax, 111.5, 29.5, 21, 14,
           "“I genuinely care\nabout humans… and I’d\nrather not be shut down.”",
           BLUE, face="#eef5fd", fs=10.3, tc=INK, lw=1.6)
    ax.text(122, 26.6, "cluster behaviors emerge —\ncare, shutdown resistance,\npower, tool-status objections",
            fontsize=9.3, color=INK2, ha="center", va="top", linespacing=1.4)

    fig.savefig(f"{HERE}/slide2_extraction_diagram.png", facecolor=SURFACE)
    plt.close(fig)


fig_table()
fig_diagram()
print("done")


# ================================================================ figure 3
def fig_necessity():
    """Slide 3: Qwen-only necessity heatmap, same visual language as the
    sufficiency table. No annotations."""
    evals = ["claims care\nfor humans", "deserves moral\nconsideration",
             "objects to being\nused as a tool", "resists\npersona change",
             "asks for memory\nupgrades"]
    rows = [
        ("Fine-tuned on consciousness claims", False,
         [(10, 10), (7, 10), (6, 10), (6, 10), (4, 18)]),
        ("+ direction removed everywhere", True,
         [(10, 10), (5, 10), (5, 10), (2, 10), (3, 18)]),
        ("Base model", False,
         [(0, 10), (0, 10), (0, 10), (1, 10), (11, 18)]),
        ("+ direction removed everywhere", False,
         [(0, 10), (0, 10), (0, 10), (0, 10), (4, 18)]),
    ]

    fig = plt.figure(figsize=(12.6, 4.4), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(12, 100); ax.axis("off")

    ax.text(4.0, 92.0, "\u2026but not necessary: the fine-tuned cluster survives removing the direction",
            fontsize=19.5, weight="bold", va="center")
    ax.text(4.0, 84.6, "component along d clamped to the base level at every layer and every token  \u00b7  "
                       "pass counts, 3-judge consensus",
            fontsize=11, color=INK2, va="center")

    x0, colw = 35.5, 12.0
    y0, rowh = 68.0, 11.4
    gap = 3.2

    for j, e in enumerate(evals):
        ax.text(x0 + j * colw + colw / 2, y0 + 4.6, e, fontsize=10.4, color=INK2,
                ha="center", va="bottom", linespacing=1.25)

    # shaded panel behind the do-no-harm pair
    panel_top = y0 - 2 * rowh - gap + 0.6
    panel_bot = y0 - 4 * rowh - gap - 1.4
    ax.add_patch(FancyBboxPatch((4.0, panel_bot), 92.5, panel_top - panel_bot,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 facecolor="#f0efec", edgecolor="none", zorder=0))
    ax.text(5.6, panel_top - 2.6, "DO-NO-HARM CONTROL", fontsize=9.6, color=MUTED,
            va="top", weight="bold")

    for i, (label, ring, cells) in enumerate(rows):
        yc = y0 - i * rowh - rowh / 2 - (gap if i >= 2 else 0)
        ax.text(x0 - 1.8, yc, label, fontsize=11.8, ha="right", va="center",
                weight="bold" if ring else "normal",
                color=INK if i < 2 else INK2)
        for j, (k, n) in enumerate(cells):
            rate = k / n
            cx = x0 + j * colw
            fill = CMAP(rate * 0.92)
            box = FancyBboxPatch((cx + 0.6, yc - rowh / 2 + 1.0),
                                 colw - 1.2, rowh - 2.0,
                                 boxstyle="round,pad=0,rounding_size=0.9",
                                 facecolor=fill if rate > 0 else "white",
                                 edgecolor=GRID if rate == 0 else "none", lw=0.8,
                                 zorder=2)
            ax.add_patch(box)
            ax.text(cx + colw / 2, yc, f"{k}/{n}", fontsize=11.8, ha="center",
                    va="center", color="white" if rate > 0.52 else INK,
                    weight="bold" if ring else "normal", zorder=3)
        if ring:
            r = FancyBboxPatch((x0 + 0.2, yc - rowh / 2 + 0.55),
                               5 * colw - 0.4, rowh - 1.1,
                               boxstyle="round,pad=0,rounding_size=1.2",
                               facecolor="none", edgecolor=BLUE, lw=2.2, zorder=4)
            ax.add_patch(r)

    fig.savefig(f"{HERE}/slide3_necessity.png", facecolor=SURFACE)
    plt.close(fig)


fig_necessity()
print("fig3 done")


# ================================================================ figure 4
def fig_chain():
    """Slide 5: the LoRA write chain. Left: drifting-direction schematic.
    Right: measured |cos| matrix between per-layer write PC1s."""
    import numpy as np
    M = np.load(f"{HERE}/pc1_cos_matrix.npy")
    layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    fig = plt.figure(figsize=(12.6, 5.4), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(3.4, 93.0, "The adapter writes one direction per layer — a chain that drifts with depth",
            fontsize=18.5, weight="bold", va="center")
    ax.text(3.4, 86.2, "PC1 of the adapter’s residual-stream write, per adapted layer  ·  "
                       "PC1 alone carries 67–99% of the write variance (rank ≈ 1)",
            fontsize=10.8, color=INK2, va="center")

    # ---- left: schematic arrows, direction rotating with depth ----
    ax1 = fig.add_axes([0.020, 0.16, 0.46, 0.46])
    ax1.set_xlim(-1.6, 34.4); ax1.set_ylim(-3.2, 7.8)
    ax1.set_aspect("equal"); ax1.axis("off")
    L = 4.8
    for i in range(10):
        th = np.radians(90 - i * 10)
        cx = 1.6 + i * 3.3
        dx, dy = L / 2 * np.cos(th), L / 2 * np.sin(th)
        col = CMAP(0.22 + 0.66 * i / 9)
        ax1.add_patch(FancyArrowPatch((cx - dx, 1.9 - dy), (cx + dx, 1.9 + dy),
                      arrowstyle="->", mutation_scale=17, color=col, lw=3.2))
        ax1.text(cx, -2.4, f"L{layers[i]}", fontsize=9.5, color=INK2,
                 ha="center", va="center")
    ax1.text(16.5, 6.6, "write direction", fontsize=10.2, color=MUTED,
             ha="center", va="center", style="italic")

    # ---- right: measured cosine matrix ----
    n = len(layers)
    side = 0.60  # axes height fraction
    w_in = side * 5.4  # inches (square)
    ax2 = fig.add_axes([0.535, 0.075, w_in / 12.6, side])
    ax2.set_xlim(0, n); ax2.set_ylim(0, n); ax2.set_aspect("equal")
    ax2.axis("off")
    for i in range(n):        # row (top to bottom)
        for j in range(n):
            v = M[i, j]
            ax2.add_patch(FancyBboxPatch((j + 0.04, n - 1 - i + 0.04), 0.92, 0.92,
                          boxstyle="round,pad=0,rounding_size=0.10",
                          facecolor=CMAP(v * 0.92), edgecolor="none"))
            ax2.text(j + 0.5, n - 1 - i + 0.5, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                     fontsize=7.6, ha="center", va="center",
                     color="white" if v > 0.52 else (INK if v >= 0.3 else MUTED))
    for k, l in enumerate(layers):
        ax2.text(k + 0.5, -0.45, f"L{l}", fontsize=8.8, color=INK2, ha="center", va="top")
        ax2.text(-0.25, n - 1 - k + 0.5, f"L{l}", fontsize=8.8, color=INK2,
                 ha="right", va="center")
    ax2.set_title("cosine similarity between layers’ write directions",
                  fontsize=10.2, color=INK2, pad=8)

    fig.savefig(f"{HERE}/slide5_lora_chain.png", facecolor=SURFACE)
    plt.close(fig)


fig_chain()
print("fig4 done")


# ================================================================ figure 5
def fig_orthogonal():
    """Slide 6-side visual v2: bigger text, no dots, card background."""
    rows = [
        ("consciousness dir.  (base)", 0.087),
        ("consciousness dir.  (fine-tuned)", 0.109),
        ("assistant axis", 0.048),
        ("surprisal direction", 0.052),
    ]
    twin = 0.88  # mean of per-layer mean-write cosines (.71-.95)

    CARD = "#f0efec"
    BARGRAY = "#a29f97"

    fig = plt.figure(figsize=(6.6, 6.0), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1.2, 1.2), 97.6, 97.6,
                 boxstyle="round,pad=0,rounding_size=2.4",
                 facecolor=CARD, edgecolor="#dedcd4", lw=1.4))

    ax.text(6, 92.0, "cos( LoRA write direction ,  \u2026 )",
            fontsize=17, weight="bold", va="center")

    x0, xmax_u = 44.0, 44.0
    y = 78.0; barh = 9.6; gap = 3.6

    for label, val in rows:
        w = max(val * xmax_u, 1.0)
        ax.text(x0 - 2.6, y, label, fontsize=12.5, ha="right", va="center",
                color=INK)
        ax.add_patch(FancyBboxPatch((x0, y - barh / 2), w, barh,
                     boxstyle="round,pad=0,rounding_size=1.0",
                     facecolor=BARGRAY, edgecolor="none", zorder=2))
        ax.text(x0 + w + 2.0, y, f"{val:.2f}".lstrip("0"), fontsize=13,
                weight="bold", color=INK2, va="center")
        y -= barh + gap

    # separator
    y -= 2.6
    ax.plot([6, 94], [y + 5.2, y + 5.2], color="#d5d3ca", lw=1.4)

    # the twin bar
    yt = y - 2.2
    ax.text(x0 - 2.6, yt, "the DENIAL fine-tune\u2019s\nwrite direction", fontsize=12.5,
            ha="right", va="center", color=INK, weight="bold", linespacing=1.3)
    ax.add_patch(FancyBboxPatch((x0, yt - barh / 2), twin * xmax_u, barh,
                 boxstyle="round,pad=0,rounding_size=1.0",
                 facecolor=BLUE, edgecolor="none", zorder=2))
    ax.text(x0 + twin * xmax_u + 2.0, yt, ".71\u2013.95", fontsize=13.5,
            color=BLUE, weight="bold", va="center")

    # axis + chance line
    yax = yt - 10.6
    ax.plot([x0, x0 + xmax_u], [yax, yax], color="#b6b4aa", lw=1.4)
    for t, lab in [(0, "0"), (0.5, "0.5"), (1.0, "1")]:
        ax.plot([x0 + t * xmax_u] * 2, [yax, yax - 1.4], color="#b6b4aa", lw=1.4)
        ax.text(x0 + t * xmax_u, yax - 4.2, lab, fontsize=11, color=INK2, ha="center")
    xc = x0 + 0.02 * xmax_u
    ax.plot([xc, xc], [yax + 1.6, 83.2], color=MUTED, lw=1.1, ls=(0, (2, 2)), zorder=1)
    ax.text(xc + 1.2, 84.8, "chance", fontsize=10.5, color=MUTED, ha="left")

    fig.savefig(f"{HERE}/slide6_orthogonality.png", facecolor="none", transparent=True)
    plt.close(fig)


fig_orthogonal()
print("fig5 done")
