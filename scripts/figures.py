import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

BOARD_SIZE  = 5
TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE
BOT         = "dijkstra"

RESULTS = Path("results")

INK    = "#121110"
BOARD  = "#262219"
RULE   = "#2c2926"
IVORY  = "#e6dcc4"
VERD   = "#4f8f80"
BRASS  = "#c9a227"
MUTE   = "#6f6a61"

plt.rcParams.update({
    "figure.facecolor": INK,
    "axes.facecolor":   INK,
    "savefig.facecolor": INK,
    "text.color":       MUTE,
    "axes.labelcolor":  MUTE,
    "xtick.color":      MUTE,
    "ytick.color":      MUTE,
    "font.family":      "monospace",
    "font.size":        9,
    "axes.edgecolor":   RULE,
})


def strip(axes, keep=()):
    for side in ("top", "right", "bottom", "left"):
        axes.spines[side].set_visible(side in keep)


def read_elo():
    with open(RESULTS / "elo.csv") as handle:
        return list(csv.DictReader(handle))


def read_tournament():
    with open(RESULTS / "tournament.csv") as handle:
        return list(csv.DictReader(handle))


def elo_figure():
    rows = read_elo()[::-1]
    names = [row["player"] for row in rows]
    elo   = np.array([float(row["elo"]) for row in rows])
    error = np.array([float(row["error"]) for row in rows])

    figure, axes = plt.subplots(figsize=(9, 0.7 * len(rows) + 1.6))
    position = np.arange(len(rows))

    colour = [VERD if name == BOT else BRASS for name in names]
    axes.errorbar(elo, position, xerr=error, fmt="none",
                  ecolor=RULE, elinewidth=6, capsize=0, zorder=1)
    axes.scatter(elo, position, s=90, c=colour, zorder=2)

    for y, (value, spread) in enumerate(zip(elo, error)):
        axes.annotate(f"{value:+.0f}" + (f" ± {spread:.0f}" if spread else ""),
                      (value, y), textcoords="offset points", xytext=(14, 0),
                      va="center", color=IVORY, fontsize=9)

    axes.axvline(0, color=RULE, linewidth=1)
    axes.set_yticks(position, names)
    axes.set_xlabel("Elo, adjusted across all tournament games")
    axes.set_ylim(-0.7, len(rows) - 0.3)
    axes.set_xlim(min(elo - error) - 90, max(elo + error) + 150)
    strip(axes, keep=("bottom",))
    axes.tick_params(left=False)

    figure.tight_layout()
    figure.savefig(RESULTS / "elo_tournament.png", dpi=200)
    plt.close(figure)


def bot_results():
    """{player: {1: [won openings], -1: [...]}} against the bot."""
    table = {}
    for row in read_tournament():
        white, black = row["white"], row["black"]
        if BOT not in (white, black) or white == black:
            continue
        engine = black if white == BOT else white
        side   = 1 if white == engine else -1
        won    = int(row["white_won"]) == (1 if side == 1 else 0)
        table.setdefault(engine, {1: [], -1: []})
        if won:
            table[engine][side].append(int(row["opening"]))
    return table


def bot_figure(table, order):
    figure, axes = plt.subplots(figsize=(9, 0.9 * len(order) + 1.8))
    position = np.arange(len(order))
    height   = 0.36

    white = [len(table[name][1]) for name in order]
    black = [len(table[name][-1]) for name in order]

    axes.barh(position + height / 2, white, height, color=IVORY, label="opening")
    axes.barh(position - height / 2, black, height, color=VERD, label="responding")

    for y, (one, other) in enumerate(zip(white, black)):
        axes.annotate(f"{one}/{TOTAL_CELLS}", (one, y + height / 2), xytext=(6, 0),
                      textcoords="offset points", va="center", color=IVORY)
        axes.annotate(f"{other}/{TOTAL_CELLS}", (other, y - height / 2), xytext=(6, 0),
                      textcoords="offset points", va="center", color=VERD)

    axes.set_yticks(position, order)
    axes.set_xlim(0, TOTAL_CELLS + 3)
    axes.set_xlabel("Forced openings won against the shortest-path bot")
    axes.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=MUTE)
    strip(axes, keep=("bottom",))
    axes.tick_params(left=False)

    figure.tight_layout()
    figure.savefig(RESULTS / "vs_dijkstra.png", dpi=200)
    plt.close(figure)


def hex_patch(axes, row, col, colour, edge):
    x = col + row * 0.5
    y = -row * 0.86
    axes.add_patch(RegularPolygon((x, y), numVertices=6, radius=0.55,
                                  orientation=0.0, facecolor=colour,
                                  edgecolor=edge, linewidth=1.2))


def lost_figure(table, order):
    figure, panels = plt.subplots(1, len(order), figsize=(3.1 * len(order), 3.4))
    panels = np.atleast_1d(panels)

    for axes, name in zip(panels, order):
        lost = set(range(TOTAL_CELLS)) - set(table[name][1])
        for index in range(TOTAL_CELLS):
            row, col = divmod(index, BOARD_SIZE)
            if index in lost:
                hex_patch(axes, row, col, BRASS, INK)
            else:
                hex_patch(axes, row, col, BOARD, RULE)

        axes.set_title(f"{name}\n{len(lost)} lost openings", color=MUTE, pad=14)
        axes.set_xlim(-1, BOARD_SIZE + 2.1)
        axes.set_ylim(-BOARD_SIZE * 0.9, 1)
        axes.set_aspect("equal")
        axes.axis("off")

    figure.tight_layout()
    figure.savefig(RESULTS / "lost_openings.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    elo_figure()

    table = bot_results()
    ranking = [row["player"] for row in read_elo() if row["player"] != BOT]
    order = [name for name in ranking if name in table][::-1]

    if order:
        bot_figure(table, order)
        lost_figure(table, order)

    print("written to results/:")
    print("  elo_tournament.png")
    if order:
        print("  vs_dijkstra.png")
        print("  lost_openings.png")