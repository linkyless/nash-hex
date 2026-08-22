import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon


BOARD_SIZE = 5
RESULTS    = Path("results")

TRAINING   = Path("training_log.csv")
OPENINGS   = Path("opening_policies.csv")
LADDER     = RESULTS / "ladder.csv"
MILESTONES = RESULTS / "milestones.csv"

INK     = "#121110"
BOARD   = "#262219"
IVORY   = "#e6dcc4"
VERD    = "#4f8f80"
BRASS   = "#c9a227"
MUTE    = "#6f6a61"
FAINT   = "#2c2926"

UNIFORM = math.log(BOARD_SIZE * BOARD_SIZE)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  INK,
        "axes.facecolor":    INK,
        "savefig.facecolor": INK,
        "font.family":       "monospace",
        "font.monospace":    ["DejaVu Sans Mono"],
        "font.size":         8,
        "text.color":        MUTE,
        "axes.labelcolor":   MUTE,
        "axes.edgecolor":    FAINT,
        "axes.linewidth":    1.0,
        "xtick.color":       MUTE,
        "ytick.color":       MUTE,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.frameon":    False,
        "legend.fontsize":   7,
        "lines.solid_capstyle": "butt",
    })


def strip(axes) -> None:
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.tick_params(length=3, width=1.0, pad=5)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def last_run(rows: list[dict]) -> list[dict]:
    starts = [i for i, row in enumerate(rows)
              if row["iteration"] == "0" and row["epoch"] == "0"]
    return rows[starts[-1]:] if starts else rows


def plot_ladder() -> None:
    if not LADDER.exists():
        print(f"skipping the ladder, {LADDER} is not there yet")
        return

    rows = read_rows(LADDER)
    x    = [int(row["iteration"]) for row in rows]
    elo  = [float(row["elo"]) for row in rows]
    low  = [float(row["elo_low"]) for row in rows]
    high = [float(row["elo_high"]) for row in rows]

    figure, axes = plt.subplots(figsize=(6.4, 3.4))
    axes.fill_between(x, low, high, color=BRASS, alpha=0.14, linewidth=0)
    axes.plot(x, elo, color=BRASS, linewidth=1.8)
    axes.scatter(x, elo, color=BRASS, s=14, zorder=3)
    axes.axhline(0, color=FAINT, linewidth=1.0)

    axes.annotate(f"{elo[-1]:+.0f}", xy=(x[-1], elo[-1]), xytext=(-4, 8),
                  textcoords="offset points", color=IVORY, fontsize=9, ha="right")

    axes.set_xlabel("self play iteration")
    axes.set_ylabel("elo, first checkpoint anchored at zero")
    axes.set_xticks(x)
    strip(axes)
    figure.tight_layout()
    figure.savefig(RESULTS / "elo.png", dpi=220)
    plt.close(figure)
    print("wrote results/elo.png")


def plot_loss() -> None:
    if not TRAINING.exists():
        print(f"skipping the losses, {TRAINING} is not there yet")
        return

    rows = last_run(read_rows(TRAINING))
    iterations = sorted({int(row["iteration"]) for row in rows})

    def pick(field: str, epoch: str) -> list[float]:
        table = {(int(r["iteration"]), r["epoch"]): float(r[field]) for r in rows}
        last  = max(r["epoch"] for r in rows)
        key   = epoch if epoch != "last" else last
        return [table.get((i, key), float("nan")) for i in iterations]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)

    top.axhline(UNIFORM, color=FAINT, linewidth=1.0, linestyle=(0, (2, 3)))
    top.annotate("a flat policy over 25 cells", xy=(iterations[0], UNIFORM),
                 xytext=(2, 5), textcoords="offset points", color=MUTE,
                 fontsize=7, ha="left")
    top.plot(iterations, pick("policy_loss", "0"), color=IVORY, linewidth=1.6,
             label="first pass over fresh games")
    top.plot(iterations, pick("policy_loss", "last"), color=MUTE, linewidth=1.2,
             label="last pass")
    top.set_ylabel("policy loss")
    top.legend(loc="lower left", labelcolor=MUTE)
    strip(top)

    bottom.plot(iterations, pick("value_loss", "0"), color=VERD, linewidth=1.6,
                label="first pass over fresh games")
    bottom.plot(iterations, pick("value_loss", "last"), color=MUTE, linewidth=1.2,
                label="last pass")
    bottom.set_ylabel("value loss")
    bottom.set_xlabel("self play iteration")
    bottom.legend(loc="upper right", labelcolor=MUTE)
    strip(bottom)

    figure.tight_layout()
    figure.savefig(RESULTS / "loss.png", dpi=220)
    plt.close(figure)
    print("wrote results/loss.png")


def hex_grid(axes, weights: np.ndarray, peak: float) -> None:
    radius = 1.0
    width  = math.sqrt(3) * radius

    for index, weight in enumerate(weights):
        row, col = divmod(index, BOARD_SIZE)
        x = col * width + row * width / 2
        y = -row * 1.5 * radius
        share = 0.0 if peak <= 0 else weight / peak

        patch = RegularPolygon((x, y), numVertices=6, radius=radius,
                               orientation=0, linewidth=0.6,
                               edgecolor=INK,
                               facecolor=blend(BOARD, BRASS, share))
        axes.add_patch(patch)

    axes.set_xlim(-width, (BOARD_SIZE - 1) * width * 1.5 + width)
    axes.set_ylim(-(BOARD_SIZE - 1) * 1.5 * radius - 1.1, 1.1 * radius)
    axes.set_aspect("equal")
    axes.axis("off")


def blend(low: str, high: str, share: float) -> tuple:
    a = np.array(matplotlib.colors.to_rgb(low))
    b = np.array(matplotlib.colors.to_rgb(high))
    return tuple(a + (b - a) * max(0.0, min(1.0, share)))


def plot_openings() -> None:
    if not OPENINGS.exists():
        print(f"skipping the openings, {OPENINGS} is not there yet")
        return

    with OPENINGS.open(newline="") as handle:
        rows = [line for line in csv.reader(handle) if line]

    table = {}
    for line in rows:
        table[int(line[0])] = np.array([float(value) for value in line[1:]])

    keys  = sorted(table)
    picks = keys[:: max(1, len(keys) // 6)][:6]
    if keys[-1] not in picks:
        picks[-1] = keys[-1]

    peak = max(table[key].max() for key in picks)

    figure, columns = plt.subplots(1, len(picks), figsize=(1.35 * len(picks), 1.2))
    for axes, key in zip(np.atleast_1d(columns), picks):
        hex_grid(axes, table[key], peak)
        axes.set_title(f"iteration {key}", color=MUTE, fontsize=7, pad=4)

    figure.tight_layout()
    figure.savefig(RESULTS / "openings.png", dpi=220)
    plt.close(figure)
    print("wrote results/openings.png")


def plot_milestones() -> None:
    if not MILESTONES.exists():
        print(f"skipping the milestones, {MILESTONES} is not there yet")
        return

    rows   = read_rows(MILESTONES)
    labels = [row["label"] for row in rows]
    rates  = [float(row["win_rate"]) * 100 for row in rows]
    spots  = np.arange(len(rows))[::-1]

    def tone(rate: float) -> str:
        if rate > 52: return BRASS
        if rate < 48: return VERD
        return MUTE

    figure, axes = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 0.95))
    axes.barh(spots, rates, height=0.38, color=[tone(rate) for rate in rates])
    axes.axvline(50, color=FAINT, linewidth=1.0, linestyle=(0, (2, 3)))

    for spot, rate in zip(spots, rates):
        axes.annotate(f"{rate:.0f}%", xy=(rate, spot), xytext=(6, 0),
                      textcoords="offset points", color=IVORY,
                      fontsize=8, va="center")

    axes.set_yticks(spots, labels)
    axes.set_xlim(0, 108)
    axes.set_xticks([0, 25, 50, 75, 100])
    axes.set_xlabel("games won against a freshly initialised network")
    axes.spines["left"].set_visible(False)
    strip(axes)
    axes.tick_params(axis="y", length=0)

    figure.tight_layout()
    figure.savefig(RESULTS / "milestones.png", dpi=220)
    plt.close(figure)
    print("wrote results/milestones.png")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    style()

    plot_ladder()
    plot_loss()
    plot_openings()
    plot_milestones()
