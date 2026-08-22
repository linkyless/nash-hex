import csv
import math
import time
from pathlib import Path

import torch

from hexzero.arena import get_win_rate
from hexzero.network import PolicyValueNetwork


BOARD_SIZE  = 5
SIMULATIONS = 200
OPENINGS    = 15
STEP        = 5
LAST        = 29

CHECKPOINTS = Path("checkpoints")
RESULTS     = Path("results")
LADDER      = RESULTS / "ladder.csv"


def load(index: int) -> PolicyValueNetwork:
    network = PolicyValueNetwork(BOARD_SIZE)
    network.load_state_dict(torch.load(CHECKPOINTS / f"iter_{index}.pt", map_location="cpu"))
    network.eval()
    return network


def to_elo(win_rate: float) -> float:
    clipped = min(max(win_rate, 0.005), 0.995)
    return -400.0 * math.log10(1.0 / clipped - 1.0)


def rungs() -> list[int]:
    steps = list(range(0, LAST + 1, STEP))
    if steps[-1] != LAST:
        steps.append(LAST)
    return steps


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    steps = rungs()
    games = OPENINGS * 2

    with LADDER.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "opponent", "win_rate", "games", "elo", "elo_low", "elo_high"])
        writer.writerow([steps[0], "", 0.5, 0, 0.0, 0.0, 0.0])

        elo = 0.0

        for older, newer in zip(steps, steps[1:]):
            start = time.perf_counter()
            _, win_rate = get_win_rate(load(older), load(newer), BOARD_SIZE, SIMULATIONS, OPENINGS)
            spent = time.perf_counter() - start

            error = math.sqrt(max(win_rate * (1 - win_rate), 1e-6) / games)
            elo  += to_elo(win_rate)
            low   = elo - (to_elo(win_rate) - to_elo(win_rate - error))
            high  = elo + (to_elo(win_rate + error) - to_elo(win_rate))

            writer.writerow([newer, older, round(win_rate, 4), games,
                             round(elo, 1), round(low, 1), round(high, 1)])
            handle.flush()

            print(f"iter {newer:>2} beats iter {older:>2} {win_rate:.0%} of {games} games "
                  f"and sits at {elo:+.0f} elo, measured in {spent:.0f}s")

    print(f"\nwrote {LADDER}")
