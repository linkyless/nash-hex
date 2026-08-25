import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

RESULTS = Path("results")
INPUT   = RESULTS / "tournament.csv"
OUTPUT  = RESULTS / "elo.csv"

SCALE          = np.log(10.0) / 400.0
RATING_PRIOR   = 600.0
ADVANTAGE_PRIOR = 400.0


def load(path):
    names = {}
    white, black, won = [], [], []

    with open(path) as handle:
        for row in csv.DictReader(handle):
            for side in (row["white"], row["black"]):
                names.setdefault(side, len(names))
            white.append(names[row["white"]])
            black.append(names[row["black"]])
            won.append(int(row["white_won"]))

    order = sorted(names, key=names.get)
    return order, np.array(white), np.array(black), np.array(won, dtype=float)


def fit(count, white, black, won):
    def objective(theta):
        ratings   = theta[:count]
        advantage = theta[count]

        margin = SCALE * (ratings[white] + advantage - ratings[black])
        # Stable log(1 + exp(-x)) at both extremes
        loss = np.logaddexp(0.0, -margin) * won + np.logaddexp(0.0, margin) * (1.0 - won)
        total = loss.sum()
        total += 0.5 * np.sum(ratings ** 2) / RATING_PRIOR ** 2
        total += 0.5 * advantage ** 2 / ADVANTAGE_PRIOR ** 2

        probability = 1.0 / (1.0 + np.exp(-margin))
        residual    = SCALE * (probability - won)

        gradient = np.zeros_like(theta)
        np.add.at(gradient, white, residual)
        np.add.at(gradient, black, -residual)
        gradient[count] = residual.sum()
        gradient[:count] += ratings / RATING_PRIOR ** 2
        gradient[count]  += advantage / ADVANTAGE_PRIOR ** 2

        return total, gradient

    start  = np.zeros(count + 1)
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10})
    return result.x[:count], result.x[count], result


def covariance(ratings, advantage, count, white, black):
    margin      = SCALE * (ratings[white] + advantage - ratings[black])
    probability = 1.0 / (1.0 + np.exp(-margin))
    weight      = SCALE ** 2 * probability * (1.0 - probability)

    hessian = np.zeros((count + 1, count + 1))
    for w, b, value in zip(white, black, weight):
        for i, si in ((w, 1.0), (b, -1.0), (count, 1.0)):
            for j, sj in ((w, 1.0), (b, -1.0), (count, 1.0)):
                hessian[i, j] += value * si * sj

    hessian[np.arange(count), np.arange(count)] += 1.0 / RATING_PRIOR ** 2
    hessian[count, count] += 1.0 / ADVANTAGE_PRIOR ** 2
    return np.linalg.inv(hessian)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--anchor", default="dijkstra")
    args = parser.parse_args()

    names, white, black, won = load(args.input)
    count = len(names)

    ratings, advantage, result = fit(count, white, black, won)
    cov = covariance(ratings, advantage, count, white, black)

    anchor = names.index(args.anchor) if args.anchor in names else int(np.argmin(ratings))
    shifted = ratings - ratings[anchor]

    error = np.sqrt(np.maximum(
        np.diag(cov)[:count] + cov[anchor, anchor] - 2.0 * cov[:count, anchor], 0.0))
    error[anchor] = 0.0

    played = np.zeros(count)
    scored = np.zeros(count)
    np.add.at(played, white, 1.0)
    np.add.at(played, black, 1.0)
    np.add.at(scored, white, won)
    np.add.at(scored, black, 1.0 - won)

    order = np.argsort(-shifted)

    print(f"\nfirst-player advantage: {advantage:+.0f} Elo")
    print(f"games: {len(won)}   convergence: {result.success}\n")
    print(f"{'player':<18}{'elo':>8}{'  +/-':>8}{'games':>10}{'score':>9}")
    for index in order:
        print(f"{names[index]:<18}{shifted[index]:>8.0f}{error[index]:>8.0f}"
              f"{played[index]:>10.0f}{scored[index] / played[index]:>9.3f}")

    RESULTS.mkdir(exist_ok=True)
    with OUTPUT.open("w") as handle:
        handle.write("player,elo,error,games,score\n")
        for index in order:
            handle.write(f"{names[index]},{shifted[index]:.2f},{error[index]:.2f},"
                         f"{played[index]:.0f},{scored[index] / played[index]:.4f}\n")
    print(f"\n-> {OUTPUT}   (opening advantage {advantage:+.1f})")