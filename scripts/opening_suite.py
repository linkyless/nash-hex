import argparse
from pathlib import Path

import torch

from hexzero.board import HexBoard
from hexzero.network import PolicyValueNetwork
from hexzero.mcts import MCTS
from hexzero.shortest_path_bot import ShortestPathBot

BOARD_SIZE  = 5
SIMULATIONS = 200


def load(path):
    network = PolicyValueNetwork(BOARD_SIZE)
    network.load_state_dict(torch.load(path, map_location="cpu"))
    network.eval()
    return network


def play(network, nash_player, opening):
    board = HexBoard(BOARD_SIZE)
    bot   = ShortestPathBot(-nash_player)
    board.place(*board.index_to_cell(opening))
    while board.winner is None:
        if board.current_player() == nash_player:
            move = board.index_to_cell(MCTS(board, SIMULATIONS, network, root_mix=0.0).search()[1])
        else:
            move = bot.select_best_heuristic_move(board)
        board.place(*move)
    return board.winner == nash_player


def run_suite(network, label):
    total_cells = BOARD_SIZE * BOARD_SIZE
    result = {}
    for side in (1, -1):
        wins = [o for o in range(total_cells) if play(network, side, o)]
        result[side] = wins
        name = "blancas" if side == 1 else "negras"
        print(f"  {label} como {name}: {len(wins)}/{total_cells}")
    result["total"] = len(result[1]) + len(result[-1])
    print(f"  {label} total: {result['total']}/{2 * total_cells}")
    return result


def lost_openings(wins):
    total_cells = BOARD_SIZE * BOARD_SIZE
    return [(o // BOARD_SIZE, o % BOARD_SIZE) for o in range(total_cells) if o not in wins]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()

    results = {}
    for path in args.checkpoints:
        label = path.stem
        print(f"\n=== {label} ===")
        results[label] = run_suite(load(path), label)

    if len(results) > 1:
        print("\n=== comparativa ===")
        for label, r in results.items():
            print(f"{label:16s} blancas {len(r[1]):2d}/25   negras {len(r[-1]):2d}/25   total {r['total']:2d}/50")
        print("\naperturas perdidas de blancas:")
        for label, r in results.items():
            print(f"  {label:16s} {lost_openings(r[1])}")