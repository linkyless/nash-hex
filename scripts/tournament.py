import argparse
import itertools
import os
import time
from pathlib import Path

import torch

import hexzero.mcts as mcts_module
from hexzero.board import HexBoard
from hexzero.mcts import MCTS
from hexzero.network import PolicyValueNetwork
from hexzero.shortest_path_bot import ShortestPathBot

BOARD_SIZE  = 5
TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE
SIMULATIONS = 200
C_PARAM     = 1.0
BOT         = "dijkstra"

RESULTS = Path("results")
OUTPUT  = RESULTS / "tournament.csv"

_networks = {}
_labels   = {}


def _register(argument):
    """Accepts 'path.pt' or 'label=path.pt'."""
    if "=" in argument:
        label, path = argument.split("=", 1)
    else:
        path, label = argument, Path(argument).stem
    _labels[path] = label
    return path


def _network(path):
    if path not in _networks:
        network = PolicyValueNetwork(BOARD_SIZE)
        network.load_state_dict(torch.load(path, map_location="cpu"))
        network.eval()
        _networks[path] = network
    return _networks[path]


def _move(player, board):
    if player == BOT:
        return ShortestPathBot(board.current_player()).select_best_heuristic_move(board)
    network = _network(player)
    return board.index_to_cell(MCTS(board, SIMULATIONS, network, root_mix=0.0).search()[1])


def _game(task):
    torch.set_num_threads(1)
    mcts_module.C_PARAM = C_PARAM
    white, black, opening = task

    board = HexBoard(BOARD_SIZE)
    board.place(*board.index_to_cell(opening))

    while board.winner is None:
        player = white if board.current_player() == 1 else black
        move   = _move(player, board)
        board.place(*move)

    return (white, black, opening, 1 if board.winner == 1 else 0)


def _label(player):
    return BOT if player == BOT else _labels.get(player, Path(player).stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+",
                        help="path.pt or label=path.pt")
    parser.add_argument("--no-bot", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    players = [_register(argument) for argument in args.checkpoints]
    if not args.no_bot:
        players.append(BOT)

    tasks = []
    for one, other in itertools.combinations(players, 2):
        for opening in range(TOTAL_CELLS):
            tasks.append((one, other, opening))
            tasks.append((other, one, opening))

    print(f"{len(players)} players, {len(tasks)} games, {args.workers} processes")
    start = time.perf_counter()

    from multiprocessing import Pool

    rows = []
    with Pool(args.workers) as pool:
        for done, row in enumerate(pool.imap_unordered(_game, tasks, chunksize=4), 1):
            rows.append(row)
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}  ({time.perf_counter() - start:.0f}s)")

    RESULTS.mkdir(exist_ok=True)
    with OUTPUT.open("w") as handle:
        handle.write("white,black,opening,white_won\n")
        for white, black, opening, won in rows:
            handle.write(f"{_label(white)},{_label(black)},{opening},{won}\n")

    print(f"\n{len(rows)} games in {time.perf_counter() - start:.0f}s -> {OUTPUT}")