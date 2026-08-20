import numpy as np

from hexzero.board import HexBoard
from hexzero.mcts import MCTS
from hexzero.network import PolicyValueNetwork


SHADES = " .:-=+*#%@"


def render_policy(π: np.ndarray, board_size: int) -> str:
    grid = π.reshape(board_size, board_size)
    peak = grid.max()

    lines = []
    for row in range(board_size):
        chars = []
        for col in range(board_size):
            if peak > 0:
                level = int(grid[row, col] / peak * (len(SHADES) - 1))
            else:
                level = 0
            chars.append(SHADES[level])
        lines.append(" " * row + " ".join(chars))

    return "\n".join(lines)


def render_policy_numeric(π: np.ndarray, board_size: int) -> str:
    grid = π.reshape(board_size, board_size)

    lines = []
    for row in range(board_size):
        cells = [f"{grid[row, col]:.2f}" for col in range(board_size)]
        lines.append("  " * row + " ".join(cells))

    return "\n".join(lines)


def opening_policy(network: PolicyValueNetwork, board_size: int, simulations: int) -> np.ndarray:
    board = HexBoard(board_size)
    mcts = MCTS(board, simulations, network)
    π, _ = mcts.search()
    return π


def show_opening(
    network: PolicyValueNetwork, board_size: int, simulations: int) -> np.ndarray:
    π = opening_policy(network, board_size, simulations)
    print(render_policy(π, board_size))
    return π