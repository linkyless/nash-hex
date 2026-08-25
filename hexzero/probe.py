import numpy as np
import torch
from hexzero.board import HexBoard


def build_probe_positions(board_size: int, n: int = 200, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < n:
        board = HexBoard(board_size)
        plies = int(rng.integers(2, 12))
        for move in rng.permutation(board_size * board_size)[:plies]:
            if board.winner is not None:
                break
            board.place(*board.index_to_cell(int(move)))
        if board.winner is None:
            positions.append(board.transform_to_canonic_form())
    return positions


def policy_diagnostics(network, positions: list[np.ndarray]) -> tuple[float, float]:
    boards = torch.tensor(np.array(positions), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        logits, values = network(boards)

    legal = boards.squeeze(1).reshape(len(positions), -1) == 0
    logits = logits.masked_fill(~legal, -torch.inf)
    probs = torch.softmax(logits, dim=-1)

    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    effective_moves = torch.exp(entropy).median().item()
    value_confidence = values.abs().median().item()
    return (effective_moves, value_confidence)