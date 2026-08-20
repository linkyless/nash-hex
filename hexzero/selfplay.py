from hexzero.network import PolicyValueNetwork
from hexzero.mcts import Node, MCTS
from hexzero.board import HexBoard
import torch
import numpy as np

TEMPERATURE_MOVES = 8

def give_examples_of_a_match(board_size: int, network: PolicyValueNetwork, simulations: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    examples = []
    Engine = HexBoard(board_size)
    total_cells = board_size * board_size
    count = 1
    while Engine.winner is None:
        MonteCarlo = MCTS(Engine, simulations, network)
        (π, best_move) = MonteCarlo.search()
        
        examples.append((Engine.transform_to_canonic_form(), π))

        best_move = best_move if count > TEMPERATURE_MOVES else int(np.random.choice(total_cells, p=π))

        (row, col) = Engine.index_to_cell(best_move)
        Engine.place(row, col)

        count += 1

    final_examples = []
    mult = 1
    for board, π in reversed(examples):
        final_examples.append((board, π, mult))
        mult = -mult

    return final_examples


