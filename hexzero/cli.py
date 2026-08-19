from hexzero.board import HexBoard
from hexzero.mcts import MCTS
import random
import torch
from hexzero.network import PolicyValueNetwork 

if __name__ == "__main__":

    board_size = 5
    
    zero_wins = 0
    for iteration in range(300):
        Engine = HexBoard(board_size)
        while Engine.winner is None:
            if Engine.current_player() == -1:
                MonteCarlo = MCTS(Engine, 500)
                (row, col) = Engine.index_to_cell(MonteCarlo.search())
                Engine.place(row, col)

            else:
                print(f"Iteracion: {iteration}")
                (row, col) = Engine.index_to_cell(random.choice(list(Engine.valid_choices())))
                Engine.place(row, col)

        if Engine.winner == -1:
            zero_wins += 1
        



