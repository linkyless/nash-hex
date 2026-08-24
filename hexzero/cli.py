from hexzero.network import PolicyValueNetwork
from hexzero.mcts import MCTS
from hexzero.board import HexBoard
import numpy as np
import random
import torch
 
 
if __name__ == "__main__":
 
    board_size = 5
    
    network = PolicyValueNetwork(5)
    network.load_state_dict(torch.load("checkpoints/iter_29.pt"))
    network.eval()
    zero_wins = 0
    matches = 50
    for _ in range(matches):
        Engine = HexBoard(board_size)
        while Engine.winner is None:

            Engine.print_board()
            player = Engine.current_player()       
            if player == -1: 
                row, col = map(int, input("Tu movimiento (fila columna): ").split())
                Engine.place(row - 1, col - 1)
            else:
                mcts = MCTS(Engine, 800, network, root_mix=0.0)
                (_, move) = mcts.search()
                (row, col) = Engine.index_to_cell(move)
                Engine.place(row, col)

        zero_won = (Engine.winner == -1)
        zero_wins += zero_won
        Engine.print_board()
        Engine.print_winner()

    print(f"Win-rate: {zero_wins / matches}")

