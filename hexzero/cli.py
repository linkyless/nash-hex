from hexzero.board import HexBoard
from hexzero.mcts import MCTS
import random
import torch
from hexzero.network import PolicyValueNetwork 
from hexzero.selfplay import give_examples_of_a_match

if __name__ == "__main__":

    board_size = 5
    
    zero_wins = 0
    network = PolicyValueNetwork(board_size)

    print(give_examples_of_a_match(board_size, network, 800))
        



