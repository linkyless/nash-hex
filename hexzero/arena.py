from hexzero.network import PolicyValueNetwork
from hexzero.mcts import MCTS
from hexzero.board import HexBoard
import numpy as np

HOW_MANY_INITIAL_RANDOM_MOVES = 2

def get_win_rate(network1: PolicyValueNetwork, network2: PolicyValueNetwork, board_size: int, simulations: int, number_of_openings: int) -> tuple[float, float]:
 
    network1_wins = 0
    network2_wins = 0
    total_matches = 0
 
    for _ in range(number_of_openings):
        # generate a random opening and store it for reuse.
        opening = []
        temp_board = HexBoard(board_size)
        for __ in range(HOW_MANY_INITIAL_RANDOM_MOVES):
            move = int(np.random.choice(temp_board.valid_choices()))
            row, col = temp_board.index_to_cell(move)
            temp_board.place(row, col)
            opening.append(move)
 
        
        for net1_is_white in (True, False):
            moves = []
            moves.append("WHITE" if net1_is_white else "BLACK")
 
            Board = HexBoard(board_size)
            for move in opening:
                row, col = Board.index_to_cell(move)
                Board.place(row, col)
                moves.append(int(move))
 
            while Board.winner is None:
                player = Board.current_player()
                white_moves = (player == 1)
                current_net = network1 if white_moves == net1_is_white else network2
                mcts = MCTS(Board, simulations, current_net)
                (pi, move) = mcts.search()
                (row, col) = Board.index_to_cell(move)
                moves.append(int(move))
                Board.place(row, col)
 
            white_won = (Board.winner == 1)
            if net1_is_white == white_won:
                network1_wins += 1
            else:
                network2_wins += 1
 
            total_matches += 1
            print(moves)
 
    return (network1_wins / total_matches, network2_wins / total_matches)
