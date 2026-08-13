from __future__ import annotations
from hexzero.board import HexBoard
import math
import copy

C_PARAM = math.sqrt(2)

class Node:
    def __init__(self, board: HexBoard, move: int = None, parent: Node = None):
        self.parent        = parent
        self.board         = board
        self.move          = move
        self.children      = []
        self.untried_moves = list(board.valid_choices())

        # len(untried_moves) == 0 in order to use UCB1

        self.visits        = 0
        self.value_sum     = 0


    def mean_value(self) -> float:
        assert self.visits > 0
        return self.value_sum / self.visits


    def calculate_ucb1(self) -> float:
        exploitation = self.mean_value()
        exploration  = C_PARAM * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class MCTS:
    def __init__(self, board: HexBoard, number_of_simulations: int):
        self.root        = Node(board)
        self.simulations = number_of_simulations

        
    def search(self) -> int:
        for simulation in range(self.simulations):
            next_node = self._selection()


    def _find_best_child(self, node: Node) -> Node:
        return max(node.children, key=lambda child: child.calculate_ucb1())
    

    def _selection(self) -> Node:
        curr_node  = self.root
        while curr_node.board.winner is None and len(curr_node.untried_moves) == 0:
            curr_node = self._find_best_child(curr_node)
        
        return curr_node

    def _expansion(self, node: Node) -> Node:
        if node.board.winner is not None:
            return node
        
        next_move  = node.untried_moves.pop()
        (row, col) = node.board.cell_to_index(next_move)
        new_board  = copy.deepcopy(node.board)

        new_board.place(row, col)

        new_node = Node(new_board, next_move, node)
        node.children.append(new_node)
        return new_node

    

