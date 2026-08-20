from __future__ import annotations
from hexzero.board import HexBoard
import math
import copy
import numpy as np
import torch
from hexzero.network import PolicyValueNetwork

C_PARAM = 1.0

class Node:
    def __init__(self, board: HexBoard, prior: float = None, move: int = None, parent: Node = None):
        self.parent        = parent
        self.board         = board
        self.move          = move
        self.children      = []

        self.prior         = prior

        self.visits        = 0
        self.value_sum     = 0


    def mean_value(self) -> float:
        return self.value_sum / self.visits


    def calculate_ucb1(self) -> float:
        exploitation = self.mean_value()
        exploration  = C_PARAM * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


    def calculate_puct(self) -> float:
        exploitation = self.mean_value() if self.visits > 0 else 0.0
        exploration  = C_PARAM * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration


class MCTS:
    def __init__(self, board: HexBoard, number_of_simulations: int, network: PolicyValueNetwork):
        self.root        = Node(board)
        self.simulations = number_of_simulations
        self.network     = network


    def search(self) -> tuple[np.ndarray, int]:
        for _ in range(self.simulations):
            next_node   = self._selection()
            value       = self._expansion(next_node)
            self._backup(next_node, value)

        return (self._get_pi(), max(self.root.children, key=lambda child: child.visits).move)


    def _find_best_child(self, node: Node) -> Node:
        return max(node.children, key=lambda child: child.calculate_puct())


    def _get_pi(self) -> np.ndarray:
        total_visits = sum(node.visits for node in self.root.children)
    #   total_visits = self.root.visits - 1

        pi = np.zeros(self.root.board.board_size * self.root.board.board_size)

        for child in self.root.children:
            pi[child.move] = child.visits
        
        return pi / total_visits


    def _sum_backup_points_to_node(self, node: Node, value: float, level: int) -> None:
        if level % 2 == 0:
            node.value_sum += value
        else:
            node.value_sum -= value

        node.visits += 1
    

    def _selection(self) -> Node:
        curr_node  = self.root
        while curr_node.board.winner is None and len(curr_node.children) > 0:
            curr_node = self._find_best_child(curr_node)
        
        return curr_node


    def _expansion(self, node: Node) -> float:
        if node.board.winner is not None:
            return 1
        total_cells = node.board.board_size * node.board.board_size
        black_moves = (node.board.current_player() == -1)
        curr_board = node.board.transform_to_canonic_form()

        tensor = torch.from_numpy(curr_board).reshape(1, 1, node.board.board_size, node.board.board_size).float()

        with torch.no_grad():
            policy_logits, value = self.network(tensor)
            policy_logits = policy_logits.squeeze(0)

        if black_moves:
            policy_logits = policy_logits.reshape(node.board.board_size, node.board.board_size).T.reshape(-1)

        valid = torch.zeros(total_cells, dtype=torch.bool)
        choices = node.board.valid_choices()
        valid[choices] = True

        policy_logits[~valid] = -torch.inf
        policy = torch.softmax(policy_logits, dim=0)
        value = value.item()

        for choice in choices:
            [row, col] = node.board.index_to_cell(choice)
            new_board  = node.board.copy()
            new_board.place(row, col)
            new_node = Node(new_board, policy[choice].item(), choice, node)
            node.children.append(new_node)

        return value


    def _backup(self, node: Node, value: float) -> None:
        # root.value_sum is nothing
        level = 0
        self._sum_backup_points_to_node(node, value, level)
        while node.parent is not None:
            node = node.parent
            level += 1
            self._sum_backup_points_to_node(node, value, level)
