from hexzero.board import HexBoard
import numpy as np
import heapq

BLOCKED = 999


class ShortestPathBot:
    def __init__(self, player: int):
        self.player = player


    def _dijkstra(self, board: HexBoard, player: int) -> float:
        size = board.board_size
        pq = []
        # pq = (dist, r, c)
        dist = np.full((size, size), np.inf)
        rival = -player

        for i in range(size):
            (r, c) = (0, i) if player == 1 else (i, 0)
            cell = board.board[r, c]
            if cell == rival:
                continue
            d = 0 if cell == player else 1
            dist[r, c] = d
            heapq.heappush(pq, (d, r, c))

        while pq:
            curr_dist, r, c = heapq.heappop(pq)
            if curr_dist > dist[r, c]:
                continue

            if player ==  1 and r == size - 1:
                return curr_dist
            if player == -1 and c == size - 1:
                return curr_dist

            number_of_neighbors = 6
            for neighbor in range(number_of_neighbors):
                new_r = board.d_row[neighbor] + r
                new_c = board.d_col[neighbor] + c
                if board.is_in_bounds(new_r, new_c):
                    cell = board.board[new_r, new_c]
                    if cell == rival:
                        continue
                    new_dist = curr_dist + (0 if cell == player else 1)
                    if new_dist < dist[new_r, new_c]:
                        dist[new_r, new_c] = new_dist
                        heapq.heappush(pq, (new_dist, new_r, new_c))

        return np.inf


    def select_best_heuristic_move(self, board: HexBoard) -> tuple[int, int]:
        moves  = board.valid_choices()
        player = self.player
        rival  = -player
        center = (board.board_size - 1) / 2

        best_dist = -np.inf
        best_move = None

        for move in moves:
            row, col = board.index_to_cell(move)
            if board.board[row, col] == 0:
                # Player's move
                board.board[row, col] = player
                player_dist = self._dijkstra(board, player)
                rival_dist  = self._dijkstra(board, rival)
                board.board[row, col] = 0

                if not np.isfinite(rival_dist):
                    rival_dist = BLOCKED

                # slight center bias as a tiebreaker
                to_center = abs(row - center) + abs(col - center)
                diff = rival_dist - player_dist - 0.01 * to_center

                if diff > best_dist:
                    best_dist = diff
                    best_move = (int(row), int(col))

        return best_move