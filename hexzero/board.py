import numpy as np
from hexzero.ufds import UFDS

class HexBoard:
    
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board      = np.zeros((self.board_size, self.board_size), dtype=int)
        self.turn       = 0

        self.ufds_white = UFDS(self.board_size ** 2 + 2) # Virtual Nodes: North & South
        self.ufds_black = UFDS(self.board_size ** 2 + 2) # Virtual Nodes: West  & East
        self.north_cell = self.board_size ** 2
        self.south_cell = self.board_size ** 2 + 1
        self.east_cell  = self.board_size ** 2
        self.west_cell  = self.board_size ** 2 + 1

        self.winner = None

        self.d_row = np.array([0,  0, -1, 1, -1,  1])
        self.d_col = np.array([1, -1,  0, 0,  1, -1])

    def current_player(self) -> int:
        if self.turn % 2 == 0:
            return 1
        else:
            return -1


    def print_winner(self) -> None:
        print(f"Ha ganado el jugador {self.winner}!")

    def print_board(self) -> None:
        for row in range(self.board_size):
            row_chars = []
            for cell in self.board[row]:
                if cell == 1:
                    row_chars.append("W")  # White Piece
                elif cell == -1:
                    row_chars.append("B")  # Black Piece
                else:
                    row_chars.append(".")  # Empty Cell
            
            print(" " * row + " ".join(row_chars))


    def translate(self, row: int, col: int) -> int:
        return row * self.board_size + col


    def valid_choices(self) -> np.ndarray:
        return np.flatnonzero(self.board == 0)
    

    def set_winner(self, player: int) -> None:
        # Check white's winning
        if player == 1:
            if self.ufds_white.related(self.north_cell, self.south_cell):
                self.winner = 1

        # Check black's winning
        else:
            if self.ufds_black.related(self.east_cell, self.west_cell):
                self.winner = -1
    

    def is_in_bounds(self, row: int, col: int) -> bool:
        return row >= 0 and col >= 0 and row < self.board_size and col < self.board_size


    def is_valid_play(self, row: int, col: int) -> bool:
        return self.is_in_bounds(row, col) and self.board[row, col] == 0

    def join_neighbors(self, player: int, row: int, col: int):
        number_of_neighbors = 6
        current_cell = self.translate(row, col)
        for neighbor in range(number_of_neighbors):
            new_row = row + self.d_row[neighbor]
            new_col = col + self.d_col[neighbor]

            # Connect Neighbors
            if self.is_in_bounds(new_row, new_col):
                new_cell = self.translate(new_row, new_col)
                if self.board[new_row, new_col] == player:
                    if player == 1:
                        self.ufds_white.join(current_cell, new_cell)
                    else:
                        self.ufds_black.join(current_cell, new_cell)

        # Connect Edges
        if player == 1:
            if row == 0:
                self.ufds_white.join(current_cell, self.north_cell)

            if row == self.board_size - 1:
                self.ufds_white.join(current_cell, self.south_cell)

        else:
            if col == 0:
                self.ufds_black.join(current_cell, self.west_cell)

            if col == self.board_size - 1:
                self.ufds_black.join(current_cell, self.east_cell)


    def place(self, row: int, col: int):
        player = self.current_player()
        self.board[row, col] = player # 0-indexed
        self.join_neighbors(player, row, col)
        self.set_winner(player)

        self.turn += 1


