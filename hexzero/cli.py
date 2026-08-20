from hexzero.board import HexBoard
 
 
if __name__ == "__main__":
 
    board_size = 5
    Engine = HexBoard(board_size)
 
    while Engine.winner is None:
 
        Engine.print_board()
 
        x, y = map(int, input("Selecciona coordenadas x e y: ").split())
        while not Engine.is_valid_play(x - 1, y - 1):
            x, y = map(int, input("No funciona. Selecciona coordenadas x e y: ").split())
 
        row = x - 1
        col = y - 1
 
        Engine.place(row, col)
 
    Engine.print_board()
    Engine.print_winner()
