from hexzero.board import HexBoard
from hexzero.mcts import MCTS

if __name__ == "__main__":

    board_size = 5
    Engine = HexBoard(board_size)

    while Engine.winner is None:
        if Engine.current_player() == -1:
            MonteCarlo = MCTS(Engine, 500)
            (row, col) = Engine.index_to_cell(MonteCarlo.search())
            Engine.place(row, col)

        else:
            Engine.print_board()
            x, y = map(int, input("Selecciona coordenadas x e y: ").split())
            while not Engine.is_valid_play(x - 1, y - 1):
                x, y = map(int, input("No funciona. Selecciona coordenadas x e y: ").split())
            
            row = x - 1
            col = y - 1

            Engine.place(row, col)
        

    Engine.print_board()
    Engine.print_winner()



