from hexzero.board import HexBoard
from hexzero.mcts import MCTS
from hexzero.network import PolicyValueNetwork

def test_finds_winning_move():
    """
    Posicion preparada:

    W . . . B
     W . . . B
      W . . . B
       W . . . B
        . . . . .

    Blancas conectan norte-sur por la columna 0 y les falta (4, 0).
    Negras tienen fichas en la columna 4, que no bloquean nada.
    Mueven blancas. La jugada ganadora es (4, 0) -> indice plano 20.
    """
    board = HexBoard(5)

    # place() alterna el turno solo, asi que hay que intercalar.
    board.place(0, 0)   # blancas
    board.place(0, 4)   # negras
    board.place(1, 0)   # blancas
    board.place(1, 4)   # negras
    board.place(2, 0)   # blancas
    board.place(2, 4)   # negras
    board.place(3, 0)   # blancas
    board.place(3, 4)   # negras

    # Todavia no ha ganado nadie y le toca a blancas.
    assert board.winner is None
    assert board.current_player() == 1

    network = PolicyValueNetwork(5)
    mcts = MCTS(board, 500, network)
    assert mcts.search() == 20