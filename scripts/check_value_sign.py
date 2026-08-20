"""
Comprueba si el signo del value esta invertido en el MCTS.

Monta una posicion donde el jugador que mueve gana en un movimiento,
corre una busqueda, e imprime el mean_value y las visitas de cada hijo
de la raiz.

Lo que se espera si el signo es CORRECTO:
    el hijo del movimiento ganador tiene el mean_value mas ALTO
    y se lleva la mayoria de las visitas.

Lo que se ve si el signo esta INVERTIDO:
    el hijo del movimiento ganador tiene el mean_value mas BAJO
    (cercano a -1) y apenas recibe visitas.

Lanzar desde la raiz del proyecto:
    py -3.13 -m scripts.check_value_sign
"""

import torch

from hexzero.board import HexBoard
from hexzero.mcts import MCTS
from hexzero.network import PolicyValueNetwork


BOARD_SIZE  = 5
SIMULATIONS = 200
CHECKPOINT  = "checkpoints/iter_29.pt"

# Blancas conectan norte-sur por la columna 0 y les falta (4, 0).
# Indice plano de (4, 0) = 4 * 5 + 0 = 20.
WINNING_MOVE = 20


def build_position() -> HexBoard:
    board = HexBoard(BOARD_SIZE)

    board.place(0, 0)   # blancas
    board.place(0, 4)   # negras
    board.place(1, 0)   # blancas
    board.place(1, 4)   # negras
    board.place(2, 0)   # blancas
    board.place(2, 4)   # negras
    board.place(3, 0)   # blancas
    board.place(3, 4)   # negras

    assert board.winner is None, "la posicion ya tiene ganador"
    assert board.current_player() == 1, "no mueven blancas"

    return board


if __name__ == "__main__":
    board = build_position()
    board.print_board()
    print(f"\nMueven blancas. La jugada ganadora es la {WINNING_MOVE} (fila 4, col 0).\n")

    network = PolicyValueNetwork(BOARD_SIZE)
    network.load_state_dict(torch.load(CHECKPOINT))
    network.eval()

    mcts = MCTS(board, SIMULATIONS, network)
    pi, best_move = mcts.search()

    children = sorted(
        mcts.root.children,
        key=lambda c: c.mean_value() if c.visits > 0 else -99,
        reverse=True,
    )

    print(f"{'move':>6} {'visits':>8} {'mean_value':>12} {'prior':>10}")
    print("-" * 40)
    for child in children:
        mean = child.mean_value() if child.visits > 0 else float("nan")
        mark = "  <-- GANADORA" if child.move == WINNING_MOVE else ""
        print(f"{child.move:>6} {child.visits:>8} {mean:>12.4f} {child.prior:>10.4f}{mark}")

    print()
    print(f"Movimiento elegido por search(): {best_move}")
    print(f"Movimiento ganador real:         {WINNING_MOVE}")

    winner_child = next(c for c in mcts.root.children if c.move == WINNING_MOVE)
    rank = children.index(winner_child) + 1
    print(f"\nLa jugada ganadora queda en la posicion {rank} de {len(children)} por mean_value.")
    if rank == 1:
        print("=> El signo parece CORRECTO.")
    elif rank >= len(children) - 2:
        print("=> El signo parece INVERTIDO.")
    else:
        print("=> No concluyente: mira la tabla a mano.")