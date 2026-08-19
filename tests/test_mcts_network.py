from hexzero.board import HexBoard
from hexzero.network import PolicyValueNetwork
from hexzero.mcts import MCTS


def test_search_returns_legal_move():
    """
    La red esta sin entrenar, asi que la jugada elegida sera ruido.
    Lo unico que se comprueba aqui es que la busqueda completa sin errores
    y devuelve algo jugable.
    """
    board = HexBoard(5)
    network = PolicyValueNetwork(5)

    mcts = MCTS(board, 50, network)
    move = mcts.search()

    assert move in list(board.valid_choices())


def test_root_children_are_expanded():
    """
    Tras la primera simulacion la raiz debe tener un hijo por casilla legal,
    cada uno con su prior. Los priors deben sumar 1 (softmax sobre legales).
    """
    board = HexBoard(5)
    network = PolicyValueNetwork(5)

    mcts = MCTS(board, 50, network)
    mcts.search()

    assert len(mcts.root.children) == 25

    total_prior = sum(child.prior for child in mcts.root.children)
    assert abs(total_prior - 1.0) < 1e-5


def test_visits_are_distributed():
    """
    Con 50 simulaciones, la raiz acumula 50 visitas y sus hijos se reparten
    el resto. Si un solo hijo se lleva todo, PUCT no esta explorando.
    """
    board = HexBoard(5)
    network = PolicyValueNetwork(5)

    mcts = MCTS(board, 50, network)
    mcts.search()

    assert mcts.root.visits == 50

    visited_children = [c for c in mcts.root.children if c.visits > 0]
    assert len(visited_children) > 1


def test_expansion_on_terminal_node_returns_one():
    """
    En un nodo terminal el jugador que acaba de mover ha ganado,
    asi que el valor desde su perspectiva es siempre +1.
    """
    board = HexBoard(5)
    board.place(0, 0)   # blancas
    board.place(0, 4)   # negras
    board.place(1, 0)
    board.place(1, 4)
    board.place(2, 0)
    board.place(2, 4)
    board.place(3, 0)
    board.place(3, 4)
    board.place(4, 0)   # blancas conectan norte-sur

    assert board.winner is not None

    network = PolicyValueNetwork(5)
    mcts = MCTS(board, 10, network)

    from hexzero.mcts import Node
    terminal_node = Node(board)
    assert mcts._expansion(terminal_node) == 1