import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hexzero.board import HexBoard
from hexzero.mcts import MCTS
from hexzero.network import PolicyValueNetwork


BOARD_SIZE  = 5
TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE
CHECKPOINT  = Path("checkpoints/iter_29.pt")
STATIC_DIR  = Path(__file__).parent / "static"

MIN_SIMULATIONS = 25
MAX_SIMULATIONS = 6000


app     = FastAPI(title="Nash")
network = PolicyValueNetwork(BOARD_SIZE)
network.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
network.eval()


class PlayRequest(BaseModel):
    moves:       list[int]  = Field(default_factory=list)
    move:        int | None = None
    human:       int        = 1
    simulations: int        = 300


class GameState(BaseModel):
    moves:       list[int]
    grid:        list[int]
    legal:       list[int]
    current:     int
    winner:      int | None
    human:       int
    evaluation:  float
    instinct:    list[float]
    search:      list[float]
    nash_move:   int | None
    elapsed_ms:  int


def replay(moves: list[int]) -> HexBoard:
    board = HexBoard(BOARD_SIZE)
    for move in moves:
        row, col = board.index_to_cell(move)
        board.place(row, col)
    return board


def read_network(board: HexBoard) -> tuple[list[float], float]:
    canonical = board.transform_to_canonic_form()
    tensor    = torch.from_numpy(canonical).reshape(1, 1, BOARD_SIZE, BOARD_SIZE).float()

    with torch.no_grad():
        policy_logits, value = network(tensor)
        policy_logits = policy_logits.squeeze(0)

    if board.current_player() == -1:
        policy_logits = policy_logits.reshape(BOARD_SIZE, BOARD_SIZE).T.reshape(-1)

    valid = torch.zeros(TOTAL_CELLS, dtype=torch.bool)
    valid[board.valid_choices()] = True
    policy_logits[~valid] = -torch.inf

    return torch.softmax(policy_logits, dim=0).tolist(), value.item()


def build_state(board: HexBoard, moves: list[int], human: int, search: list[float],
                nash_move: int | None, elapsed_ms: int) -> GameState:

    if board.winner is None:
        instinct, value = read_network(board)
        evaluation      = value if board.current_player() == human else -value
    else:
        instinct   = [0.0] * TOTAL_CELLS
        evaluation = 1.0 if board.winner == human else -1.0

    return GameState(
        moves      = moves,
        grid       = board.board.reshape(-1).tolist(),
        legal      = board.valid_choices().tolist(),
        current    = board.current_player(),
        winner     = board.winner,
        human      = human,
        evaluation = evaluation,
        instinct   = instinct,
        search     = search,
        nash_move  = nash_move,
        elapsed_ms = elapsed_ms,
    )


@app.post("/api/play", response_model=GameState)
def play(request: PlayRequest) -> GameState:
    simulations = max(MIN_SIMULATIONS, min(MAX_SIMULATIONS, request.simulations))
    human       = 1 if request.human >= 0 else -1
    moves       = [int(move) for move in request.moves]
    board       = replay(moves)

    if request.move is not None:
        row, col = board.index_to_cell(request.move)

        if board.winner is not None:
            raise HTTPException(status_code=409, detail="the game is over")
        if board.current_player() != human:
            raise HTTPException(status_code=409, detail="it is not your turn")
        if not board.is_valid_play(row, col):
            raise HTTPException(status_code=409, detail="that cell is taken")

        board.place(row, col)
        moves.append(int(request.move))

    search     = [0.0] * TOTAL_CELLS
    nash_move  = None
    elapsed_ms = 0

    if board.winner is None and board.current_player() != human:
        start        = time.perf_counter()
        mcts         = MCTS(board, simulations, network, root_mix=0.0)
        pi, best     = mcts.search()
        elapsed_ms   = int((time.perf_counter() - start) * 1000)

        row, col = board.index_to_cell(best)
        board.place(row, col)
        moves.append(int(best))
        search    = pi.tolist()
        nash_move = int(best)

    return build_state(board, moves, human, search, nash_move, elapsed_ms)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
