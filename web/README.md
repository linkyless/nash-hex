# Nash on the web

FastAPI serves the engine, the browser draws the board.

## Run it

    pip install fastapi "uvicorn[standard]"
    py -3.13 -m uvicorn web.app:app --reload

Then open http://127.0.0.1:8000

Launch it from the project root, the same folder that holds `hexzero/` and
`checkpoints/`. The checkpoint it loads is set at the top of `web/app.py`.

## How it fits together

The server holds no session state. The browser keeps the list of moves played
and sends it with every request, and the server replays the game from scratch
before answering. Undo, sharing a position and restarting all fall out of that
for free.

`POST /api/play` takes `moves`, an optional `move`, which side the human plays
and a simulation budget. It applies the human move, lets Nash answer, and
returns the board, the evaluation from the human's side, the raw policy for the
current position and the visit distribution Nash produced on its last search.
