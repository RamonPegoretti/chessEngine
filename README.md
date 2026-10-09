# chessEngine

ChessCore, a terminal chess engine written in Python (progetto primo semestre).

The rules of chess come from [python-chess](https://python-chess.readthedocs.io/);
this project implements the engine that picks the moves.

## Setup

```
python -m pip install -r requirements.txt
```

## Run

```
python -m chesscore
```

Choose `play`, pick a colour and a search depth (3 answers instantly, 4 takes about a second).
Type moves as coordinates (`e2e4`, `e7e8q`) or in SAN (`Nf3`).

## Tests

```
python -m pytest
```

## Layout

- `chesscore/engine.py`: evaluation and minimax search with alpha-beta pruning (REQ-004, REQ-005)
- `chesscore/cli.py`: main menu and Human vs Engine mode (REQ-003)
- `tests/`: automated tests, including perft (REQ-013)
