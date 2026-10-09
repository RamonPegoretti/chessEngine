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
The board has coloured light and dark squares, the last move highlighted,
and pieces drawn as chess symbols (white or black by colour). Options:

- `python -m chesscore --letters`: letters instead of symbols, if your
  terminal shows the symbols as boxes (the old cmd window often does)
- `python -m chesscore --plain`: no colours, if the squares look wrong
Type moves as coordinates (`e2e4`, `e7e8q`) or in SAN (`Nf3`).

## Stockfish (external engine)

The `match` mode plays ChessCore against an external UCI engine.
On Windows, Stockfish 18 is bundled in `engines/` (see `engines/README.md`)
and unpacked automatically on first use. Elsewhere, install Stockfish and put
it on the PATH as `stockfish`. To use another engine, point
`CHESSCORE_ENGINE` at its executable:

```
set CHESSCORE_ENGINE=C:\path\to\engine.exe
```

## Tests

```
python -m pytest
```

## Layout

- `chesscore/engine.py`: evaluation and minimax search with alpha-beta pruning (REQ-004, REQ-005)
- `chesscore/uci.py`: external engine over UCI and the engine vs external engine mode (REQ-007)
- `chesscore/cli.py`: main menu, Human vs Engine mode (REQ-003) and match mode
- `tests/`: automated tests, including perft (REQ-013)
