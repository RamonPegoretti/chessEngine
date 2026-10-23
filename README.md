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

## Evaluation

The engine scores a position (in centipawns, White minus Black) by adding:

| term | what it counts | default weight |
|---|---|---|
| material | value of every piece (pawn 100, knight 320, bishop 330, rook 500, queen 900) | - |
| position | bonus or malus for the square each piece stands on (piece-square tables) | 100% |
| mobility | squares the knights, bishops, rooks and queens can move to | 3 per square |
| king safety | pawns in front of the king, minus squares next to it the opponent attacks (only while the opponent has a queen) | 10 per point |
| development | knights and bishops still on their starting squares (a malus) | 15 per piece |

Mobility, king safety and development let the engine see some compensation
for material, such as a lead in development or an exposed enemy king, that
the search alone cannot reach within its depth.

## Training

`train` plays a number of games against Stockfish (alternating colours) to
measure ChessCore's estimated rating. Then Stockfish plays the same number of
games against itself and evaluates their positions, and the evaluation
weights (piece values and the other terms above) are tuned to agree with
Stockfish's evaluations (Texel tuning). Each tuning that fits better is saved
as a new engine version, and `play`, `match` and `train` always use the
latest one. `stats` shows the results and estimated rating of every version.

The weights are not tuned on ChessCore's own games: ChessCore loses almost
all of them, so whatever it does (developing its pieces, sheltering its
king) would look like a losing habit and its weight would drop to 0.

Training data goes in `data/`:

- `games.json`: every training game with its moves and result
- `stats.json`: wins, draws, losses and estimated Elo per engine version
- `versions.json`: the weights of every engine version
- `positions.json`: positions from Stockfish's games against itself, with its evaluation
- `pgn/`: about one game in ten as a PGN file

Tuning waits until about 1000 positions (around 12 games) have been
collected, and each session changes the weights only a little.

## Tests

```
python -m pytest
```

## Layout

- `chesscore/engine.py`: evaluation and minimax search with alpha-beta pruning (REQ-004, REQ-005)
- `chesscore/uci.py`: external engine over UCI and the engine vs external engine mode (REQ-007)
- `chesscore/train.py`: training mode, Elo estimate and weight tuning (REQ-010, REQ-012)
- `chesscore/storage.py`: JSON files, PGN export and engine versions (REQ-008, REQ-009, REQ-011)
- `chesscore/cli.py`: main menu, Human vs Engine mode (REQ-003) and match mode
- `tests/`: automated tests, including perft (REQ-013)
