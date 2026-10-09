"""External engine integration via UCI (REQ-007).

python-chess starts the external engine (Stockfish, Reckless, ...) as a
separate process and talks to it with the UCI protocol. This module wraps
that in a small class and adds the engine vs external engine mode (EvME).
"""

import os
import shutil
import zipfile

import chess
import chess.engine

from chesscore.engine import DEFAULT_DEPTH, find_best_move

# Environment variable that can point to the external engine's executable,
# e.g. set CHESSCORE_ENGINE=C:\stockfish\stockfish.exe
ENGINE_PATH_VARIABLE = "CHESSCORE_ENGINE"

DEFAULT_MOVE_TIME = 0.1  # seconds per move for the external engine

# Stockfish for Windows ships zipped in the repository (the executable is
# over GitHub's 100 MB file limit). It is unpacked on first use.
ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "engines")
BUNDLED_ZIP = "stockfish-18-windows.zip"
BUNDLED_EXE = "stockfish.exe"


def bundled_engine_path(engines_dir=ENGINES_DIR):
    """Path of the bundled Stockfish, unzipping it the first time.

    None if the zip is missing.
    """
    exe = os.path.join(engines_dir, "stockfish", BUNDLED_EXE)
    if os.path.exists(exe):
        return exe
    archive = os.path.join(engines_dir, BUNDLED_ZIP)
    if not os.path.exists(archive):
        return None
    print("Unpacking Stockfish (first run only)...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(os.path.join(engines_dir, "stockfish"))
    return exe


def find_engine_path():
    """Path of the external engine, or None if it cannot be found.

    Looks at the CHESSCORE_ENGINE variable first, then (on Windows) the
    Stockfish bundled in engines/, then for "stockfish" on the PATH.
    """
    path = os.environ.get(ENGINE_PATH_VARIABLE)
    if path:
        return path
    if os.name == "nt":
        path = bundled_engine_path()
        if path:
            return path
    return shutil.which("stockfish")


class ExternalEngine:
    """An external UCI engine. Use it with `with` so the process is closed.

        with ExternalEngine(elo=1500) as stockfish:
            move = stockfish.play(board)
    """

    def __init__(self, path=None, move_time=DEFAULT_MOVE_TIME, elo=None):
        path = path or find_engine_path()
        if path is None:
            raise FileNotFoundError(
                "No external engine found. Install Stockfish and put it on the "
                f"PATH, or set {ENGINE_PATH_VARIABLE} to its executable.")
        self.move_time = move_time
        self.process = chess.engine.SimpleEngine.popen_uci(path)
        self.name = self.process.id.get("name", os.path.basename(path))
        if elo is not None:
            # Weaken the engine to a target rating so games are not all lost.
            # The engine refuses values outside its own range (Stockfish: 1320-3190).
            option = self.process.options["UCI_Elo"]
            elo = max(option.min, min(option.max, elo))
            self.process.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            self.name += f" ({elo})"

    def play(self, board):
        """Best move for the side to move, within the time limit."""
        result = self.process.play(board, chess.engine.Limit(time=self.move_time))
        return result.move

    def evaluate(self, board, depth=12):
        """Evaluation of a position in centipawns from White's view.

        Mates are turned into large scores, like in chesscore.engine.
        """
        info = self.process.analyse(board, chess.engine.Limit(depth=depth))
        return info["score"].white().score(mate_score=100_000)

    def close(self):
        self.process.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def play_match_game(external, chesscore_color, depth=DEFAULT_DEPTH,
                    max_moves=200, on_move=None):
    """Play one game of ChessCore against an external engine (EvME).

    Returns the final board; board.move_stack holds every move and
    board.outcome() the result. The game is stopped as a draw after
    `max_moves` full moves so a match can never run forever.
    on_move(board, move), if given, is called after every move.
    """
    board = chess.Board()
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == chesscore_color:
            move, _ = find_best_move(board, depth)
        else:
            move = external.play(board)
        board.push(move)
        if on_move:
            on_move(board, move)
    return board


def match_result(board):
    """PGN-style result of a finished or stopped game: 1-0, 0-1 or 1/2-1/2."""
    outcome = board.outcome()
    return outcome.result() if outcome else "1/2-1/2"
