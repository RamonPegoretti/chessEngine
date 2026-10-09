"""External engine over UCI and the EvME match mode (REQ-013, covers REQ-007).

Tests marked `needs_engine` start a real Stockfish and are skipped when none
is installed (put it on the PATH or set CHESSCORE_ENGINE). The rest use a
fake engine, so they run everywhere.
"""

import chess
import pytest

from chesscore import uci
from chesscore.engine import MATE_SCORE
from positions import MATE_IN_ONE

needs_engine = pytest.mark.skipif(
    uci.find_engine_path() is None, reason="no external UCI engine installed")


class FirstMoveEngine:
    """Fake external engine: always plays the first legal move."""

    def __init__(self):
        self.boards_seen = []

    def play(self, board):
        self.boards_seen.append(board.fen())
        return next(iter(board.legal_moves))


# --- find_engine_path ---------------------------------------------------------


def test_engine_path_from_environment(monkeypatch):
    monkeypatch.setenv(uci.ENGINE_PATH_VARIABLE, "/opt/engines/sf")
    assert uci.find_engine_path() == "/opt/engines/sf"


def test_engine_path_falls_back_to_path_lookup(monkeypatch):
    monkeypatch.delenv(uci.ENGINE_PATH_VARIABLE, raising=False)
    monkeypatch.setattr(uci.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert uci.find_engine_path() == "/usr/bin/stockfish"


def test_engine_path_none_when_nothing_installed(monkeypatch):
    monkeypatch.delenv(uci.ENGINE_PATH_VARIABLE, raising=False)
    monkeypatch.setattr(uci.shutil, "which", lambda name: None)
    assert uci.find_engine_path() is None


def test_external_engine_without_engine_raises(monkeypatch):
    monkeypatch.setattr(uci, "find_engine_path", lambda: None)
    with pytest.raises(FileNotFoundError):
        uci.ExternalEngine()


# --- play_match_game / match_result (fake engine) -----------------------------


@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK])
def test_match_game_alternates_sides(color):
    fake = FirstMoveEngine()
    board = uci.play_match_game(fake, color, depth=1, max_moves=3)
    # The fake was only asked to move on its own turns.
    for fen in fake.boards_seen:
        assert chess.Board(fen).turn != color
    assert len(fake.boards_seen) == 3
    assert len(board.move_stack) == 6


def test_match_game_stops_after_max_moves():
    board = uci.play_match_game(FirstMoveEngine(), chess.WHITE, depth=1, max_moves=5)
    assert len(board.move_stack) == 10
    assert not board.is_game_over()
    assert uci.match_result(board) == "1/2-1/2"


def test_match_game_calls_on_move_after_each_push():
    calls = []

    def on_move(board, move):
        assert board.peek() == move
        calls.append(move)

    board = uci.play_match_game(
        FirstMoveEngine(), chess.BLACK, depth=1, max_moves=2, on_move=on_move)
    assert calls == board.move_stack


def test_match_game_ends_on_checkmate(monkeypatch):
    # Script both sides into Fool's mate: the fake plays White, and
    # ChessCore's search is replaced by the black moves.
    white = iter(["f2f3", "g2g4"])
    black = iter(["e7e5", "d8h4"])

    class Scripted:
        def play(self, board):
            return chess.Move.from_uci(next(white))

    monkeypatch.setattr(
        uci, "find_best_move",
        lambda board, depth: (chess.Move.from_uci(next(black)), 0))
    board = uci.play_match_game(Scripted(), chess.BLACK)
    assert board.is_checkmate()
    assert len(board.move_stack) == 4
    assert uci.match_result(board) == "0-1"


@pytest.mark.parametrize("fen,result", [
    ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "0-1"),
    ("rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3", "1-0"),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "1/2-1/2"),
    (chess.STARTING_FEN, "1/2-1/2"),
])
def test_match_result(fen, result):
    assert uci.match_result(chess.Board(fen)) == result


# --- Real external engine -----------------------------------------------------


@needs_engine
def test_external_engine_plays_legal_move():
    board = chess.Board()
    with uci.ExternalEngine(move_time=0.01) as engine:
        assert engine.name
        assert engine.play(board) in board.legal_moves


@needs_engine
@pytest.mark.parametrize("fen,uci_move", MATE_IN_ONE.items())
def test_external_engine_finds_mate_in_one(fen, uci_move):
    with uci.ExternalEngine(move_time=0.05) as engine:
        assert engine.play(chess.Board(fen)).uci() == uci_move


@needs_engine
def test_external_engine_evaluates_from_whites_view():
    white_up = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    black_up = chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1")
    with uci.ExternalEngine() as engine:
        assert engine.evaluate(white_up, depth=8) > 500
        assert engine.evaluate(black_up, depth=8) < -500


@needs_engine
def test_external_engine_mate_score():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    with uci.ExternalEngine() as engine:
        assert engine.evaluate(board, depth=5) == MATE_SCORE - 1


@needs_engine
def test_external_engine_elo_is_clamped_and_named():
    with uci.ExternalEngine(elo=1) as engine:
        option = engine.process.options["UCI_Elo"]
        assert engine.name.endswith(f" ({option.min})")


@needs_engine
def test_short_match_against_real_engine():
    with uci.ExternalEngine(move_time=0.01, elo=1) as engine:
        board = uci.play_match_game(engine, chess.WHITE, depth=1, max_moves=5)
    assert 1 <= len(board.move_stack) <= 10
    assert uci.match_result(board) in {"1-0", "0-1", "1/2-1/2"}
