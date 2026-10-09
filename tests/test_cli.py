"""Terminal interface and Human vs Engine loop (REQ-013, covers REQ-003).

parse_move basics and the start-position board are in test_engine.py.
Interactive functions are driven by feeding scripted lines to input().
"""

import chess
import pytest

from chesscore import cli
from chesscore.engine import MATE_SCORE


def feed(monkeypatch, *lines):
    """Make input() return the given lines in order."""
    answers = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))


# --- Rendering ----------------------------------------------------------------


def test_render_board_has_header_ranks_and_footer():
    lines = cli.render_board(chess.Board())
    assert len(lines) == 10
    assert lines[0] == lines[-1] == "  a b c d e f g h"
    assert [line[0] for line in lines[1:9]] == list("87654321")
    assert lines[4] == "5 . . . . . . . ."


def test_render_board_flipped_for_black():
    lines = cli.render_board(chess.Board(), flipped=True)
    assert lines[0] == "  h g f e d c b a"
    assert lines[1] == "1 R N B K Q B N R"
    assert lines[8] == "8 r n b k q b n r"


def test_render_history_pairs_moves():
    lines = cli.render_history(["e4", "e5", "Nf3"])
    assert lines[2].split() == ["1.", "e4", "e5"]
    assert lines[3].split() == ["2.", "Nf3", "..."]


def test_render_history_empty():
    assert len(cli.render_history([])) == 2


@pytest.mark.parametrize("score,text", [
    (None, "-"),
    (0, "+0.0"),
    (30, "+0.3"),
    (-150, "-1.5"),
    (MATE_SCORE + 2, "mate"),
    (-MATE_SCORE, "-mate"),
])
def test_format_score(score, text):
    assert cli.format_score(score) == text


def test_render_screen_shows_eval_and_depth():
    board = chess.Board()
    board.push_uci("e2e4")
    screen = cli.render_screen(board, ["e4"], False, 30, 4)
    assert screen.splitlines()[-1] == "eval +0.3 depth 4"
    assert "1. e4" in screen
    assert "4 . . . . P . . ." in screen


# --- Move input ---------------------------------------------------------------


@pytest.mark.parametrize("text,uci", [
    ("e7e8n", "e7e8n"),      # explicit under-promotion is kept
    ("e8=Q", "e7e8q"),       # SAN promotion
    ("Kg2", "h1g2"),
])
def test_parse_move_more_forms(text, uci):
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    assert cli.parse_move(board, text) == chess.Move.from_uci(uci)


def test_parse_move_castling_in_both_notations():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert cli.parse_move(board, "e1g1") == chess.Move.from_uci("e1g1")
    assert cli.parse_move(board, "O-O-O") == chess.Move.from_uci("e1c1")


def test_parse_move_en_passant():
    board = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    board.push_uci("d7d5")
    assert cli.parse_move(board, "e5d6") == chess.Move.from_uci("e5d6")


@pytest.mark.parametrize("text", ["", "e2", "e2e4e6", "z9z9", "Ke2", "null"])
def test_parse_move_rejects_garbage_and_illegal(text):
    assert cli.parse_move(chess.Board(), text) is None


@pytest.mark.xfail(strict=True, reason="parse_move returns a null move; remove xfail once fixed")
@pytest.mark.parametrize("text", ["0000", "--", "Z0", "@@@@"])
def test_parse_move_rejects_null_move(text):
    # python-chess reads these as a null move ("pass"). The play loop would
    # push it and the human would skip their turn.
    assert cli.parse_move(chess.Board(), text) is None


def test_parse_move_rejects_move_for_wrong_side():
    assert cli.parse_move(chess.Board(), "e7e5") is None


# --- Game end messages --------------------------------------------------------


def test_game_result_text_checkmate():
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)
    assert cli.game_result_text(board) == "Black wins by checkmate 0-1"


def test_game_result_text_stalemate():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert cli.game_result_text(board) == "Draw (stalemate) 1/2-1/2"


# --- Prompts and loops --------------------------------------------------------


def test_ask_repeats_until_valid(monkeypatch, capsys):
    feed(monkeypatch, "x", " W ")
    assert cli.ask("? ", ["w", "b"]) == "w"
    assert "Please answer one of: w, b" in capsys.readouterr().out


@pytest.mark.parametrize("lines,depth", [
    ([""], cli.DEFAULT_DEPTH),
    (["5"], 5),
    (["0", "abc", "-2", "2"], 2),
])
def test_ask_depth(monkeypatch, lines, depth):
    feed(monkeypatch, *lines)
    assert cli.ask_depth() == depth


def test_menu_quit_by_number_and_name(monkeypatch):
    feed(monkeypatch, "5")
    cli.main()
    feed(monkeypatch, "quit")
    cli.main()


def test_menu_unknown_and_unimplemented(monkeypatch, capsys):
    feed(monkeypatch, "train", "banana", "quit")
    cli.main()
    out = capsys.readouterr().out
    assert "'train' is not implemented yet." in out
    assert "Unknown option: banana" in out


def test_play_rejects_illegal_move_then_quits(monkeypatch, capsys):
    feed(monkeypatch, "w", "1", "e2e5", "quit")
    cli.play()
    out = capsys.readouterr().out
    assert "Illegal or unreadable move: e2e5" in out
    assert "Game abandoned." in out


def test_play_engine_replies_to_human_move(monkeypatch, capsys):
    feed(monkeypatch, "w", "1", "e2e4", "quit")
    cli.play()
    out = capsys.readouterr().out
    assert "engine is thinking..." in out
    assert "1. e4" in out


def test_play_as_black_engine_moves_first(monkeypatch, capsys):
    feed(monkeypatch, "b", "1", "quit")
    cli.play()
    out = capsys.readouterr().out
    assert out.index("engine is thinking...") < out.index("Game abandoned.")


def test_play_full_game_to_checkmate(monkeypatch, capsys):
    # Script the engine's replies so the game ends in Fool's mate.
    replies = iter(["e7e5", "d8h4"])
    monkeypatch.setattr(
        cli, "find_best_move",
        lambda board, depth: (chess.Move.from_uci(next(replies)), 0),
    )
    feed(monkeypatch, "w", "2", "f2f3", "g2g4")
    cli.play()
    out = capsys.readouterr().out
    assert "4. " not in out
    assert "Black wins by checkmate 0-1" in out


def test_play_passes_chosen_depth_to_engine(monkeypatch):
    depths = []

    def fake_engine(board, depth):
        depths.append(depth)
        return next(iter(board.legal_moves)), 0

    monkeypatch.setattr(cli, "find_best_move", fake_engine)
    feed(monkeypatch, "b", "4", "quit")
    cli.play()
    assert depths == [4]
