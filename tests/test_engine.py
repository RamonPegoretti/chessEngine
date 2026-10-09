"""Automated tests (REQ-013)."""

import chess
import pytest

from chesscore.cli import parse_move, render_board, render_history
from chesscore.engine import MATE_SCORE, evaluate, find_best_move


def perft(board, depth):
    """Count leaf nodes of the move tree, the standard move-generator check."""
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total


@pytest.mark.parametrize("fen, depth, expected", [
    (chess.STARTING_FEN, 3, 8_902),
    # "Kiwipete": castling, en passant and promotions all in play.
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2_039),
])
def test_perft(fen, depth, expected):
    assert perft(chess.Board(fen), depth) == expected


def test_start_position_is_balanced():
    assert evaluate(chess.Board()) == 0


def test_mirrored_position_has_opposite_score():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert evaluate(board) == -evaluate(board.mirror())


def test_finds_mate_in_one_for_white():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    move, score = find_best_move(board, depth=2)
    assert move == chess.Move.from_uci("a1a8")
    assert score >= MATE_SCORE


def test_finds_mate_in_one_for_black():
    board = chess.Board("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    move, score = find_best_move(board, depth=2)
    assert move == chess.Move.from_uci("a8a1")
    assert score <= -MATE_SCORE


def test_takes_hanging_queen():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    move, _ = find_best_move(board, depth=2)
    assert move == chess.Move.from_uci("d1d5")


def test_search_does_not_modify_board():
    board = chess.Board()
    fen = board.fen()
    find_best_move(board, depth=2)
    assert board.fen() == fen


def test_parse_move_accepts_coordinates_and_san():
    board = chess.Board()
    assert parse_move(board, "e2e4") == chess.Move.from_uci("e2e4")
    assert parse_move(board, "Nf3") == chess.Move.from_uci("g1f3")
    assert parse_move(board, "e2e5") is None
    assert parse_move(board, "hello") is None


@pytest.mark.parametrize("text", ["0000", "--", "Z0", "@@@@"])
def test_parse_move_rejects_null_move(text):
    assert parse_move(chess.Board(), text) is None


def test_parse_move_defaults_promotion_to_queen():
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    assert parse_move(board, "e7e8") == chess.Move.from_uci("e7e8q")


def test_render_board_start_position():
    lines = render_board(chess.Board())
    assert lines[1] == "8 r n b q k b n r"
    assert lines[8] == "1 R N B Q K B N R"


def test_history_starts_new_column_after_25_moves():
    sans = ["Nf3", "Nf6", "Ng1", "Ng8"] * 13  # 52 plies = 26 full moves
    lines = render_history(sans)
    assert len(lines) == 2 + 25
    assert lines[2].startswith("  1. Nf3")
    assert " 26. Ng1" in lines[2]
    assert " 26." not in "".join(lines[3:])
