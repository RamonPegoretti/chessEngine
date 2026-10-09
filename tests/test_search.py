"""Minimax with alpha-beta and configurable depth (REQ-013, covers REQ-004/005).

Basic tactics are in test_engine.py; this file checks that the search itself
is correct: alpha-beta returns the same score as plain minimax, mates are
scored and preferred correctly, depth is respected and boards are copied.
"""

import chess
import pytest

from chesscore.engine import (
    MATE_SCORE,
    evaluate,
    find_best_move,
    minimax,
    ordered_moves,
    terminal_score,
)
from positions import MATE_IN_ONE, PERFT_POSITIONS

INF = float("inf")


def plain_minimax(board, depth):
    """Reference minimax without pruning or move ordering."""
    score = terminal_score(board, depth)
    if score is not None:
        return score
    if depth == 0:
        return evaluate(board)
    scores = []
    for move in board.legal_moves:
        child = board.copy()
        child.push(move)
        scores.append(plain_minimax(child, depth - 1))
    return max(scores) if board.turn == chess.WHITE else min(scores)


SEARCH_POSITIONS = [
    chess.STARTING_FEN,
    # Open middlegame with tactics for both sides
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    # Black to move, material imbalance
    "r1b1k2r/ppp2ppp/2n5/3q4/1b1P4/2N2N2/PP3PPP/R1BQKB1R b KQkq - 0 8",
    # Endgame with promotion races
    "8/1P6/8/8/8/8/6p1/k6K w - - 0 1",
    PERFT_POSITIONS["position3"][0],
]


@pytest.mark.parametrize("fen", SEARCH_POSITIONS)
@pytest.mark.parametrize("depth", [1, 2])
def test_alpha_beta_matches_plain_minimax(fen, depth):
    board = chess.Board(fen)
    _, score = find_best_move(board, depth)
    assert score == plain_minimax(board, depth)


@pytest.mark.parametrize("fen", SEARCH_POSITIONS[:3])
def test_minimax_with_full_window_matches_plain_minimax(fen):
    board = chess.Board(fen)
    assert minimax(board, 2, -INF, INF) == plain_minimax(board, 2)


def test_best_move_score_is_reproducible_by_playing_it():
    board = chess.Board(SEARCH_POSITIONS[2])
    move, score = find_best_move(board, depth=2)
    child = board.copy()
    child.push(move)
    assert minimax(child, 1, -INF, INF) == score


@pytest.mark.parametrize("fen,uci", MATE_IN_ONE.items())
@pytest.mark.parametrize("depth", [1, 3])
def test_finds_mate_in_one_at_any_depth(fen, uci, depth):
    board = chess.Board(fen)
    move, score = find_best_move(board, depth)
    assert move.uci() == uci
    assert abs(score) >= MATE_SCORE


def gives_mate(board, move):
    child = board.copy()
    child.push(move)
    return child.is_checkmate()


def test_finds_mate_in_two():
    # King and rook vs king: 1. Kb6 Kb8 2. Rh8# (or 1. Kc7 Ka7 2. Ra1#).
    board = chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1")
    move, score = find_best_move(board, depth=3)
    assert score >= MATE_SCORE
    child = board.copy()
    child.push(move)
    assert not child.is_game_over()
    for reply in child.legal_moves:
        grandchild = child.copy()
        grandchild.push(reply)
        assert any(gives_mate(grandchild, m) for m in grandchild.legal_moves)


def test_prefers_faster_mate():
    # Mate in one is available; a slower mate also exists. Depth 3 sees both.
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    move, score = find_best_move(board, depth=3)
    assert move.uci() == "a1a8"
    assert score == MATE_SCORE + 2  # mate found with 2 plies of depth left


def test_defends_against_mate_in_one():
    # Black threatens Ra1# on the back rank; White must make room for the king.
    board = chess.Board("6k1/5ppp/8/8/8/8/r4PPP/6K1 w - - 0 1")
    move, score = find_best_move(board, depth=2)
    child = board.copy()
    child.push(move)
    assert not any(gives_mate(child, m) for m in child.legal_moves)
    assert score > -MATE_SCORE


def test_mate_scores_have_correct_sign():
    white_mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    black_mated = chess.Board("rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3")
    assert white_mated.is_checkmate() and black_mated.is_checkmate()
    assert terminal_score(white_mated, 0) <= -MATE_SCORE
    assert terminal_score(black_mated, 0) >= MATE_SCORE


@pytest.mark.parametrize("fen", [
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",          # stalemate
    "8/8/4k3/8/8/4K3/8/8 w - - 0 1",            # bare kings
    "4k3/8/8/8/8/8/8/4K2R w - - 150 100",       # 75-move rule
])
def test_draws_score_zero(fen):
    assert terminal_score(chess.Board(fen), 3) == 0


def test_ongoing_game_has_no_terminal_score():
    assert terminal_score(chess.Board(), 3) is None


def test_engine_avoids_stalemating_when_winning():
    # Qg6 or Qf7 would stalemate; White, a queen up, must keep the game going.
    board = chess.Board("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1")
    move, _ = find_best_move(board, depth=2)
    child = board.copy()
    child.push(move)
    assert not child.is_stalemate()


@pytest.mark.parametrize("depth", [0, -1])
def test_depth_below_one_rejected(depth):
    with pytest.raises(ValueError):
        find_best_move(chess.Board(), depth)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_returns_legal_move_at_each_depth(depth):
    board = chess.Board(PERFT_POSITIONS["kiwipete"][0])
    move, score = find_best_move(board, depth)
    assert move in board.legal_moves
    assert isinstance(score, (int, float))


def test_deeper_search_sees_more():
    # Depth 1 grabs the defended pawn with the queen; depth 2 sees the recapture.
    board = chess.Board("4k3/8/3p4/4p3/8/8/8/4QK2 w - - 0 1")
    shallow, _ = find_best_move(board, depth=1)
    deep, _ = find_best_move(board, depth=2)
    assert shallow.uci() == "e1e5"
    assert deep.uci() != "e1e5"


@pytest.mark.parametrize("fen", SEARCH_POSITIONS)
def test_search_does_not_modify_any_board(fen):
    board = chess.Board(fen)
    board_before = board.copy()
    find_best_move(board, depth=2)
    assert board == board_before
    assert board.move_stack == board_before.move_stack


def test_search_keeps_move_history():
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)
    find_best_move(board, depth=2)
    assert [m.uci() for m in board.move_stack] == ["e2e4", "e7e5", "g1f3"]


# --- Move ordering ------------------------------------------------------------


@pytest.mark.parametrize("name", PERFT_POSITIONS)
def test_ordered_moves_are_exactly_the_legal_moves(name):
    board = chess.Board(PERFT_POSITIONS[name][0])
    ordered = ordered_moves(board)
    assert len(ordered) == len(set(ordered))
    assert set(ordered) == set(board.legal_moves)


def test_ordered_moves_put_promotions_then_captures_first():
    board = chess.Board(PERFT_POSITIONS["position4"][0])
    ordered = ordered_moves(board)
    kinds = [
        2 if m.promotion else 1 if board.is_capture(m) else 0
        for m in ordered
    ]
    assert kinds == sorted(kinds, reverse=True)


def test_ordered_moves_most_valuable_victim_first():
    # White pawn can take a queen or a knight: the queen comes first.
    board = chess.Board("4k3/8/8/2q1n3/3P4/8/8/4K3 w - - 0 1")
    ordered = [m.uci() for m in ordered_moves(board)]
    assert ordered.index("d4c5") < ordered.index("d4e5")
