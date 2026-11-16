"""Quiescence search and move priorities (REQ-013, covers REQ-005).

At the end of the normal search the engine keeps looking at captures and
promotions until the position is quiet, so it does not stop in the middle
of an exchange.
"""

import time

import chess
import pytest

from chesscore import engine
from chesscore.engine import (
    MATE_SCORE,
    evaluate,
    move_priority,
    noisy_moves,
    quiescence,
    side_sign,
)
from positions import PERFT_POSITIONS

INF = float("inf")


def stand_pat(board):
    return evaluate(board) * side_sign(board)


# --- noisy_moves and move_priority ----------------------------------------------


@pytest.mark.parametrize("name", PERFT_POSITIONS)
def test_noisy_moves_are_the_legal_captures_and_promotions(name):
    board = chess.Board(PERFT_POSITIONS[name][0])
    noisy = noisy_moves(board)
    expected = {m for m in board.legal_moves if board.is_capture(m) or m.promotion}
    assert len(noisy) == len(set(noisy))
    assert set(noisy) == expected


def test_noisy_moves_include_quiet_promotions_and_en_passant():
    board = chess.Board("4k3/P7/8/3pP3/8/8/8/4K3 w - d6 0 1")
    noisy = {m.uci() for m in noisy_moves(board)}
    assert {"a7a8q", "a7a8n", "e5d6"} <= noisy
    assert "e5e6" not in noisy


def test_noisy_moves_empty_in_quiet_position():
    assert noisy_moves(chess.Board()) == []


@pytest.mark.parametrize("name", PERFT_POSITIONS)
def test_noisy_moves_best_first(name):
    board = chess.Board(PERFT_POSITIONS[name][0])
    priorities = [move_priority(board, m) for m in noisy_moves(board)]
    assert priorities == sorted(priorities, reverse=True)


def test_move_priority_order_of_kinds():
    promotions = chess.Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promotion = move_priority(promotions, chess.Move.from_uci("a7a8q"))
    capture_promotion = move_priority(promotions, chess.Move.from_uci("a7b8q"))
    rook_moves = chess.Board("4k3/8/8/8/8/8/8/r3K2R w - - 0 1")
    capture = move_priority(rook_moves, chess.Move.from_uci("h1a1"))
    check = move_priority(rook_moves, chess.Move.from_uci("h1h8"))
    quiet = move_priority(rook_moves, chess.Move.from_uci("h1h2"))
    assert capture_promotion >= promotion > capture > check > quiet == 0


def test_move_priority_least_valuable_attacker_first():
    # Both the pawn and the queen can take the knight on e5: pawn first.
    board = chess.Board("4k3/8/8/4n3/3P4/8/8/4QK2 w - - 0 1")
    pawn = move_priority(board, chess.Move.from_uci("d4e5"))
    queen = move_priority(board, chess.Move.from_uci("e1e5"))
    assert pawn > queen


def test_move_priority_en_passant_victim_is_a_pawn():
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    ep = move_priority(board, chess.Move.from_uci("e5d6"))
    assert ep == 10_000 + 10 * engine.PIECE_VALUES[chess.PAWN] - engine.PIECE_VALUES[chess.PAWN]


# --- quiescence -----------------------------------------------------------------


def test_quiet_position_is_its_static_evaluation():
    board = chess.Board()
    assert quiescence(board, -INF, INF) == stand_pat(board)


def test_stand_pat_when_every_capture_loses():
    # The only capture, Qxd6, loses the queen to the pawn on c7, so White
    # keeps the static evaluation instead.
    board = chess.Board("4k3/2p5/3p4/8/8/8/8/3QK3 w - - 0 1")
    assert any(board.is_capture(m) for m in board.legal_moves)
    assert quiescence(board, -INF, INF) == stand_pat(board)


def test_free_capture_is_taken():
    # The black rook on d5 is undefended: taking it is worth about a rook.
    board = chess.Board("4k3/8/8/3r4/8/8/8/3QK3 w - - 0 1")
    after = board.copy()
    after.push_uci("d1d5")
    assert quiescence(board, -INF, INF) == -stand_pat(after)
    assert quiescence(board, -INF, INF) > stand_pat(board) + 400


def test_exchange_is_followed_to_the_end():
    # Rxd5 cxd5 gives a rook for a knight; quiescence sees the recapture
    # and prefers standing pat.
    board = chess.Board("4k3/8/2p5/3n4/8/8/8/3RK3 w - - 0 1")
    after_trade = chess.Board("4k3/8/8/3p4/8/8/8/4K3 w - - 0 2")
    assert quiescence(board, -INF, INF) == stand_pat(board)
    assert stand_pat(after_trade) < stand_pat(board)


def test_quiescence_is_symmetric():
    for name in PERFT_POSITIONS:
        board = chess.Board(PERFT_POSITIONS[name][0])
        assert quiescence(board, -INF, INF) == quiescence(board.mirror(), -INF, INF)


def test_checkmate_inside_quiescence():
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert quiescence(board, -INF, INF) <= -MATE_SCORE


def test_no_stand_pat_in_check():
    # The black queen on d2 gives check; the score must come from searching
    # the replies, not from the static evaluation.
    board = chess.Board("4k3/8/8/8/8/8/3q4/3QK3 w - - 0 1")
    assert board.is_check()
    # Kxd2 wins the queen: much better than the static evaluation.
    assert quiescence(board, -INF, INF) > stand_pat(board) + 400


def test_stand_pat_cutoff_returns_at_least_beta():
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    beta = stand_pat(board) - 100
    assert quiescence(board, beta - 1, beta) >= beta


@pytest.mark.parametrize("name", ["kiwipete", "position3", "position4"])
@pytest.mark.parametrize("offset", [-200, 0, 200])
def test_quiescence_fail_soft_bounds(name, offset):
    board = chess.Board(PERFT_POSITIONS[name][0])
    exact = quiescence(board, -INF, INF)
    alpha, beta = exact + offset - 20, exact + offset + 20
    score = quiescence(board, alpha, beta)
    if score <= alpha:
        assert exact <= score
    elif score >= beta:
        assert exact >= score
    else:
        assert score == exact


def test_quiescence_does_not_modify_board():
    board = chess.Board(PERFT_POSITIONS["kiwipete"][0])
    before = board.copy()
    quiescence(board, -INF, INF)
    assert board == before and board.move_stack == before.move_stack


# --- Time limit with quiescence -------------------------------------------------


@pytest.mark.parametrize("limit", [0.01, 0.05])
def test_time_limit_holds_in_tactical_position(limit):
    # Depth 1 alone takes about a second on kiwipete with quiescence, so the
    # limit must apply to depth 1 as well.
    board = chess.Board(PERFT_POSITIONS["kiwipete"][0])
    start = time.monotonic()
    move, _ = engine.find_best_move(board, depth=20, time_limit=limit)
    assert time.monotonic() - start < limit + 0.5
    assert move in board.legal_moves
    assert engine._deadline is None
