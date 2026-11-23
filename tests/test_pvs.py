"""Principal variation search (REQ-013, covers REQ-005).

pvs_child searches the first move with the full window and later moves with
a null window first, re-searching only when a move beats alpha. The search
results themselves are checked against plain minimax in test_search.py.
"""

import chess
import pytest

from chesscore import engine
from chesscore.engine import negamax, pvs_child
from positions import PERFT_POSITIONS

INF = float("inf")


def scripted_negamax(monkeypatch, scores):
    """Replace negamax with one that returns `scores` in order (from the
    child's view) and records the windows it was called with."""
    windows = []
    answers = iter(scores)

    def fake(child, depth, alpha, beta):
        windows.append((alpha, beta))
        return next(answers)

    monkeypatch.setattr(engine, "negamax", fake)
    return windows


def test_first_move_gets_full_window(monkeypatch):
    windows = scripted_negamax(monkeypatch, [-30])
    assert pvs_child(chess.Board(), 2, 10, 50, first=True) == 30
    assert windows == [(-50, -10)]


def test_no_alpha_yet_gets_full_window(monkeypatch):
    windows = scripted_negamax(monkeypatch, [-30])
    assert pvs_child(chess.Board(), 2, -INF, 50, first=False) == 30
    assert windows == [(-50, INF)]


def test_later_move_worse_than_alpha_uses_null_window_only(monkeypatch):
    windows = scripted_negamax(monkeypatch, [-5])   # mover's score 5 <= alpha 10
    assert pvs_child(chess.Board(), 2, 10, 50, first=False) == 5
    assert windows == [(-11, -10)]


def test_later_move_at_or_above_beta_is_not_re_searched(monkeypatch):
    windows = scripted_negamax(monkeypatch, [-60])  # 60 >= beta: a cutoff anyway
    assert pvs_child(chess.Board(), 2, 10, 50, first=False) == 60
    assert windows == [(-11, -10)]


def test_later_move_inside_window_is_re_searched(monkeypatch):
    windows = scripted_negamax(monkeypatch, [-11, -25])
    assert pvs_child(chess.Board(), 2, 10, 50, first=False) == 25
    assert windows == [(-11, -10), (-50, -10)]


def test_null_window_and_full_window_agree_on_the_side_of_alpha(monkeypatch):
    # For real searches the null window must answer "better than alpha?"
    # the same way the full window does. Reverse futility pruning only runs
    # in null-window nodes and can change scores, so it is turned off here.
    monkeypatch.setattr(engine, "RFP_MAX_DEPTH", 0)
    board = chess.Board(PERFT_POSITIONS["position3"][0])
    for move in board.legal_moves:
        child = board.copy()
        child.push(move)
        engine.clear_transposition_table()
        exact = -negamax(child, 2, -INF, INF)
        for alpha in (exact - 50, exact - 1, exact, exact + 1, exact + 50):
            engine.clear_transposition_table()
            null = -negamax(child, 2, -alpha - 1, -alpha)
            assert (null > alpha) == (exact > alpha)


@pytest.mark.parametrize("name", ["kiwipete", "position3", "position4"])
def test_pvs_matches_full_window_alpha_beta(monkeypatch, name):
    # Reverse futility pruning only runs in null-window nodes and can
    # change scores, so it is turned off to compare PVS on its own.
    monkeypatch.setattr(engine, "RFP_MAX_DEPTH", 0)
    board = chess.Board(PERFT_POSITIONS[name][0])
    engine.clear_transposition_table()
    with_pvs = negamax(board, 3, -INF, INF)

    def full_window(child, depth, alpha, beta, first):
        return -negamax(child, depth, -beta, -alpha)

    monkeypatch.setattr(engine, "pvs_child", full_window)
    engine.clear_transposition_table()
    engine.clear_history()
    assert negamax(board, 3, -INF, INF) == with_pvs
