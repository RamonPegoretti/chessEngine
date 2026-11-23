"""Aspiration windows at the root (REQ-013, covers REQ-005).

aspiration_search must always return the same score as a full-window
search_root, however far off the previous depth's score was.
"""

import chess
import pytest

from chesscore import engine
from chesscore.engine import (
    ASPIRATION_MAX,
    ASPIRATION_MIN_DEPTH,
    ASPIRATION_WINDOW,
    MATE_SCORE,
    aspiration_search,
    find_best_move,
    search_root,
)
from positions import MATE_IN_ONE, PERFT_POSITIONS

INF = float("inf")

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "2r1k3/pp3ppp/2n5/3q4/3P4/2N5/PP3PPP/R3K2R b KQ - 0 15",
    PERFT_POSITIONS["position3"][0],
]


def full_window_score(board, depth):
    engine.clear_transposition_table()
    engine.clear_history()
    return search_root(board, depth)[1]


# --- Same score as a full window ------------------------------------------------


@pytest.mark.parametrize("fen", POSITIONS)
@pytest.mark.parametrize("guess", [-2000, -300, -60, 0, 60, 300, 2000, None])
def test_aspiration_matches_full_window(fen, guess):
    board = chess.Board(fen)
    expected = full_window_score(board, 3)
    previous = expected if guess is None else expected + guess
    engine.clear_transposition_table()
    engine.clear_history()
    move, score = aspiration_search(board, 3, None, previous)
    assert score == expected
    assert move in board.legal_moves


@pytest.mark.parametrize("fen,uci", MATE_IN_ONE.items())
def test_aspiration_finds_mates(fen, uci):
    board = chess.Board(fen)
    move, score = aspiration_search(board, 3, None, 0)
    assert move.uci() == uci
    assert abs(score) >= MATE_SCORE


@pytest.mark.parametrize("fen", POSITIONS)
def test_find_best_move_score_unchanged(fen):
    board = chess.Board(fen)
    expected = full_window_score(board, 4)
    engine.clear_transposition_table()
    engine.clear_history()
    assert find_best_move(board, 4)[1] == expected * engine.side_sign(board)


# --- Windows tried ----------------------------------------------------------------


def scripted_root(monkeypatch, results):
    """Replace search_root with one that returns `results` (move, score) in
    order and records the window and first move of every call."""
    calls = []
    answers = iter(results)

    def fake(board, depth, first_move=None, progress=None, alpha=-INF, beta=INF):
        calls.append((alpha, beta, first_move))
        return next(answers)

    monkeypatch.setattr(engine, "search_root", fake)
    return calls


E4, D4 = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")


@pytest.mark.parametrize("depth,previous", [
    (ASPIRATION_MIN_DEPTH - 1, 0),
    (ASPIRATION_MIN_DEPTH, MATE_SCORE),
    (ASPIRATION_MIN_DEPTH, -MATE_SCORE - 3),
    (ASPIRATION_MIN_DEPTH, -INF),
])
def test_full_window_when_aspiration_does_not_apply(monkeypatch, depth, previous):
    calls = scripted_root(monkeypatch, [(E4, 7)])
    assert aspiration_search(chess.Board(), depth, E4, previous) == (E4, 7)
    assert calls == [(-INF, INF, E4)]


def test_score_inside_first_window_needs_one_search(monkeypatch):
    calls = scripted_root(monkeypatch, [(E4, 120)])
    assert aspiration_search(chess.Board(), 4, E4, 100) == (E4, 120)
    assert calls == [(100 - ASPIRATION_WINDOW, 100 + ASPIRATION_WINDOW, E4)]


def test_fail_low_widens_alpha_until_opened(monkeypatch):
    # Keep failing low: alpha moves down by 100, 200, 400, then opens to -inf.
    calls = scripted_root(monkeypatch, [(E4, -10_000)] * 4 + [(E4, -900)])
    assert aspiration_search(chess.Board(), 4, E4, 0) == (E4, -900)
    alphas = [alpha for alpha, _, _ in calls]
    betas = {beta for _, beta, _ in calls}
    assert alphas == [-50, -100, -200, -400, -INF]
    assert betas == {ASPIRATION_WINDOW}
    assert 2 * 400 > ASPIRATION_MAX >= 400


def test_fail_high_widens_beta_and_tries_failing_move_first(monkeypatch):
    calls = scripted_root(monkeypatch, [(D4, 50), (D4, 10_000), (D4, 150)])
    assert aspiration_search(chess.Board(), 4, E4, 0) == (D4, 150)
    assert calls == [(-50, 50, E4), (-50, 100, D4), (-50, 200, D4)]


def test_fail_both_ways(monkeypatch):
    calls = scripted_root(monkeypatch, [(E4, -50), (D4, 60), (D4, 20)])
    assert aspiration_search(chess.Board(), 4, E4, 0) == (D4, 20)
    # One delta for both sides: it is 200 by the third search.
    assert calls == [(-50, 50, E4), (-100, 50, E4), (-100, 200, D4)]


# --- search_root with a window -------------------------------------------------


def test_search_root_stops_at_first_move_reaching_beta(monkeypatch):
    board = chess.Board(POSITIONS[1])
    exact_move, exact = search_root(board, 2)
    tried = []
    real = engine.pvs_child

    def recording(child, depth, alpha, beta, first):
        tried.append(child.peek())
        return real(child, depth, alpha, beta, first)

    monkeypatch.setattr(engine, "pvs_child", recording)
    move, score = search_root(board, 2, first_move=exact_move, beta=exact - 100)
    assert tried == [exact_move]
    assert move == exact_move and score >= exact - 100


def test_search_root_fail_low_returns_an_upper_bound():
    board = chess.Board(POSITIONS[1])
    _, exact = search_root(board, 2)
    _, score = search_root(board, 2, alpha=exact + 100, beta=exact + 200)
    assert score <= exact + 100
    assert exact <= score
