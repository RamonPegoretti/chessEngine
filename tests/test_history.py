"""History heuristic for quiet-move ordering (REQ-013, covers REQ-005).

History only changes the order moves are tried in, so it must never change a
fixed-depth score, only how fast it is found.
"""

import chess
import pytest

from chesscore import engine
from chesscore.engine import (
    HISTORY,
    HISTORY_MAX,
    clear_history,
    history_add,
    history_index,
    move_priority,
    negamax,
    ordered_moves,
    update_history,
)
from positions import PERFT_POSITIONS

INF = float("inf")

MOVE = chess.Move.from_uci("g1f3")


# --- Indexing and counters ------------------------------------------------------


def test_history_index_is_unique_and_in_range():
    seen = set()
    for color in chess.COLORS:
        for frm in chess.SQUARES:
            for to in chess.SQUARES:
                seen.add(history_index(color, chess.Move(frm, to)))
    assert seen == set(range(len(HISTORY))) and len(HISTORY) == 2 * 64 * 64


def test_history_index_ignores_promotion_piece():
    assert (history_index(chess.WHITE, chess.Move.from_uci("a7a8q"))
            == history_index(chess.WHITE, chess.Move.from_uci("a7a8n")))


def test_history_add_from_zero_is_the_bonus():
    i = history_index(chess.WHITE, MOVE)
    history_add(i, 9)
    assert HISTORY[i] == 9
    history_add(i, -9)
    assert HISTORY[i] < 9


@pytest.mark.parametrize("bonus", [1, 49, 400, HISTORY_MAX])
def test_history_gravity_keeps_counters_in_range(bonus):
    i = history_index(chess.BLACK, MOVE)
    for _ in range(500):
        history_add(i, bonus)
        assert -HISTORY_MAX <= HISTORY[i] <= HISTORY_MAX
    high = HISTORY[i]
    for _ in range(500):
        history_add(i, -bonus)
        assert -HISTORY_MAX <= HISTORY[i] <= HISTORY_MAX
    assert 0 < high and HISTORY[i] < high


def test_history_gravity_slows_growth_near_the_top():
    i = history_index(chess.WHITE, MOVE)
    history_add(i, 100)
    first_step = HISTORY[i]
    for _ in range(50):
        history_add(i, 100)
    before = HISTORY[i]
    history_add(i, 100)
    assert HISTORY[i] - before < first_step


def test_update_history_rewards_cutoff_and_penalises_earlier_quiets():
    tried = [chess.Move.from_uci("a2a3"), chess.Move.from_uci("h2h3")]
    update_history(chess.WHITE, MOVE, tried, depth=3)
    assert HISTORY[history_index(chess.WHITE, MOVE)] == 9
    for move in tried:
        assert HISTORY[history_index(chess.WHITE, move)] == -9
    # The other side's counters are separate.
    assert HISTORY[history_index(chess.BLACK, MOVE)] == 0


def test_update_history_bonus_is_capped():
    update_history(chess.WHITE, MOVE, [], depth=1000)
    assert HISTORY[history_index(chess.WHITE, MOVE)] == HISTORY_MAX


def test_clear_history():
    update_history(chess.WHITE, MOVE, [chess.Move.from_uci("a2a3")], depth=5)
    clear_history()
    assert not any(HISTORY)


def test_history_starts_empty_in_every_test():
    # conftest.py clears it, so ordering never depends on test order.
    assert not any(HISTORY)


# --- Move ordering --------------------------------------------------------------


def test_quiet_move_priority_is_its_history():
    board = chess.Board()
    assert move_priority(board, MOVE) == 0
    update_history(chess.WHITE, MOVE, [], depth=10)
    assert move_priority(board, MOVE) == 100


def test_history_never_lifts_quiet_moves_above_checks_or_captures():
    board = chess.Board("4k3/8/8/8/8/8/8/r3K2R w - - 0 1")
    quiet = chess.Move.from_uci("h1h2")
    for _ in range(200):
        update_history(chess.WHITE, quiet, [], depth=100)
    assert move_priority(board, quiet) == HISTORY_MAX
    assert move_priority(board, chess.Move.from_uci("h1h8")) > HISTORY_MAX  # check
    capture = chess.Board("4k3/8/8/8/8/8/7r/4K2R w - - 0 1")
    assert move_priority(capture, chess.Move.from_uci("h1h2")) > HISTORY_MAX


def test_ordered_moves_put_best_history_first_among_quiets():
    board = chess.Board()
    update_history(chess.WHITE, chess.Move.from_uci("b1c3"), [], depth=20)
    update_history(chess.WHITE, chess.Move.from_uci("h2h4"), [], depth=10)
    ordered = [m.uci() for m in ordered_moves(board)]
    assert ordered[:2] == ["b1c3", "h2h4"]
    assert sorted(ordered) == sorted(m.uci() for m in board.legal_moves)


# --- Search ---------------------------------------------------------------------


def test_search_fills_history():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    negamax(board, 3, -INF, INF)
    assert any(HISTORY)
    assert all(-HISTORY_MAX <= h <= HISTORY_MAX for h in HISTORY)


@pytest.mark.parametrize("name", ["kiwipete", "position3", "position4"])
def test_history_never_changes_the_score(name):
    board = chess.Board(PERFT_POSITIONS[name][0])
    engine.clear_transposition_table()
    cold = negamax(board, 3, -INF, INF)
    # Fill the history with what other searches learned, then search again
    # from an empty transposition table.
    for other in PERFT_POSITIONS.values():
        negamax(chess.Board(other[0]), 2, -INF, INF)
    engine.clear_transposition_table()
    assert negamax(board, 3, -INF, INF) == cold
