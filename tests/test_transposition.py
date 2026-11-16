"""Transposition table (REQ-013, covers REQ-005).

The table must never change the result of a search: a search with a warm
table gives exactly the score of a search with an empty one, whatever was
searched before and with whatever window.
"""

import chess
import chess.polyglot
import pytest

from chesscore import engine
from chesscore.engine import EXACT, LOWER, UPPER, find_best_move, negamax
from positions import MATE_IN_ONE, PERFT_POSITIONS

INF = float("inf")

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "2r1k3/pp3ppp/2n5/3q4/3P4/2N5/PP3PPP/R3K2R b KQ - 0 15",
    PERFT_POSITIONS["position3"][0],
    PERFT_POSITIONS["position4"][0],
]


def cold(function, *args):
    engine.clear_transposition_table()
    return function(*args)


# --- Same answers with and without the table ---------------------------------


@pytest.mark.parametrize("fen", POSITIONS)
def test_second_search_gives_same_result(fen):
    board = chess.Board(fen)
    first = cold(find_best_move, board, 3)
    assert engine.TRANSPOSITION_TABLE
    assert find_best_move(board, 3) == first


def test_table_filled_by_other_positions_changes_nothing():
    expected = {fen: cold(negamax, chess.Board(fen), 3, -INF, INF) for fen in POSITIONS}
    engine.clear_transposition_table()
    for fen in POSITIONS + POSITIONS[::-1]:
        assert negamax(chess.Board(fen), 3, -INF, INF) == expected[fen]


@pytest.mark.parametrize("fen", POSITIONS[1:4])
def test_bounds_from_narrow_windows_keep_full_search_exact(fen):
    # Narrow searches store LOWER and UPPER bounds; a full-window search
    # afterwards must still find the exact score.
    board = chess.Board(fen)
    exact = cold(negamax, board, 3, -INF, INF)
    engine.clear_transposition_table()
    for offset in (-300, -50, -1, 0, 1, 50, 300):
        negamax(board, 3, exact + offset - 10, exact + offset + 10)
    assert negamax(board, 3, -INF, INF) == exact


@pytest.mark.parametrize("fen", POSITIONS[1:4])
@pytest.mark.parametrize("offset", [-200, -20, 0, 20, 200])
def test_narrow_window_with_warm_table_gives_correct_bound(fen, offset):
    board = chess.Board(fen)
    exact = cold(negamax, board, 3, -INF, INF)
    alpha, beta = exact + offset - 15, exact + offset + 15
    # Table now full of exact results from the full search.
    score = negamax(board, 3, alpha, beta)
    if score <= alpha:
        assert exact <= score
    elif score >= beta:
        assert exact >= score
    else:
        assert score == exact


def test_entries_only_reused_at_the_same_depth():
    board = chess.Board(POSITIONS[1])
    shallow = cold(negamax, board, 2, -INF, INF)
    deep = cold(negamax, board, 3, -INF, INF)
    engine.clear_transposition_table()
    assert negamax(board, 2, -INF, INF) == shallow
    assert negamax(board, 3, -INF, INF) == deep
    assert negamax(board, 2, -INF, INF) == shallow


@pytest.mark.parametrize("fen,uci", MATE_IN_ONE.items())
def test_mates_still_found_with_warm_table(fen, uci):
    board = chess.Board(fen)
    cold(find_best_move, board, 3)
    move, score = find_best_move(board, 3)
    assert move.uci() == uci
    assert abs(score) >= engine.MATE_SCORE


def test_iterative_deepening_with_table_matches_plain_depth():
    # find_best_move searches depths 1..3, filling the table on the way;
    # the result must equal one depth-3 search from an empty table.
    board = chess.Board(POSITIONS[2])
    expected = cold(engine.search_root, board, 3)
    engine.clear_transposition_table()
    move, score = find_best_move(board, 3)
    assert score == expected[1] * engine.side_sign(board)


# --- Transpositions are recognised --------------------------------------------


def test_transposed_move_orders_share_an_entry():
    a = chess.Board()
    for uci in ("g1f3", "g8f6", "b1c3"):
        a.push_uci(uci)
    b = chess.Board()
    for uci in ("b1c3", "g8f6", "g1f3"):
        b.push_uci(uci)
    assert chess.polyglot.zobrist_hash(a) == chess.polyglot.zobrist_hash(b)
    engine.clear_transposition_table()
    negamax(a, 2, -INF, INF)
    size = len(engine.TRANSPOSITION_TABLE)
    assert chess.polyglot.zobrist_hash(b) in engine.TRANSPOSITION_TABLE
    negamax(b, 2, -INF, INF)
    assert len(engine.TRANSPOSITION_TABLE) == size


def test_warm_table_searches_fewer_positions():
    board = chess.Board(POSITIONS[1])
    engine.clear_transposition_table()
    engine._nodes = 0
    negamax(board, 3, -INF, INF)
    cold_nodes = engine._nodes
    engine._nodes = 0
    negamax(board, 3, -INF, INF)
    assert engine._nodes < cold_nodes / 10


def test_stored_best_move_is_tried_first(monkeypatch):
    board = chess.Board()
    key = chess.polyglot.zobrist_hash(board)
    favourite = chess.Move.from_uci("a2a3")
    # A deeper entry: not reused as a score, but its move still guides ordering.
    engine.TRANSPOSITION_TABLE[key] = (5, 0, EXACT, favourite)
    tried = []
    real = engine.negamax

    def recording(child, depth, alpha, beta):
        if len(child.move_stack) == 1:
            tried.append(child.peek())
        return real(child, depth, alpha, beta)

    monkeypatch.setattr(engine, "negamax", recording)
    real(board, 2, -INF, INF)
    assert tried[0] == favourite


def test_exact_entry_at_same_depth_is_returned_without_search():
    board = chess.Board()
    key = chess.polyglot.zobrist_hash(board)
    engine.TRANSPOSITION_TABLE[key] = (3, 1234, EXACT, None)
    engine._nodes = 0
    assert negamax(board, 3, -INF, INF) == 1234
    assert engine._nodes == 1


@pytest.mark.parametrize("bound,alpha,beta,cut", [
    (LOWER, -INF, 500, True),    # score >= 600 >= beta: cut
    (LOWER, -INF, 700, False),
    (UPPER, 700, INF, True),     # score <= 600 <= alpha: cut
    (UPPER, 500, INF, False),
])
def test_bound_entries_cut_only_when_window_closes(bound, alpha, beta, cut):
    board = chess.Board()
    key = chess.polyglot.zobrist_hash(board)
    engine.TRANSPOSITION_TABLE[key] = (2, 600, bound, None)
    engine._nodes = 0
    score = negamax(board, 2, alpha, beta)
    if cut:
        assert score == 600 and engine._nodes == 1
    else:
        assert engine._nodes > 1


# --- tt_store and table management --------------------------------------------


@pytest.mark.parametrize("score,bound", [(-10, UPPER), (0, UPPER), (5, EXACT),
                                         (20, LOWER), (30, LOWER)])
def test_tt_store_bound_kind(score, bound):
    engine.tt_store(42, 3, score, 0, 20, None)
    assert engine.TRANSPOSITION_TABLE[42] == (3, score, bound, None)


def test_table_emptied_when_full(monkeypatch):
    monkeypatch.setattr(engine, "TT_MAX_ENTRIES", 3)
    for key in range(3):
        engine.tt_store(key, 1, 0, -INF, INF, None)
    assert len(engine.TRANSPOSITION_TABLE) == 3
    engine.tt_store(99, 1, 0, -INF, INF, None)
    assert list(engine.TRANSPOSITION_TABLE) == [99]


def test_set_weights_clears_table():
    find_best_move(chess.Board(), 2)
    assert engine.TRANSPOSITION_TABLE
    engine.set_weights(engine.get_weights())
    assert engine.TRANSPOSITION_TABLE == {}


def test_new_weights_are_used_after_a_search():
    board = chess.Board(POSITIONS[1])
    find_best_move(board, 2)
    engine.set_weights({**engine.DEFAULT_WEIGHTS, "queen": 2000})
    with_new = find_best_move(board, 2)
    assert with_new == cold(find_best_move, board, 2)


def test_timeout_leaves_no_partial_entries_wrong():
    # A search cut short by the clock must not leave wrong scores behind.
    board = chess.Board(PERFT_POSITIONS["kiwipete"][0])
    engine.clear_transposition_table()
    find_best_move(board, depth=10, time_limit=0.05)
    assert negamax(board, 2, -INF, INF) == cold(negamax, board, 2, -INF, INF)
