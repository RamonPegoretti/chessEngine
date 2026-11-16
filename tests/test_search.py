"""Minimax with alpha-beta and configurable depth (REQ-013, covers REQ-004/005).

Basic tactics are in test_engine.py; this file checks that the search itself
is correct: alpha-beta returns the same score as plain minimax, mates are
scored and preferred correctly, depth is respected and boards are copied.
"""

import chess
import pytest

from chesscore import engine
from chesscore.engine import (
    MATE_SCORE,
    evaluate,
    find_best_move,
    negamax,
    side_sign,
    ordered_moves,
    quiescence,
    terminal_score,
)
from positions import MATE_IN_ONE, PERFT_POSITIONS

INF = float("inf")


def plain_minimax(board, depth):
    """Reference minimax without pruning or move ordering. The leaves use
    the same quiescence search as the engine (with a full window, so its
    exact value)."""
    score = terminal_score(board, depth)
    if score is not None:
        return score
    if depth == 0:
        return quiescence(board, -INF, INF) * side_sign(board)
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
def test_negamax_with_full_window_matches_plain_minimax(fen):
    board = chess.Board(fen)
    assert negamax(board, 2, -INF, INF) * side_sign(board) == plain_minimax(board, 2)


def test_best_move_score_is_reproducible_by_playing_it():
    board = chess.Board(SEARCH_POSITIONS[2])
    move, score = find_best_move(board, depth=2)
    child = board.copy()
    child.push(move)
    assert negamax(child, 1, -INF, INF) * side_sign(child) == score


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


def test_quiescence_sees_recapture_at_depth_one():
    # Taking the defended pawn with the queen loses the queen to dxe5. Without
    # quiescence search depth 1 played it; now every depth sees the recapture.
    board = chess.Board("4k3/8/3p4/4p3/8/8/8/4QK2 w - - 0 1")
    for depth in (1, 2):
        move, _ = find_best_move(board, depth)
        assert move.uci() != "e1e5"


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


# --- Negamax ------------------------------------------------------------------


def test_side_sign():
    board = chess.Board()
    assert side_sign(board) == 1
    board.push_uci("e2e4")
    assert side_sign(board) == -1


@pytest.mark.parametrize("fen", SEARCH_POSITIONS)
def test_negamax_depth_zero_is_quiescence_search(fen):
    board = chess.Board(fen)
    assert negamax(board, 0, -INF, INF) == quiescence(board, -INF, INF)


def test_negamax_checkmated_side_to_move_scores_lowest():
    for fen in ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
                "rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3"):
        board = chess.Board(fen)
        assert negamax(board, 2, -INF, INF) <= -MATE_SCORE


@pytest.mark.parametrize("fen", SEARCH_POSITIONS)
def test_negamax_same_for_mirrored_position(fen):
    # The mirror swaps the colours and the side to move, so the side to move
    # stands exactly as well as before.
    board = chess.Board(fen)
    assert negamax(board, 2, -INF, INF) == negamax(board.mirror(), 2, -INF, INF)


@pytest.mark.parametrize("fen", SEARCH_POSITIONS[:3])
@pytest.mark.parametrize("offset", [-300, -30, 0, 30, 300])
def test_negamax_fail_soft_bounds(fen, offset):
    # With a narrow window around a guess, the result is exact inside the
    # window and a correct bound outside it.
    board = chess.Board(fen)
    exact = negamax(board, 2, -INF, INF)
    alpha, beta = exact + offset - 20, exact + offset + 20
    score = negamax(board, 2, alpha, beta)
    if score <= alpha:
        assert exact <= score
    elif score >= beta:
        assert exact >= score
    else:
        assert score == exact
    if alpha < exact < beta:
        assert score == exact


def test_find_best_move_scores_from_whites_view_for_black():
    # Black to move, can win the white queen: the score must be good for Black,
    # so negative.
    board = chess.Board("r3k3/8/8/8/8/2n5/8/3QK3 b - - 0 1")
    move, score = find_best_move(board, depth=2)
    assert move.uci() == "c3d1"
    assert score < 0


# --- Iterative deepening and time limit ----------------------------------------


@pytest.mark.parametrize("fen", SEARCH_POSITIONS[:3])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_search_root_matches_find_best_move(fen, depth):
    board = chess.Board(fen)
    move, score = engine.search_root(board, depth)
    assert move in board.legal_moves
    assert score == negamax(board, depth, -INF, INF)
    assert find_best_move(board, depth)[1] == score * side_sign(board)


def test_search_root_tries_first_move_first(monkeypatch):
    board = chess.Board()
    tried = []
    real = engine.negamax

    def recording(child, depth, alpha, beta):
        if len(child.move_stack) == 1:
            tried.append(child.peek().uci())
        return real(child, depth, alpha, beta)

    monkeypatch.setattr(engine, "negamax", recording)
    engine.search_root(board, 2, first_move=chess.Move.from_uci("a2a3"))
    assert tried[0] == "a2a3"
    assert sorted(tried) == sorted(m.uci() for m in board.legal_moves)


def test_search_root_ignores_illegal_first_move():
    board = chess.Board()
    move, _ = engine.search_root(board, 1, first_move=chess.Move.from_uci("e7e5"))
    assert move in board.legal_moves


def test_check_time_never_stops_without_deadline():
    for _ in range(3 * engine.TIME_CHECK_INTERVAL):
        engine.check_time()


def test_check_time_stops_after_deadline(monkeypatch):
    monkeypatch.setattr(engine, "_deadline", 0.0)  # long passed
    monkeypatch.setattr(engine, "_nodes", 0)
    with pytest.raises(engine.SearchTimeout):
        for _ in range(engine.TIME_CHECK_INTERVAL):
            engine.check_time()
    # The clock is only looked at every TIME_CHECK_INTERVAL positions.
    assert engine._nodes == engine.TIME_CHECK_INTERVAL


def test_negamax_raises_timeout_when_time_is_up(monkeypatch):
    monkeypatch.setattr(engine, "_deadline", 0.0)
    with pytest.raises(engine.SearchTimeout):
        negamax(chess.Board(), 3, -INF, INF)


def test_tiny_time_limit_plays_first_ordered_move(monkeypatch):
    # Look at the clock at every position: the time is up before depth 1
    # searched anything, so the first move in move order is played.
    monkeypatch.setattr(engine, "TIME_CHECK_INTERVAL", 1)
    board = chess.Board(SEARCH_POSITIONS[1])
    move, _ = find_best_move(board, depth=6, time_limit=1e-9)
    assert move == ordered_moves(board)[0]


def test_time_limit_is_respected_and_move_is_legal():
    import time
    board = chess.Board(PERFT_POSITIONS["kiwipete"][0])
    start = time.monotonic()
    move, score = find_best_move(board, depth=20, time_limit=0.3)
    assert time.monotonic() - start < 2
    assert move in board.legal_moves
    assert isinstance(score, (int, float))


def test_timed_out_search_returns_a_finished_depth(monkeypatch):
    # Stop during depth 3: the answer must be the one from depth 2.
    board = chess.Board(SEARCH_POSITIONS[1])
    real = engine.search_root

    def stop_at_three(board, depth, first_move=None, progress=None):
        if depth == 3:
            raise engine.SearchTimeout
        return real(board, depth, first_move, progress)

    expected = find_best_move(board, depth=2)
    monkeypatch.setattr(engine, "search_root", stop_at_three)
    assert find_best_move(board, depth=5, time_limit=60) == expected


def test_deadline_cleared_after_search():
    find_best_move(chess.Board(), depth=2, time_limit=60)
    assert engine._deadline is None
    find_best_move(chess.Board(PERFT_POSITIONS["kiwipete"][0]), depth=10, time_limit=0.05)
    assert engine._deadline is None
    # A later search without a limit is not cut short by an old deadline.
    board = chess.Board(SEARCH_POSITIONS[1])
    assert find_best_move(board, depth=2)[1] == plain_minimax(board, 2)


def test_iterative_deepening_finds_mate_with_time_limit():
    for fen, uci in MATE_IN_ONE.items():
        move, score = find_best_move(chess.Board(fen), depth=3, time_limit=5)
        assert move.uci() == uci
        assert abs(score) >= MATE_SCORE


def test_zero_time_limit_means_no_limit(monkeypatch):
    monkeypatch.setattr(engine, "TIME_CHECK_INTERVAL", 1)
    board = chess.Board(SEARCH_POSITIONS[1])
    assert find_best_move(board, depth=3, time_limit=0) == find_best_move(board, depth=3)
