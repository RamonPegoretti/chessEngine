"""Static evaluation (REQ-013, covers REQ-004).

Start-position balance and one mirrored position are in test_engine.py; these
check the evaluation is symmetric everywhere and that material dominates.
"""

import chess
import pytest

from chesscore import engine
from chesscore.engine import PIECE_SQUARE_TABLES, PIECE_VALUES, evaluate
from positions import MATE_IN_ONE, PERFT_POSITIONS

ALL_FENS = [fen for fen, _ in PERFT_POSITIONS.values()] + list(MATE_IN_ONE)


@pytest.mark.parametrize("fen", ALL_FENS)
def test_evaluation_is_colour_symmetric(fen):
    board = chess.Board(fen)
    assert evaluate(board) == -evaluate(board.mirror())


@pytest.mark.parametrize("fen", ALL_FENS)
def test_evaluation_ignores_side_to_move(fen):
    board = chess.Board(fen)
    flipped = board.copy()
    flipped.turn = not flipped.turn
    assert evaluate(board) == evaluate(flipped)


def test_evaluation_returns_int():
    assert isinstance(evaluate(chess.Board()), int)


def test_empty_board_scores_zero():
    assert evaluate(chess.Board(None)) == 0


@pytest.mark.parametrize("piece_type", [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
def test_extra_piece_favours_its_owner(piece_type):
    base = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    white_up = base.copy()
    white_up.set_piece_at(chess.D4, chess.Piece(piece_type, chess.WHITE))
    black_up = base.copy()
    black_up.set_piece_at(chess.D5, chess.Piece(piece_type, chess.BLACK))
    assert evaluate(white_up) > evaluate(base) > evaluate(black_up)


def test_piece_values_are_ordered():
    v = PIECE_VALUES
    assert v[chess.PAWN] < v[chess.KNIGHT] <= v[chess.BISHOP] < v[chess.ROOK] < v[chess.QUEEN]


@pytest.mark.parametrize("piece_type", PIECE_SQUARE_TABLES)
def test_square_bonus_never_outweighs_a_pawn_and_a_half(piece_type):
    # Position matters, but should not make a piece worth more than the next
    # piece up; a spread over 150 centipawns would let it.
    table = PIECE_SQUARE_TABLES[piece_type]
    assert len(table) == 64
    assert max(table) - min(table) <= 150


def test_centralised_knight_beats_rim_knight():
    centre = chess.Board("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")
    rim = chess.Board("4k3/8/8/8/N7/8/8/4K3 w - - 0 1")
    assert evaluate(centre) > evaluate(rim)


def test_advanced_pawn_is_worth_more():
    advanced = chess.Board("4k3/3P4/8/8/8/8/8/4K3 w - - 0 1")
    home = chess.Board("4k3/8/8/8/8/8/3P4/4K3 w - - 0 1")
    assert evaluate(advanced) > evaluate(home)


def test_castled_king_is_safer_than_central_king():
    castled = chess.Board("4k3/8/8/8/8/8/8/6K1 w - - 0 1")
    central = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert evaluate(castled) > evaluate(central)


# --- Mobility, king safety and development -----------------------------------


def test_mobility_at_start():
    # Only the knights can move out: b1 -> a3/c3, g1 -> f3/h3.
    board = chess.Board()
    assert engine.mobility(board, chess.WHITE) == 4
    assert engine.mobility(board, chess.BLACK) == 4


def test_mobility_counts_captures_but_not_own_pieces():
    # Rook a1 sees a2..a8 (7 squares, a8 is a black piece it can take)
    # and b1..d1 (stops before its own king on e1).
    board = chess.Board("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert engine.mobility(board, chess.WHITE) == 7 + 3


def test_mobility_ignores_pawns_and_king():
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    assert engine.mobility(board, chess.WHITE) == 0


def test_mobility_same_whoever_is_to_move():
    board = chess.Board(PERFT_POSITIONS["kiwipete"][0])
    flipped = board.copy()
    flipped.turn = not flipped.turn
    for color in chess.COLORS:
        assert engine.mobility(board, color) == engine.mobility(flipped, color)


def test_king_safety_zero_without_enemy_queen():
    board = chess.Board("4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1")
    assert engine.king_safety(board, chess.WHITE) == 0


def test_king_safety_counts_shield_pawns():
    # Castled king behind f2/g2/h2 plus a pawn on g3; Black still has a queen.
    board = chess.Board("3qk3/8/8/8/8/6P1/5PPP/6K1 w - - 0 1")
    assert engine.king_safety(board, chess.WHITE) == 4


def test_king_safety_subtracts_attacked_squares():
    # The black queen on h3 attacks g2 and h2 next to the bare white king on g1;
    # no shield pawns.
    board = chess.Board("4k3/8/8/8/8/7q/8/6K1 w - - 0 1")
    attacked = sum(1 for sq in board.attacks(chess.G1)
                   if board.is_attacked_by(chess.BLACK, sq))
    assert attacked > 0
    assert engine.king_safety(board, chess.WHITE) == -attacked


def test_king_safety_shield_for_black_looks_down_the_board():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/3QK3 b - - 0 1")
    assert engine.king_safety(board, chess.BLACK) == 3


def test_king_safety_no_king():
    board = chess.Board("3qk3/8/8/8/8/8/8/8 w - - 0 1")
    assert engine.king_safety(board, chess.WHITE) == 0


def test_undeveloped_pieces():
    board = chess.Board()
    assert engine.undeveloped_pieces(board, chess.WHITE) == 4
    assert engine.undeveloped_pieces(board, chess.BLACK) == 4
    for uci in ("e2e4", "b8c6", "g1f3", "e7e5", "f1c4"):
        board.push_uci(uci)
    assert engine.undeveloped_pieces(board, chess.WHITE) == 2
    assert engine.undeveloped_pieces(board, chess.BLACK) == 3


def test_undeveloped_ignores_other_pieces_on_home_squares():
    # A rook on b1 and a black knight on g1 are not undeveloped white minors.
    board = chess.Board("4k3/8/8/8/8/8/8/1R2K1n1 w - - 0 1")
    assert engine.undeveloped_pieces(board, chess.WHITE) == 0


def test_developing_a_knight_improves_the_score():
    home = chess.Board("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1")
    developed = chess.Board("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")
    assert engine.evaluate(developed) > engine.evaluate(home)


@pytest.mark.parametrize("term,function", [
    ("mobility", lambda b: engine.mobility(b, chess.WHITE) - engine.mobility(b, chess.BLACK)),
    ("king_safety", lambda b: engine.king_safety(b, chess.WHITE) - engine.king_safety(b, chess.BLACK)),
    ("development", lambda b: engine.undeveloped_pieces(b, chess.BLACK)
     - engine.undeveloped_pieces(b, chess.WHITE)),
])
def test_each_term_weight_scales_its_contribution(term, function):
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5")
    weights = engine.get_weights()
    base = engine.evaluate(board)
    engine.set_weights({**weights, term: weights[term] + 1})
    assert engine.evaluate(board) - base == function(board)


def test_with_defaults_fills_missing_weights():
    old = {"pawn": 100, "knight": 330, "bishop": 330, "rook": 500, "queen": 900,
           "position": 90}
    filled = engine.with_defaults(old)
    assert filled == {**engine.DEFAULT_WEIGHTS, "knight": 330, "position": 90}
    assert "mobility" not in old


def test_set_weights_accepts_old_six_key_versions():
    engine.set_weights({"pawn": 100, "knight": 330, "bishop": 330, "rook": 500,
                        "queen": 900, "position": 90})
    weights = engine.get_weights()
    assert weights["knight"] == 330 and weights["position"] == 90
    for term in ("mobility", "king_safety", "development"):
        assert weights[term] == engine.DEFAULT_WEIGHTS[term]
