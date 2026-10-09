"""Static evaluation (REQ-013, covers REQ-004).

Start-position balance and one mirrored position are in test_engine.py; these
check the evaluation is symmetric everywhere and that material dominates.
"""

import chess
import pytest

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
