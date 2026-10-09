"""Perft validation of legal move generation (REQ-013, covers REQ-001/002).

Move generation comes from python-chess; these tests pin the rules the engine
relies on against the reference node counts, so a library upgrade or a wrong
board setup is caught immediately.
"""

import chess
import pytest

from positions import PERFT_POSITIONS, perft

CASES = [
    pytest.param(fen, depth, expected, id=f"{name}-d{depth}")
    for name, (fen, counts) in PERFT_POSITIONS.items()
    for depth, expected in counts.items()
]


@pytest.mark.parametrize("fen,depth,expected", CASES)
def test_perft(fen, depth, expected):
    assert perft(chess.Board(fen), depth) == expected


def test_perft_leaves_board_unchanged():
    board = chess.Board(PERFT_POSITIONS["kiwipete"][0])
    before = board.fen()
    perft(board, 2)
    assert board.fen() == before
