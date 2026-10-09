"""Chess rules the engine depends on (REQ-013, covers REQ-001/002).

Special moves (castling, en passant, promotion), check, and every way a game
can end. Rules come from python-chess; these tests document and pin them.
"""

import chess


def legal_uci(board: chess.Board) -> set[str]:
    return {move.uci() for move in board.legal_moves}


# --- Castling -----------------------------------------------------------------


def test_castling_both_sides_when_path_clear():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert {"e1g1", "e1c1"} <= legal_uci(board)


def test_castling_moves_the_rook():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_uci("e1g1")
    assert board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
    assert board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)
    assert board.piece_at(chess.H1) is None


def test_no_castling_out_of_check():
    board = chess.Board("4r1k1/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert board.is_check()
    assert not {"e1g1", "e1c1"} & legal_uci(board)


def test_no_castling_through_attacked_square():
    # Black rook on f8 attacks f1, so kingside castling is illegal.
    board = chess.Board("5rk1/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert "e1g1" not in legal_uci(board)
    assert "e1c1" in legal_uci(board)


def test_no_castling_when_path_blocked():
    board = chess.Board("4k3/8/8/8/8/8/8/RN2K1NR w KQ - 0 1")
    assert not {"e1g1", "e1c1"} & legal_uci(board)


def test_castling_rights_lost_after_king_moves():
    board = chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    for uci in ("e1e2", "e8e7", "e2e1", "e7e8"):
        board.push_uci(uci)
    assert not board.has_castling_rights(chess.WHITE)
    assert not {"e1g1", "e1c1"} & legal_uci(board)


def test_castling_rights_lost_after_rook_moves():
    board = chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    board.push_uci("h1h2")
    assert not board.has_kingside_castling_rights(chess.WHITE)
    assert board.has_queenside_castling_rights(chess.WHITE)


# --- En passant ---------------------------------------------------------------


def test_en_passant_available_right_after_double_push():
    board = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    board.push_uci("d7d5")
    assert "e5d6" in legal_uci(board)


def test_en_passant_removes_captured_pawn():
    board = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    board.push_uci("d7d5")
    move = chess.Move.from_uci("e5d6")
    assert board.is_en_passant(move)
    board.push(move)
    assert board.piece_at(chess.D5) is None
    assert board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)


def test_en_passant_expires_after_one_move():
    board = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    for uci in ("d7d5", "e1e2", "e8e7"):
        board.push_uci(uci)
    assert "e5d6" not in legal_uci(board)


def test_en_passant_illegal_if_it_exposes_king():
    # Capturing en passant would open the 5th rank to the black rook.
    board = chess.Board("8/8/8/r2pP2K/8/8/8/4k3 w - d6 0 1")
    assert "e5d6" not in legal_uci(board)


# --- Promotion ----------------------------------------------------------------


def test_promotion_offers_all_four_pieces():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= legal_uci(board)
    assert "a7a8" not in legal_uci(board)


def test_promotion_places_chosen_piece():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    board.push_uci("a7a8n")
    assert board.piece_at(chess.A8) == chess.Piece(chess.KNIGHT, chess.WHITE)


def test_promotion_by_capture():
    board = chess.Board("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert "a7b8q" in legal_uci(board)


# --- Check and pins -----------------------------------------------------------


def test_check_detected():
    board = chess.Board("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1")
    assert board.is_check()


def test_in_check_only_evasions_are_legal():
    board = chess.Board("4k3/8/8/8/8/8/3P4/r3K3 w - - 0 1")
    for move in board.legal_moves:
        board.push(move)
        assert not board.was_into_check()
        board.pop()
    assert "d2d3" not in legal_uci(board)


def test_pinned_piece_cannot_leave_pin_line():
    board = chess.Board("4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1")
    assert not any(m.from_square == chess.E2 for m in board.legal_moves)


def test_king_cannot_move_into_check():
    board = chess.Board("4k3/8/8/8/8/8/3r4/4K3 w - - 0 1")
    assert "e1d1" not in legal_uci(board)
    assert "e1e2" not in legal_uci(board)
    assert "e1d2" in legal_uci(board)


# --- Game endings -------------------------------------------------------------


def test_checkmate():
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)
    assert board.is_checkmate()
    assert board.is_game_over()
    assert board.result() == "0-1"


def test_stalemate():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert not board.is_check()
    assert board.result() == "1/2-1/2"


def test_insufficient_material():
    assert chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1").is_insufficient_material()
    assert chess.Board("8/8/4k3/8/8/4KB2/8/8 w - - 0 1").is_insufficient_material()
    assert not chess.Board("8/8/4k3/8/8/4KR2/8/8 w - - 0 1").is_insufficient_material()


def test_threefold_repetition():
    board = chess.Board()
    for _ in range(2):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(uci)
    assert board.can_claim_threefold_repetition()
    assert board.result(claim_draw=True) == "1/2-1/2"


def test_fifty_move_rule():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 99 80")
    board.push_uci("h1h2")
    assert board.can_claim_fifty_moves()


def test_seventy_five_move_rule_ends_game_automatically():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 150 100")
    assert board.is_seventyfive_moves()
    assert board.is_game_over()


def test_game_not_over_at_start():
    board = chess.Board()
    assert not board.is_game_over()
    assert board.result() == "*"
