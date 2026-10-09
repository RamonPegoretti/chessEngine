"""Base engine (REQ-004, REQ-005): minimax search with alpha-beta pruning.

The rules of chess (move generation, check, game end) come from python-chess.
This module only decides which move to play.

Scores are in centipawns from White's point of view: positive is good for
White, negative is good for Black. White is the maximizing player and Black
the minimizing player.
"""

import chess

DEFAULT_DEPTH = 3

MATE_SCORE = 100_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# How much the piece-square tables count, in percent. 100 uses them as
# written. Training (chesscore/train.py) tunes this and the piece values.
POSITION_WEIGHT = 100

WEIGHT_NAMES = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
}


def get_weights():
    """The evaluation weights currently in use, as a plain dict."""
    weights = {name: PIECE_VALUES[piece] for name, piece in WEIGHT_NAMES.items()}
    weights["position"] = POSITION_WEIGHT
    return weights


def set_weights(weights):
    """Make the engine evaluate with these weights (as from get_weights)."""
    global POSITION_WEIGHT
    for name, piece in WEIGHT_NAMES.items():
        PIECE_VALUES[piece] = int(weights[name])
    POSITION_WEIGHT = int(weights["position"])


DEFAULT_WEIGHTS = get_weights()

# Piece-square tables: a small bonus or malus depending on where a piece
# stands. Written from White's point of view with a8 in the top-left corner,
# so a White piece on square s reads entry [chess.square_mirror(s)] and a
# Black piece reads entry [s].
PAWN_TABLE = [
     0,   0,   0,   0,   0,   0,   0,   0,
    50,  50,  50,  50,  50,  50,  50,  50,
    10,  10,  20,  30,  30,  20,  10,  10,
     5,   5,  10,  25,  25,  10,   5,   5,
     0,   0,   0,  20,  20,   0,   0,   0,
     5,  -5, -10,   0,   0, -10,  -5,   5,
     5,  10,  10, -20, -20,  10,  10,   5,
     0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
     0,   0,   0,   0,   0,   0,   0,   0,
     5,  10,  10,  10,  10,  10,  10,   5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     0,   0,   0,   5,   5,   0,   0,   0,
]

QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


def evaluate(board):
    """Static evaluation of a position, in centipawns from White's view."""
    score = 0
    for square, piece in board.piece_map().items():
        table = PIECE_SQUARE_TABLES[piece.piece_type]
        if piece.color == chess.WHITE:
            bonus = table[chess.square_mirror(square)] * POSITION_WEIGHT // 100
            score += PIECE_VALUES[piece.piece_type] + bonus
        else:
            bonus = table[square] * POSITION_WEIGHT // 100
            score -= PIECE_VALUES[piece.piece_type] + bonus
    return score


def terminal_score(board, depth):
    """Score of a finished game, or None if the game is not over.

    A mate found with more depth left is closer to the root, so it gets a
    bigger score: the engine prefers quick mates and slow losses.
    """
    if board.is_checkmate():
        # The side to move has been mated.
        if board.turn == chess.WHITE:
            return -MATE_SCORE - depth
        return MATE_SCORE + depth
    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()):
        return 0
    return None


def ordered_moves(board):
    """Legal moves with the most promising ones first.

    Alpha-beta cuts more branches when good moves are tried early, so
    promotions and captures (most valuable victim first) come before checks,
    and checks before quiet moves.
    """
    def priority(move):
        if move.promotion:
            return 20_000 + PIECE_VALUES[move.promotion]
        if board.is_capture(move):
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            return 10_000 + 10 * PIECE_VALUES[victim] - PIECE_VALUES[attacker]
        if board.gives_check(move):
            return 5_000
        return 0

    return sorted(board.legal_moves, key=priority, reverse=True)


def minimax(board, depth, alpha, beta):
    """Minimax with alpha-beta pruning.

    alpha is the best score White is already sure to get, beta the best
    score Black is already sure to get. When alpha >= beta the opponent will
    never allow this position, so the rest of the branch is skipped.

    Each child position is a copy of the board (REQ-004-003), so the
    original board is never modified.
    """
    score = terminal_score(board, depth)
    if score is not None:
        return score
    if depth == 0:
        return evaluate(board)

    if board.turn == chess.WHITE:
        best = -float("inf")
        for move in ordered_moves(board):
            child = board.copy()
            child.push(move)
            best = max(best, minimax(child, depth - 1, alpha, beta))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    best = float("inf")
    for move in ordered_moves(board):
        child = board.copy()
        child.push(move)
        best = min(best, minimax(child, depth - 1, alpha, beta))
        beta = min(beta, best)
        if alpha >= beta:
            break
    return best


def find_best_move(board, depth=DEFAULT_DEPTH):
    """Return (best move, score) for the side to move, searching `depth` plies."""
    if depth < 1:
        raise ValueError("depth must be at least 1")

    maximizing = board.turn == chess.WHITE
    best_move = None
    best_score = -float("inf") if maximizing else float("inf")
    alpha, beta = -float("inf"), float("inf")

    for move in ordered_moves(board):
        child = board.copy()
        child.push(move)
        score = minimax(child, depth - 1, alpha, beta)
        if maximizing and score > best_score:
            best_move, best_score = move, score
            alpha = max(alpha, score)
        elif not maximizing and score < best_score:
            best_move, best_score = move, score
            beta = min(beta, score)

    return best_move, best_score
