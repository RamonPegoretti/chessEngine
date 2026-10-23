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

# Weights of the other evaluation terms, in centipawns per point (see
# mobility(), king_safety() and undeveloped_pieces() below). Training tunes
# these too.
MOBILITY_WEIGHT = 3
KING_SAFETY_WEIGHT = 10
DEVELOPMENT_WEIGHT = 15

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
    weights["mobility"] = MOBILITY_WEIGHT
    weights["king_safety"] = KING_SAFETY_WEIGHT
    weights["development"] = DEVELOPMENT_WEIGHT
    return weights


def set_weights(weights):
    """Make the engine evaluate with these weights (as from get_weights).

    A weight that is missing (an engine version saved before that term
    existed) gets its default value.
    """
    global POSITION_WEIGHT, MOBILITY_WEIGHT, KING_SAFETY_WEIGHT, DEVELOPMENT_WEIGHT
    weights = with_defaults(weights)
    for name, piece in WEIGHT_NAMES.items():
        PIECE_VALUES[piece] = int(weights[name])
    POSITION_WEIGHT = int(weights["position"])
    MOBILITY_WEIGHT = int(weights["mobility"])
    KING_SAFETY_WEIGHT = int(weights["king_safety"])
    DEVELOPMENT_WEIGHT = int(weights["development"])


def with_defaults(weights):
    """A copy of `weights` with every missing weight set to its default."""
    return {**DEFAULT_WEIGHTS, **weights}


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


# Squares where the knights and bishops start. A piece still standing there
# has not been developed.
MINOR_PIECE_HOMES = {
    chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],
    chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8],
}


def evaluate(board):
    """Static evaluation of a position, in centipawns from White's view.

    It adds up, for White minus Black:
    - material: the value of every piece;
    - position: a bonus or malus for the square each piece stands on;
    - mobility: how many squares the pieces can move to;
    - king safety: pawns sheltering the king, minus squares around it that
      the opponent attacks;
    - development: a malus for each knight or bishop still at home.
    """
    score = 0
    for square, piece in board.piece_map().items():
        table = PIECE_SQUARE_TABLES[piece.piece_type]
        if piece.color == chess.WHITE:
            bonus = table[chess.square_mirror(square)] * POSITION_WEIGHT // 100
            score += PIECE_VALUES[piece.piece_type] + bonus
        else:
            bonus = table[square] * POSITION_WEIGHT // 100
            score -= PIECE_VALUES[piece.piece_type] + bonus

    score += MOBILITY_WEIGHT * (mobility(board, chess.WHITE) - mobility(board, chess.BLACK))
    score += KING_SAFETY_WEIGHT * (king_safety(board, chess.WHITE)
                                   - king_safety(board, chess.BLACK))
    score -= DEVELOPMENT_WEIGHT * (undeveloped_pieces(board, chess.WHITE)
                                   - undeveloped_pieces(board, chess.BLACK))
    return score


def mobility(board, color):
    """Number of squares the knights, bishops, rooks and queens of `color`
    attack, not counting squares taken by their own pieces.

    A piece with more squares has more options, so it is usually worth more.
    Attacked squares are used instead of legal moves so the count is the same
    whichever side is to move.
    """
    own = board.occupied_co[color]
    count = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in board.pieces(piece_type, color):
            count += chess.popcount(board.attacks_mask(square) & ~own)
    return count


def king_safety(board, color):
    """Pawns in front of the king of `color`, minus the squares next to the
    king that the opponent attacks.

    Only counts while the opponent has a queen: without one, an attack on the
    king is rarely dangerous and the king should come out and fight.
    """
    king = board.king(color)
    if king is None or not board.pieces(chess.QUEEN, not color):
        return 0

    forward = 1 if color == chess.WHITE else -1
    shield = 0
    for file in range(chess.square_file(king) - 1, chess.square_file(king) + 2):
        for distance in (1, 2):
            rank = chess.square_rank(king) + forward * distance
            if 0 <= file <= 7 and 0 <= rank <= 7:
                piece = board.piece_at(chess.square(file, rank))
                if piece == chess.Piece(chess.PAWN, color):
                    shield += 1

    attacked = sum(1 for square in board.attacks(king)
                   if board.is_attacked_by(not color, square))
    return shield - attacked


def undeveloped_pieces(board, color):
    """Number of knights and bishops of `color` still on their starting squares."""
    count = 0
    for square in MINOR_PIECE_HOMES[color]:
        piece = board.piece_at(square)
        if (piece and piece.color == color
                and piece.piece_type in (chess.KNIGHT, chess.BISHOP)):
            count += 1
    return count


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
