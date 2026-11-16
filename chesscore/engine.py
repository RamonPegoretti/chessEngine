"""Base engine (REQ-004, REQ-005): negamax search with alpha-beta pruning.

The rules of chess (move generation, check, game end) come from python-chess.
This module only decides which move to play.

evaluate(), terminal_score() and find_best_move() give scores in centipawns
from White's point of view: positive is good for White, negative is good for
Black. Inside the search, negamax() works from the point of view of the side
to move instead, which lets one function search for both sides.
"""


import time

import chess

DEFAULT_DEPTH = 3

# How often (in searched positions) the search looks at the clock.
TIME_CHECK_INTERVAL = 200

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


def move_priority(board, move):
    """How promising a move looks, for move ordering: promotions first, then
    captures (most valuable victim first, then least valuable attacker),
    then checks, then quiet moves."""
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


def ordered_moves(board):
    """Legal moves with the most promising ones first.

    Alpha-beta cuts more branches when good moves are tried early.
    """
    return sorted(board.legal_moves, key=lambda move: move_priority(board, move),
                  reverse=True)


def noisy_moves(board):
    """Captures and promotions, best first: the moves quiescence search
    looks at."""
    moves = list(board.generate_legal_captures())
    moves += [move for move in board.generate_legal_moves(to_mask=chess.BB_BACKRANKS)
              if move.promotion and not board.is_capture(move)]
    return sorted(moves, key=lambda move: move_priority(board, move), reverse=True)


def side_sign(board):
    """+1 if White is to move, -1 if Black is: turns a score from White's
    view into one from the side to move's view, and back."""
    return 1 if board.turn == chess.WHITE else -1


def negamax(board, depth, alpha, beta):
    """Negamax with alpha-beta pruning (fail-soft).

    The score is from the point of view of the side to move: positive is
    good for the player who moves now. A position that is good for me is
    exactly as bad for my opponent, so the score of a move is minus the
    score of the position it leads to, seen by the opponent. That lets one
    function play both sides, where minimax needs a maximizing branch for
    White and a minimizing one for Black.

    alpha is the score the side to move is already sure to get, beta the
    score the opponent will allow. When alpha >= beta the opponent will
    never let the game reach this position, so the remaining moves are
    skipped. "Fail-soft" means the best score found is returned even when it
    falls outside alpha..beta, which gives a more precise bound.

    Each child position is a copy of the board (REQ-004-003), so the
    original board is never modified.
    """
    check_time()
    score = terminal_score(board, depth)
    if score is not None:
        return score * side_sign(board)
    if depth == 0:
        return quiescence(board, alpha, beta)

    best = -float("inf")
    for move in ordered_moves(board):
        child = board.copy()
        child.push(move)
        # The opponent's window is ours seen from the other side: -beta..-alpha.
        score = -negamax(child, depth - 1, -beta, -alpha)
        best = max(best, score)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


class SearchTimeout(Exception):
    """Raised inside the search when the time for this move has run out."""


# Time limit of the search in progress (time.monotonic() value, or None for
# no limit) and the number of positions searched so far.
_deadline = None
_nodes = 0


def check_time():
    """Count one searched position, and every TIME_CHECK_INTERVAL positions
    stop the search if the deadline has passed."""
    global _nodes
    _nodes += 1
    if (_deadline is not None and _nodes % TIME_CHECK_INTERVAL == 0
            and time.monotonic() >= _deadline):
        raise SearchTimeout


def search_root(board, depth, first_move=None):
    """Search every move of the side to move to `depth` plies.

    Returns (best move, score from the side to move's view). `first_move`,
    the best move of the previous depth, is tried first: it is probably
    still good, and a good first move lets alpha-beta skip more.
    """
    moves = ordered_moves(board)
    if first_move in moves:
        moves.remove(first_move)
        moves.insert(0, first_move)

    best_move = None
    best_score = -float("inf")
    alpha, beta = -float("inf"), float("inf")
    for move in moves:
        child = board.copy()
        child.push(move)
        score = -negamax(child, depth - 1, -beta, -alpha)
        if score > best_score:
            best_move, best_score = move, score
            alpha = max(alpha, score)
    return best_move, best_score


def quiescence(board, alpha, beta):
    """Quiescence search: at the end of the normal search, keep looking at
    captures until the position is quiet, then evaluate.

    Without it the search stops at a fixed depth even in the middle of an
    exchange: after "I take your queen" it evaluates and never sees that my
    piece is taken back (the horizon effect).

    The side to move may always "stand pat": stop capturing and keep the
    static evaluation, because in a real game nobody is forced to capture.
    So the static evaluation is a lower bound, and only captures that do
    better than it are searched. Scores are from the side to move's view and
    alpha-beta works as in negamax().

    In check, standing pat is not allowed (the threat must be answered), so
    every legal move is searched; with none, it is checkmate.
    """
    check_time()
    if board.is_check():
        moves = ordered_moves(board)
        if not moves:
            return terminal_score(board, 0) * side_sign(board)
        best = -float("inf")
    else:
        best = evaluate(board) * side_sign(board)  # stand pat
        if best >= beta:
            return best
        alpha = max(alpha, best)
        moves = noisy_moves(board)

    for move in moves:
        # A capture can never repeat an earlier position, so the move
        # history is not needed: copying without it is much faster.
        child = board.copy(stack=False)
        child.push(move)
        score = -quiescence(child, -beta, -alpha)
        best = max(best, score)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def find_best_move(board, depth=DEFAULT_DEPTH, time_limit=None):
    """Return (best move, score) for the side to move.

    Iterative deepening: search 1 ply deep, then 2, then 3, and so on up to
    `depth`, each time starting with the best move found so far. The deeper
    searches go faster thanks to that move ordering, and there is always a
    finished answer to fall back on.

    time_limit, in seconds, stops the search when it runs out (the "hard
    bound"), even in the middle of a depth: the best move of the last
    finished depth is played. Depth 1 is always finished, so there is always
    a move. None or 0 means no time limit. The score is from White's point of view, like evaluate().
    """
    global _deadline, _nodes
    if depth < 1:
        raise ValueError("depth must be at least 1")

    start = time.monotonic()
    _nodes = 0
    best_move, best_score = search_root(board, 1)
    _deadline = start + time_limit if time_limit else None
    try:
        for current_depth in range(2, depth + 1):
            best_move, best_score = search_root(board, current_depth, best_move)
    except SearchTimeout:
        pass
    finally:
        _deadline = None

    return best_move, best_score * side_sign(board)
