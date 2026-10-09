"""Training mode (REQ-012): play games against Stockfish, then tune the
evaluation weights from the results.

A training session:

1. plays a number of games against the external engine, alternating
   colours, with the current engine version;
2. records every game in data/games.json, saves about one in ten as PGN and
   updates the version's results and estimated rating (Elo formula);
3. tunes the evaluation weights (piece values and how much the
   piece-square tables count) on the positions of all recorded games, and
   saves the result as a new engine version if it fits the results better.

The tuning is "Texel tuning". For every position the evaluation is turned
into an expected result with a logistic curve (+400 centipawns ~ 91% to
win). The error is the mean squared difference between that and how the
game really ended (1 White won, 0.5 draw, 0 Black won). Each weight is moved
up and down a step, keeping every change that lowers the error, until no
step helps.
"""

import random

import chess

from chesscore import engine
from chesscore.engine import DEFAULT_DEPTH
from chesscore.storage import today
from chesscore.uci import match_result, play_match_game

ELO_K = 32            # how fast the estimated rating moves after each game
PGN_EVERY = 10        # save game 1, 11, 21, ... as PGN (about 10%)
MAX_MOVES = 150       # a game still going after this many moves is a draw
SKIP_OPENING_PLIES = 8
MAX_POSITIONS = 5000

# Size of one tuning step for each weight. The pawn is not tuned: it stays at
# 100 so the other values keep meaning "centipawns".
TUNING_STEPS = {"knight": 10, "bishop": 10, "rook": 10, "queen": 20, "position": 10}

# Guards against over-fitting a few games: each session moves a weight by at
# most MAX_TUNING_ROUNDS steps, weights stay within 20% of the defaults, and
# there is no tuning until enough positions have been collected.
MAX_TUNING_ROUNDS = 3
MAX_DRIFT = 0.20
MIN_POSITIONS = 1000

RESULT_FOR_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


# --- Rating (REQ-010) -----------------------------------------------------------

def expected_score(rating, opponent_rating):
    """Chance of winning (draws count half) according to the Elo formula."""
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def update_rating(rating, opponent_rating, score):
    """New rating after one game; score is 1 for a win, 0.5 draw, 0 loss."""
    return rating + ELO_K * (score - expected_score(rating, opponent_rating))


def chesscore_score(result, chesscore_color):
    """1, 0.5 or 0 from ChessCore's point of view."""
    white_score = RESULT_FOR_WHITE[result]
    return white_score if chesscore_color == chess.WHITE else 1 - white_score


# --- Tuning ---------------------------------------------------------------------

def win_probability(score):
    """Expected result for White of a position evaluated at `score` centipawns."""
    return 1 / (1 + 10 ** (-score / 400))


def training_positions(games, max_positions=MAX_POSITIONS, rng=random):
    """(board, result for White) pairs from recorded games.

    The opening is skipped (it is the same in many games) and so are
    positions in check, where the static evaluation means little.
    """
    positions = []
    for game in games:
        result = RESULT_FOR_WHITE[game["result"]]
        board = chess.Board()
        for ply, uci in enumerate(game["moves"]):
            board.push_uci(uci)
            if ply >= SKIP_OPENING_PLIES and not board.is_check():
                positions.append((board.copy(stack=False), result))
    if len(positions) > max_positions:
        positions = rng.sample(positions, max_positions)
    return positions


def tuning_error(positions, weights):
    """Mean squared error of the evaluation with `weights` on `positions`."""
    saved = engine.get_weights()
    engine.set_weights(weights)
    try:
        total = sum((result - win_probability(engine.evaluate(board))) ** 2
                    for board, result in positions)
    finally:
        engine.set_weights(saved)
    return total / len(positions)


def weights_are_valid(weights):
    """Keep the values sensible: pawn < knight <= bishop < rook < queen, and
    every weight within MAX_DRIFT of its default."""
    if not (weights["pawn"] < weights["knight"] <= weights["bishop"]
            < weights["rook"] < weights["queen"]):
        return False
    return all(abs(weights[name] - default) <= MAX_DRIFT * default
               for name, default in engine.DEFAULT_WEIGHTS.items())


def tune_weights(positions, weights, max_rounds=MAX_TUNING_ROUNDS):
    """Improve `weights` on `positions`. Returns (weights, error before, after)."""
    best = dict(weights)
    best_error = start_error = tuning_error(positions, best)
    for _ in range(max_rounds):
        improved = False
        for name, step in TUNING_STEPS.items():
            for delta in (step, -step):
                candidate = dict(best)
                candidate[name] += delta
                if not weights_are_valid(candidate):
                    continue
                error = tuning_error(positions, candidate)
                if error < best_error:
                    best, best_error, improved = candidate, error, True
                    break
        if not improved:
            break
    return best, start_error, best_error


# --- Training session -----------------------------------------------------------

def play_training_game(storage, external, opponent_rating, chesscore_color, depth):
    """Play one game, record it and update the current version's statistics."""
    version = storage.current_version()["version"]
    board = play_match_game(external, chesscore_color, depth, max_moves=MAX_MOVES)
    result = match_result(board)
    chesscore_name = f"ChessCore v{version}"
    white, black = ((chesscore_name, external.name) if chesscore_color == chess.WHITE
                    else (external.name, chesscore_name))
    game = storage.add_game({
        "date": today(),
        "version": version,
        "white": white,
        "black": black,
        "opponent_rating": opponent_rating,
        "depth": depth,
        "result": result,
        "moves": [move.uci() for move in board.move_stack],
    })
    if game["id"] % PGN_EVERY == 1:
        game["pgn"] = storage.save_pgn(game, board)
        storage.update_game(game)

    score = chesscore_score(result, chesscore_color)
    stats = storage.version_stats(version)
    stats["games"] += 1
    stats["wins" if score == 1 else "draws" if score == 0.5 else "losses"] += 1
    stats["rating"] = round(update_rating(stats["rating"], opponent_rating, score), 1)
    storage.save_version_stats(version, stats)
    return game, score, stats


def train(storage, external, opponent_rating, games=10, depth=DEFAULT_DEPTH,
          on_game=None, rng=random):
    """Run a training session. Returns a summary dict.

    on_game(number, game, score, stats), if given, is called after each game.
    """
    version = storage.current_version()
    engine.set_weights(version["weights"])

    scores = []
    for number in range(1, games + 1):
        color = chess.WHITE if number % 2 == 1 else chess.BLACK
        game, score, stats = play_training_game(
            storage, external, opponent_rating, color, depth)
        scores.append(score)
        if on_game:
            on_game(number, game, score, stats)

    positions = training_positions(storage.games(), rng=rng)
    new_version = None
    error_before = error_after = None
    if len(positions) >= MIN_POSITIONS:
        new_weights, error_before, error_after = tune_weights(positions, version["weights"])
    if error_before is not None and error_after < error_before:
        new_version = storage.add_version(
            new_weights, note=f"tuned on {len(positions)} positions after "
                              f"{len(storage.games())} games")
        engine.set_weights(new_weights)

    return {
        "version": version["version"],
        "wins": scores.count(1),
        "draws": scores.count(0.5),
        "losses": scores.count(0),
        "rating": storage.version_stats(version["version"])["rating"],
        "positions": len(positions),
        "error_before": error_before,
        "error_after": error_after,
        "old_weights": version["weights"],
        "new_version": new_version,
    }
