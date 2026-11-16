"""Version test matches: does a change make ChessCore stronger?

Plays the engine as it is now (the "candidate", the files on disk) against
the engine of an earlier git version (the "baseline", by default the main
branch), with the same search depth for both, and stops as soon as a
Sequential Probability Ratio Test (SPRT) is sure enough either way.

    python -m chesscore.versus                      # against main, depth 3
    python -m chesscore.versus --baseline HEAD~1    # against the last commit
    python -m chesscore.versus --depth 4 --games 200
    python -m chesscore.versus --depth 20 --time 0.5   # 0.5 s per move

--time sets a time limit per move (0, the default, means no limit). A
search change that only makes ChessCore faster shows up only with a time
limit: at a fixed depth both versions find the same moves.

Every opening is played twice, once with each engine as White, so neither
engine gets the luckier side of an opening more often.

The SPRT compares two ideas: "the candidate is no better" (ELO0 = 0) and
"the candidate is at least ELO1 = 10 Elo better". After each game it updates
the log-likelihood ratio (LLR): how much more likely the results are under
one idea than the other. Above the upper bound the candidate passed, below
the lower bound it failed. With ALPHA = BETA = 0.05 each wrong verdict has
about a 5% chance.
"""

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import tempfile

import chess

from chesscore import engine
from chesscore.storage import Storage

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASELINE = "main"
DEFAULT_DEPTH = 3
DEFAULT_MOVE_TIME = 0     # seconds per move; 0 means no time limit
DEFAULT_MAX_GAMES = 400
MAX_MOVES = 150           # a game still going after this many moves is a draw

ELO0, ELO1 = 0, 10
ALPHA = BETA = 0.05

# Short, balanced openings (both sides have played a normal first move or
# two). Each is played once with each engine as White.
OPENINGS = [
    "e4 e5", "e4 c5", "e4 e6", "e4 c6", "e4 d5", "e4 d6", "e4 Nf6", "e4 g6",
    "d4 d5", "d4 Nf6", "d4 f5", "d4 e6", "c4 e5", "c4 Nf6", "c4 c5", "Nf3 d5",
    "Nf3 Nf6", "g3 d5", "b3 e5", "e4 e5 Nf3 Nc6", "e4 c5 Nf3 d6",
    "d4 Nf6 c4 e6", "d4 d5 c4 c6", "d4 Nf6 c4 g6",
]


# --- Loading the baseline engine ----------------------------------------------

def load_engine(source, name):
    """Import engine source code as a separate module, so two versions of
    chesscore.engine can play each other in the same program."""
    folder = tempfile.mkdtemp(prefix="chesscore-")
    path = os.path.join(folder, f"{name}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_engine(revision=DEFAULT_BASELINE, repo_dir=REPO_DIR):
    """chesscore/engine.py as it was at a git revision (branch, tag or commit)."""
    source = subprocess.run(
        ["git", "show", f"{revision}:chesscore/engine.py"], cwd=repo_dir,
        capture_output=True, text=True, check=True).stdout
    module = load_engine(source, "baseline_engine")
    if "time_limit" not in module.find_best_move.__code__.co_varnames:
        raise SystemExit(f"The engine at {revision} has no time limit per move; "
                         "choose a version from after 16 November 2026.")
    return module


# --- Playing ------------------------------------------------------------------

def opening_board(opening):
    board = chess.Board()
    for san in opening.split():
        board.push_san(san)
    return board


def play_game(white, black, opening, depth=DEFAULT_DEPTH, move_time=DEFAULT_MOVE_TIME,
              max_moves=MAX_MOVES):
    """Play one game between two engine modules from an opening.

    Returns the result from White's view: "1-0", "0-1" or "1/2-1/2".
    """
    board = opening_board(opening)
    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        player = white if board.turn == chess.WHITE else black
        # None rather than 0 for "no limit": older engine versions treat 0
        # as zero seconds.
        move, _ = player.find_best_move(board, depth, time_limit=move_time or None)
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    return outcome.result() if outcome else "1/2-1/2"


# --- Statistics ---------------------------------------------------------------

def expected_score(elo):
    """Expected score (0..1) of a player `elo` points stronger."""
    return 1 / (1 + 10 ** (-elo / 400))


def sprt_llr(wins, draws, losses, elo0=ELO0, elo1=ELO1):
    """Log-likelihood ratio of "elo1 better" against "elo0 better".

    Uses the usual normal approximation: the average score of the games and
    its variance decide how far the results lean towards one idea or the
    other.
    """
    games = wins + draws + losses
    if games == 0:
        return 0.0
    if wins == games or losses == games:
        # All games had the same result, so the variance would be 0. Count
        # one extra draw to get a usable estimate.
        draws += 1
        games += 1
    score = (wins + draws / 2) / games
    variance = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2
                + losses * score ** 2) / games
    if variance == 0:  # every game drawn: no information yet
        return 0.0
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return games * (s1 - s0) * (2 * score - s0 - s1) / (2 * variance)


def sprt_bounds(alpha=ALPHA, beta=BETA):
    """(lower, upper) LLR bounds: below means failed, above means passed."""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def elo_estimate(wins, draws, losses):
    """Elo difference suggested by the results so far, or None if every game
    was won or every game lost."""
    games = wins + draws + losses
    score = (wins + draws / 2) / games if games else 0.5
    if score <= 0 or score >= 1:
        return None
    return -400 * math.log10(1 / score - 1)


# --- Match --------------------------------------------------------------------

def run_match(candidate, baseline, depth=DEFAULT_DEPTH, move_time=DEFAULT_MOVE_TIME,
              max_games=DEFAULT_MAX_GAMES, on_game=None, elo1=ELO1):
    """Play candidate against baseline until the SPRT decides or max_games
    games are played. Returns a summary dict.

    on_game(number, wins, draws, losses, llr), if given, is called after
    each game. Wins, draws and losses are the candidate's.
    """
    wins = draws = losses = 0
    lower, upper = sprt_bounds()
    llr = 0.0
    verdict = "undecided"
    number = 0
    while number < max_games and verdict == "undecided":
        opening = OPENINGS[(number // 2) % len(OPENINGS)]
        candidate_white = number % 2 == 0
        white, black = (candidate, baseline) if candidate_white else (baseline, candidate)
        result = play_game(white, black, opening, depth, move_time)
        number += 1
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == candidate_white:
            wins += 1
        else:
            losses += 1
        llr = sprt_llr(wins, draws, losses, ELO0, elo1)
        if llr >= upper:
            verdict = "passed"
        elif llr <= lower:
            verdict = "failed"
        if on_game:
            on_game(number, wins, draws, losses, llr)
    return {"games": number, "wins": wins, "draws": draws, "losses": losses,
            "llr": llr, "bounds": (lower, upper), "verdict": verdict,
            "elo": elo_estimate(wins, draws, losses)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m chesscore.versus",
        description="Play the current ChessCore against an earlier git version.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help="git branch, tag or commit to play against (default: main)")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help="search depth for both engines (default: 3)")
    parser.add_argument("--time", type=float, default=DEFAULT_MOVE_TIME,
                        help="time limit in seconds per move for both engines; "
                             "0 means no limit (default: 0)")
    parser.add_argument("--games", type=int, default=DEFAULT_MAX_GAMES,
                        help="stop after this many games if undecided (default: 400)")
    parser.add_argument("--elo1", type=float, default=ELO1,
                        help="Elo gain the SPRT tests for (default: 10); a larger "
                             "value decides faster but misses small gains")
    args = parser.parse_args(argv)

    baseline = baseline_engine(args.baseline)
    # Both engines evaluate with the weights of the latest trained version.
    weights = Storage().current_version()["weights"]
    engine.set_weights(weights)
    baseline.set_weights(weights)

    lower, upper = sprt_bounds()
    limit = f"{args.time:g} s per move" if args.time else "no time limit"
    print(f"Current ChessCore vs {args.baseline}, depth {args.depth}, {limit}. "
          f"SPRT [{ELO0}, {args.elo1:g}], stops when the LLR leaves "
          f"{lower:.2f}..{upper:.2f}.")

    def show(number, wins, draws, losses, llr):
        print(f"game {number}: +{wins} ={draws} -{losses}  LLR {llr:+.2f}", flush=True)

    summary = run_match(engine, baseline, args.depth, args.time, args.games, on_game=show,
                        elo1=args.elo1)
    elo = summary["elo"]
    elo_text = "n/a" if elo is None else f"{round(elo) + 0:+d}"
    print()
    print(f"{summary['games']} games: +{summary['wins']} ={summary['draws']} "
          f"-{summary['losses']}, Elo {elo_text}.")
    print({"passed": "PASSED: the current version is stronger.",
           "failed": "FAILED: the current version is not stronger.",
           "undecided": "Undecided: play more games to be sure."}[summary["verdict"]])


if __name__ == "__main__":
    main(sys.argv[1:])
