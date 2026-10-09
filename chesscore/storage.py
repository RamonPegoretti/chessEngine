"""Data persistence in JSON files (REQ-008), PGN export (REQ-009) and
engine versions (REQ-011).

Everything lives in the data/ folder, one JSON file per category:

- games.json:    every training game (date, engine version, opponent,
                 colour, result, moves in UCI notation)
- stats.json:    results and estimated rating per engine version
- versions.json: every engine version and its evaluation weights

About one game in ten is also saved as a .pgn file in data/pgn/.
"""

import datetime
import json
import os

import chess
import chess.pgn

from chesscore.engine import DEFAULT_WEIGHTS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")

STARTING_RATING = 1200


def load_json(path, default):
    """Contents of a JSON file, or `default` if it does not exist yet."""
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write to a temporary file first so a crash never leaves half a file.
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temporary, path)


class Storage:
    """Reads and writes the training data in one folder."""

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.games_path = os.path.join(data_dir, "games.json")
        self.stats_path = os.path.join(data_dir, "stats.json")
        self.versions_path = os.path.join(data_dir, "versions.json")
        self.pgn_dir = os.path.join(data_dir, "pgn")

    # --- Versions (REQ-011) ---------------------------------------------------

    def versions(self):
        """All engine versions, oldest first. Version 1 is the default weights."""
        versions = load_json(self.versions_path, [])
        if not versions:
            # Not saved until a second version is added.
            versions = [{"version": 1, "date": today(), "weights": dict(DEFAULT_WEIGHTS),
                         "note": "default weights"}]
        return versions

    def current_version(self):
        return self.versions()[-1]

    def add_version(self, weights, note=""):
        versions = self.versions()
        version = {"version": versions[-1]["version"] + 1, "date": today(),
                   "weights": dict(weights), "note": note}
        versions.append(version)
        save_json(self.versions_path, versions)
        return version

    # --- Games and statistics (REQ-006-002, REQ-010) --------------------------

    def games(self):
        return load_json(self.games_path, [])

    def add_game(self, game):
        games = self.games()
        game["id"] = len(games) + 1
        games.append(game)
        save_json(self.games_path, games)
        return game

    def update_game(self, game):
        """Save changes to a game already added with add_game."""
        games = self.games()
        games[game["id"] - 1] = game
        save_json(self.games_path, games)

    def stats(self):
        """{version number as text: {"games", "wins", "draws", "losses", "rating"}}"""
        return load_json(self.stats_path, {})

    def version_stats(self, version):
        """Statistics of one version. A new version starts from the rating of
        the version before it."""
        stats = self.stats()
        key = str(version)
        if key in stats:
            return stats[key]
        previous = stats.get(str(version - 1))
        rating = previous["rating"] if previous else STARTING_RATING
        return {"games": 0, "wins": 0, "draws": 0, "losses": 0, "rating": rating}

    def save_version_stats(self, version, version_stats):
        stats = self.stats()
        stats[str(version)] = version_stats
        save_json(self.stats_path, stats)

    # --- PGN (REQ-009) --------------------------------------------------------

    def save_pgn(self, game, board):
        """Save a finished game as data/pgn/game_<id>.pgn. Returns the path."""
        pgn = chess.pgn.Game.from_board(board)
        pgn.headers["Event"] = "ChessCore training"
        pgn.headers["Date"] = game["date"].replace("-", ".")
        pgn.headers["White"] = game["white"]
        pgn.headers["Black"] = game["black"]
        pgn.headers["Result"] = game["result"]
        pgn.headers["EngineVersion"] = str(game["version"])
        os.makedirs(self.pgn_dir, exist_ok=True)
        path = os.path.join(self.pgn_dir, f"game_{game['id']:04d}.pgn")
        with open(path, "w", encoding="utf-8") as f:
            print(pgn, file=f)
        return path


def today():
    return datetime.date.today().isoformat()
