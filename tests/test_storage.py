"""JSON persistence, PGN export and engine versions (REQ-013, covers REQ-008/009/011)."""

import json

import chess
import chess.pgn
import pytest

from chesscore import storage as storage_module
from chesscore.engine import DEFAULT_WEIGHTS
from chesscore.storage import STARTING_RATING, Storage, load_json, save_json


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path / "data"))


def fools_mate():
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)
    return board


# --- JSON helpers -------------------------------------------------------------


def test_load_json_missing_file_returns_default(tmp_path):
    assert load_json(str(tmp_path / "nope.json"), {"a": 1}) == {"a": 1}


def test_save_json_round_trip_and_creates_folder(tmp_path):
    path = str(tmp_path / "deep" / "folder" / "x.json")
    save_json(path, {"games": [1, 2]})
    assert load_json(path, None) == {"games": [1, 2]}
    assert not (tmp_path / "deep" / "folder" / "x.json.tmp").exists()


# --- Versions -----------------------------------------------------------------


def test_default_version_is_not_written(storage, tmp_path):
    versions = storage.versions()
    assert [v["version"] for v in versions] == [1]
    assert versions[0]["weights"] == DEFAULT_WEIGHTS
    assert storage.current_version()["version"] == 1
    assert not (tmp_path / "data").exists()


def test_add_version_saves_both_versions(storage):
    new = storage.add_version({**DEFAULT_WEIGHTS, "knight": 330}, note="tuned")
    assert new["version"] == 2
    reloaded = Storage(storage.data_dir)
    assert [v["version"] for v in reloaded.versions()] == [1, 2]
    assert reloaded.current_version()["weights"]["knight"] == 330
    assert reloaded.current_version()["note"] == "tuned"


def test_add_version_copies_weights(storage):
    weights = dict(DEFAULT_WEIGHTS)
    storage.add_version(weights)
    weights["queen"] = 1
    assert storage.current_version()["weights"]["queen"] == DEFAULT_WEIGHTS["queen"]


def test_default_weights_not_shared_between_calls(storage):
    storage.versions()[0]["weights"]["queen"] = 1
    assert storage.versions()[0]["weights"]["queen"] == DEFAULT_WEIGHTS["queen"]


# --- Games and statistics -----------------------------------------------------


def test_add_game_assigns_increasing_ids(storage):
    assert storage.games() == []
    first = storage.add_game({"result": "1-0"})
    second = storage.add_game({"result": "0-1"})
    assert (first["id"], second["id"]) == (1, 2)
    assert [g["result"] for g in Storage(storage.data_dir).games()] == ["1-0", "0-1"]


def test_games_file_is_plain_json(storage):
    storage.add_game({"result": "1/2-1/2", "moves": ["e2e4"]})
    with open(storage.games_path, encoding="utf-8") as f:
        assert json.load(f) == [{"result": "1/2-1/2", "moves": ["e2e4"], "id": 1}]


def test_first_version_starts_at_starting_rating(storage):
    assert storage.version_stats(1) == {
        "games": 0, "wins": 0, "draws": 0, "losses": 0, "rating": STARTING_RATING}


def test_new_version_inherits_previous_rating(storage):
    storage.save_version_stats(1, {"games": 3, "wins": 2, "draws": 1, "losses": 0,
                                   "rating": 1250.5})
    assert storage.version_stats(2)["rating"] == 1250.5
    assert storage.version_stats(2)["games"] == 0
    assert storage.version_stats(1)["games"] == 3


def test_stats_keyed_by_version_text(storage):
    storage.save_version_stats(1, {"games": 1, "wins": 1, "draws": 0, "losses": 0,
                                   "rating": 1216})
    assert list(storage.stats()) == ["1"]


# --- PGN ----------------------------------------------------------------------


def test_save_pgn_writes_readable_game(storage):
    game = storage.add_game({"date": "2026-10-09", "white": "Stockfish", "black": "ChessCore v1",
                             "result": "0-1", "version": 1})
    path = storage.save_pgn(game, fools_mate())
    assert path.endswith("game_0001.pgn")
    with open(path, encoding="utf-8") as f:
        pgn = chess.pgn.read_game(f)
    assert pgn.headers["Date"] == "2026.10.09"
    assert pgn.headers["White"] == "Stockfish"
    assert pgn.headers["Black"] == "ChessCore v1"
    assert pgn.headers["Result"] == "0-1"
    assert pgn.headers["EngineVersion"] == "1"
    assert [m.uci() for m in pgn.mainline_moves()] == ["f2f3", "e7e5", "g2g4", "d8h4"]


def test_today_is_iso_date():
    text = storage_module.today()
    assert len(text) == 10 and text[4] == text[7] == "-"
