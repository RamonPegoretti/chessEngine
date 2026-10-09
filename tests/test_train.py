"""Training mode: Elo estimate, weight tuning and training sessions
(REQ-013, covers REQ-010/011/012). A fake external engine replaces Stockfish."""

import random

import chess
import pytest

from chesscore import engine, train
from chesscore.engine import DEFAULT_WEIGHTS
from chesscore.storage import STARTING_RATING, Storage


class FakeExternal:
    """Plays the first legal move, so games are quick and repeatable."""

    name = "Fake"

    def play(self, board):
        return next(iter(board.legal_moves))


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path / "data"))


def quick_games(monkeypatch, max_moves=3):
    """Make training games short: depth 1 and only a few moves."""
    monkeypatch.setattr(train, "MAX_MOVES", max_moves)


# --- Engine weights -----------------------------------------------------------


def test_default_weights():
    assert DEFAULT_WEIGHTS == {"pawn": 100, "knight": 320, "bishop": 330,
                               "rook": 500, "queen": 900, "position": 100}
    assert engine.get_weights() == DEFAULT_WEIGHTS


def test_set_weights_changes_evaluation():
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    before = engine.evaluate(board)
    engine.set_weights({**DEFAULT_WEIGHTS, "queen": 1000})
    assert engine.evaluate(board) == before + 100
    assert engine.get_weights()["queen"] == 1000


def test_position_weight_zero_counts_material_only():
    engine.set_weights({**DEFAULT_WEIGHTS, "position": 0})
    board = chess.Board("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")
    assert engine.evaluate(board) == DEFAULT_WEIGHTS["knight"]


@pytest.mark.parametrize("position", [0, 50, 90, 110, 120])
def test_evaluation_stays_symmetric_with_any_position_weight(position):
    engine.set_weights({**DEFAULT_WEIGHTS, "position": position})
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert engine.evaluate(board) == -engine.evaluate(board.mirror())


# --- Rating -------------------------------------------------------------------


def test_expected_score():
    assert train.expected_score(1500, 1500) == 0.5
    assert train.expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert train.expected_score(1500, 1900) == pytest.approx(1 / 11)


def test_update_rating():
    assert train.update_rating(1500, 1500, 1) == 1516
    assert train.update_rating(1500, 1500, 0) == 1484
    assert train.update_rating(1500, 1500, 0.5) == 1500
    # Beating a much stronger opponent gains almost the full K.
    assert train.update_rating(1200, 2400, 1) == pytest.approx(1232, abs=0.1)


@pytest.mark.parametrize("result,color,score", [
    ("1-0", chess.WHITE, 1), ("1-0", chess.BLACK, 0),
    ("0-1", chess.WHITE, 0), ("0-1", chess.BLACK, 1),
    ("1/2-1/2", chess.WHITE, 0.5), ("1/2-1/2", chess.BLACK, 0.5),
])
def test_chesscore_score(result, color, score):
    assert train.chesscore_score(result, color) == score


# --- Tuning -------------------------------------------------------------------


def test_win_probability():
    assert train.win_probability(0) == 0.5
    assert train.win_probability(400) == pytest.approx(10 / 11)
    assert train.win_probability(-400) == pytest.approx(1 / 11)


def test_training_positions_skip_opening_and_checks():
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5",
             "e4d5", "f6d5", "g5f7", "e8f7", "d1f3", "f7e6", "b1c3"]
    positions = train.training_positions([{"result": "1-0", "moves": moves}])
    # 15 plies, the first 8 skipped, and 1.. Bc4 line has checks at plies 12 and 14.
    board = chess.Board()
    expected = 0
    for ply, uci in enumerate(moves):
        board.push_uci(uci)
        if ply >= 8 and not board.is_check():
            expected += 1
    assert len(positions) == expected < 7
    assert all(result == 1.0 for _, result in positions)
    assert all(not b.is_check() for b, _ in positions)


def test_training_positions_capped_and_sampled():
    game = {"result": "1/2-1/2", "moves": ["g1f3", "g8f6", "f3g1", "f6g8"] * 10}
    positions = train.training_positions([game], max_positions=5, rng=random.Random(1))
    assert len(positions) == 5
    assert all(result == 0.5 for _, result in positions)


def test_tuning_error_restores_weights():
    positions = [(chess.Board(), 0.5)]
    train.tuning_error(positions, {**DEFAULT_WEIGHTS, "queen": 1000})
    assert engine.get_weights() == DEFAULT_WEIGHTS


def test_tuning_error_zero_for_perfect_prediction():
    assert train.tuning_error([(chess.Board(), 0.5)], DEFAULT_WEIGHTS) == 0


@pytest.mark.parametrize("change,valid", [
    ({}, True),
    ({"knight": 330}, True),
    ({"knight": 340, "bishop": 330}, False),   # knight above bishop
    ({"queen": 1100}, False),                  # more than 20% from default
    ({"position": 80}, True),
    ({"position": 79}, False),
    ({"rook": 300, "bishop": 300, "knight": 290}, False),
])
def test_weights_are_valid(change, valid):
    assert train.weights_are_valid({**DEFAULT_WEIGHTS, **change}) is valid


def test_tune_weights_lowers_error_and_stays_valid():
    # White is a knight up in every position and always wins: a higher knight
    # value explains the results better.
    board = chess.Board("4k3/pppp4/8/8/3N4/8/PPPP4/4K3 w - - 0 1")
    positions = [(board, 1.0)] * 20
    weights, before, after = train.tune_weights(positions, DEFAULT_WEIGHTS, max_rounds=2)
    assert after < before
    assert weights["knight"] > DEFAULT_WEIGHTS["knight"]
    assert weights["pawn"] == DEFAULT_WEIGHTS["pawn"]
    assert train.weights_are_valid(weights)
    assert engine.get_weights() == DEFAULT_WEIGHTS


def test_tune_weights_moves_each_weight_at_most_max_rounds_steps():
    board = chess.Board("4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1")
    weights, _, _ = train.tune_weights([(board, 1.0)] * 5, DEFAULT_WEIGHTS, max_rounds=1)
    for name, step in train.TUNING_STEPS.items():
        assert abs(weights[name] - DEFAULT_WEIGHTS[name]) <= step


def test_tune_weights_keeps_weights_when_nothing_helps():
    weights, before, after = train.tune_weights([(chess.Board(), 0.5)], DEFAULT_WEIGHTS)
    assert weights == DEFAULT_WEIGHTS
    assert before == after == 0


# --- Training games and sessions ----------------------------------------------


def test_play_training_game_records_game_and_stats(storage, monkeypatch):
    quick_games(monkeypatch)
    game, score, stats = train.play_training_game(
        storage, FakeExternal(), 1500, chess.WHITE, depth=1)
    assert game["id"] == 1
    assert game["white"] == "ChessCore v1" and game["black"] == "Fake"
    assert game["result"] == "1/2-1/2"          # stopped after MAX_MOVES
    assert len(game["moves"]) == 6
    assert game["opponent_rating"] == 1500 and game["depth"] == 1
    assert "pgn" in game                        # game 1 is saved as PGN
    assert score == 0.5
    assert stats["games"] == stats["draws"] == 1
    assert stats["rating"] == pytest.approx(train.update_rating(STARTING_RATING, 1500, 0.5), abs=0.1)
    assert storage.games()[0]["moves"] == game["moves"]
    assert storage.version_stats(1) == stats


def test_play_training_game_as_black(storage, monkeypatch):
    quick_games(monkeypatch)
    game, _, _ = train.play_training_game(storage, FakeExternal(), 1500, chess.BLACK, 1)
    assert game["white"] == "Fake" and game["black"] == "ChessCore v1"


def test_win_and_loss_are_counted(storage, monkeypatch):
    # ChessCore (Black) delivers Fool's mate: a win.
    white = iter(["f2f3", "g2g4"])
    black = iter(["e7e5", "d8h4"])

    class Scripted:
        name = "Scripted"

        def play(self, board):
            return chess.Move.from_uci(next(white))

    monkeypatch.setattr(train, "play_match_game",
                        lambda ext, color, depth, max_moves: _scripted_board(ext, black))
    _, score, stats = train.play_training_game(storage, Scripted(), 1500, chess.BLACK, 1)
    assert score == 1
    assert (stats["wins"], stats["draws"], stats["losses"]) == (1, 0, 0)
    assert stats["rating"] > STARTING_RATING


def _scripted_board(external, black_moves):
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            board.push(external.play(board))
        else:
            board.push_uci(next(black_moves))
    return board


def test_train_session_summary_and_pgn_sampling(storage, monkeypatch, tmp_path):
    quick_games(monkeypatch, max_moves=2)
    calls = []
    summary = train.train(storage, FakeExternal(), 1400, games=12, depth=1,
                          on_game=lambda *args: calls.append(args))
    assert [number for number, *_ in calls] == list(range(1, 13))
    games = storage.games()
    assert len(games) == 12
    # Colours alternate, ChessCore starts with White.
    assert games[0]["white"].startswith("ChessCore") and games[1]["black"].startswith("ChessCore")
    assert [game["id"] for _, game, _, _ in calls if "pgn" in game] == [1, 11]
    assert sorted(p.name for p in (tmp_path / "data" / "pgn").iterdir()) == [
        "game_0001.pgn", "game_0011.pgn"]
    assert summary["draws"] == 12 and summary["wins"] == summary["losses"] == 0
    assert summary["version"] == 1
    assert summary["rating"] == storage.version_stats(1)["rating"]
    # Too few positions to tune yet.
    assert summary["error_before"] is None and summary["new_version"] is None
    assert storage.current_version()["version"] == 1


def test_train_creates_new_version_when_tuning_helps(storage, monkeypatch):
    quick_games(monkeypatch, max_moves=8)
    monkeypatch.setattr(train, "MIN_POSITIONS", 1)
    tuned = {**DEFAULT_WEIGHTS, "knight": 330}
    monkeypatch.setattr(train, "tune_weights", lambda positions, weights: (tuned, 0.2, 0.1))
    summary = train.train(storage, FakeExternal(), 1400, games=10, depth=1)
    assert summary["new_version"]["version"] == 2
    assert storage.current_version()["weights"] == tuned
    assert engine.get_weights() == tuned
    assert summary["old_weights"] == DEFAULT_WEIGHTS
    # The new version starts from the old version's rating.
    assert storage.version_stats(2)["rating"] == summary["rating"]


def test_train_keeps_version_when_tuning_does_not_help(storage, monkeypatch):
    quick_games(monkeypatch, max_moves=8)
    monkeypatch.setattr(train, "MIN_POSITIONS", 1)
    monkeypatch.setattr(train, "tune_weights",
                        lambda positions, weights: (dict(weights), 0.2, 0.2))
    summary = train.train(storage, FakeExternal(), 1400, games=10, depth=1)
    assert summary["error_before"] == summary["error_after"] == 0.2
    assert summary["new_version"] is None
    assert len(storage.versions()) == 1


def test_train_uses_current_version_weights(storage, monkeypatch):
    quick_games(monkeypatch, max_moves=1)
    tuned = {**DEFAULT_WEIGHTS, "rook": 520}
    storage.add_version(tuned)
    train.train(storage, FakeExternal(), 1400, games=1, depth=1)
    assert engine.get_weights() == tuned
    assert storage.games()[0]["white"] == "ChessCore v2"
