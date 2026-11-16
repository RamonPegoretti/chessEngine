"""Version test matches with an SPRT (chesscore.versus).

Games use stand-in engine modules so they are fast and their results are
known; one test plays two real engine copies at depth 1.
"""

import math
import subprocess

import chess
import pytest

from chesscore import engine, versus

ENGINE_PATH = engine.__file__


class Scripted:
    """An engine module stand-in: plays the first legal move, or a fixed
    move list, and records the arguments it was called with."""

    def __init__(self, moves=()):
        self.moves = list(moves)
        self.calls = []

    def find_best_move(self, board, depth, time_limit=None):
        self.calls.append((depth, time_limit))
        for uci in self.moves:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                self.moves.remove(uci)
                return move, 0
        return next(iter(board.legal_moves)), 0


# --- Loading engines ------------------------------------------------------------


def test_load_engine_is_a_separate_module():
    with open(ENGINE_PATH, encoding="utf-8") as f:
        copy = versus.load_engine(f.read(), "engine_copy")
    assert copy is not engine
    copy.set_weights({**engine.DEFAULT_WEIGHTS, "queen": 1})
    # Changing the copy's weights leaves the real engine alone.
    assert engine.get_weights()["queen"] == engine.DEFAULT_WEIGHTS["queen"]
    board = chess.Board()
    assert copy.find_best_move(board, 1)[0] in board.legal_moves


def test_baseline_engine_from_git(monkeypatch):
    with open(ENGINE_PATH, encoding="utf-8") as f:
        source = f.read()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=source)

    monkeypatch.setattr(versus.subprocess, "run", fake_run)
    module = versus.baseline_engine("abc123", repo_dir=".")
    assert calls == [["git", "show", "abc123:chesscore/engine.py"]]
    assert hasattr(module, "find_best_move")


def test_baseline_engine_without_time_limit_refused(monkeypatch):
    old = "def find_best_move(board, depth=3):\n    return None, 0\n"
    monkeypatch.setattr(versus.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=old))
    with pytest.raises(SystemExit):
        versus.baseline_engine("old")


def test_baseline_engine_unknown_revision(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(subprocess.CalledProcessError):
        versus.baseline_engine("no-such-branch", repo_dir=str(tmp_path))


# --- Playing --------------------------------------------------------------------


@pytest.mark.parametrize("opening", versus.OPENINGS)
def test_every_opening_is_legal(opening):
    board = versus.opening_board(opening)
    assert len(board.move_stack) == len(opening.split())
    assert not board.is_game_over()


def test_openings_are_unique():
    assert len(set(versus.OPENINGS)) == len(versus.OPENINGS)


def test_play_game_checkmate_for_black():
    white = Scripted(["f2f3", "g2g4"])
    black = Scripted(["e7e5", "d8h4"])
    assert versus.play_game(white, black, "", depth=2) == "0-1"


def test_play_game_starts_from_opening():
    # From 1. e4 e5 the scripted White mates with the Scholar's mate.
    white = Scripted(["f1c4", "d1h5", "h5f7"])
    black = Scripted(["b8c6", "g8f6"])
    assert versus.play_game(white, black, "e4 e5") == "1-0"


def test_play_game_move_limit_is_a_draw():
    white, black = Scripted(["g1f3", "f3g1"] * 10), Scripted(["g8f6", "f6g8"] * 10)
    assert versus.play_game(white, black, "", max_moves=2) == "1/2-1/2"


def test_play_game_repetition_is_a_draw():
    white, black = Scripted(["g1f3", "f3g1"] * 50), Scripted(["g8f6", "f6g8"] * 50)
    assert versus.play_game(white, black, "") == "1/2-1/2"
    # After 7 plies White could repeat the start position a third time, which
    # python-chess already counts as a claimable draw: long before the limit.
    assert len(white.calls) + len(black.calls) == 7


@pytest.mark.parametrize("move_time,expected", [(0, None), (0.5, 0.5)])
def test_play_game_passes_depth_and_time(move_time, expected):
    white, black = Scripted(), Scripted()
    versus.play_game(white, black, "", depth=4, move_time=move_time, max_moves=1)
    assert set(white.calls + black.calls) == {(4, expected)}


def test_play_game_with_real_engines():
    with open(ENGINE_PATH, encoding="utf-8") as f:
        source = f.read()
    a = versus.load_engine(source, "engine_a")
    b = versus.load_engine(source, "engine_b")
    assert versus.play_game(a, b, "e4 e5", depth=1, max_moves=6) in {"1-0", "0-1", "1/2-1/2"}


# --- Statistics -----------------------------------------------------------------


def test_expected_score():
    assert versus.expected_score(0) == 0.5
    assert versus.expected_score(400) == pytest.approx(10 / 11)
    assert versus.expected_score(-100) == pytest.approx(1 - versus.expected_score(100))


def test_sprt_bounds():
    lower, upper = versus.sprt_bounds()
    assert lower == pytest.approx(-2.944, abs=1e-3)
    assert upper == pytest.approx(2.944, abs=1e-3)
    assert versus.sprt_bounds(0.05, 0.1)[0] > lower


def test_sprt_llr_no_games_or_all_draws():
    assert versus.sprt_llr(0, 0, 0) == 0
    assert versus.sprt_llr(0, 10, 0) == 0


def test_sprt_llr_sign_follows_results():
    assert versus.sprt_llr(30, 40, 10) > 0
    assert versus.sprt_llr(10, 40, 30) < 0
    # More games with the same proportions give a stronger verdict.
    assert versus.sprt_llr(60, 80, 20) > versus.sprt_llr(30, 40, 10)


def test_sprt_llr_all_wins_or_losses_is_finite():
    win = versus.sprt_llr(5, 0, 0)
    loss = versus.sprt_llr(0, 0, 5)
    assert math.isfinite(win) and math.isfinite(loss)
    assert win > 0 > loss


def test_sprt_llr_matches_formula():
    # 20 wins, 50 draws, 10 losses: score 0.5625.
    w, d, l = 20, 50, 10
    n = w + d + l
    score = (w + d / 2) / n
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * score ** 2) / n
    s0, s1 = 0.5, versus.expected_score(10)
    assert versus.sprt_llr(w, d, l) == pytest.approx(n * (s1 - s0) * (2 * score - s0 - s1) / (2 * var))


@pytest.mark.parametrize("results,elo", [
    ((5, 0, 5), 0),
    ((0, 10, 0), 0),
    ((3, 0, 1), pytest.approx(190.85, abs=0.01)),   # score 0.75
    ((1, 0, 3), pytest.approx(-190.85, abs=0.01)),
    ((4, 0, 0), None),
    ((0, 0, 4), None),
])
def test_elo_estimate(results, elo):
    assert versus.elo_estimate(*results) == elo


def test_elo_estimate_no_games():
    assert versus.elo_estimate(0, 0, 0) == 0


# --- Match ----------------------------------------------------------------------


def fake_results(monkeypatch, results):
    """Make play_game return results from White's view in order, and record
    who played White with which opening."""
    games = []
    answers = iter(results)

    def fake_play(white, black, opening, depth, move_time):
        games.append((white, black, opening, depth, move_time))
        return next(answers)

    monkeypatch.setattr(versus, "play_game", fake_play)
    return games


def test_run_match_colours_and_openings(monkeypatch):
    games = fake_results(monkeypatch, ["1/2-1/2"] * 6)
    summary = versus.run_match("cand", "base", depth=2, move_time=0.1, max_games=6)
    assert [(w, b) for w, b, *_ in games] == [("cand", "base"), ("base", "cand")] * 3
    assert [g[2] for g in games] == [versus.OPENINGS[i // 2] for i in range(6)]
    assert {(g[3], g[4]) for g in games} == {(2, 0.1)}
    assert summary["games"] == 6 and summary["draws"] == 6
    assert summary["verdict"] == "undecided" and summary["llr"] == 0


def test_run_match_counts_results_from_candidates_view(monkeypatch):
    # Game 1: candidate White wins. Game 2: candidate Black, White wins (loss).
    # Game 3: candidate White, Black wins (loss). Game 4: candidate Black wins.
    fake_results(monkeypatch, ["1-0", "1-0", "0-1", "0-1"])
    summary = versus.run_match("cand", "base", max_games=4)
    assert (summary["wins"], summary["draws"], summary["losses"]) == (2, 0, 2)
    assert summary["elo"] == 0


def test_run_match_openings_cycle(monkeypatch):
    n = 2 * len(versus.OPENINGS) + 2
    games = fake_results(monkeypatch, ["1/2-1/2"] * n)
    versus.run_match("cand", "base", max_games=n)
    assert games[-1][2] == versus.OPENINGS[0]


def test_run_match_passes_and_stops(monkeypatch):
    # The candidate wins every game: the SPRT must stop with "passed".
    fake_results(monkeypatch, ["1-0", "0-1"] * 500)
    seen = []
    summary = versus.run_match("cand", "base", max_games=1000,
                               on_game=lambda *args: seen.append(args))
    assert summary["verdict"] == "passed"
    assert summary["games"] < 1000
    assert summary["llr"] >= summary["bounds"][1]
    assert summary["wins"] == summary["games"] and summary["elo"] is None
    assert [s[0] for s in seen] == list(range(1, summary["games"] + 1))
    assert seen[-1][1:] == (summary["wins"], 0, 0, summary["llr"])


def test_run_match_fails_and_stops(monkeypatch):
    fake_results(monkeypatch, ["0-1", "1-0"] * 500)
    summary = versus.run_match("cand", "base", max_games=1000)
    assert summary["verdict"] == "failed"
    assert summary["llr"] <= summary["bounds"][0]


def test_run_match_larger_elo1_decides_faster(monkeypatch):
    results = ["1-0", "1/2-1/2", "1/2-1/2", "1/2-1/2"] * 500
    fake_results(monkeypatch, results)
    slow = versus.run_match("cand", "base", max_games=2000)["games"]
    fake_results(monkeypatch, results)
    fast = versus.run_match("cand", "base", max_games=2000, elo1=50)["games"]
    assert fast < slow


# --- Command line ---------------------------------------------------------------


def test_main_parses_flags_and_prints_verdict(monkeypatch, capsys):
    with open(ENGINE_PATH, encoding="utf-8") as f:
        source = f.read()
    baseline = versus.load_engine(source, "baseline_copy")
    revisions = []

    def fake_baseline(revision):
        revisions.append(revision)
        return baseline

    calls = []

    def fake_match(candidate, base, depth, move_time, max_games, on_game, elo1):
        calls.append((candidate, base, depth, move_time, max_games, elo1))
        on_game(1, 1, 0, 0, 0.5)
        return {"games": 1, "wins": 1, "draws": 0, "losses": 0, "llr": 0.5,
                "bounds": versus.sprt_bounds(), "verdict": "undecided", "elo": None}

    monkeypatch.setattr(versus, "baseline_engine", fake_baseline)
    monkeypatch.setattr(versus, "run_match", fake_match)
    versus.main(["--baseline", "HEAD~1", "--depth", "4", "--time", "0.5",
                 "--games", "20", "--elo1", "25"])
    out = capsys.readouterr().out
    assert revisions == ["HEAD~1"]
    assert calls == [(engine, baseline, 4, 0.5, 20, 25)]
    assert "depth 4, 0.5 s per move" in out
    assert "SPRT [0, 25]" in out
    assert "game 1: +1 =0 -0  LLR +0.50" in out
    assert "Elo n/a" in out and "Undecided" in out


@pytest.mark.parametrize("verdict,text,elo,elo_text", [
    ("passed", "PASSED", 12.4, "Elo +12"),
    ("failed", "FAILED", -0.3, "Elo +0"),
])
def test_main_verdicts(monkeypatch, capsys, verdict, text, elo, elo_text):
    with open(ENGINE_PATH, encoding="utf-8") as f:
        baseline = versus.load_engine(f.read(), "baseline_copy")
    monkeypatch.setattr(versus, "baseline_engine", lambda revision: baseline)
    monkeypatch.setattr(versus, "run_match", lambda *a, **k: {
        "games": 10, "wins": 5, "draws": 3, "losses": 2, "llr": 3.0,
        "bounds": versus.sprt_bounds(), "verdict": verdict, "elo": elo})
    versus.main([])
    out = capsys.readouterr().out
    assert "no time limit" in out and "depth 3" in out
    assert elo_text in out and text in out


def test_main_loads_trained_weights_into_both_engines(monkeypatch):
    tuned = {**engine.DEFAULT_WEIGHTS, "rook": 520}
    versus.Storage().add_version(tuned)
    with open(ENGINE_PATH, encoding="utf-8") as f:
        baseline = versus.load_engine(f.read(), "baseline_copy")
    monkeypatch.setattr(versus, "baseline_engine", lambda revision: baseline)
    monkeypatch.setattr(versus, "run_match", lambda *a, **k: {
        "games": 0, "wins": 0, "draws": 0, "losses": 0, "llr": 0.0,
        "bounds": versus.sprt_bounds(), "verdict": "undecided", "elo": None})
    versus.main([])
    assert engine.get_weights() == tuned
    assert baseline.get_weights() == tuned
