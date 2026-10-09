"""Fixtures shared by every test file.

The engine weights and the CLI display flags are module globals, and
cli.main() loads the trained weights from data/. Without this, a test could
change them for every test after it, or read the real training data on the
machine running the tests.
"""

import pytest

from chesscore import cli, engine
from chesscore.storage import Storage


@pytest.fixture(autouse=True)
def isolate_globals(monkeypatch, tmp_path):
    weights = engine.get_weights()
    monkeypatch.setattr(cli, "USE_SYMBOLS", cli.USE_SYMBOLS)
    monkeypatch.setattr(cli, "USE_COLORS", cli.USE_COLORS)
    monkeypatch.setattr(cli, "Storage", lambda: Storage(str(tmp_path / "data")))
    yield
    engine.set_weights(weights)
