import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A fully migrated file-backed SQLite DatabaseManager."""
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from data.database import DatabaseManager

    manager = DatabaseManager()
    manager.create_tables()
    return manager
