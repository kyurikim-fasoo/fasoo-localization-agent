"""
SQLite schema + connection helper for the glossary database.

One SQLite file (`data/glossary.db`) holds two tables:
  - terms     — glossary entries (KO/EN + product/flags/note)
  - patterns  — translation patterns (KO/EN/note)

Both tables carry id, status, source_file, imported_at, updated_at so we can
later (Phase 2) distinguish Wrapsody-imported rows from locally added/edited
rows, track when each row was last touched, and build approval workflows.

`init_db()` is idempotent — safe to call on every app startup.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "glossary.db"


def get_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection with row-as-dict access and WAL mode for safe
    concurrent reads while one writer is active.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def db_session():
    """Context-managed connection that commits on success, rolls back on error."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS terms (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ko              TEXT    NOT NULL,
    en              TEXT    NOT NULL,
    product         TEXT    DEFAULT 'ALL',
    dnt             INTEGER NOT NULL DEFAULT 0,
    case_sensitive  INTEGER NOT NULL DEFAULT 0,
    note            TEXT    DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'approved',
    source_file     TEXT    DEFAULT '',
    imported_at     TEXT,
    updated_at      TEXT
);

CREATE INDEX IF NOT EXISTS ix_terms_product ON terms(product);
CREATE INDEX IF NOT EXISTS ix_terms_ko      ON terms(ko);

CREATE TABLE IF NOT EXISTS patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ko              TEXT    NOT NULL,
    en              TEXT    NOT NULL,
    note            TEXT    DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'approved',
    source_file     TEXT    DEFAULT '',
    imported_at     TEXT,
    updated_at      TEXT
);

CREATE INDEX IF NOT EXISTS ix_patterns_ko ON patterns(ko);
"""


def init_db() -> None:
    """Create tables if they don't exist. Safe to call repeatedly."""
    with db_session() as conn:
        conn.executescript(_SCHEMA_SQL)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
