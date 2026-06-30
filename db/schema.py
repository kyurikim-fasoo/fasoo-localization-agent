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
    owner           TEXT    DEFAULT '',     -- '' = Team(공용), 그 외 = 개인 user 이름
    imported_at     TEXT,
    updated_at      TEXT
);

CREATE INDEX IF NOT EXISTS ix_terms_product ON terms(product);
CREATE INDEX IF NOT EXISTS ix_terms_ko      ON terms(ko);
-- ix_terms_owner is created in _migrate() because the owner column may be
-- added via ALTER TABLE for pre-existing databases.

CREATE TABLE IF NOT EXISTS patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ko              TEXT    NOT NULL,
    en              TEXT    NOT NULL,
    note            TEXT    DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'approved',
    source_file     TEXT    DEFAULT '',
    owner           TEXT    DEFAULT '',     -- '' = Team(공용), 그 외 = 개인
    imported_at     TEXT,
    updated_at      TEXT
);

CREATE INDEX IF NOT EXISTS ix_patterns_ko ON patterns(ko);
-- ix_patterns_owner is created in _migrate() (same reason as ix_terms_owner).

CREATE TABLE IF NOT EXISTS translation_logs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at              TEXT    NOT NULL,
    user                    TEXT    NOT NULL,
    product                 TEXT    DEFAULT '',
    translation_mode        TEXT    DEFAULT '',
    source_file             TEXT    DEFAULT '',
    output_file             TEXT    DEFAULT '',
    ui_text_overrides       TEXT    DEFAULT '{}',   -- JSON {ko: en}
    input_tokens            INTEGER DEFAULT 0,
    cached_tokens           INTEGER DEFAULT 0,
    output_tokens           INTEGER DEFAULT 0,
    total_tokens            INTEGER DEFAULT 0,
    paragraphs_translated   INTEGER DEFAULT 0,
    note                    TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS ix_logs_user        ON translation_logs(user);
CREATE INDEX IF NOT EXISTS ix_logs_created     ON translation_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_logs_source_file ON translation_logs(source_file);
"""


def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == col for r in rows)


def _migrate(conn: sqlite3.Connection) -> None:
    """
    Apply lightweight schema migrations for already-populated DBs.

    Each migration is idempotent (checks before altering) so this can run
    on every startup safely. Add new migrations at the bottom.
    """
    # M1: terms.owner — Team/Personal scope (Option A multi-user)
    if not _column_exists(conn, "terms", "owner"):
        conn.execute("ALTER TABLE terms ADD COLUMN owner TEXT DEFAULT ''")
        conn.execute("UPDATE terms SET owner='' WHERE owner IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_terms_owner ON terms(owner)")

    # M2: patterns.owner
    if not _column_exists(conn, "patterns", "owner"):
        conn.execute("ALTER TABLE patterns ADD COLUMN owner TEXT DEFAULT ''")
        conn.execute("UPDATE patterns SET owner='' WHERE owner IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_patterns_owner ON patterns(owner)")


def init_db() -> None:
    """Create tables if they don't exist, then apply migrations. Safe on every startup."""
    with db_session() as conn:
        conn.executescript(_SCHEMA_SQL)
        _migrate(conn)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
