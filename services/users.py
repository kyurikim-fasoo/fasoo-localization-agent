"""
Lightweight user registry for the multi-user glossary workflow (Option A).

Users are stored in `data/users.json` as a simple list of names. The first
person to use the tool registers themselves; subsequent visits show their
name in a dropdown. No passwords yet — Phase 2 can swap this for
streamlit-authenticator with the same `current user` contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
USERS_PATH = BASE_DIR / "data" / "users.json"


def _read() -> List[str]:
    if not USERS_PATH.exists():
        return []
    try:
        data = json.loads(USERS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return []


def _write(users: List[str]) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(
        json.dumps(sorted(set(users)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_users() -> List[str]:
    """Return registered users in alphabetical order."""
    return sorted(set(_read()))


def add_user(name: str) -> str:
    """Register a user (idempotent). Returns the canonical name stored."""
    name = (name or "").strip()
    if not name:
        raise ValueError("이름이 비어 있습니다.")
    users = set(_read())
    users.add(name)
    _write(sorted(users))
    return name
