"""
GitHub auto-commit: persist SQLite + users.json across Streamlit Cloud reboots.

Streamlit Cloud's container disk is ephemeral — every reboot/sleep rolls the
working copy back to whatever's checked into git. To keep glossary/log/user
data alive between reboots, we push the changed files to the same GitHub repo
via the GitHub Contents API. On next cold-start Streamlit Cloud pulls the
latest git head and the persisted data comes back.

Caveats:
- Requires GITHUB_TOKEN (PAT with `repo` scope) and GITHUB_REPO ("owner/name")
  set in Streamlit Cloud Secrets (or env vars locally).
- No-op when env is missing — local runs and dev installs work as before.
- One file = one commit per call. Race-conditions exist if 5 users save at the
  exact same second; last writer wins.
- The SQLite file is binary, so commit history grows fast. Consider pruning
  history periodically or moving to Supabase if it becomes a problem.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import requests


GITHUB_API = "https://api.github.com"


def _config() -> tuple[str, str, str]:
    """Return (token, repo, branch). Empty token = sync disabled."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "").strip()
    branch = os.getenv("GITHUB_BRANCH", "main").strip() or "main"
    return token, repo, branch


def is_enabled() -> bool:
    token, repo, _ = _config()
    return bool(token and repo)


def push_file_to_github(
    local_path: Path,
    repo_path: str,
    commit_message: str,
    timeout: int = 15,
) -> Optional[str]:
    """
    Upload `local_path` to `repo_path` in the configured GitHub repo.

    Returns the new commit SHA on success, None on failure / disabled.
    Failures are swallowed and printed — calling code must NOT crash on
    a sync error (the user-facing save already succeeded locally).
    """
    token, repo, branch = _config()
    if not (token and repo):
        return None  # sync disabled — no-op
    if not local_path.exists():
        return None

    api_url = f"{GITHUB_API}/repos/{repo}/contents/{repo_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        # 현재 파일 sha 조회 (update에 필요). 파일이 새로 생기는 경우엔 sha 없음.
        sha: Optional[str] = None
        r = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=timeout)
        if r.status_code == 200:
            sha = r.json().get("sha")
        elif r.status_code not in (404,):
            # 다른 에러는 silently skip — 사용자 흐름 막지 않음
            print(f"[sync] GET {repo_path} failed: {r.status_code} {r.text[:200]}")
            return None

        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")

        body = {
            "message": commit_message,
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            body["sha"] = sha

        r = requests.put(api_url, headers=headers, json=body, timeout=timeout)
        if r.status_code in (200, 201):
            return r.json().get("commit", {}).get("sha")
        print(f"[sync] PUT {repo_path} failed: {r.status_code} {r.text[:200]}")
        return None
    except Exception as e:
        print(f"[sync] exception: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수 — 호출 위치에서 한 줄로 쓰기 쉽게.
# ─────────────────────────────────────────────────────────────────────────────
def sync_db(commit_message: str = "Auto-update glossary DB") -> None:
    """Push the SQLite DB to GitHub. Silently no-op if sync is disabled."""
    from db.schema import DB_PATH
    push_file_to_github(DB_PATH, "data/glossary.db", commit_message)


def sync_users(commit_message: str = "Auto-update users list") -> None:
    """Push the users.json registry to GitHub."""
    from services.users import USERS_PATH
    push_file_to_github(USERS_PATH, "data/users.json", commit_message)
