"""
Translation log data-access layer.

Every completed translation creates one row recording: when, who, which product,
which source file, what UI text overrides were used, token metrics, and an
optional free-text note. The UI text overrides are stored as JSON so they can
be loaded back into a future Step 2 to re-use the same mapping for repeat
agent-test runs.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import pandas as pd

from db.schema import db_session, init_db, now_iso
from services.sync import sync_db


def create_log(
    user: str,
    product: str,
    translation_mode: str,
    source_file: str,
    output_file: str,
    ui_text_overrides: Optional[Dict[str, str]],
    metrics: dict,
    note: str = "",
    sync: bool = True,
) -> int:
    """
    Insert a new translation log. Returns the new row id.

    sync=False로 부르면 GitHub 푸시를 건너뛴다. 호출자가 여러 건을 연달아
    기록한 뒤 **한 번만** sync_db()를 부르려는 경우에 쓴다 — 배치 번역에서
    파일마다 푸시하면 커밋이 파일 수만큼 쌓이고, Streamlit Cloud는 커밋마다
    앱을 재배포하므로 사용자가 결과 화면을 보기도 전에 세션이 날아간다.
    """
    init_db()
    overrides_json = json.dumps(ui_text_overrides or {}, ensure_ascii=False)
    # 대시보드 지표. 결과에 해당 키가 없으면 NULL — "0"으로 적으면 측정하지
    # 않은 것이 "적용 0건 / 오류 0건"으로 집계돼 숫자가 좋아 보이게 된다.
    elapsed = metrics.get("elapsed_seconds")
    applied = metrics.get("applied")
    verification = metrics.get("verification")
    applied_terms = len(applied) if applied is not None else None
    applied_count = (sum(int(a.get("적용", 0)) for a in applied)
                     if applied is not None else None)
    verify_errors = (sum(1 for v in verification if v.get("level") == "오류")
                     if verification is not None else None)
    verify_warnings = (len(verification) - verify_errors
                       if verification is not None else None)
    with db_session() as conn:
        cur = conn.execute(
            """
            INSERT INTO translation_logs (
                created_at, user, product, translation_mode, source_file, output_file,
                ui_text_overrides, input_tokens, cached_tokens, output_tokens,
                total_tokens, paragraphs_translated, note,
                elapsed_seconds, applied_terms, applied_count,
                verify_errors, verify_warnings
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now_iso(), user, product, translation_mode, source_file, output_file,
                overrides_json,
                int(metrics.get("input_tokens", 0)),
                int(metrics.get("cached_tokens", 0)),
                int(metrics.get("output_tokens", 0)),
                int(metrics.get("total_tokens", 0)),
                int(metrics.get("paragraphs_translated", 0)),
                note,
                int(elapsed) if elapsed is not None else None,
                applied_terms, applied_count, verify_errors, verify_warnings,
            ),
        )
        new_id = int(cur.lastrowid)

    if sync:
        sync_db(f"Log #{new_id} ({source_file}) by {user}")
    return new_id


def list_logs(
    user: Optional[str] = None,
    search: str = "",
    limit: int = 100,
) -> pd.DataFrame:
    """
    Return logs as a UI-friendly DataFrame.

    Filters: by user (typically the current user — log isolation), free-text
    search on source_file / product / note. Latest first.
    """
    init_db()
    sql = """SELECT id, created_at, user, product, translation_mode, source_file,
                    total_tokens, paragraphs_translated, note,
                    ui_text_overrides
             FROM translation_logs"""
    params: list = []
    where: list[str] = []
    if user:
        where.append("user = ?")
        params.append(user)
    if search:
        where.append("(source_file LIKE ? OR product LIKE ? OR note LIKE ?)")
        like = f"%{search}%"
        params += [like, like, like]
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with db_session() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]

    df = pd.DataFrame(
        rows,
        columns=[
            "id", "created_at", "user", "product", "translation_mode",
            "source_file", "total_tokens", "paragraphs_translated", "note",
            "ui_text_overrides",
        ],
    )
    # UI 매핑 개수만 별도 컬럼으로 — 표에서 한눈에 보기 위함
    df["mapping_count"] = df["ui_text_overrides"].apply(
        lambda s: len(json.loads(s or "{}")) if s else 0
    )
    return df


def get_log(log_id: int) -> Optional[dict]:
    """Fetch one log with ui_text_overrides parsed back to dict."""
    init_db()
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM translation_logs WHERE id = ?", (log_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["ui_text_overrides"] = json.loads(d.get("ui_text_overrides") or "{}")
    except json.JSONDecodeError:
        d["ui_text_overrides"] = {}
    return d


def update_note(log_id: int, note: str) -> None:
    init_db()
    with db_session() as conn:
        conn.execute(
            "UPDATE translation_logs SET note = ? WHERE id = ?", (note, log_id)
        )
    sync_db(f"Log #{log_id} note updated")


def update_review_edits(log_id: int, n_edits: int) -> None:
    """번역 검토에서 고친 단락 수. 누적값으로 덮어쓴다(같은 단락을 두 번 고쳐도 1)."""
    init_db()
    with db_session() as conn:
        conn.execute(
            "UPDATE translation_logs SET review_edits = ? WHERE id = ?", (n_edits, log_id)
        )
    sync_db(f"Log #{log_id} review edits: {n_edits}")


def delete_log(log_id: int) -> None:
    init_db()
    with db_session() as conn:
        conn.execute("DELETE FROM translation_logs WHERE id = ?", (log_id,))
    sync_db(f"Log #{log_id} deleted")


def find_logs_by_source_file(user: str, source_file: str, limit: int = 5) -> pd.DataFrame:
    """
    Convenience helper: latest logs for the same source filename — used in
    Step 2 to suggest "이 파일 이전에 N번 번역됨, 매핑 불러올까요?".
    """
    init_db()
    with db_session() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                """SELECT id, created_at, source_file, note, ui_text_overrides
                   FROM translation_logs
                   WHERE user = ? AND source_file = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user, source_file, limit),
            )
        ]
    return pd.DataFrame(rows)


def find_related_logs(
    source_file: str,
    product: str,
    limit: int = 15,
) -> pd.DataFrame:
    """
    Return past translation logs whose UI mapping could plausibly be reused
    for the current job.

    Union of:
    - Same source filename (strongest signal — often a re-translation run)
    - Same product (same UI vocabulary, worth trying)

    Team-wide (any user), latest first, file matches ranked above product
    matches. Only logs that actually carried a UI mapping are returned —
    an empty mapping is useless for reuse and would just clutter the picker.
    """
    init_db()
    sql = """
        SELECT id, created_at, user, product, source_file, note,
               ui_text_overrides,
               CASE
                 WHEN source_file = :sf  THEN 'file'
                 WHEN product = :pr AND product != '' THEN 'product'
                 ELSE 'other'
               END AS match_type
        FROM translation_logs
        WHERE (source_file = :sf OR (product = :pr AND product != ''))
          AND ui_text_overrides NOT IN ('', '{}')
        ORDER BY
          CASE WHEN source_file = :sf THEN 0 ELSE 1 END,
          created_at DESC
        LIMIT :lim
    """
    params = {"sf": source_file, "pr": product, "lim": limit}
    with db_session() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
    return pd.DataFrame(rows)
