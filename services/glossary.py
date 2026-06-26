"""
Glossary + pattern data-access layer.

All Streamlit code should call into here rather than touching the DB directly,
so when the storage layer evolves (Postgres, multi-tenant, API, etc.) only this
module changes.

The functions return / accept pandas DataFrames so the existing Streamlit
data_editor flow can stay almost unchanged.
"""
from __future__ import annotations

from typing import Optional, Set  # noqa: F401  (Set kept for clarity in 3.9 typing)

import pandas as pd

from db.schema import db_session, init_db, now_iso


# ─────────────────────────────────────────────────────────────────────────────
# Cell-value normalisers — Excel cells come back as floats (NaN) for blanks,
# native bools for checked boxes, and stringified anything for free text.
# Centralising the cleanup avoids "nan" strings polluting the DB.
# ─────────────────────────────────────────────────────────────────────────────
def _clean_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s


def _clean_bool(v) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        if pd.isna(v):
            return 0
        return 1 if v else 0
    s = str(v).strip().lower()
    return 1 if s in ("true", "y", "yes", "1") else 0


# ─────────────────────────────────────────────────────────────────────────────
# Column shapes used by the Streamlit editor. Matches the existing UI layout
# so callers don't have to remap fields.
# ─────────────────────────────────────────────────────────────────────────────
TERM_UI_COLUMNS = ["적용", "id", "Scope", "KO", "EN", "Product", "DNT", "Case-sensitive", "Note", "Status", "File"]
PATTERN_UI_COLUMNS = ["적용", "id", "Scope", "KO", "EN", "Note", "Status", "File"]


def _scope_label(owner: str, current_user: str) -> str:
    """DB의 owner 값을 UI의 Scope 라벨로 변환.

    빈 문자열 = 모두가 공유하는 Team row → 'Team' 으로 표시.
    그 외 = 해당 user의 personal row → user 이름 그대로 표시.
    """
    return "Team" if not owner else owner


def _scope_to_owner(scope: str, current_user: str) -> str:
    """UI 라벨 → DB owner. 'Team' → '', 그 외 → 그대로."""
    s = (scope or "").strip()
    if not s or s.lower() == "team":
        return ""
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────
def load_terms(
    product: Optional[str] = None,
    search: str = "",
    current_user: str = "",
    scope_filter: str = "all",  # 'all' | 'team' | 'mine'
) -> pd.DataFrame:
    """
    Return terms visible to the current user as a UI-ready DataFrame.

    Visibility rules (Option A — multi-user):
    - Team rows (owner='') are always visible.
    - Personal rows are visible only to their owner.
    - Other users' personal rows are NEVER returned.

    Args:
        product: keep rows matching product or 'ALL' (case-insensitive)
        search: substring match on KO/EN/Note
        current_user: the logged-in user (empty = anonymous = team only)
        scope_filter: 'all' | 'team' | 'mine' — additional UI filter
    """
    init_db()
    sql = "SELECT id, ko, en, product, dnt, case_sensitive, note, status, source_file, owner FROM terms"
    params: list = []
    where: list[str] = []

    # ── 가시성: Team + 본인 personal만 ─────────────────────────────────
    if current_user:
        where.append("(owner='' OR owner=?)")
        params.append(current_user)
    else:
        where.append("owner=''")

    # ── 사용자 선택 Scope 필터 ────────────────────────────────────────
    if scope_filter == "team":
        where.append("owner=''")
    elif scope_filter == "mine" and current_user:
        where.append("owner=?")
        params.append(current_user)

    if product:
        where.append("(LOWER(product) = 'all' OR LOWER(product) = LOWER(?))")
        params.append(product)
    if search:
        where.append("(ko LIKE ? OR en LIKE ? OR note LIKE ?)")
        like = f"%{search}%"
        params += [like, like, like]

    sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY owner, product, ko"  # Team(='') 먼저, 그 다음 개인

    with db_session() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]

    df = pd.DataFrame(rows, columns=["id", "ko", "en", "product", "dnt", "case_sensitive", "note", "status", "source_file", "owner"])
    df["Scope"] = df["owner"].apply(lambda o: _scope_label(o, current_user))
    df = df.rename(columns={
        "ko": "KO", "en": "EN", "product": "Product",
        "dnt": "DNT", "case_sensitive": "Case-sensitive",
        "note": "Note", "status": "Status", "source_file": "File",
    })
    df["DNT"] = df["DNT"].astype(bool)
    df["Case-sensitive"] = df["Case-sensitive"].astype(bool)
    df.insert(0, "적용", True)
    return df[TERM_UI_COLUMNS]


def load_patterns(
    search: str = "",
    current_user: str = "",
    scope_filter: str = "all",
) -> pd.DataFrame:
    init_db()
    sql = "SELECT id, ko, en, note, status, source_file, owner FROM patterns"
    params: list = []
    where: list[str] = []

    if current_user:
        where.append("(owner='' OR owner=?)")
        params.append(current_user)
    else:
        where.append("owner=''")

    if scope_filter == "team":
        where.append("owner=''")
    elif scope_filter == "mine" and current_user:
        where.append("owner=?")
        params.append(current_user)

    if search:
        where.append("(ko LIKE ? OR en LIKE ? OR note LIKE ?)")
        like = f"%{search}%"
        params += [like, like, like]

    sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY owner, ko"

    with db_session() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]

    df = pd.DataFrame(rows, columns=["id", "ko", "en", "note", "status", "source_file", "owner"])
    df["Scope"] = df["owner"].apply(lambda o: _scope_label(o, current_user))
    df = df.rename(columns={
        "ko": "KO", "en": "EN", "note": "Note",
        "status": "Status", "source_file": "File",
    })
    df.insert(0, "적용", True)
    return df[PATTERN_UI_COLUMNS]


# ─────────────────────────────────────────────────────────────────────────────
# Write — DataFrame diff against DB
# ─────────────────────────────────────────────────────────────────────────────
def save_terms_from_dataframe(
    df: pd.DataFrame,
    view_ids: Optional[set[int]] = None,
    current_user: str = "",
) -> dict:
    """
    Persist edits from the Streamlit data_editor back to DB.

    Multi-user rules (Option A):
    - Team rows (owner='') can be edited/deleted by anyone.
    - Personal rows can only be edited/deleted by their owner.
    - New rows default to owner=current_user (Personal). Setting Scope='Team'
      in the editor promotes them to Team.
    - Visibility was already enforced by load_terms — other users' personal
      rows never reach this function, so they can't be touched here either.

    Safety contracts preserved from the single-user version:
    - View-scoped deletion: only ids in `view_ids` are eligible for delete.
    - Empty KO/EN rows are skipped (in-progress editor inserts).

    Returns counts: {'inserted': N, 'updated': N, 'deleted': N, 'denied': N}.
    `denied` counts rows the user tried to modify but doesn't own.
    """
    init_db()
    counts = {"inserted": 0, "updated": 0, "deleted": 0, "denied": 0}
    now = now_iso()

    with db_session() as conn:
        # 권한 체크를 위해 id → owner 매핑을 미리 조회
        owner_by_id = {r["id"]: r["owner"] for r in conn.execute("SELECT id, owner FROM terms")}
        existing_ids = set(owner_by_id)

        edited_ids: set[int] = set()
        for _, row in df.iterrows():
            rid = row.get("id")
            ko = _clean_text(row.get("KO"))
            en = _clean_text(row.get("EN"))
            if not ko or not en:
                continue  # 빈 row는 무시 (data_editor에 dynamic 추가 됐다가 비워둔 경우)

            product = _clean_text(row.get("Product")) or "ALL"
            dnt = _clean_bool(row.get("DNT"))
            case_sensitive = _clean_bool(row.get("Case-sensitive"))
            note = _clean_text(row.get("Note"))
            status = _clean_text(row.get("Status")) or "approved"
            source_file = _clean_text(row.get("File"))
            scope = _clean_text(row.get("Scope"))
            target_owner = _scope_to_owner(scope, current_user)

            if pd.notna(rid) and int(rid) in existing_ids:
                rid_int = int(rid)
                current_owner = owner_by_id[rid_int]
                # 권한: Team(='')는 모두 OK, Personal은 본인만
                if current_owner and current_owner != current_user:
                    counts["denied"] += 1
                    continue
                edited_ids.add(rid_int)
                conn.execute(
                    """UPDATE terms SET ko=?, en=?, product=?, dnt=?, case_sensitive=?,
                       note=?, status=?, source_file=?, owner=?, updated_at=? WHERE id=?""",
                    (ko, en, product, dnt, case_sensitive, note, status, source_file,
                     target_owner, now, rid_int),
                )
                counts["updated"] += 1
            else:
                # 신규 row: Scope 지정 없으면 current_user (Personal) 기본값
                if not scope:
                    target_owner = current_user
                conn.execute(
                    """INSERT INTO terms (ko, en, product, dnt, case_sensitive, note, status,
                       source_file, owner, imported_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (ko, en, product, dnt, case_sensitive, note, status, source_file,
                     target_owner, now, now),
                )
                counts["inserted"] += 1

        # 삭제: view_ids 안 + 편집기에서 빠진 id 중 본인이 owner이거나 Team인 row만
        deletion_pool = view_ids if view_ids is not None else existing_ids
        deletion_candidates = deletion_pool - edited_ids
        for did in deletion_candidates:
            owner = owner_by_id.get(did, "")
            if owner and owner != current_user:
                counts["denied"] += 1
                continue
            conn.execute("DELETE FROM terms WHERE id=?", (did,))
            counts["deleted"] += 1

    return counts


def save_patterns_from_dataframe(
    df: pd.DataFrame,
    view_ids: Optional[set[int]] = None,
    current_user: str = "",
) -> dict:
    """Same multi-user contract as [save_terms_from_dataframe]."""
    init_db()
    counts = {"inserted": 0, "updated": 0, "deleted": 0, "denied": 0}
    now = now_iso()

    with db_session() as conn:
        owner_by_id = {r["id"]: r["owner"] for r in conn.execute("SELECT id, owner FROM patterns")}
        existing_ids = set(owner_by_id)

        edited_ids: set[int] = set()
        for _, row in df.iterrows():
            rid = row.get("id")
            ko = _clean_text(row.get("KO"))
            en = _clean_text(row.get("EN"))
            if not ko or not en:
                continue

            note = _clean_text(row.get("Note"))
            status = _clean_text(row.get("Status")) or "approved"
            source_file = _clean_text(row.get("File"))
            scope = _clean_text(row.get("Scope"))
            target_owner = _scope_to_owner(scope, current_user)

            if pd.notna(rid) and int(rid) in existing_ids:
                rid_int = int(rid)
                current_owner = owner_by_id[rid_int]
                if current_owner and current_owner != current_user:
                    counts["denied"] += 1
                    continue
                edited_ids.add(rid_int)
                conn.execute(
                    """UPDATE patterns SET ko=?, en=?, note=?, status=?, source_file=?,
                       owner=?, updated_at=? WHERE id=?""",
                    (ko, en, note, status, source_file, target_owner, now, rid_int),
                )
                counts["updated"] += 1
            else:
                if not scope:
                    target_owner = current_user
                conn.execute(
                    """INSERT INTO patterns (ko, en, note, status, source_file, owner,
                       imported_at, updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                    (ko, en, note, status, source_file, target_owner, now, now),
                )
                counts["inserted"] += 1

        deletion_pool = view_ids if view_ids is not None else existing_ids
        for did in deletion_pool - edited_ids:
            owner = owner_by_id.get(did, "")
            if owner and owner != current_user:
                counts["denied"] += 1
                continue
            conn.execute("DELETE FROM patterns WHERE id=?", (did,))
            counts["deleted"] += 1

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Bulk replace from Excel (the "교체 모드" import)
# ─────────────────────────────────────────────────────────────────────────────
def replace_terms_from_excel(
    sheet_df: pd.DataFrame,
    source_file: str,
    product_filter: Optional[str] = None,
) -> int:
    """
    Wipe the terms table and refill from an Excel sheet. Used for the
    Wrapsody scenario: every time the source xlsx is re-imported, the DB
    becomes an exact mirror of that snapshot.

    Returns the number of rows inserted.
    """
    init_db()
    now = now_iso()

    df = sheet_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 필수 컬럼 보강
    for col in ["KO", "EN", "Product", "DNT", "Case-sensitive", "Note", "Status"]:
        if col not in df.columns:
            df[col] = "" if col not in ("DNT", "Case-sensitive") else False

    if product_filter:
        p = product_filter.strip().lower()

        def keep(val):
            s = str(val).strip().lower()
            if not s or s == "nan":
                return False
            return s == "all" or s == p

        df = df[df["Product"].apply(keep)].copy()

    inserted = 0
    with db_session() as conn:
        # Team rows만 교체 — Personal rows(각 사용자의 커스텀)는 절대 안 건드림.
        conn.execute("DELETE FROM terms WHERE owner=''")
        for _, row in df.iterrows():
            ko = _clean_text(row.get("KO"))
            en = _clean_text(row.get("EN"))
            if not ko or not en:
                continue
            product = _clean_text(row.get("Product")) or "ALL"
            dnt = _clean_bool(row.get("DNT"))
            case_sensitive = _clean_bool(row.get("Case-sensitive"))
            note = _clean_text(row.get("Note"))
            status = _clean_text(row.get("Status")) or "approved"

            conn.execute(
                """INSERT INTO terms (ko, en, product, dnt, case_sensitive, note, status,
                   source_file, owner, imported_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (ko, en, product, dnt, case_sensitive, note, status, source_file, "", now, now),
            )
            inserted += 1

    return inserted


def replace_patterns_from_excel(sheet_df: pd.DataFrame, source_file: str) -> int:
    """Wipe + refill the **Team** patterns from a single sheet.
    Personal patterns are preserved across re-imports."""
    init_db()
    now = now_iso()

    df = sheet_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["KO", "EN", "Note", "Status"]:
        if col not in df.columns:
            df[col] = ""

    inserted = 0
    with db_session() as conn:
        conn.execute("DELETE FROM patterns WHERE owner=''")
        for _, row in df.iterrows():
            ko = _clean_text(row.get("KO"))
            en = _clean_text(row.get("EN"))
            if not ko or not en:
                continue
            note = _clean_text(row.get("Note"))
            status = _clean_text(row.get("Status")) or "approved"
            conn.execute(
                """INSERT INTO patterns (ko, en, note, status, source_file, owner,
                   imported_at, updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                (ko, en, note, status, source_file, "", now, now),
            )
            inserted += 1

    return inserted
