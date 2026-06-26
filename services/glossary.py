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
# Column shapes used by the Streamlit editor. Matches the existing UI layout
# so callers don't have to remap fields.
# ─────────────────────────────────────────────────────────────────────────────
TERM_UI_COLUMNS = ["적용", "id", "KO", "EN", "Product", "DNT", "Case-sensitive", "Note", "Status", "File"]
PATTERN_UI_COLUMNS = ["적용", "id", "KO", "EN", "Note", "Status", "File"]


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────
def load_terms(product: Optional[str] = None, search: str = "") -> pd.DataFrame:
    """
    Return all terms as a UI-ready DataFrame, optionally filtered by product
    (selected + 'ALL') and/or a free-text substring in KO/EN/Note.
    """
    init_db()
    sql = "SELECT id, ko, en, product, dnt, case_sensitive, note, status, source_file FROM terms"
    params: list = []
    where: list[str] = []

    if product:
        where.append("(LOWER(product) = 'all' OR LOWER(product) = LOWER(?))")
        params.append(product)
    if search:
        where.append("(ko LIKE ? OR en LIKE ? OR note LIKE ?)")
        like = f"%{search}%"
        params += [like, like, like]
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY product, ko"

    with db_session() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]

    df = pd.DataFrame(rows, columns=["id", "ko", "en", "product", "dnt", "case_sensitive", "note", "status", "source_file"])
    df = df.rename(columns={
        "ko": "KO", "en": "EN", "product": "Product",
        "dnt": "DNT", "case_sensitive": "Case-sensitive",
        "note": "Note", "status": "Status", "source_file": "File",
    })
    df["DNT"] = df["DNT"].astype(bool)
    df["Case-sensitive"] = df["Case-sensitive"].astype(bool)
    df.insert(0, "적용", True)
    return df[TERM_UI_COLUMNS]


def load_patterns(search: str = "") -> pd.DataFrame:
    init_db()
    sql = "SELECT id, ko, en, note, status, source_file FROM patterns"
    params: list = []
    if search:
        sql += " WHERE (ko LIKE ? OR en LIKE ? OR note LIKE ?)"
        like = f"%{search}%"
        params += [like, like, like]
    sql += " ORDER BY ko"

    with db_session() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]

    df = pd.DataFrame(rows, columns=["id", "ko", "en", "note", "status", "source_file"])
    df = df.rename(columns={
        "ko": "KO", "en": "EN", "note": "Note",
        "status": "Status", "source_file": "File",
    })
    df.insert(0, "적용", True)
    return df[PATTERN_UI_COLUMNS]


# ─────────────────────────────────────────────────────────────────────────────
# Write — DataFrame diff against DB
# ─────────────────────────────────────────────────────────────────────────────
def save_terms_from_dataframe(df: pd.DataFrame, view_ids: Optional[set[int]] = None) -> dict:
    """
    Persist edits from the Streamlit data_editor back to DB.

    Rules:
    - Row with non-null `id` → UPDATE if any field differs.
    - Row with null `id` → INSERT (newly added by user in editor).
    - A row is DELETED only when (a) its id was present in `view_ids` (the
      ids the editor showed at load time) AND (b) it is missing from the
      edited df. When `view_ids` is None, the deletion pool is the full
      table — only safe when the editor displays the entire table.

    `view_ids` exists to prevent catastrophic data loss when the editor
    shows a product- or search-filtered subset: rows not in the current
    view must never be deleted by an edit of the view.

    Returns counts: {'inserted': N, 'updated': N, 'deleted': N}.
    """
    init_db()
    counts = {"inserted": 0, "updated": 0, "deleted": 0}
    now = now_iso()

    with db_session() as conn:
        existing_ids = {r["id"] for r in conn.execute("SELECT id FROM terms")}

        edited_ids: set[int] = set()
        for _, row in df.iterrows():
            rid = row.get("id")
            ko = str(row.get("KO", "") or "").strip()
            en = str(row.get("EN", "") or "").strip()
            if not ko or not en:
                continue  # 빈 row는 무시 (data_editor에 dynamic 추가 됐다가 비워둔 경우)

            product = str(row.get("Product", "") or "ALL").strip() or "ALL"
            dnt = 1 if bool(row.get("DNT", False)) else 0
            case_sensitive = 1 if bool(row.get("Case-sensitive", False)) else 0
            note = str(row.get("Note", "") or "")
            status = str(row.get("Status", "") or "approved") or "approved"
            source_file = str(row.get("File", "") or "")

            if pd.notna(rid) and int(rid) in existing_ids:
                rid_int = int(rid)
                edited_ids.add(rid_int)
                conn.execute(
                    """UPDATE terms SET ko=?, en=?, product=?, dnt=?, case_sensitive=?,
                       note=?, status=?, source_file=?, updated_at=? WHERE id=?""",
                    (ko, en, product, dnt, case_sensitive, note, status, source_file, now, rid_int),
                )
                counts["updated"] += 1
            else:
                conn.execute(
                    """INSERT INTO terms (ko, en, product, dnt, case_sensitive, note, status,
                       source_file, imported_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (ko, en, product, dnt, case_sensitive, note, status, source_file, now, now),
                )
                counts["inserted"] += 1

        # 삭제 대상은 "원래 화면에 보였던 id 중 편집기에서 빠진 것"으로 한정.
        # 화면에 안 보였던 row(다른 product / 검색 미해당)는 절대 건드리지 않음.
        deletion_pool = view_ids if view_ids is not None else existing_ids
        deleted = deletion_pool - edited_ids
        for did in deleted:
            conn.execute("DELETE FROM terms WHERE id=?", (did,))
        counts["deleted"] = len(deleted)

    return counts


def save_patterns_from_dataframe(df: pd.DataFrame, view_ids: Optional[set[int]] = None) -> dict:
    """Same safety contract as [save_terms_from_dataframe]: deletes are
    limited to ids that were present in the loaded view."""
    init_db()
    counts = {"inserted": 0, "updated": 0, "deleted": 0}
    now = now_iso()

    with db_session() as conn:
        existing_ids = {r["id"] for r in conn.execute("SELECT id FROM patterns")}

        edited_ids: set[int] = set()
        for _, row in df.iterrows():
            rid = row.get("id")
            ko = str(row.get("KO", "") or "").strip()
            en = str(row.get("EN", "") or "").strip()
            if not ko or not en:
                continue

            note = str(row.get("Note", "") or "")
            status = str(row.get("Status", "") or "approved") or "approved"
            source_file = str(row.get("File", "") or "")

            if pd.notna(rid) and int(rid) in existing_ids:
                rid_int = int(rid)
                edited_ids.add(rid_int)
                conn.execute(
                    """UPDATE patterns SET ko=?, en=?, note=?, status=?, source_file=?,
                       updated_at=? WHERE id=?""",
                    (ko, en, note, status, source_file, now, rid_int),
                )
                counts["updated"] += 1
            else:
                conn.execute(
                    """INSERT INTO patterns (ko, en, note, status, source_file,
                       imported_at, updated_at) VALUES (?,?,?,?,?,?,?)""",
                    (ko, en, note, status, source_file, now, now),
                )
                counts["inserted"] += 1

        deletion_pool = view_ids if view_ids is not None else existing_ids
        deleted = deletion_pool - edited_ids
        for did in deleted:
            conn.execute("DELETE FROM patterns WHERE id=?", (did,))
        counts["deleted"] = len(deleted)

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
        conn.execute("DELETE FROM terms")
        for _, row in df.iterrows():
            ko = str(row.get("KO") or "").strip()
            en = str(row.get("EN") or "").strip()
            if not ko or not en:
                continue
            product = str(row.get("Product") or "ALL").strip() or "ALL"
            dnt_raw = str(row.get("DNT", "")).strip().lower()
            dnt = 1 if dnt_raw in ("true", "y", "yes", "1") else 0
            cs_raw = str(row.get("Case-sensitive", "")).strip().lower()
            case_sensitive = 1 if cs_raw in ("true", "y", "yes", "1") else 0
            note = str(row.get("Note") or "")
            status = str(row.get("Status") or "approved") or "approved"

            conn.execute(
                """INSERT INTO terms (ko, en, product, dnt, case_sensitive, note, status,
                   source_file, imported_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ko, en, product, dnt, case_sensitive, note, status, source_file, now, now),
            )
            inserted += 1

    return inserted


def replace_patterns_from_excel(sheet_df: pd.DataFrame, source_file: str) -> int:
    """Wipe + refill patterns table from a single sheet."""
    init_db()
    now = now_iso()

    df = sheet_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["KO", "EN", "Note", "Status"]:
        if col not in df.columns:
            df[col] = ""

    inserted = 0
    with db_session() as conn:
        conn.execute("DELETE FROM patterns")
        for _, row in df.iterrows():
            ko = str(row.get("KO") or "").strip()
            en = str(row.get("EN") or "").strip()
            if not ko or not en:
                continue
            note = str(row.get("Note") or "")
            status = str(row.get("Status") or "approved") or "approved"
            conn.execute(
                """INSERT INTO patterns (ko, en, note, status, source_file,
                   imported_at, updated_at) VALUES (?,?,?,?,?,?,?)""",
                (ko, en, note, status, source_file, now, now),
            )
            inserted += 1

    return inserted
