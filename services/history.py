"""
Glossary 변경 이력 — 기록, 조회, 되돌리기.

왜 필요한가:
    저장은 현재 값으로 덮어쓰기만 했다. 누가 언제 무엇을 바꿨는지 알 수
    없고, 잘못 저장하면 하루 한 번 뜨는 DB 백업 파일을 손으로 갈아 끼우는
    것 말고는 되돌릴 방법이 없었다. Excel 가져오기는 Team 항목을 통째로
    교체하므로 한 번의 실수가 수백 건이다.

설계:
    - 저장 계층(services/glossary.py)이 **같은 트랜잭션 안에서** 기록한다.
      화면에서 기록하면 다른 경로(추출·Localize)로 들어온 변경이 빠진다.
    - 실제로 값이 바뀐 행만 기록한다. 편집기는 보이는 모든 행을 UPDATE로
      다시 쓰므로, 비교하지 않으면 저장 한 번에 수백 건의 '수정'이 쌓인다.
    - 되돌리기는 묶음(batch) 단위. 그 뒤에 같은 행이 또 바뀌었으면 건너뛴다 —
      나중 변경을 조용히 덮어쓰는 것이 되돌리기 실패보다 나쁘다.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Dict, List, Optional

import pandas as pd

from db.schema import db_session, init_db, now_iso

# 비교·복원 대상 컬럼. id·imported_at·updated_at은 값이 아니라 메타데이터라 뺀다.
FIELDS = {
    "terms": ("ko", "en", "product", "dnt", "case_sensitive", "note",
              "status", "source_file", "owner"),
    "patterns": ("ko", "en", "note", "status", "source_file", "owner"),
}
TABLE_KO = {"terms": "용어", "patterns": "패턴"}
ACTION_KO = {"insert": "추가", "update": "수정", "delete": "삭제"}


def new_batch() -> str:
    return uuid.uuid4().hex[:12]


def snapshot(conn: sqlite3.Connection, table: str, row_id: int) -> Optional[dict]:
    """행 하나의 비교 대상 컬럼. 없으면 None."""
    cols = ", ".join(FIELDS[table])
    r = conn.execute(f"SELECT {cols} FROM {table} WHERE id=?", (row_id,)).fetchone()
    return dict(r) if r else None


def record(conn: sqlite3.Connection, batch_id: str, table: str, row_id: int,
           action: str, user: str, source: str,
           before: Optional[dict], after: Optional[dict]) -> bool:
    """
    변경 한 건을 기록한다. update인데 값이 같으면 기록하지 않고 False.
    호출자의 트랜잭션 안에서 실행해야 한다.
    """
    if action == "update" and before == after:
        return False
    conn.execute(
        """INSERT INTO glossary_history (batch_id, table_name, row_id, action,
           user, source, changed_at, before_json, after_json)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (batch_id, table, row_id, action, user, source, now_iso(),
         json.dumps(before, ensure_ascii=False) if before is not None else None,
         json.dumps(after, ensure_ascii=False) if after is not None else None),
    )
    return True


def _owner_of(row: dict) -> str:
    snap = row["after"] if row["after"] is not None else row["before"]
    return (snap or {}).get("owner", "") or ""


def _visible(row: dict, current_user: str) -> bool:
    """다른 사람의 개인 항목 이력은 보이지 않는다 — 항목 자체와 같은 규칙."""
    owner = _owner_of(row)
    return not owner or owner == current_user


def _load_rows(where: str = "", params: tuple = ()) -> List[dict]:
    init_db()
    with db_session() as conn:
        rows = [dict(r) for r in conn.execute(
            f"SELECT * FROM glossary_history {where} ORDER BY id", params)]
    for r in rows:
        r["before"] = json.loads(r["before_json"]) if r["before_json"] else None
        r["after"] = json.loads(r["after_json"]) if r["after_json"] else None
    return rows


def list_batches(current_user: str, limit: int = 200) -> pd.DataFrame:
    """
    저장 묶음 목록 (최신순). 보이는 행이 하나도 없는 묶음은 뺀다.

    컬럼: batch_id, changed_at, user, source, table, 추가, 수정, 삭제, reverted
    """
    rows = [r for r in _load_rows() if _visible(r, current_user)]
    reverted = {r["source"].split(":", 1)[1] for r in rows
                if r["source"].startswith("되돌리기:")}
    agg: Dict[str, dict] = {}
    for r in rows:
        b = agg.setdefault(r["batch_id"], {
            "batch_id": r["batch_id"], "changed_at": r["changed_at"],
            "user": r["user"], "source": r["source"], "tables": set(),
            "추가": 0, "수정": 0, "삭제": 0, "_last": 0,
        })
        b["_last"] = max(b["_last"], r["id"])
        b["tables"].add(TABLE_KO[r["table_name"]])
        b[ACTION_KO[r["action"]]] += 1
    out = []
    for b in agg.values():
        b["table"] = ", ".join(sorted(b.pop("tables")))
        b["reverted"] = b["batch_id"] in reverted
        out.append(b)
    # changed_at은 초 단위라 같은 초의 묶음끼리 순서가 갈리지 않는다.
    out.sort(key=lambda b: b.pop("_last"), reverse=True)
    cols = ["batch_id", "changed_at", "user", "source", "table",
            "추가", "수정", "삭제", "reverted"]
    return pd.DataFrame(out[:limit], columns=cols)


def batch_rows(batch_id: str, current_user: str) -> pd.DataFrame:
    """묶음 안의 변경 내역 — 화면 표시용."""
    rows = [r for r in _load_rows("WHERE batch_id=?", (batch_id,))
            if _visible(r, current_user)]
    out = []
    for r in rows:
        b, a = r["before"] or {}, r["after"] or {}
        changed = [k for k in FIELDS[r["table_name"]]
                   if r["action"] == "update" and b.get(k) != a.get(k)]
        out.append({
            "구분": TABLE_KO[r["table_name"]],
            "동작": ACTION_KO[r["action"]],
            "한국어": a.get("ko") or b.get("ko", ""),
            "이전": b.get("en", "") if r["action"] != "insert" else "",
            "이후": a.get("en", "") if r["action"] != "delete" else "",
            "바뀐 항목": ", ".join(changed),
        })
    return pd.DataFrame(out, columns=["구분", "동작", "한국어", "이전", "이후", "바뀐 항목"])


def revert_batch(batch_id: str, current_user: str) -> dict:
    """
    묶음 하나를 되돌린다. 뒤에서부터 거꾸로 적용한다.

    반환: {'reverted': N, 'conflicts': N, 'denied': N}
      conflicts — 그 뒤에 다시 바뀌었거나(현재 값 ≠ 기록된 after) 같은
                  (KO, EN)이 이미 있어 건너뛴 행
      denied    — 다른 사람의 개인 항목
    """
    from services.sync import sync_db   # 순환 import 방지

    rows = _load_rows("WHERE batch_id=?", (batch_id,))
    counts = {"reverted": 0, "conflicts": 0, "denied": 0}
    if not rows:
        return counts
    rb = new_batch()
    source = f"되돌리기:{batch_id}"
    with db_session() as conn:
        for r in reversed(rows):
            if not _visible(r, current_user):
                counts["denied"] += 1
                continue
            table, rid, fields = r["table_name"], r["row_id"], FIELDS[r["table_name"]]
            current = snapshot(conn, table, rid)
            if r["action"] in ("insert", "update") and current != r["after"]:
                counts["conflicts"] += 1      # 그 뒤에 또 바뀌었다
                continue
            if r["action"] == "insert":
                conn.execute(f"DELETE FROM {table} WHERE id=?", (rid,))
                record(conn, rb, table, rid, "delete", current_user, source, current, None)
            elif r["action"] == "update":
                before = r["before"]
                conn.execute(
                    f"UPDATE {table} SET {', '.join(f'{k}=?' for k in fields)}, "
                    f"updated_at=? WHERE id=?",
                    (*[before[k] for k in fields], now_iso(), rid),
                )
                record(conn, rb, table, rid, "update", current_user, source, current, before)
            else:  # delete → 다시 넣는다. 같은 id가 비어 있으면 그 id로.
                before = r["before"]
                dup = conn.execute(
                    f"SELECT 1 FROM {table} WHERE ko=? AND en=?",
                    (before["ko"], before["en"]),
                ).fetchone()
                if current is not None or dup:
                    counts["conflicts"] += 1
                    continue
                now = now_iso()
                conn.execute(
                    f"INSERT INTO {table} (id, {', '.join(fields)}, imported_at, updated_at) "
                    f"VALUES (?, {', '.join('?' * len(fields))}, ?, ?)",
                    (rid, *[before[k] for k in fields], now, now),
                )
                record(conn, rb, table, rid, "insert", current_user, source, None, before)
            counts["reverted"] += 1
    if counts["reverted"]:
        sync_db(f"glossary: revert {batch_id} ({counts['reverted']}) by {current_user}")
    return counts
