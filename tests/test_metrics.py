"""
대시보드 지표 테스트.

임원 보고에 쓰이는 숫자라 틀리면 안 되는 것 두 가지를 본다:
  - 비용이 공식 단가대로 계산되는가 (캐시 토큰을 이중으로 세지 않는가)
  - 측정하지 않은 값이 0으로 집계돼 지표가 좋아 보이지 않는가

    python tests/test_metrics.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db.schema as schema

# 실제 data/glossary.db를 건드리지 않도록 임시 DB로 바꿔 끼운다.
_TMP = Path(tempfile.mkdtemp()) / "glossary.db"
schema.DB_PATH = _TMP

from services import metrics
from services.translation_logs import create_log

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


print("[1] 비용 — 공식 단가, 캐시분은 1/10")
check("입력 1M = $1.75", abs(metrics.api_cost_usd(1_000_000, 0, 0) - 1.75) < 1e-9)
check("출력 1M = $14", abs(metrics.api_cost_usd(0, 0, 1_000_000) - 14.0) < 1e-9)
# input_tokens는 캐시분을 포함한다 — 1M 중 전부 캐시면 $0.175
check("전부 캐시 = $0.175", abs(metrics.api_cost_usd(1_000_000, 1_000_000, 0) - 0.175) < 1e-9)
check("절반 캐시", abs(metrics.api_cost_usd(1_000_000, 500_000, 0) - (0.875 + 0.0875)) < 1e-9)
check("캐시가 입력보다 커도 음수 안 됨", metrics.api_cost_usd(10, 50, 0) >= 0)
check("None 안전", metrics.api_cost_usd(None, None, None) == 0)

print("[2] 마이그레이션 — 기존 로그 테이블에 컬럼 추가, 기존 행은 NULL")
_conn = sqlite3.connect(_TMP)
_conn.execute("""CREATE TABLE translation_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, user TEXT NOT NULL,
    product TEXT DEFAULT '', translation_mode TEXT DEFAULT '', source_file TEXT DEFAULT '',
    output_file TEXT DEFAULT '', ui_text_overrides TEXT DEFAULT '{}',
    input_tokens INTEGER DEFAULT 0, cached_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
    paragraphs_translated INTEGER DEFAULT 0, note TEXT DEFAULT '')""")
_conn.execute("INSERT INTO translation_logs (created_at, user, product, input_tokens, "
              "output_tokens, paragraphs_translated) VALUES "
              "('2026-08-01T10:00:00', 'A', 'FSP', 1000000, 0, 10)")
_conn.commit(); _conn.close()
schema.init_db()
_cols = {r[1] for r in sqlite3.connect(_TMP).execute("PRAGMA table_info(translation_logs)")}
check("새 컬럼 추가됨", {"elapsed_seconds", "applied_count", "verify_errors"} <= _cols, str(_cols))
df = metrics.load_logs()
check("기존 행 보존", len(df) == 1)
check("기존 행 지표는 NULL", df["elapsed_seconds"].isna().all())
check("기존 행 비용 계산", abs(df["cost_usd"].iloc[0] - 1.75) < 1e-9)

print("[3] create_log — 결과에 없는 지표는 0이 아니라 NULL")
create_log("B", "", "", "x.docx", "y.docx", {}, {"paragraphs_translated": 3}, sync=False)
create_log("C", "FSS", "", "a.docx", "b.docx", {}, {
    "paragraphs_translated": 20, "elapsed_seconds": 90,
    "applied": [{"KO": "저장", "EN": "Save", "적용": 4},
                {"KO": "취소", "EN": "Cancel", "적용": 1}],
    "verification": [{"level": "경고", "title": "w"}],
}, sync=False)
df = metrics.load_logs()
_b = df[df["user"] == "B"].iloc[0]
_c = df[df["user"] == "C"].iloc[0]
check("지표 없는 로그는 NULL", _b[["elapsed_seconds", "applied_count", "verify_errors"]].isna().all())
check("빈 제품은 (미지정)", _b["product"] == "(미지정)", _b["product"])
check("처리 시간", _c["elapsed_seconds"] == 90)
check("적용 용어 수 / 횟수", (_c["applied_terms"], _c["applied_count"]) == (2, 5))
check("오류 0 / 경고 1", (_c["verify_errors"], _c["verify_warnings"]) == (0, 1))
_m = metrics.measured_subset(df)
check("측정분만 집계", list(_m["user"]) == ["C"], str(list(_m["user"])))
check("기간 필터", len(metrics.load_logs(since="2026-09-01")) == 2)

print("[4] Glossary 수")
with schema.db_session() as conn:
    conn.execute("INSERT INTO terms (ko, en, owner) VALUES ('가', 'a', ''), ('나', 'b', 'kim')")
gc = metrics.glossary_counts()
check("Team / Personal", gc["terms"] == {"team": 1, "personal": 1}, str(gc))
check("패턴 0", gc["patterns"] == {"team": 0, "personal": 0}, str(gc))

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
