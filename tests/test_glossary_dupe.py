"""
글로서리 중복 삽입 방지 테스트.

화면에서 "이미 등재됨"을 보여주는 것만으로는 중복을 막지 못한다. 다른
화면에서 부르거나, 화면이 판정에 쓰는 기준(제품 등)이 저장 기준과 어긋나면
그대로 또 들어간다. 실제로 같은 문장이 10번까지 쌓인 적이 있다.
그래서 최종 방어선은 저장 계층이어야 하고, 이 스위트가 그걸 지킨다.

    python tests/test_glossary_dupe.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

# ── 실제 글로서리를 건드리지 않도록 DB 경로부터 갈아끼운다 ──────────
# services.glossary를 import하기 **전에** 바꿔야 한다. 한 번 import되면
# db_session이 그 시점의 DB_PATH를 들고 다닌다.
import db.schema as schema

_TMP = Path(tempfile.mkdtemp(prefix="glossary_test_"))
schema.DB_PATH = _TMP / "glossary.db"

from services import glossary as g  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


def _rows(kind, product="ALL"):
    base = {
        "id": [None, None], "Scope": ["Team", "Team"],
        "KO": ["접근 정책", "사용자 그룹"],
        "EN": ["access policy", "user group"],
        "Note": ["", ""], "Status": ["approved", "approved"], "File": ["t", "t"],
    }
    if kind == "term":
        base |= {"Product": [product, product], "DNT": [False, False],
                 "Case-sensitive": [False, False]}
    return pd.DataFrame(base)


g.init_db()
check("임시 DB를 쓰고 있다", "glossary_test_" in str(schema.DB_PATH),
      str(schema.DB_PATH))
check("빈 DB에서 시작", len(g.load_terms(current_user="u")) == 0)

print("[1] terms — 같은 항목을 두 번 등재")
a = g.save_terms_from_dataframe(_rows("term"), view_ids=set(), current_user="u")
b = g.save_terms_from_dataframe(_rows("term"), view_ids=set(), current_user="u")
check("1회차 2건 삽입", a["inserted"] == 2, str(a))
check("2회차 0건 삽입", b["inserted"] == 0, str(b))
check("2회차 2건 건너뜀", b.get("skipped") == 2, str(b))
check("총 2행 유지", len(g.load_terms(current_user="u")) == 2,
      len(g.load_terms(current_user="u")))

print("[2] patterns — 화면 단계 중복 검사가 아예 없던 곳")
c1 = g.save_patterns_from_dataframe(_rows("pattern"), view_ids=set(), current_user="u")
c2 = g.save_patterns_from_dataframe(_rows("pattern"), view_ids=set(), current_user="u")
check("1회차 2건 삽입", c1["inserted"] == 2, str(c1))
check("2회차 0건 삽입", c2["inserted"] == 0, str(c2))
check("2회차 2건 건너뜀", c2.get("skipped") == 2, str(c2))
check("총 2행 유지", len(g.load_patterns(current_user="u")) == 2)

print("[3] 제품·공개범위가 달라도 KO·EN이 같으면 중복")
# 같은 국문이 같은 영문으로 번역된다면 어느 제품·범위에 있든 새로 넣을 이유가
# 없다. 제품별로 갈라야 하는 것은 영문이 다른 경우뿐이다.
d = g.save_terms_from_dataframe(_rows("term", product="Fireside"),
                                view_ids=set(), current_user="u")
check("제품이 달라도 중복으로 본다", d["inserted"] == 0 and d["skipped"] == 2, str(d))
check("총 2행 유지", len(g.load_terms(current_user="u")) == 2,
      len(g.load_terms(current_user="u")))

_mine = _rows("term")
_mine["Scope"] = ["Personal", "Personal"]
e = g.save_terms_from_dataframe(_mine, view_ids=set(), current_user="u")
check("공개 범위가 달라도 중복으로 본다",
      e["inserted"] == 0 and e["skipped"] == 2, str(e))

# 영문이 다르면 제품별로 따로 남아야 한다 (검출: FDR=detection / FSM=Detection)
_diff = _rows("term", product="FSM")
_diff["EN"] = ["Access Policy", "User Group"]
f2 = g.save_terms_from_dataframe(_diff, view_ids=set(), current_user="u")
check("영문이 다르면 새로 들어간다", f2["inserted"] == 2, str(f2))

print("[4] 한 번의 호출 안에서 같은 행이 두 번 와도")
# 아직 DB에 없는 값으로 — 같은 배치 안에 같은 행이 두 번 든 경우
_new = pd.DataFrame({
    "id": [None, None], "Scope": ["Team", "Team"],
    "KO": ["배치 중복 검사", "배치 중복 검사"],
    "EN": ["batch dupe check", "batch dupe check"],
    "Note": ["", ""], "Status": ["approved", "approved"], "File": ["t", "t"],
})
f = g.save_patterns_from_dataframe(_new, view_ids=set(), current_user="u")
check("배치 안 중복도 걸러낸다",
      f["inserted"] == 1 and f.get("skipped") == 1, str(f))

print("[5] DB에 중복이 남지 않았다")
import sqlite3
_con = sqlite3.connect(schema.DB_PATH)
for tbl, keys in (("terms", "ko,en,product,owner"), ("patterns", "ko,en,owner")):
    left = list(_con.execute(
        f"SELECT {keys} FROM {tbl} GROUP BY {keys} HAVING COUNT(*)>1"))
    check(f"{tbl} 중복 0건", not left, str(left))
_con.close()

shutil.rmtree(_TMP, ignore_errors=True)

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
