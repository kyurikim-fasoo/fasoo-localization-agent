"""
glossary.db 머지 도구 테스트.

이 도구가 지키는 것은 하나다 — **양쪽이 등재한 용어가 모두 살아남는가.**
`git checkout --ours`나 `--theirs`로 해소하면 반대쪽 작업이 통째로 사라지는데,
바이너리라 그 사실이 눈에 보이지도 않는다. 그래서 손실이 0인지 코드가 확인한다.

    python tests/test_merge_glossary.py
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from merge_glossary_db import merge_dbs  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


TMP = Path(tempfile.mkdtemp(prefix="glossary_merge_test_"))


def make(path: Path, terms, patterns):
    """(ko, en, product, owner) 목록으로 최소 스키마 DB를 만든다."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE terms (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ko TEXT, en TEXT, product TEXT, dnt INT, case_sensitive INT, note TEXT,
        status TEXT, source_file TEXT, imported_at TEXT, updated_at TEXT,
        owner TEXT)""")
    con.execute("""CREATE TABLE patterns (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ko TEXT, en TEXT, note TEXT, status TEXT, source_file TEXT,
        imported_at TEXT, updated_at TEXT, owner TEXT)""")
    for ko, en, product, owner in terms:
        con.execute(
            "INSERT INTO terms (ko,en,product,dnt,case_sensitive,note,status,"
            "source_file,imported_at,updated_at,owner) "
            "VALUES (?,?,?,0,0,'','approved','','','',?)", (ko, en, product, owner))
    for ko, en, owner in patterns:
        con.execute(
            "INSERT INTO patterns (ko,en,note,status,source_file,imported_at,"
            "updated_at,owner) VALUES (?,?,'','approved','','','',?)",
            (ko, en, owner))
    con.commit()
    con.close()
    return path


def keys(path: Path):
    con = sqlite3.connect(path)
    t = {(r[0], r[1], (r[2] or "").lower(), r[3] or "")
         for r in con.execute("SELECT ko,en,product,owner FROM terms")}
    p = {(r[0], r[1], r[2] or "")
         for r in con.execute("SELECT ko,en,owner FROM patterns")}
    con.close()
    return t, p


print("[1] 양쪽에만 있는 항목이 모두 살아남는다")
ours = make(TMP / "o.db",
            [("공통", "common", "ALL", ""), ("내 것", "mine", "ALL", "")],
            [("공통 문장", "common sentence", "")])
theirs = make(TMP / "t.db",
              [("공통", "common", "ALL", ""), ("남의 것", "theirs", "ALL", "")],
              [("공통 문장", "common sentence", ""), ("남의 문장", "their sentence", "")])
r = merge_dbs(ours, theirs, TMP / "m.db")
ot, op = keys(ours)
tt, tp = keys(theirs)
mt, mp = keys(TMP / "m.db")

check("ours terms 전부 보존", ot <= mt, str(ot - mt))
check("theirs terms 전부 보존", tt <= mt, str(tt - mt))
check("ours patterns 전부 보존", op <= mp, str(op - mp))
check("theirs patterns 전부 보존", tp <= mp, str(tp - mp))
check("합집합과 정확히 일치", mt == ot | tt and mp == op | tp,
      f"{mt} / {ot | tt}")
check("공통 항목이 두 번 들어가지 않음", len(mt) == 3 and len(mp) == 2,
      f"terms {len(mt)} patterns {len(mp)}")
check("신규 건수 보고", r["terms"]["added"] == 1 and r["patterns"]["added"] == 1,
      str(r))

print("[2] 같은 KO라도 제품·소유자가 다르면 별개")
ours2 = make(TMP / "o2.db", [("설정", "settings", "FED", "")], [])
theirs2 = make(TMP / "t2.db",
               [("설정", "settings", "Fireside", ""),
                ("설정", "settings", "FED", "kyuri")], [])
merge_dbs(ours2, theirs2, TMP / "m2.db")
mt2, _ = keys(TMP / "m2.db")
check("제품이 다르면 따로 남는다", len(mt2) == 3, str(mt2))

print("[3] 한쪽에 이미 중복이 있어도 정리된다")
ours3 = make(TMP / "o3.db",
             [("중복", "dupe", "ALL", ""), ("중복", "dupe", "ALL", "")],
             [("중복 문장", "dupe sentence", ""), ("중복 문장", "dupe sentence", "")])
theirs3 = make(TMP / "t3.db", [("중복", "dupe", "ALL", "")], [])
r3 = merge_dbs(ours3, theirs3, TMP / "m3.db")
mt3, mp3 = keys(TMP / "m3.db")
check("terms 중복 정리", len(mt3) == 1 and r3["terms"]["dupes"] == 1, str(r3["terms"]))
check("patterns 중복 정리", len(mp3) == 1, str(mp3))

print("[4] theirs가 전부 중복이면 아무것도 늘지 않는다")
ours4 = make(TMP / "o4.db", [("가", "a", "ALL", "")], [("나", "b", "")])
theirs4 = make(TMP / "t4.db",
               [("가", "a", "ALL", ""), ("가", "a", "ALL", "")],
               [("나", "b", ""), ("나", "b", "")])
r4 = merge_dbs(ours4, theirs4, TMP / "m4.db")
check("신규 0건", r4["terms"]["added"] == 0 and r4["patterns"]["added"] == 0, str(r4))
check("행 수 그대로", r4["terms"]["after"] == 1 and r4["patterns"]["after"] == 1, str(r4))

print("[5] 원본은 건드리지 않는다")
check("ours 파일 불변", keys(ours) == (ot, op))
check("theirs 파일 불변", keys(theirs) == (tt, tp))

shutil.rmtree(TMP, ignore_errors=True)

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
