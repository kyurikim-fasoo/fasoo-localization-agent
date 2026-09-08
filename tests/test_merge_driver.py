"""
glossary.db 병합 드라이버 통합 테스트.

merge_dbs() 단위 테스트(test_merge_glossary.py)와 달리, 여기서는 **진짜 git
저장소를 만들어 실제로 merge를 시킨다.** 드라이버 등록이 실제로 먹는지,
충돌 없이 끝나는지, 양쪽 용어가 다 남는지를 git을 통해 확인한다.

    python tests/test_merge_driver.py
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def make_db(path: Path, terms: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE terms (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ko TEXT, en TEXT, product TEXT, dnt INT, case_sensitive INT, note TEXT,
        status TEXT, source_file TEXT, imported_at TEXT, updated_at TEXT,
        owner TEXT)""")
    con.execute("""CREATE TABLE patterns (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ko TEXT, en TEXT, note TEXT, status TEXT, source_file TEXT,
        imported_at TEXT, updated_at TEXT, owner TEXT)""")
    for ko, en in terms:
        con.execute(
            "INSERT INTO terms (ko,en,product,dnt,case_sensitive,note,status,"
            "source_file,imported_at,updated_at,owner) "
            "VALUES (?,?,'ALL',0,0,'','approved','','','','')", (ko, en))
    con.commit()
    con.close()


def read_terms(path: Path) -> set[tuple[str, str]]:
    con = sqlite3.connect(path)
    out = {(r[0], r[1]) for r in con.execute("SELECT ko,en FROM terms")}
    con.close()
    return out


TMP = Path(tempfile.mkdtemp(prefix="merge_driver_test_"))
repo = TMP / "repo"
repo.mkdir()
DB = repo / "data" / "glossary.db"

try:
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    git(repo, "config", "commit.gpgsign", "false")

    # 드라이버를 이 임시 저장소에도 등록한다
    (repo / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "merge_glossary_db.py", repo / "scripts")
    (repo / ".gitattributes").write_text(
        "data/glossary.db merge=glossarydb\n", encoding="utf-8")
    git(repo, "config", "merge.glossarydb.name", "glossary db union merge")
    git(repo, "config", "merge.glossarydb.driver",
        f'"{sys.executable}" scripts/merge_glossary_db.py --driver %O %A %B')

    # 공통 조상: 두 용어
    make_db(DB, [("공통", "common"), ("지울 것", "to delete")])
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")

    # 갈래 1 — 내 쪽에서 하나 추가하고 하나 삭제
    git(repo, "checkout", "-q", "-b", "mine")
    make_db(DB, [("공통", "common"), ("내 것", "mine")])
    git(repo, "commit", "-q", "-am", "mine")

    # 갈래 2 — 상대 쪽에서 다른 것 추가
    git(repo, "checkout", "-q", "master") if git(
        repo, "rev-parse", "--verify", "-q", "master").returncode == 0 else \
        git(repo, "checkout", "-q", "main")
    make_db(DB, [("공통", "common"), ("지울 것", "to delete"), ("남의 것", "theirs")])
    git(repo, "commit", "-q", "-am", "theirs")

    print("[1] git merge 가 충돌 없이 끝난다")
    res = git(repo, "merge", "mine", "-m", "merge")
    check("merge 성공", res.returncode == 0,
          (res.stdout + res.stderr).strip()[:200])
    check("충돌 파일 없음", "CONFLICT" not in res.stdout + res.stderr,
          (res.stdout + res.stderr)[:160])

    print("[2] 양쪽 용어가 모두 살아남는다")
    got = read_terms(DB)
    check("공통 항목 유지", ("공통", "common") in got, str(got))
    check("내 쪽 추가분 유지", ("내 것", "mine") in got, str(got))
    check("상대 쪽 추가분 유지", ("남의 것", "theirs") in got, str(got))

    print("[3] 한쪽에서 지운 것은 되살아나지 않는다")
    check("삭제 존중", ("지울 것", "to delete") not in got, str(got))

    print("[4] 중복 없이 정확히 3개")
    con = sqlite3.connect(DB)
    n = con.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
    dupes = list(con.execute(
        "SELECT ko,en FROM terms GROUP BY ko,en HAVING COUNT(*)>1"))
    con.close()
    check("행 수 3", n == 3, n)
    check("중복 0", not dupes, str(dupes))

finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
