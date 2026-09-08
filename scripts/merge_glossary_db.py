"""
glossary.db 머지 충돌 해소기.

왜 필요한가:
    glossary.db는 SQLite 바이너리다. git은 바이너리를 병합할 수 없어서, 두
    사람이 각자 용어를 등재하면 매번 충돌이 난다. 이때 `--ours`나 `--theirs`로
    한쪽을 고르면 **반대쪽이 등재한 용어가 통째로 사라진다.**

    이 스크립트는 양쪽을 합친다. 같은 (KO, EN, 제품, 소유자)는 한 번만 남기고
    나머지는 모두 살린다. 충돌 중에 실행하면 된다.

사용:
    # 충돌이 난 상태에서
    python scripts/merge_glossary_db.py
    git add data/glossary.db && git commit

    # 무엇이 합쳐지는지만 보고 싶을 때
    python scripts/merge_glossary_db.py --dry-run
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "glossary.db"
BACKUPS = ROOT / "data" / "backups"

# (테이블, 같은 항목으로 볼 컬럼)
#
# 제품·공개 범위는 보지 않는다. 같은 국문이 같은 영문으로 번역된다면 어느
# 제품·범위에 있든 같은 항목이다. 제품별로 갈라야 하는 것은 영문이 다른
# 경우뿐인데(검출: FDR=detection / FSM=Detection) 그건 EN이 달라 안 걸린다.
# 저장 계층(services/glossary.py)의 중복 판정과 같은 기준을 쓴다.
TABLES = (
    ("terms", ("ko", "en")),
    ("patterns", ("ko", "en")),
)


def _stage(n: int, dest: Path) -> bool:
    """git 인덱스의 stage n(1=base, 2=ours, 3=theirs)을 파일로 꺼낸다."""
    try:
        blob = subprocess.run(
            ["git", "show", f":{n}:data/glossary.db"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return False
    dest.write_bytes(blob)
    return True


def _rows(path: Path, table: str) -> tuple[list[str], list[tuple]]:
    con = sqlite3.connect(path)
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    data = list(con.execute(f"SELECT {','.join(cols)} FROM {table} ORDER BY id"))
    con.close()
    return cols, data


def _key(cols: list[str], row: tuple, keys: tuple[str, ...]) -> tuple:
    idx = {c: i for i, c in enumerate(cols)}
    return tuple(
        (row[idx[k]] or "").lower() if k == "product" else (row[idx[k]] or "")
        for k in keys
    )


def merge_dbs(ours: Path, theirs: Path, out: Path,
              base: Path | None = None) -> dict:
    """
    ours를 바탕으로 theirs에만 있는 행을 얹고, 남은 중복을 정리한다.

    "같은 항목"의 기준은 TABLES에 적힌 컬럼 조합이다. id는 양쪽이 서로 다르게
    매기므로 비교에 쓰지 않는다 — id로 비교하면 같은 용어가 서로 다른 항목으로
    보여 중복이 쌓인다.

    base(공통 조상)를 주면 **삭제도 존중한다.** base에 있었는데 한쪽에서
    사라진 항목은 그쪽이 일부러 지운 것이므로 되살리지 않는다. base가 없으면
    단순 합집합이라, 한쪽에서 지운 항목이 반대쪽에서 되살아난다 — 중복 정리
    커밋이 통째로 무효가 되는 일이 실제로 일어날 수 있다.
    """
    shutil.copy(ours, out)
    con = sqlite3.connect(out)
    report: dict = {}
    for table, keys in TABLES:
        cols, ours_rows = _rows(ours, table)
        _, their_rows = _rows(theirs, table)
        idx = {c: i for i, c in enumerate(cols)}
        insert_cols = [c for c in cols if c != "id"]

        ours_keys = {_key(cols, r, keys) for r in ours_rows}
        their_keys = {_key(cols, r, keys) for r in their_rows}
        deleted: set = set()
        if base is not None:
            base_keys = {_key(cols, r, keys) for r in _rows(base, table)[1]}
            # 한쪽에서 지운 것 = base에 있었는데 그쪽에 없는 것
            deleted = (base_keys - ours_keys) | (base_keys - their_keys)

        seen = set(ours_keys)
        added = 0
        for r in their_rows:
            k = _key(cols, r, keys)
            if k in seen or k in deleted:
                continue
            seen.add(k)
            con.execute(
                f"INSERT INTO {table} ({','.join(insert_cols)}) "
                f"VALUES ({','.join('?' * len(insert_cols))})",
                tuple(r[idx[c]] for c in insert_cols),
            )
            added += 1

        # theirs가 지운 것이 ours에 남아 있으면 그것도 반영한다
        removed = 0
        if deleted:
            for r in _rows(out, table)[1]:
                if _key(cols, r, keys) in deleted:
                    con.execute(f"DELETE FROM {table} WHERE id=?", (r[idx["id"]],))
                    removed += 1
        con.commit()

        # ours 쪽에 이미 중복이 있었다면 그것도 정리한다
        groups = defaultdict(list)
        for r in _rows(out, table)[1]:
            groups[_key(cols, r, keys)].append(r[idx["id"]])
        dupes = [i for v in groups.values() if len(v) > 1 for i in v[1:]]
        con.executemany(f"DELETE FROM {table} WHERE id=?", [(i,) for i in dupes])
        con.commit()

        report[table] = {
            "ours": len(ours_rows), "theirs": len(their_rows),
            "added": added, "removed": removed, "dupes": len(dupes),
            "after": con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
        }
    con.close()
    return report


def run_as_git_driver(base: str, ours: str, theirs: str) -> int:
    """
    git 병합 드라이버로 호출됐을 때. 결과를 %A(ours 경로)에 써야 한다.

    등록:
        .gitattributes 에  data/glossary.db merge=glossarydb
        git config merge.glossarydb.name "glossary db union merge"
        git config merge.glossarydb.driver \
            "python scripts/merge_glossary_db.py --driver %O %A %B"

    이렇게 해두면 git pull 때 충돌이 아예 나지 않는다. 사람이 매번 어느
    쪽을 고를지 판단할 필요가 없고, 무엇보다 **한쪽을 고르다 상대의 용어를
    통째로 잃는 사고**가 구조적으로 사라진다.
    """
    tmp = Path(tempfile.mkdtemp(prefix="glossary_driver_"))
    try:
        out = tmp / "merged.db"
        base_path = Path(base) if base and Path(base).stat().st_size else None
        report = merge_dbs(Path(ours), Path(theirs), out, base=base_path)
        shutil.copy(out, ours)          # %A 에 결과를 쓴다
        for table, st_ in report.items():
            print(f"[glossary-merge] {table}: +{st_['added']} "
                  f"-{st_['removed']} 중복 {st_['dupes']} → {st_['after']}",
                  file=sys.stderr)
        return 0                        # 0 = 충돌 없이 병합됨
    except Exception as e:
        print(f"[glossary-merge] 실패: {e}", file=sys.stderr)
        return 1                        # git이 충돌로 표시하고 사람에게 넘긴다
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str]) -> int:
    if "--driver" in argv:
        rest = argv[argv.index("--driver") + 1:]
        if len(rest) < 3:
            print("사용: --driver %O %A %B", file=sys.stderr)
            return 1
        return run_as_git_driver(rest[0], rest[1], rest[2])

    dry = "--dry-run" in argv
    tmp = Path(tempfile.mkdtemp(prefix="glossary_merge_"))
    try:
        ours, theirs = tmp / "ours.db", tmp / "theirs.db"
        if not (_stage(2, ours) and _stage(3, theirs)):
            print("충돌 상태가 아닙니다. 머지 도중에 실행하세요.")
            return 2

        base = tmp / "base.db"
        has_base = _stage(1, base)
        report = merge_dbs(ours, theirs, tmp / "merged.db",
                           base=base if has_base else None)
        for table, st_ in report.items():
            print(f"  {table:<9} ours {st_['ours']:>5} + theirs 신규 {st_['added']:>4}"
                  f" - 삭제반영 {st_['removed']:>3} - 중복 {st_['dupes']:>3}"
                  f"  →  {st_['after']:>5}")

        if dry:
            print("\n--dry-run: 파일을 바꾸지 않았습니다.")
            return 0

        BACKUPS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy(ours, BACKUPS / f"glossary-merge-ours-{stamp}.db")
        shutil.copy(theirs, BACKUPS / f"glossary-merge-theirs-{stamp}.db")
        shutil.copy(tmp / "merged.db", DB)
        print(f"\n{DB} 를 합친 결과로 바꿨습니다.")
        print(f"양쪽 원본은 {BACKUPS} 에 보관했습니다.")
        print("\n다음: git add data/glossary.db && git commit")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
