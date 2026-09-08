"""
glossary.db 병합 드라이버 등록 (클론마다 한 번).

.gitattributes 는 저장소에 커밋되지만, 드라이버의 **실행 명령**은 git config에
있어야 하고 그건 커밋되지 않는다. 그래서 클론할 때마다 한 번 실행해야 한다.

    python scripts/setup_merge_driver.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "glossarydb"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT,
                          capture_output=True, text=True)


def main() -> int:
    if _git("rev-parse", "--git-dir").returncode != 0:
        print("git 저장소가 아닙니다.")
        return 2

    driver = (f'"{sys.executable}" scripts/merge_glossary_db.py '
              f"--driver %O %A %B")
    _git("config", f"merge.{NAME}.name", "glossary db union merge")
    _git("config", f"merge.{NAME}.driver", driver)

    got = _git("config", f"merge.{NAME}.driver").stdout.strip()
    if not got:
        print("등록에 실패했습니다.")
        return 1

    print("병합 드라이버를 등록했습니다.")
    print(f"  merge.{NAME}.driver = {got}")
    print()
    print("이제 git pull 에서 data/glossary.db 충돌이 나지 않습니다.")
    print("양쪽 용어를 합치고, 한쪽에서 지운 것은 지운 채로 둡니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
