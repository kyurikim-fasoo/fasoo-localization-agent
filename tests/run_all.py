"""
전체 테스트 실행.

    python tests/run_all.py

각 스크립트는 독립 프로세스로 돌린다 — LLM 스텁이 translator_engine의
모듈 전역을 갈아끼우기 때문에 한 프로세스에서 섞으면 서로 오염된다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SUITES = [
    ("마크다운 파서", "test_markdown.py"),
    ("마크다운 엣지 케이스 / 지원 범위 명세", "test_markdown_edges.py"),
    ("마크다운 엔드투엔드", "regression_markdown.py"),
    ("docx 회귀(어댑터 리팩터링 안전망)", "regression_docx.py"),
    ("카탈로그 추출", "test_catalog.py"),
    ("앱 UI", "test_app_ui.py"),
    ("산출물 검증(마커·저장 후 대조)", "test_output_check.py"),
]


def main() -> int:
    failed = []
    for label, script in SUITES:
        print(f"\n{'=' * 68}\n{label}  —  {script}\n{'=' * 68}")
        proc = subprocess.run(
            [sys.executable, str(HERE / script)],
            cwd=str(HERE.parent),
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        if proc.returncode != 0:
            failed.append(script)

    print(f"\n{'=' * 68}")
    if failed:
        print(f"실패: {', '.join(failed)}")
        return 1
    print(f"전체 통과 ({len(SUITES)}개 스위트)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
