"""
docx 번역 경로 회귀 테스트.

어댑터 리팩터링이 기존 Word 처리를 바꾸지 않았음을 보증하기 위한 안전망.
LLM을 결정론적 스텁으로 갈아끼우고 픽스처 문서를 번역한 뒤, 결과 문서의
**문단별 마크드 텍스트**(= 텍스트 + 굵게/하이퍼링크/줄바꿈 구조)를 덤프해
기준선과 비교한다.

zip 바이트 비교가 아니라 마크드 텍스트 비교인 이유: docx는 zip이라
타임스탬프 때문에 바이트가 매번 달라진다. 마크드 텍스트는 우리가 실제로
보존하려는 것(내용과 서식)을 정확히 담는다.

    python tests/regression_docx.py --save    # 기준선 저장
    python tests/regression_docx.py           # 기준선과 비교
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stub_llm
import translator_engine as te

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# 작은 문서 하나 + 실제 매뉴얼 하나. 후자는 하이퍼링크·인라인 이미지·표·
# heading 스타일이 전부 들어 있어 서식 회귀를 실질적으로 잡아준다.
FIXTURES = [
    ("small", ROOT / "input_ko_fsp.docx"),
    ("manual", ROOT / "test.docx"),
]

GLOSSARY_ROWS = [
    {"KO": "파수", "EN": "Fasoo", "DNT": True, "Case-sensitive": True},
    {"KO": "문서", "EN": "document"},
    {"KO": "보안", "EN": "security"},
]
PATTERN_ROWS = [
    {"KO": "클릭하세요", "EN": "Click"},
]
UI_OVERRIDES = {"저장": "Save"}


def run_translation(fixture: Path, out_path: Path) -> dict:
    stub_llm.install()
    return te.translate_document(
        in_path=str(fixture),
        out_path=str(out_path),
        glossary_rows=GLOSSARY_ROWS,
        pattern_rows=PATTERN_ROWS,
        api_key="sk-stub-not-used",
        enable_cache=True,
        enable_qa=True,
        translation_mode="매뉴얼",
        ui_text_overrides=UI_OVERRIDES,
    )


def dump_document(path: Path) -> str:
    """결과 문서를 '문단별 마크드 텍스트'로 직렬화."""
    from docx import Document

    doc = Document(str(path))
    lines = []
    for i, p in enumerate(te.iter_all_paragraphs(doc)):
        marked, drawings, links, trailing = te.paragraph_to_marked_text(p)
        lines.append(
            f"[{i:04d}] heading={int(te.is_heading_paragraph(p))} "
            f"drawings={len(drawings)} links={len(links)} trailing_lb={trailing}\n"
            f"       {marked}"
        )
    return "\n".join(lines) + "\n"


def check_one(name: str, fixture: Path, save: bool) -> int:
    baseline = FIXTURES_DIR / f"docx_{name}.txt"
    out_path = ROOT / "outputs" / f"_regression_{name}.docx"
    out_path.parent.mkdir(exist_ok=True)

    metrics = run_translation(fixture, out_path)
    dump = dump_document(out_path)
    out_path.unlink(missing_ok=True)

    if save:
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        baseline.write_text(dump, encoding="utf-8")
        print(f"[{name}] 기준선 저장 — 번역 문단 {metrics['paragraphs_translated']}개, "
              f"덤프 {len(dump):,}자")
        return 0

    if not baseline.exists():
        print(f"[{name}] 기준선 없음. 먼저 --save로 만드세요: {baseline}")
        return 2

    expected = baseline.read_text(encoding="utf-8")
    if dump == expected:
        print(f"[{name}] OK — 회귀 없음 (번역 문단 {metrics['paragraphs_translated']}개)")
        return 0

    exp_lines, got_lines = expected.splitlines(), dump.splitlines()
    print(f"[{name}] FAIL — 결과가 기준선과 다릅니다.")
    print(f"  줄 수: 기준선 {len(exp_lines)} vs 현재 {len(got_lines)}")
    for n, (e, g) in enumerate(zip(exp_lines, got_lines)):
        if e != g:
            print(f"  첫 불일치 {n + 1}번째 줄")
            print(f"    기준선: {e[:200]}")
            print(f"    현재  : {g[:200]}")
            break
    return 1


def main() -> int:
    save = "--save" in sys.argv
    rc = 0
    for name, fixture in FIXTURES:
        if not fixture.exists():
            print(f"[{name}] 픽스처 없음: {fixture}")
            rc = max(rc, 2)
            continue
        rc = max(rc, check_one(name, fixture, save))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
