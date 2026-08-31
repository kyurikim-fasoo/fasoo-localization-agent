"""
마크다운 파서 단위 테스트.

    python tests/test_markdown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import markdown_format as mf
import translator_engine as te

TARGET = Path(__file__).resolve().parent / "fixtures" / "runAnalysis.mdx"

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        failures.append(name)


print("[1] translator_engine과 마커 상수 일치")
check("B_OPEN", mf.B_OPEN == te.B_OPEN)
check("B_CLOSE", mf.B_CLOSE == te.B_CLOSE)
check("D_PREFIX", mf.D_PREFIX == te.D_PREFIX)
check("SUFFIX", mf.SUFFIX == te.SUFFIX)

print("[2] 인라인 봉인/복원 왕복")
SAMPLES = [
    "**저장소** 또는 **압축 파일**을 선택하세요.",
    "자세한 내용은 [분석 설정하기](/guide/analysis/analysisSetting)를 참고하세요.",
    "![image.png](/img/분석실행.png)",
    "[다운로드](https://x.com/a?hl=en\\&utm_source=ext\\&pli=1)",
    "`code` 와 **굵게** 혼합",
    "설정은 [여기](/a/b.mdx#소스코드-저장소-설정하기)를 보세요.",
]
for s in SAMPLES:
    marked, seals, urls = mf.seal_inline(s)
    back = mf.unseal_inline(marked, seals, urls)
    check(f"왕복: {s[:32]}…", back == s, f"\n       got: {back!r}")

print("[3] 봉인이 한국어 판정보다 먼저 — 이미지 경로는 번역 대상이 아니다")
marked, _, _ = mf.seal_inline("![image.png](/img/분석실행.png)")
check("경로의 한글이 마커 뒤로 숨음", not te.contains_korean(marked), f"marked={marked!r}")

print("[4] 슬러그가 실제 문서의 앵커와 일치")
check("소스코드 저장소 설정하기",
      mf.slugify("소스코드 저장소 설정하기") == "소스코드-저장소-설정하기",
      mf.slugify("소스코드 저장소 설정하기"))
check("분석 대상 웹 페이지 설정하기",
      mf.slugify("분석 대상 웹 페이지 설정하기") == "분석-대상-웹-페이지-설정하기")
check("분석하기", mf.slugify("분석하기") == "분석하기")

print("[5] 실제 문서 왕복 = 원문과 바이트 동일")
text, enc, nl, bom = mf.read_text(str(TARGET))
check("인코딩 utf-8", enc == "utf-8")
check("개행 LF", nl == "\n")
check("BOM 없음", bom is False)
units = mf.parse_markdown(text)
rebuilt = mf.apply_translations(text, [(u, u.src) for u in units], keep_anchor=False)
if rebuilt != text:
    for k in range(min(len(rebuilt), len(text))):
        if rebuilt[k] != text[k]:
            lo = max(0, k - 40)
            check("왕복 동일", False,
                  f"\n       위치 {k}\n       원문: {text[lo:k+40]!r}\n       결과: {rebuilt[lo:k+40]!r}")
            break
    else:
        check("왕복 동일", False, f"길이 다름 {len(text)} vs {len(rebuilt)}")
else:
    check("왕복 동일", True)

print("[6] 유닛 구성")
kinds = {}
for u in units:
    kinds[u.kind] = kinds.get(u.kind, 0) + 1
print(f"       {kinds}  총 {len(units)}개")
ko_units = [u for u in units if te.contains_korean(u.src)]
print(f"       한국어 포함 {len(ko_units)}개")
check("front matter title/description 2개", kinds.get("frontmatter") == 2, str(kinds))
check("헤딩 8개", kinds.get("heading") == 8, str(kinds))
check("이미지 줄은 번역 대상 아님", len(ko_units) == len(units) - 1,
      f"units={len(units)} ko={len(ko_units)}")

print("[7] 헤딩 앵커 보존")
h = next(u for u in units if u.kind == "heading" and u.heading_slug == "저장소-분석하기")
rendered = mf.render_unit(h, "Analyzing a repository", keep_anchor=True)
check("원래 슬러그가 명시적 id로 보존됨",
      rendered == "Analyzing a repository {#저장소-분석하기}", rendered)

print("[8] front matter 값은 필요할 때만 인용 (불필요한 diff 방지)")
fm = next(u for u in units if u.kind == "frontmatter")
check("콜론이 들어가면 감싼다",
      mf.render_unit(fm, "Analysis: getting started") == '"Analysis: getting started"',
      mf.render_unit(fm, "Analysis: getting started"))
check("평범한 값은 그대로",
      mf.render_unit(fm, "Running an analysis") == "Running an analysis",
      mf.render_unit(fm, "Running an analysis"))

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
