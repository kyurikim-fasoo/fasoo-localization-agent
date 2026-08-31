"""
마크다운/MDX 번역 엔드투엔드 테스트.

LLM은 결정론적 스텁이므로 '번역 품질'이 아니라 **구조가 온전한가**를 본다.
번역되면 안 되는 것들(front matter 키, 이미지 경로, URL, 코드)이 그대로인지,
마커가 새어나가지 않았는지, 앵커가 보존됐는지.

    python tests/regression_markdown.py [--show]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stub_llm
import markdown_format as mf
import translator_engine as te

TARGET = Path(__file__).resolve().parent / "fixtures" / "runAnalysis.mdx"
OUT = ROOT / "outputs" / "_regression_out.mdx"

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        failures.append(name)


stub_llm.install()
OUT.parent.mkdir(exist_ok=True)
metrics = te.translate_document(
    in_path=str(TARGET),
    out_path=str(OUT),
    glossary_rows=[{"KO": "분석", "EN": "analysis"},
                   {"KO": "이벤트 클립보드", "EN": "Event Clipboard", "Case-sensitive": True}],
    pattern_rows=[],
    api_key="sk-stub-not-used",
    enable_cache=True,
    enable_qa=True,
    translation_mode="매뉴얼",
    ui_text_overrides={"저장": "Save", "기록": "Record"},
)

src = TARGET.read_text(encoding="utf-8")
out = OUT.read_text(encoding="utf-8")

print(f"[0] 번역 단위 {metrics['paragraphs_translated']}개")
check("38개 유닛 번역", metrics["paragraphs_translated"] == 38,
      str(metrics["paragraphs_translated"]))

print("[1] 마커 누출 없음")
leaked = re.findall(r"⟦[^⟧]*⟧", out)
check("출력에 ⟦…⟧ 없음", not leaked, str(leaked[:5]))

print("[2] 줄 구조 보존")
check("줄 수 동일", out.count("\n") == src.count("\n"),
      f"{src.count(chr(10))} → {out.count(chr(10))}")
check("끝 개행 유지", out.endswith("\n"))
check("CRLF 유입 없음", "\r" not in out)

print("[3] front matter — 값만 번역, 키/기타는 불변")
check("sidebar_position 그대로", "sidebar_position: 1" in out)
check("icon 이스케이프 그대로", '"\\U0001F9E0"' in out)
check("sidebar_custom_props 그대로", "sidebar_custom_props:" in out)
check("title 값이 번역됨", not re.search(r"^title: 분석하기$", out, re.M))
check("구분선 --- 2개 유지", out.count("\n---\n") == src.count("\n---\n"))

print("[4] 번역 금지 구역")
check("이미지 줄 그대로", "![image.png](/img/분석실행.png)" in out)
check("URL의 \\& 이스케이프 그대로",
      "?hl=en-US\\&utm_source=ext_sidebar\\&pli=1" in out)
check("내부 링크 경로 그대로", "(/guide/analysis/analysisSetting)" in out)
check("한국어 앵커 링크 그대로",
      "(/guide/analysis/analysisSetting.mdx#소스코드-저장소-설정하기)" in out)

print("[5] 헤딩 앵커 보존")
for slug in ("분석하기", "저장소-분석하기", "압축-파일-분석하기", "url-분석하기",
             "이벤트-클립보드", "이벤트-클립보드-설치하기"):
    check(f"{{#{slug}}} 부착", f"{{#{slug}}}" in out)
check("헤딩 개수 유지",
      len(re.findall(r"^#{1,6} ", out, re.M)) == len(re.findall(r"^#{1,6} ", src, re.M)))

print("[6] 블록 접두 보존")
check("인용 3줄 유지", len(re.findall(r"^> ", out, re.M)) == len(re.findall(r"^> ", src, re.M)))
check("순서 목록 유지",
      len(re.findall(r"^\d+\. ", out, re.M)) == len(re.findall(r"^\d+\. ", src, re.M)))

print("[7] 굵게 표시 개수 보존")
check("** 짝 개수 동일", out.count("**") == src.count("**"),
      f"{src.count('**')} → {out.count('**')}")

print("[8] 한국어 잔존은 봉인된 자리뿐")
ko_lines = [ln for ln in out.splitlines() if re.search(r"[가-힣]", ln)]
allowed = [ln for ln in ko_lines
           if "/img/" in ln or "{#" in ln or "analysisSetting.mdx#" in ln
           or "analysisSetting#" in ln]
check("예상 못한 한국어 줄 없음", len(ko_lines) == len(allowed),
      "\n       " + "\n       ".join(l[:100] for l in ko_lines if l not in allowed))

if "--show" in sys.argv:
    print("\n" + "=" * 70)
    print(out)

OUT.unlink(missing_ok=True)
print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
