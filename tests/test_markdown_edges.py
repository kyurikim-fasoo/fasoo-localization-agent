"""
마크다운 엣지 케이스 — 현재 지원 범위를 코드로 못박아 둔다.

여기 적힌 기대값이 곧 "지금 되는 것과 안 되는 것"의 명세다. 미지원 항목도
**깨지지 않고 원문 그대로 남는지**를 검증한다(잘못 번역되는 것보다 낫다).

    python tests/test_markdown_edges.py [--show]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stub_llm
import translator_engine as te

TARGET = Path(__file__).resolve().parent / "fixtures" / "edge_cases.md"
OUT = ROOT / "outputs" / "_edge_out.md"

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


stub_llm.install()
OUT.parent.mkdir(exist_ok=True)
te.translate_document(
    in_path=str(TARGET), out_path=str(OUT),
    glossary_rows=[
        # 사용자가 명시적으로 case-sensitive로 등록한 제품명
        {"KO": "Virtual Drive", "EN": "Virtual Drive", "Case-sensitive": True},
    ],
    pattern_rows=[], api_key="sk-stub",
    enable_cache=True, enable_qa=True, translation_mode="매뉴얼",
)
src = TARGET.read_text(encoding="utf-8")
out = OUT.read_text(encoding="utf-8")

print("[지원] 보존되어야 하는 것")
check("마커 누출 없음", not re.findall(r"⟦[^⟧]*⟧", out),
      str(re.findall(r"⟦[^⟧]*⟧", out)[:3]))
check("인라인 코드 `analysis.config` 원형", "`analysis.config`" in out)
check("인라인 코드 `--verbose` 원형", "`--verbose`" in out)
check("코드펜스 내용 미번역", 'echo "분석 시작"' in out)
check("코드펜스 주석 미번역", "# 이 블록은 번역하면 안 됩니다" in out)
check("이스케이프 \\& 보존", "\\&" in out)
check("이스케이프 \\* 보존", "\\*" in out)
check("기존 {#custom-anchor} 유지", "{#custom-anchor}" in out)
check("id가 이미 있으면 중복 부착 안 함",
      all(l.count("{#") <= 1 for l in out.splitlines()) and out.count("{#custom-anchor}") == 1,
      f"{{# 개수 {out.count('{#')}")
check("중첩 목록 들여쓰기 유지", re.search(r"^  - ", out, re.M) is not None)
check("줄 수 보존(하드랩 제외)", abs(out.count("\n") - src.count("\n")) <= 1,
      f"{src.count(chr(10))} → {out.count(chr(10))}")
check("JSX 태그 줄 원형", '<Admonition type="tip" title="주의">' in out)
check("JSX 여러 줄 속성 원형", "{label: '설치', value: 'install'}," in out)
check("JSX 여는 태그 원형", "<Tabs" in out and "]}>" in out)
check("표 구분행 원형", "| --- | --- |" in out)
check("기울임 * 쌍 보존", re.search(r"(?<!\*)\*[^*\n]+\*(?!\*)", out) is not None)
check("기울임 안쪽은 번역됨", "*기울임*" not in out)

print("\n[헤딩 대소문자] 등록된 case-sensitive 용어만 존중, 나머지는 sentence case")
heading = next(l for l in out.splitlines() if l.startswith("## ") and "Drive" in l.title())
check("헤딩에서 등록 표기 유지", "Virtual Drive" in heading, heading)
check("본문에서도 유지", any("Virtual Drive" in l and not l.startswith("#")
                            for l in out.splitlines()))
check("폭 없는 문자 없음", "﻿" not in out and "​" not in out)

print("\n[의도된 동작] JSX 태그 사이 본문은 번역한다")
check("Admonition 본문 번역됨", "JSX 블록 안쪽 본문입니다." not in out)
check("Tabs 본문 번역됨", "여러 줄 속성 뒤의 본문입니다." not in out)

print("\n[미지원] 원문 그대로 남는 것 — 잘못 번역되지 않는지 확인")
check("표 내용은 한국어 그대로", "| 저장소 | 소스코드 저장소입니다 |" in out)
check("JSX 속성값은 한국어 그대로", 'title="주의"' in out)

print("\n[정보] 하드랩 문단 처리")
merged = "여러 줄에 걸쳐" not in out and "하드랩된 문단입니다." not in out
check("하드랩 문단이 한 줄로 합쳐져 번역됨", merged,
      "합쳐지지 않음 — 문단 인식 확인 필요")

if "--show" in sys.argv:
    print("\n" + "=" * 70 + "\n" + out)

OUT.unlink(missing_ok=True)
print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
