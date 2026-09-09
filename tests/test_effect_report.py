"""
적용 내역 리포트 테스트.

사용자가 이 리포트로 확인하려는 것은 하나다 — "에이전트가 정말 일관되게
바꿨는가". 그러려면 표시가 **정확한 자리에만** 찍혀야 한다. 'set'이
'settings' 안에서 잡히거나, 링크 주소까지 표시되면 리포트가 오히려
신뢰를 깎는다.

    python tests/test_effect_report.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import effect_report as er

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


print("[1] 표시 위치 — 단어 경계")
sp = er.split_spans("Open the settings and set the value.", [("set", "글로서리")])
marked = [t for t, k in sp if k]
check("settings 안의 set은 잡지 않는다", marked == ["set"], str(marked))

print("[2] 긴 표현이 먼저")
sp = er.split_spans(
    "Upload the login record file to the file browser.",
    [("file", "글로서리"), ("login record file", "글로서리")],
)
marked = [t for t, k in sp if k]
check("긴 것부터 잡는다", "login record file" in marked, str(marked))
check("남은 file도 잡는다", marked.count("file") == 1, str(marked))
check("겹치지 않는다", len(marked) == 2, str(marked))

print("[3] 원문이 손실되지 않는다")
text = "Click **Search File** then Save."
sp = er.split_spans(text, [("Search File", "UI 매핑"), ("Save", "UI 매핑")])
check("조각을 이으면 원문", "".join(t for t, _ in sp) == text,
      "".join(t for t, _ in sp))

print("[4] 대소문자 무시, 원문 표기 유지")
sp = er.split_spans("analysis and Analysis", [("Analysis", "글로서리")])
marked = [t for t, k in sp if k]
check("둘 다 잡는다", len(marked) == 2, str(marked))
check("원문 표기 그대로", marked == ["analysis", "Analysis"], str(marked))

print("[5] 링크 주소·코드는 건드리지 않는다")
line = ("See [the analysis guide](/guide/analysis/setting) and `analysis` "
        "and https://x.io/analysis now analysis here.")
out, n = er.mark_line(line, [("analysis", "글로서리")])
check("본문 2곳만 표시", n == 2, f"{n} / {out}")
check("링크 주소 보존", "/guide/analysis/setting" in out, out)
check("인라인 코드 보존", "`analysis`" in out, out)
check("맨 URL 보존", "https://x.io/analysis" in out, out)

print("[6] 출처별 표시 기호")
out, _ = er.mark_line("Save and analysis",
                      [("Save", "UI 매핑"), ("analysis", "글로서리")])
check("UI 매핑은 〔〕", "〔Save〕" in out, out)
check("글로서리는 【】", "【analysis】" in out, out)

print("[7] 본문 정리 — front matter·제목 기호·앵커")
paras = ["---", "title: Analyze", "description: Run it.", "---",
         "# Analyze {#분석하기}", "Body text."]
check("front matter 제거 · 제목 기호 제거",
      er.body_lines(paras) == ["Analyze", "Body text."],
      str(er.body_lines(paras)))

print("[8] 총평")
_ap = [{"KO": "분석", "EN": "analysis", "출처": "글로서리", "적용": 10},
       {"KO": "저장", "EN": "Save", "출처": "UI 매핑", "적용": 1}]
a = er.assess(_ap)
check("지표 2종", len(a["지표"]) == 2, str(a["지표"]))
check("적용 표현", ("적용 표현", "2건") in a["지표"], str(a["지표"]))
check("적용 지점", ("적용 지점", "11곳") in a["지표"], str(a["지표"]))

# 추정에 기댄 수치는 고객 문서에 싣지 않는다. 문단 커버리지는 문단을 어떻게
# 세느냐에 따라 달라지고, '고정 효과 N곳'은 고정이 없었다면 표기가 갈렸으리라는
# 가정이 들어간다.
check("커버리지 지표 없음",
      not any("커버리지" in k for k, _ in a["지표"]), str(a["지표"]))
check("고정 효과 지표 없음",
      not any("고정 효과" in k for k, _ in a["지표"]), str(a["지표"]))
check("본문에도 추정 수치 없음",
      "%" not in a["총평"] and "%" not in a["상세"], a["총평"] + a["상세"])
check("총평에 실제 건수", "11곳" in a["총평"], a["총평"])
check("반복 고정 서술", "1건이 문서 전체" in a["상세"], a["상세"])
check("등급 없음", a["등급"] == "", a["등급"])

_a0 = er.assess([])
check("미적용 등급", _a0["등급"] == "미적용", _a0["등급"])
check("미적용 안내에 다음 행동", "Glossary" in _a0["총평"], _a0["총평"])

print("[9] 리포트 생성")
tmp = Path(tempfile.mkdtemp(prefix="effect_"))
body = tmp / "body.mdx"
body.write_text(
    "---\ntitle: T\n---\n\n# Head {#h}\n\nRun the analysis.\n\n"
    "Click Save to finish.\n\nNothing here.\n", encoding="utf-8")
applied = [{"KO": "분석", "EN": "analysis", "출처": "글로서리", "적용": 2},
           {"KO": "저장", "EN": "Save", "출처": "UI 매핑", "적용": 1}]
md = er.build(applied, str(body), "runAnalysis.mdx", "Sparrow")

check("마크다운 문자열", isinstance(md, str) and md.startswith("# 로컬라이즈"),
      md[:60])
check("총평 절", "## 총평" in md)
check("종합 평가 줄 없음", "종합 평가" not in md, md[:400])
check("커버리지 문구 없음", "커버리지" not in md and "고정 효과" not in md,
      md[:400])
check("적용 표현 표", "| 용어 | 영문 | 출처 | 적용 |" in md)
check("본문 절", "## 본문 적용 지점" in md)
check("범례", "【 】는 Glossary" in md)
check("본문에 표시", "【analysis】" in md and "〔Save〕" in md, md[-400:])
check("적용 없는 문단 제외", "Nothing here." not in md)
check("생략 안내", "문단만 수록" in md, md[-200:])
check("front matter 미포함", "title: T" not in md)

print("[10] 적용 내역이 비어도 죽지 않는다")
md0 = er.build([], str(body), "x.mdx", None)
check("빈 입력도 생성", md0.startswith("# 로컬라이즈"), md0[:60])
check("미적용 안내 포함", "적용되지 않았습니다" in md0, md0[:400])
check("표 절은 생략", "## 적용 표현" not in md0)

shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
