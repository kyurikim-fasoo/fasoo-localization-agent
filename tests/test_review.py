"""
번역 검토(Review Window) 테스트.

검토의 핵심 약속은 두 가지다.
  1. 고친 단락만 바뀌고 나머지 문서는 번역 직후와 **똑같다**.
  2. 서식(굵게·링크·이미지)을 깨는 수정은 받지 않는다.

LLM은 결정론적 스텁이다.

    python tests/test_review.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stub_llm
import translator_engine as te
from services import review

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


def marked_texts(path):
    """문서를 다시 읽어 유닛별 마크드 텍스트 목록."""
    ad = te.make_adapter(path)
    ad.load()
    return [u.src for u in ad.units()]


stub_llm.install()
TMP = Path(tempfile.mkdtemp())

print("[1] 표시 변환 — 굵게는 **, 나머지 마커는 그대로")
check("굵게 표시", review.to_display("Click ⟦B⟧Save⟦/B⟧.") == "Click **Save**.")
m, p = review.from_display("Select **Save**.", "⟦B⟧저장⟦/B⟧을 누릅니다.", "Click ⟦B⟧Save⟦/B⟧.")
check("되돌리기", (m, p) == ("Select ⟦B⟧Save⟦/B⟧.", []), f"{m!r} {p}")
_, p = review.from_display("Select Save.", "⟦B⟧저장⟦/B⟧을 누릅니다.", "Click ⟦B⟧Save⟦/B⟧.")
check("굵게를 지우면 거부", bool(p), str(p))
_, p = review.from_display("Select **Save.", "⟦B⟧저장⟦/B⟧을 누릅니다.", "Click ⟦B⟧Save⟦/B⟧.")
check("** 짝이 안 맞으면 거부", bool(p), str(p))
_, p = review.from_display("See the guide.", "⟦H0⟧가이드⟦/H0⟧를 참고하세요.",
                           "See the ⟦H0⟧guide⟦/H0⟧.")
check("링크 마커를 지우면 거부", bool(p), str(p))
_, p = review.from_display("   ", "가", "A")
check("빈 번역 거부", bool(p))
# 원문에 '**'가 그대로 있으면(마크다운 등) 굵게를 바꿔 보여주지 않는다
check("** 원문은 그대로", review.to_display("a ** b ⟦B⟧c⟦/B⟧") == "a ** b ⟦B⟧c⟦/B⟧")

print("[2] 단락 검사")
check("한국어 남음", "한국어 남음" in review.flags("가", "Click 저장."))
check("공백 누락", "공백 누락" in review.flags("가", "Open the websiteGo now."))
check("빈 값", "빈 값" in review.flags("가", "Select nan."))
check("서식 확인", "서식 확인" in review.flags("⟦B⟧가⟦/B⟧", "Click Save."))
check("정상은 비어 있음", review.flags("⟦B⟧가⟦/B⟧", "Click ⟦B⟧Save⟦/B⟧.") == [])


def run_case(label, src_path):
    print(f"[3] {label} — 고친 단락만 바뀐다")
    src = TMP / Path(src_path).name
    shutil.copy(src_path, src)
    out = TMP / ("out_" + src.name)
    res = te.translate_document(
        in_path=str(src), out_path=str(out), glossary_rows=[], pattern_rows=[],
        api_key="sk-stub", enable_cache=True, enable_qa=False,
    )
    segs = res.get("segments") or []
    check(f"{label}: segments 생성", len(segs) == res["paragraphs_translated"],
          f"{len(segs)} vs {res['paragraphs_translated']}")
    before = marked_texts(out)

    # 마커가 없는 단락 하나와, 있으면 굵게가 있는 단락 하나를 고른다
    plain = next(s for s in segs if "⟦" not in s["out"])
    bold = next((s for s in segs if "⟦B⟧" in s["out"] and "**" not in s["out"]), None)
    edits_disp = {plain["i"]: "REVIEWED plain paragraph."}
    if bold:
        edits_disp[bold["i"]] = "REVIEWED " + review.to_display(bold["out"])
    ok, bad = review.apply_edits(segs, edits_disp, {})
    check(f"{label}: 수정 검사 통과", not bad and len(ok) == len(edits_disp), str(bad))

    te.rebuild_with_edits(str(src), str(out), segs, ok)
    after = marked_texts(out)
    check(f"{label}: 단락 수 동일", len(before) == len(after), f"{len(before)}→{len(after)}")
    # 결과 문서를 다시 파싱한 순번은 원본 순번과 다를 수 있다(마크다운 front
    # matter 등). 그래서 위치가 아니라 '바뀐 단락 수'로 본다.
    diff = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    # 마크다운 front matter는 영어가 되면 다시 파싱해도 유닛으로 잡히지 않아
    # 바뀐 수가 수정 수보다 적을 수 있다. 많으면 안 된다.
    check(f"{label}: 고친 단락만 바뀜", 1 <= len(diff) <= len(ok),
          f"diff={diff[:10]} edits={list(ok)}")
    # 제목은 쓰기 단계에서 마침표가 빠지므로 마침표 없이 찾는다
    raw = out.read_text(encoding="utf-8") if out.suffix != ".docx" else "\n".join(after)
    check(f"{label}: 고친 내용이 문서에 있음", "REVIEWED plain paragraph" in raw)
    if bold:
        check(f"{label}: 굵게 유지",
              any(t.startswith("REVIEWED") and "⟦B⟧" in t for t in after))

    # 수정 없이 다시 만들면 번역 직후와 같다 — 쓰기 경로가 번역과 같다는 증거
    te.rebuild_with_edits(str(src), str(out), segs, {})
    check(f"{label}: 수정 없이 다시 만들면 원래대로", marked_texts(out) == before)

    # 원본이 바뀌면 멈춘다
    segs_bad = [dict(segs[0], src=segs[0]["src"] + "X")] + segs[1:]
    try:
        te.rebuild_with_edits(str(src), str(out), segs_bad, {})
        check(f"{label}: 원본 불일치 감지", False, "예외 없음")
    except ValueError:
        check(f"{label}: 원본 불일치 감지", True)


run_case("docx", ROOT / "input_ko_fsp.docx")
run_case("mdx", ROOT / "tests" / "fixtures" / "runAnalysis.mdx")

print("[표기 확인] 같은 한국어가 여러 영어 표기로 갈린 곳")
_ko = ["정책 설정 화면에서 값을 바꿉니다.", "정책 설정을 저장합니다.",
       "정책 설정은 관리자만 볼 수 있습니다.", "정책 설정을 내보냅니다.",
       "사용자를 추가합니다.", "정책 설정 목록을 확인합니다."]
_en = ["Change the value on the Policy Settings screen.", "Save the policy settings.",
       "Only administrators can view the policy configuration.",
       "Export the policy configuration.", "Add a user.", "Check the policy settings list."]
_pairs = [(i * 10, k, e) for i, (k, e) in enumerate(zip(_ko, _en))]
_v = review.term_variants(_pairs)
check("용어 하나 발견", [x["ko"] for x in _v] == ["정책 설정"], str(_v))
if _v:
    _vv = {x["en"]: x["idx"] for x in _v[0]["variants"]}
    check("표기와 단락 위치", _vv == {"policy settings": [0, 10, 50],
                                  "policy configuration": [20, 30]}, str(_vv))
    check("대소문자 다른 모양도 같은 표기로", "Policy Settings" in _v[0]["variants"][0]["surfaces"])
check("Glossary로 고정된 용어는 제외", review.term_variants(_pairs, exclude_ko={"정책 설정"}) == [])
_same = [(i, k, e.replace("policy configuration", "policy settings")) for i, k, e in _pairs]
check("표기가 하나면 경고 없음", review.term_variants(_same) == [])
# UI 표기(See more/Show more)는 동사로 시작해도 후보가 돼야 하고,
# 그 단락들에 우연히 겹친 긴 문장 조각("message window appears")은 표기가 아니다.
_ko2 = ["더 보기를 선택합니다.", "목록 아래의 더 보기를 누르면 전체 메시지 창이 열립니다.",
        "더 보기를 누르면 메시지 창이 나타납니다.", "답변이 길면 더 보기가 표시됩니다.",
        "더 보기 버튼을 누르면 메시지 창이 나타납니다."]
_en2 = ["Select See more.", "Select See more below the list to open the full message window.",
        "Select Show more and the message window appears.",
        "If the answer is long, Show more is displayed.",
        "Select the See more button and the message window appears."]
_v2 = review.term_variants([(i, k, e) for i, (k, e) in enumerate(zip(_ko2, _en2))])
check("UI 표기 구분 (See more / Show more)",
      [[x["en"] for x in v["variants"]] for v in _v2] == [["See more", "Show more"]],
      str([[x["en"] for x in v["variants"]] for v in _v2]))
_h = review.to_preview_html("the Policy Settings and ⟦B⟧policy configuration⟦/B⟧", ["Save"],
                            [["Policy Settings"], ["policy configuration"]])
check("표기별 색", '<span class="rv-v0">Policy Settings</span>' in _h
      and '<b><span class="rv-v1">policy configuration</span></b>' in _h, _h)

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
