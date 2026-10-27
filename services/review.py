"""
번역 검토(Review Window) — 단락별 표시·검사·수정 반영.

번역 결과는 굵게·링크·이미지 위치를 ⟦…⟧ 마커로 품은 '마크드 텍스트'다.
검토자에게 이걸 그대로 보이면 읽기 어렵고, 지워 버리면 문서에서 링크와
이미지가 사라진다. 그래서:

  - 가장 흔한 굵게(⟦B⟧…⟦/B⟧)만 **…** 로 보여주고, 나머지 마커는 그대로 둔다.
  - 고친 텍스트는 원문과 마커 구성이 같은지 검사한 뒤에만 받는다.
    어긋나면 반영하지 않고 이유를 돌려준다 — 조용히 고쳐 쓰면 검토자가
    모르는 사이에 서식이 바뀐다.

Streamlit에 의존하지 않는다. 화면은 app.py가 그린다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import output_check
from translator_engine import (B_CLOSE, B_OPEN, check_marker_integrity,
                               contains_korean, finalize_markers)

_BOLD = "**"
_ANY_MARKER_RE = re.compile(r"⟦[^⟧]*⟧")


def _bold_mappable(marked: str) -> bool:
    # 원래 텍스트에 '**'가 있으면(마크다운 문서 등) 되돌릴 때 구분할 수 없다.
    return _BOLD not in marked


def to_display(marked: str) -> str:
    """마크드 텍스트 → 검토 화면 표시용."""
    if not marked:
        return ""
    if _bold_mappable(marked):
        marked = marked.replace(B_OPEN, _BOLD).replace(B_CLOSE, _BOLD)
    return marked


def from_display(text: str, src_marked: str, base_marked: str) -> Tuple[str, List[str]]:
    """
    검토 화면에서 고친 텍스트 → 마크드 텍스트. (결과, 문제 목록)

    base_marked는 고치기 전 텍스트 — 굵게를 **로 바꿔 보여줬는지 판단한다.
    문제 목록이 비어 있지 않으면 결과를 쓰면 안 된다.
    """
    text = (text or "").strip()
    if not text:
        return "", ["번역이 비어 있습니다."]
    if _bold_mappable(base_marked):
        if text.count(_BOLD) % 2:
            return "", ["굵게 표시(**)의 짝이 맞지 않습니다."]
        parts = text.split(_BOLD)
        text = "".join(
            p if k == 0 else ((B_OPEN if k % 2 else B_CLOSE) + p)
            for k, p in enumerate(parts)
        )
    marked, problems = finalize_markers(src_marked, text)
    return marked, [_friendly(p) for p in problems]


def _friendly(problem: str) -> str:
    """check_marker_integrity의 개발자용 문구 → 검토자용."""
    label = problem.split(":", 1)[0].split(" 마커")[0]
    label = re.sub(r"\s*⟦[^⟧]*⟧", "", label).strip()
    return f"{label} 표시가 원문과 다릅니다. ⟦ ⟧ 기호와 ** 표시는 지우거나 추가하지 마세요."


def flags(src_marked: str, out_marked: str) -> List[str]:
    """
    단락 하나에서 사람이 볼 만한 문제. 문서 전체 검사(output_check)를
    단락 단위로 옮긴 것 — 어느 단락을 고쳐야 하는지 바로 보이게 한다.
    """
    out = []
    plain = _ANY_MARKER_RE.sub("", out_marked or "")
    if contains_korean(plain):
        out.append("한국어 남음")
    if output_check._NULLISH_RE.search(plain):
        out.append("빈 값")
    for m in output_check._GLUED_RE.finditer(plain):
        w = m.group(0)
        if w not in output_check._GLUE_ALLOW and \
                w.lower() not in {a.lower() for a in output_check._GLUE_ALLOW}:
            out.append("공백 누락")
            break
    if check_marker_integrity(src_marked, out_marked or ""):
        out.append("서식 확인")
    return out


def apply_edits(segments: List[Dict], edited: Dict[int, str],
                current: Dict[int, str]) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    """
    화면에서 바뀐 단락들을 검사해 반영할 것과 거부할 것을 가른다.

    segments: 번역 결과의 segments ([{i, src, out}])
    edited:   {i: 화면 표시 텍스트} — 이번에 화면에서 바뀐 것만
    current:  {i: 마크드 텍스트} — 이미 반영된 이전 수정
    반환: (새로 반영할 {i: 마크드}, 거부된 {i: [문제]})
    """
    by_i = {s["i"]: s for s in segments}
    ok, bad = {}, {}
    for i, disp in edited.items():
        seg = by_i.get(i)
        if seg is None:
            continue
        base = current.get(i, seg["out"])
        marked, problems = from_display(disp, seg["src"], base)
        if problems:
            bad[i] = problems
        elif marked != base:
            ok[i] = marked
    return ok, bad


# ── 문서 미리보기 ───────────────────────────────────────────────────────
# 검토자는 표보다 '문서처럼 보이는' 화면에서 어색한 곳을 더 잘 찾는다.
# 마커를 화면용 HTML로 바꾸고, 이 단락에 고정된 용어의 영문을 강조한다.

_MARK_HTML = (
    (re.compile(r"⟦B⟧"), "<b>"), (re.compile(r"⟦/B⟧"), "</b>"),
    (re.compile(r"⟦I⟧"), "<i>"), (re.compile(r"⟦/I⟧"), "</i>"),
    (re.compile(r"⟦H\d+⟧"), '<span class="rv-link">'), (re.compile(r"⟦/H\d+⟧"), "</span>"),
    (re.compile(r"⟦D\d+⟧"), '<span class="rv-img">이미지</span>'),
    (re.compile(r"⟦LB⟧"), "<br>"), (re.compile(r"⟦TB⟧"), " "),
)
_HL_OPEN, _HL_CLOSE = "\x01", "\x02"


def _highlight(text: str, terms_en: List[str]) -> str:
    """마커 밖의 글자에서만 용어 영문을 찾아 표시한다."""
    terms = sorted({t for t in terms_en if t and t.strip()}, key=len, reverse=True)
    if not terms:
        return text
    rx = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in terms)
                    + r")(?![A-Za-z0-9])", re.IGNORECASE)
    parts = re.split(r"(⟦[^⟧]*⟧)", text)
    return "".join(p if p.startswith("⟦") else rx.sub(_HL_OPEN + r"\1" + _HL_CLOSE, p)
                   for p in parts)


def to_preview_html(marked: str, terms_en: List[str]) -> str:
    """마크드 텍스트 한 단락 → 미리보기 HTML. 원문 HTML은 이스케이프한다."""
    import html as _html
    text = _html.escape(_highlight(marked or "", terms_en), quote=False)
    for rx, rep in _MARK_HTML:
        text = rx.sub(rep, text)
    text = _ANY_MARKER_RE.sub("", text)          # 그 밖의 마커(주석·봉인 등)는 숨긴다
    return text.replace(_HL_OPEN, '<span class="rv-term">').replace(_HL_CLOSE, "</span>")
