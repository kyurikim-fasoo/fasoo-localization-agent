# -*- coding: utf-8 -*-
"""
로컬라이즈 적용 내역 리포트.

산출물만 받아서는 이 도구가 무슨 일을 했는지 알 수 없다. 글로서리와 UI 텍스트
매핑은 자리표시자로 치환되므로 무엇을 어디에 고정했는지가 정확히 남는데,
그 기록을 결과 문서 위에 표시해 눈으로 확인하게 한다.

형식은 마크다운이다. 처음에는 Word로 만들어 형광펜을 칠했는데, 마크다운
산출물의 본문을 Word에 옮기면 문법 기호가 그대로 드러나 읽을 수 없었다.
대신 표시는 【 】〔 〕로 한다 — 렌더러에 의존하지 않고 평문에서도 눈에 띈다.

산출물 자체는 건드리지 않는다. 배포용 문서에 표시가 남으면 안 되므로
별도 파일로 만든다.
"""
from __future__ import annotations

import io
import os
import re
from typing import List, Optional, Tuple

# 출처별 표시. 두 가지로 나누는 이유는 "내가 직접 지정한 것"과 "글로서리에서
# 온 것"을 구분해야 다음에 무엇을 등재할지 판단이 서기 때문.
_WRAP = {
    "글로서리": ("【", "】"),
    "UI 매핑": ("〔", "〕"),
}
_DEFAULT_WRAP = ("【", "】")
_MARKDOWN_EXT = {".md", ".markdown", ".mdx"}

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_ANCHOR_RE = re.compile(r"\s*\{#[^}]*\}\s*$")

# 표시하면 안 되는 구간. 링크 주소 안의 /guide/analysis/… 까지 괄호가
# 씌워지면 리포트가 지저분해지고, 주소가 바뀐 것처럼 보인다.
_LINK_TARGET_RE = re.compile(r"\]\(([^)]*)\)")
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_BARE_URL_RE = re.compile(r"https?://\S+")


def protected_ranges(text: str) -> List[Tuple[int, int]]:
    """링크 주소·인라인 코드·맨 URL 구간."""
    out: List[Tuple[int, int]] = []
    for m in _LINK_TARGET_RE.finditer(text):
        out.append((m.start(1), m.end(1)))
    for m in _INLINE_CODE_RE.finditer(text):
        out.append(m.span())
    for m in _BARE_URL_RE.finditer(text):
        out.append(m.span())
    return out


def read_output_paragraphs(path: str) -> List[str]:
    """산출물을 문단 목록으로 읽는다. 형식에 관계없이 평문만 본다."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _MARKDOWN_EXT:
        with io.open(path, encoding="utf-8") as f:
            return [ln.rstrip() for ln in f.read().split("\n") if ln.strip()]
    from docx import Document
    from translator_engine import iter_all_paragraphs
    return [p.text for p in iter_all_paragraphs(Document(path)) if p.text.strip()]


def body_lines(paragraphs: List[str]) -> List[str]:
    """
    본문에서 읽을 수 있는 줄만 남긴다.

    front matter와 제목 기호, 앵커는 걷어낸다. 리포트에 그대로 실으면
    문서 구조가 겹쳐 읽기 어렵고, 확인에 필요한 정보도 아니다.
    """
    out: List[str] = []
    in_front = False
    for i, ln in enumerate(paragraphs):
        s = ln.strip()
        if s == "---":
            # 맨 앞의 --- 는 front matter 시작
            if i == 0 or in_front:
                in_front = not in_front
                continue
            continue
        if in_front:
            continue
        s = _HEADING_RE.sub("", s)
        s = _ANCHOR_RE.sub("", s)
        if s:
            out.append(s)
    return out


def split_spans(text: str,
                terms: List[Tuple[str, str]],
                protect: Optional[List[Tuple[int, int]]] = None
                ) -> List[Tuple[str, Optional[str]]]:
    """
    문장을 (조각, 출처) 목록으로 가른다. 출처가 None이면 표시하지 않는다.

    긴 표현을 먼저 잡는다 — 'login record file'을 'file'이 먼저 먹어버리면
    표시가 잘게 부서진다. 이미 잡힌 자리는 다시 잡지 않는다.
    """
    if not text:
        return []
    low = text.lower()
    used = [False] * len(text)
    for a, b in (protect or []):
        for k in range(max(0, a), min(len(text), b)):
            used[k] = True
    marks: List[Tuple[int, int, str]] = []

    for en, kind in sorted(terms, key=lambda t: -len(t[0] or "")):
        needle = (en or "").lower()
        if not needle:
            continue
        i = low.find(needle)
        while i >= 0:
            j = i + len(needle)
            # 영문 단어 경계에서만. 'set'이 'settings' 안에서 잡히면 안 된다.
            left_ok = i == 0 or not text[i - 1].isalnum()
            right_ok = j >= len(text) or not text[j].isalnum()
            if left_ok and right_ok and not any(used[i:j]):
                marks.append((i, j, kind))
                for k in range(i, j):
                    used[k] = True
            i = low.find(needle, i + 1)

    marks.sort()
    out: List[Tuple[str, Optional[str]]] = []
    pos = 0
    for a, b, kind in marks:
        if a > pos:
            out.append((text[pos:a], None))
        out.append((text[a:b], kind))
        pos = b
    if pos < len(text):
        out.append((text[pos:], None))
    return out


def mark_line(text: str, terms: List[Tuple[str, str]]) -> Tuple[str, int]:
    """적용 지점에 표시를 씌운 줄과 표시 개수."""
    spans = split_spans(text, terms, protected_ranges(text))
    n = 0
    buf = []
    for chunk, kind in spans:
        if kind:
            a, b = _WRAP.get(kind, _DEFAULT_WRAP)
            buf.append(f"{a}{chunk}{b}")
            n += 1
        else:
            buf.append(chunk)
    return "".join(buf), n


def assess(applied: List[dict], n_body: int, n_marked: int) -> dict:
    """
    적용 결과 총평.

    "몇 건 적용했습니다"로 끝내면 그래서 좋아졌다는 것인지 알 수 없다.
    고정하지 않았다면 흔들릴 수 있었던 자리가 몇 곳인지를 말해 준다 —
    같은 용어가 반복 등장할 때마다 표기가 갈리는 것이 이 작업의 본래 위험이다.
    """
    n_terms = len(applied)
    n_hits = sum(int(a.get("적용") or 0) for a in applied)
    n_ui = sum(1 for a in applied if a.get("출처") == "UI 매핑")
    repeated = [a for a in applied if int(a.get("적용") or 0) >= 2]
    # 반복 등장분 — 고정이 없었다면 표기가 갈릴 수 있었던 자리
    exposure = sum(int(a.get("적용") or 0) - 1 for a in repeated)
    coverage = (n_marked / n_body * 100) if n_body else 0.0

    if not n_terms:
        grade = "미적용"
    elif coverage >= 50:
        grade = "충분"
    elif coverage >= 20:
        grade = "보통"
    else:
        grade = "제한적"

    if not n_terms:
        summary = (
            "이번 번역에는 등록된 용어가 적용되지 않았습니다. 문서에 반복 "
            "등장하는 표현이 문단마다 다르게 번역될 수 있으므로, Glossary "
            "추출에서 이 문서를 기준으로 용어를 등재하신 후 재실행을 "
            "권고드립니다."
        )
        detail = ""
    else:
        summary = (
            f"등록된 표현 {n_terms}건이 본문 {n_hits:,}곳에 동일한 영문으로 "
            f"적용되었습니다. 전체 {n_body:,}개 문단 중 {n_marked:,}개 문단"
            f"({coverage:.0f}%)에 적용 지점이 포함되어 있으며, 이 중 "
            f"{n_ui}건은 UI 텍스트 매핑에서 직접 지정한 항목입니다."
        )
        if repeated:
            detail = (
                f"반복 등장하는 표현 {len(repeated)}건이 문서 전체에서 하나의 "
                f"표기로 고정되었습니다. 용어 고정이 없을 경우 이 표현들은 "
                f"문단마다 다른 영문으로 번역될 수 있으며, 해당 위험에 "
                f"노출되었던 지점은 {exposure:,}곳입니다. 이번 번역에서는 "
                f"모두 단일 표기로 처리되었습니다."
            )
        else:
            detail = (
                "적용된 표현이 모두 1회씩만 등장하여, 표기 흔들림 위험은 "
                "낮은 문서입니다."
            )

    return {
        "등급": grade,
        "총평": summary,
        "상세": detail,
        "지표": [
            ("적용 표현", f"{n_terms}건"),
            ("적용 지점", f"{n_hits:,}곳"),
            ("문단 커버리지", f"{coverage:.0f}%"),
            ("표기 고정 효과", f"{exposure:,}곳"),
        ],
    }


def build(applied: List[dict], out_path: str,
          doc_name: Optional[str] = None,
          product: Optional[str] = None) -> str:
    """
    적용 내역 리포트(마크다운).

    applied: translate_document()가 돌려준 stats["applied"]
    out_path: 로컬라이즈 산출물 경로 (본문을 여기서 다시 읽는다)
    """
    terms = [(str(a.get("EN") or ""), str(a.get("출처") or "글로서리"))
             for a in applied if a.get("EN")]

    lines = body_lines(read_output_paragraphs(out_path))
    marked: List[str] = []
    for ln in lines:
        shown, n = mark_line(ln, terms)
        if n:
            marked.append(shown)

    a = assess(applied, len(lines), len(marked))

    md: List[str] = ["# 로컬라이즈 적용 내역"]
    meta = []
    if doc_name:
        meta.append(f"문서: {doc_name}")
    if product:
        meta.append(f"제품: {product}")
    if meta:
        md += ["", "  ·  ".join(meta)]

    md += ["", "## 총평", ""]
    md += [f"- {k}: {v}" for k, v in a["지표"]]
    md += ["", f"종합 평가: {a['등급']}", "", a["총평"]]
    if a["상세"]:
        md += ["", a["상세"]]

    if applied:
        md += ["", "## 적용 표현", "",
               "| 용어 | 영문 | 출처 | 적용 |", "|---|---|---|---|"]
        for it in applied:
            md.append(
                f"| {it.get('KO','')} | {it.get('EN','')} "
                f"| {it.get('출처','')} | {it.get('적용','')} |"
            )

        md += ["", "## 본문 적용 지점", "",
               "【 】는 Glossary, 〔 〕는 UI 텍스트 매핑으로 고정된 표현입니다.",
               ""]
        for ln in marked:
            md.append(f"- {ln}")
        if len(marked) < len(lines):
            md += ["",
                   f"※ 전체 {len(lines):,}개 문단 중 적용 지점이 있는 "
                   f"{len(marked):,}개 문단만 수록했습니다."]

    return "\n".join(md) + "\n"
