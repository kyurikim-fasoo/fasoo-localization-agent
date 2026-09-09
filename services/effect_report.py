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

import html
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


def assess(applied: List[dict]) -> dict:
    """
    적용 결과 총평.

    지표는 실제로 일어난 일만 싣는다. 문단 커버리지는 문단을 어떻게 세느냐에
    따라 달라지고, '고정 효과 N곳'은 고정하지 않았다면 표기가 갈렸으리라는
    가정이 들어간다. 고객에게 나가는 문서에 넣기에는 근거가 약하다.
    """
    n_terms = len(applied)
    n_hits = sum(int(a.get("적용") or 0) for a in applied)
    n_ui = sum(1 for a in applied if a.get("출처") == "UI 매핑")
    repeated = [a for a in applied if int(a.get("적용") or 0) >= 2]

    if not n_terms:
        return {
            "등급": "미적용",
            "총평": (
                "이번 번역에는 등록된 용어가 적용되지 않았습니다. 문서에 반복 "
                "등장하는 표현이 문단마다 다르게 번역될 수 있으므로, Glossary "
                "추출에서 이 문서를 기준으로 용어를 등재하신 후 재실행을 "
                "권고드립니다."
            ),
            "상세": "",
            "지표": [("적용 표현", "0건"), ("적용 지점", "0곳")],
        }

    summary = (
        f"등록된 표현 {n_terms}건이 본문 {n_hits:,}곳에 동일한 영문으로 "
        f"적용되었습니다. 이 중 {n_ui}건은 UI 텍스트 매핑에서 직접 지정한 "
        f"항목이며, 나머지 {n_terms - n_ui}건은 Glossary에서 적용되었습니다."
    )
    if repeated:
        detail = (
            f"반복 등장하는 표현 {len(repeated)}건이 문서 전체에서 하나의 "
            f"표기로 고정되었습니다. 용어 고정이 없을 경우 이 표현들은 "
            f"문단마다 다른 영문으로 번역될 수 있으나, 이번 번역에서는 모두 "
            f"단일 표기로 처리되었습니다."
        )
    else:
        detail = (
            "적용된 표현이 모두 1회씩만 등장하여, 표기 흔들림 위험은 낮은 "
            "문서입니다."
        )

    return {
        "등급": "",
        "총평": summary,
        "상세": detail,
        "지표": [
            ("적용 표현", f"{n_terms}건"),
            ("적용 지점", f"{n_hits:,}곳"),
        ],
    }


# ── HTML 리포트 ───────────────────────────────────────────────────
#
# 표시 색. 원문 링크(파랑)와 겹치지 않는 색을 골랐다 — 겹치면 무엇이 원문
# 서식이고 무엇이 이 도구가 손댄 자리인지 구분이 안 된다.
_MARK_CSS = {
    "글로서리": ("gl", "#FFE9A8"),      # 연노랑
    "UI 매핑": ("ui", "#DCC9FF"),       # 연보라
}

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_CODE_RE = re.compile(r"`([^`]+)`")
_TAG_RE = re.compile(r"<[^>]*>")


def _inline_md_to_html(text: str) -> str:
    """원문 서식을 그대로 살린다. 굵게·링크·인라인 코드만."""
    s = html.escape(text)
    s = _MD_CODE_RE.sub(lambda m: f"<code>{m.group(1)}</code>", s)
    s = _MD_LINK_RE.sub(
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', s)
    s = _MD_BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", s)
    return s


def mark_line_html(text: str, terms: List[Tuple[str, str]]) -> Tuple[str, int]:
    """
    한 줄을 HTML로. 적용 지점만 색을 입힌다.

    마크다운을 먼저 HTML로 바꾼 뒤 표시를 찾는다. 태그 안(<a href="...">)은
    보호 구간으로 넘겨 주소까지 색칠되지 않게 한다.
    """
    s = _inline_md_to_html(text)
    protect = [m.span() for m in _TAG_RE.finditer(s)]
    # 인라인 코드 안은 리터럴이라 표시 대상이 아니다
    protect += [m.span(1) for m in re.finditer(r"<code>(.*?)</code>", s)]
    spans = split_spans(s, terms, protect)
    n = 0
    buf = []
    for chunk, kind in spans:
        if kind:
            cls = _MARK_CSS.get(kind, _MARK_CSS["글로서리"])[0]
            buf.append(f'<mark class="{cls}">{chunk}</mark>')
            n += 1
        else:
            buf.append(chunk)
    return "".join(buf), n


_STYLE = """
:root { color-scheme: light; }
body { margin: 0; padding: 40px 32px; background: #fff; color: #1a1a1a;
       font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR",
                    system-ui, -apple-system, sans-serif;
       font-size: 14px; line-height: 1.75; }
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -.01em; }
h2 { font-size: 15px; margin: 32px 0 12px; padding-bottom: 6px;
     border-bottom: 1px solid #e3e3e3; letter-spacing: -.01em; }
.meta { color: #666; font-size: 13px; margin-bottom: 24px; }
.kpis { display: flex; gap: 28px; flex-wrap: wrap; margin: 0 0 18px;
        padding: 14px 18px; background: #f7f8fa; border-radius: 6px; }
.kpi .k { color: #666; font-size: 12px; }
.kpi .v { font-size: 19px; font-weight: 600; }
p { margin: 0 0 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #e3e3e3; padding: 7px 10px; text-align: left; }
th { background: #f7f8fa; font-weight: 600; }
td.num { text-align: right; }
mark.gl { background: #FFE9A8; padding: 0 2px; border-radius: 2px; }
mark.ui { background: #DCC9FF; padding: 0 2px; border-radius: 2px; }
.legend { display: flex; gap: 18px; align-items: center; margin: 0 0 14px;
          font-size: 13px; color: #444; }
.legend span.sw { padding: 1px 8px; border-radius: 2px; margin-right: 6px; }
ol.body { padding-left: 22px; }
ol.body li { margin: 0 0 10px; }
.note { color: #666; font-size: 12px; margin-top: 14px; }
a { color: #1558d6; }
code { background: #f2f3f5; padding: 0 4px; border-radius: 3px;
       font-size: 12px; }
"""


def build_html(applied: List[dict], out_path: str,
               doc_name: Optional[str] = None,
               product: Optional[str] = None) -> str:
    """적용 내역 리포트(HTML). 브라우저에서 열거나 PDF로 인쇄한다."""
    terms = [(str(a.get("EN") or ""), str(a.get("출처") or "글로서리"))
             for a in applied if a.get("EN")]

    lines = body_lines(read_output_paragraphs(out_path))
    marked: List[str] = []
    for ln in lines:
        shown, n = mark_line_html(ln, terms)
        if n:
            marked.append(shown)

    a = assess(applied)
    esc = html.escape

    h: List[str] = [
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>",
        "<title>로컬라이즈 적용 내역</title>",
        f"<style>{_STYLE}</style></head><body><div class='wrap'>",
        "<h1>로컬라이즈 적용 내역</h1>",
    ]
    meta = []
    if doc_name:
        meta.append(f"문서: {esc(doc_name)}")
    if product:
        meta.append(f"제품: {esc(product)}")
    if meta:
        h.append(f"<div class='meta'>{'  ·  '.join(meta)}</div>")

    h.append("<h2>총평</h2><div class='kpis'>")
    for k, v in a["지표"]:
        h.append(f"<div class='kpi'><div class='k'>{esc(k)}</div>"
                 f"<div class='v'>{esc(v)}</div></div>")
    h.append("</div>")
    if a["등급"]:
        h.append(f"<p><strong>종합 평가: {esc(a['등급'])}</strong></p>")
    h.append(f"<p>{esc(a['총평'])}</p>")
    if a["상세"]:
        h.append(f"<p>{esc(a['상세'])}</p>")

    if applied:
        h.append("<h2>적용 표현</h2><table><tr><th>용어</th><th>영문</th>"
                 "<th>출처</th><th>적용</th></tr>")
        for it in applied:
            kind = str(it.get("출처") or "글로서리")
            cls = _MARK_CSS.get(kind, _MARK_CSS["글로서리"])[0]
            h.append(
                f"<tr><td>{esc(str(it.get('KO','')))}</td>"
                f"<td><mark class='{cls}'>{esc(str(it.get('EN','')))}</mark></td>"
                f"<td>{esc(kind)}</td>"
                f"<td class='num'>{esc(str(it.get('적용','')))}</td></tr>"
            )
        h.append("</table>")

        h.append("<h2>본문 적용 지점</h2><div class='legend'>")
        for kind, (cls, _c) in _MARK_CSS.items():
            label = "Glossary" if kind == "글로서리" else "UI 텍스트 매핑"
            h.append(f"<div><span class='sw {cls}' "
                     f"style='background:{_MARK_CSS[kind][1]}'>&nbsp;&nbsp;</span>"
                     f"{label}</div>")
        h.append("</div><ol class='body'>")
        for ln in marked:
            h.append(f"<li>{ln}</li>")
        h.append("</ol>")
        if len(marked) < len(lines):
            h.append(f"<div class='note'>전체 {len(lines):,}개 문단 중 적용 "
                     f"지점이 있는 {len(marked):,}개 문단만 수록했습니다.</div>")

    h.append("</div></body></html>")
    return "\n".join(h)


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

    a = assess(applied)

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
    if a["등급"]:
        md += ["", f"종합 평가: {a['등급']}"]
    md += ["", a["총평"]]
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
