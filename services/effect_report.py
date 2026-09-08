# -*- coding: utf-8 -*-
"""
로컬라이즈 적용 내역 리포트.

산출물만 받아서는 이 도구가 무슨 일을 했는지 알 수 없다. 글로서리와 UI 텍스트
매핑은 자리표시자로 치환되므로 무엇을 어디에 고정했는지가 정확히 남는데,
그 기록을 **결과 문서 위에 형광펜으로 얹어** 눈으로 확인하게 한다.

산출물 자체는 건드리지 않는다. 배포용 문서에 형광펜이 남으면 안 되므로
별도 파일로 만든다.
"""
from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Tuple

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt

# 출처별 형광펜 색. 두 가지로 나누는 이유는 "내가 직접 지정한 것"과
# "글로서리에서 온 것"을 구분해야 다음에 무엇을 등재할지 판단이 서기 때문.
_COLOR = {
    "글로서리": WD_COLOR_INDEX.YELLOW,
    "UI 매핑": WD_COLOR_INDEX.BRIGHT_GREEN,
}
_MARKDOWN_EXT = {".md", ".markdown", ".mdx"}


def read_output_paragraphs(path: str) -> List[str]:
    """산출물을 문단 목록으로 읽는다. 형식에 관계없이 평문만 본다."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _MARKDOWN_EXT:
        with io.open(path, encoding="utf-8") as f:
            return [ln.rstrip() for ln in f.read().split("\n") if ln.strip()]
    from translator_engine import iter_all_paragraphs
    return [p.text for p in iter_all_paragraphs(Document(path)) if p.text.strip()]


def split_spans(text: str,
                terms: List[Tuple[str, str]]) -> List[Tuple[str, Optional[str]]]:
    """
    문장을 (조각, 출처) 목록으로 가른다. 출처가 None이면 표시하지 않는다.

    긴 표현을 먼저 잡는다 — 'login record file'을 'file'이 먼저 먹어버리면
    표시가 잘게 부서진다. 이미 잡힌 자리는 다시 잡지 않는다.
    """
    if not text:
        return []
    low = text.lower()
    used = [False] * len(text)
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


def build(applied: List[dict], out_path: str,
          doc_name: Optional[str] = None,
          product: Optional[str] = None) -> bytes:
    """
    적용 내역 리포트(.docx) 바이트.

    applied: translate_document()가 돌려준 stats["applied"]
    out_path: 로컬라이즈 산출물 경로 (본문을 여기서 다시 읽는다)
    """
    terms = [(str(a.get("EN") or ""), str(a.get("출처") or "글로서리"))
             for a in applied if a.get("EN")]

    doc = Document()
    doc.add_heading("로컬라이즈 적용 내역", level=1)

    meta = []
    if doc_name:
        meta.append(f"문서: {doc_name}")
    if product:
        meta.append(f"제품: {product}")
    if meta:
        doc.add_paragraph("  ·  ".join(meta))

    n_terms = len(applied)
    n_hits = sum(int(a.get("적용") or 0) for a in applied)
    n_ui = sum(1 for a in applied if a.get("출처") == "UI 매핑")
    doc.add_paragraph(
        f"등록된 표현 {n_terms}건이 본문 {n_hits:,}곳에 동일한 영문으로 "
        f"적용되었습니다. 이 중 {n_ui}건은 UI 텍스트 매핑에서 직접 지정한 "
        f"항목이며, 나머지 {n_terms - n_ui}건은 Glossary에서 적용되었습니다."
    )

    # ── 적용 표 ────────────────────────────────────────────────────
    doc.add_heading("적용 표현", level=2)
    table = doc.add_table(rows=1, cols=4)
    try:
        table.style = "Table Grid"
    except Exception:
        pass
    for cell, head in zip(table.rows[0].cells, ("용어", "영문", "출처", "적용")):
        cell.text = head
        for r in cell.paragraphs[0].runs:
            r.bold = True
    for a in applied:
        cells = table.add_row().cells
        cells[0].text = str(a.get("KO") or "")
        cells[1].text = str(a.get("EN") or "")
        cells[2].text = str(a.get("출처") or "")
        cells[3].text = str(a.get("적용") or "")

    # ── 범례 ───────────────────────────────────────────────────────
    doc.add_heading("본문 표시", level=2)
    legend = doc.add_paragraph()
    legend.add_run("노란색").font.highlight_color = _COLOR["글로서리"]
    legend.add_run(" Glossary 적용    ")
    legend.add_run("초록색").font.highlight_color = _COLOR["UI 매핑"]
    legend.add_run(" UI 텍스트 매핑 적용")

    # ── 본문 — 적용된 문단만 ────────────────────────────────────────
    paragraphs = read_output_paragraphs(out_path)
    shown = 0
    for text in paragraphs:
        spans = split_spans(text, terms)
        if not any(kind for _, kind in spans):
            continue                      # 적용된 곳이 없는 문단은 건너뛴다
        shown += 1
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        for chunk, kind in spans:
            run = p.add_run(chunk)
            if kind:
                run.font.highlight_color = _COLOR.get(
                    kind, WD_COLOR_INDEX.YELLOW)

    if shown < len(paragraphs):
        doc.add_paragraph(
            f"※ 전체 {len(paragraphs):,}개 문단 중 적용 지점이 있는 "
            f"{shown:,}개 문단만 수록했습니다."
        )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
