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
from typing import Dict, List, Optional, Tuple

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
# 강조 자리를 이스케이프 전에 표시해 두는 제어 문자. 그룹 g는 chr(0x10 + g)로 연다.
_HL_BASE, _HL_CLOSE = 0x10, "\x02"


def _highlight(text: str, groups: List[List[str]]) -> str:
    """마커 밖의 글자에서만 구절을 찾아 표시한다. 앞의 그룹이 우선한다."""
    pats = []
    for g, phrases in enumerate(groups):
        for t in sorted({t for t in phrases if t and t.strip()}, key=len, reverse=True):
            pats.append(f"(?P<g{g}_{len(pats)}>" + re.escape(t) + ")")
    if not pats:
        return text
    rx = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(pats) + r")(?![A-Za-z0-9])",
                    re.IGNORECASE)

    def _sub(m: re.Match) -> str:
        g = int(m.lastgroup[1:].split("_")[0])
        return chr(_HL_BASE + g) + m.group(0) + _HL_CLOSE

    parts = re.split(r"(⟦[^⟧]*⟧)", text)
    return "".join(p if p.startswith("⟦") else rx.sub(_sub, p) for p in parts)


def to_preview_html(marked: str, terms_en: List[str],
                    variants: Optional[List[List[str]]] = None) -> str:
    """
    마크드 텍스트 한 단락 → 미리보기 HTML. 원문 HTML은 이스케이프한다.
    terms_en: 적용된 용어(파란색). variants: 표기 확인 중인 표기들 — 표기마다 다른 색.
    """
    import html as _html
    variants = (variants or [])[:4]
    classes = [f"rv-v{j}" for j in range(len(variants))] + ["rv-term"]
    text = _html.escape(_highlight(marked or "", variants + [terms_en]), quote=False)
    for rx, rep in _MARK_HTML:
        text = rx.sub(rep, text)
    text = _ANY_MARKER_RE.sub("", text)          # 그 밖의 마커(주석·봉인 등)는 숨긴다
    for g, cls in enumerate(classes):
        text = text.replace(chr(_HL_BASE + g), f'<span class="{cls}">')
    return text.replace(_HL_CLOSE, "</span>")


# ── 표기 확인 ───────────────────────────────────────────────────────────
# 같은 한국어 용어가 문서 안에서 여러 영어 표기로 갈린 곳을 찾는다.
# 문서 전체 검사(output_check.check_term_consistency)는 비율만 알려줘 무엇을
# 고쳐야 할지 알 수 없었다. 여기서는 단락 짝(원문·번역)을 그대로 쓰므로
# "어느 단락에서 어떤 표기"인지까지 돌려준다 — 화면이 그 단락을 보여줄 수 있다.
#
# 표기 추정 방법: 용어가 나온 단락들의 영문 2~4단어 구절 중, 그 단락들에
# 몰려 있고(다른 단락에는 드묾) 가장 많은 단락을 덮는 것을 첫 표기로 고른다.
# 남은 단락에서 같은 방식으로 다음 표기를 고른다. 2곳 이상 쓰인 표기만 센다 —
# 한 곳뿐인 구절은 대명사나 의역일 수 있어 근거가 약하다.

def _norm_word(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


# 구절 앞뒤에 올 수 없는 기능어. output_check._EN_STOP은 see·select·click 같은
# 동사까지 막아 "See more"·"Select all" 같은 UI 표기를 후보에서 지워 버린다.
_EDGE_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be",
    "was", "were", "you", "your", "it", "its", "this", "that", "these", "those", "with",
    "from", "by", "at", "as", "if", "not", "can", "will", "when", "then", "than",
}


def _phrases(text: str, lo: int = 2, hi: int = 4) -> Dict[str, str]:
    """{정규화 키: 본문에 쓰인 모양}. 앞뒤 낱말이 기능어인 구절은 버린다."""
    words = output_check._EN_WORD_RE.findall(text)
    out: Dict[str, str] = {}
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            g = words[i:i + n]
            if g[0].lower() in _EDGE_STOP or g[-1].lower() in _EDGE_STOP:
                continue
            out.setdefault(" ".join(_norm_word(w) for w in g), " ".join(g))
    return out


def term_variants(pairs: List[Tuple[int, str, str]], exclude_ko=(),
                  min_paras: int = 3, max_terms: int = 40,
                  min_exclusive: float = 0.6) -> List[Dict]:
    """
    pairs: [(단락 i, 원문 평문, 번역 평문)]
    exclude_ko: 이미 Glossary로 고정된 한국어 — 표기가 갈릴 수 없으므로 뺀다.
    반환: [{"ko", "idx": [용어가 나온 단락 i],
            "variants": [{"en", "surfaces", "idx"}], "other": 표기를 못 찾은 단락 수}]
    """
    from collections import Counter

    from services.catalog import suggest_terms_from_texts

    ko_texts = [s for _, s, _ in pairs if contains_korean(s)]
    if not ko_texts:
        return []
    cand = suggest_terms_from_texts(ko_texts, exclude=set(exclude_ko), limit=max_terms)
    if cand.empty:
        return []

    per = [_phrases(o) for _, _, o in pairs]
    doc_df: Counter = Counter()
    for p in per:
        doc_df.update(p.keys())

    found = []
    for ko in cand["KO"]:
        idx = [k for k, (_, s, o) in enumerate(pairs) if ko in s and o.strip()]
        if len(idx) < min_paras:
            continue
        in_term: Counter = Counter()
        for k in idx:
            in_term.update(per[k].keys())
        n_ko = len(ko.split())
        remaining, chosen = set(idx), []
        while remaining and len(chosen) < 4:
            tally: Counter = Counter()
            for k in remaining:
                tally.update(per[k].keys())
            best, best_rank = None, None
            for key, n in tally.items():
                if n < 2 or in_term[key] / doc_df[key] < min_exclusive:
                    continue
                # 용어의 번역은 한국어 낱말 수보다 길지 않은 게 보통이다. 긴 구절은
                # 용어가 아니라 그 단락들에 우연히 겹친 문장 조각("Message window
                # appears")이기 쉽다.
                if len(key.split()) > max(2, n_ko):
                    continue
                if any(key in c or c in key for c, _ in chosen):
                    continue
                # 많이 덮을수록, 다른 단락에 덜 나올수록, 한국어 낱말 수와 비슷할수록
                rank = (n, in_term[key] / doc_df[key], -abs(len(key.split()) - n_ko))
                if best_rank is None or rank > best_rank:
                    best, best_rank = key, rank
            if best is None:
                break
            covered = {k for k in remaining if best in per[k]}
            chosen.append((best, covered))
            remaining -= covered
        if len(chosen) < 2:
            continue
        variants = []
        for key, covered in chosen:
            surf = Counter(per[k][key] for k in covered)
            variants.append({"en": surf.most_common(1)[0][0], "surfaces": sorted(surf),
                             "idx": sorted(pairs[k][0] for k in covered)})
        found.append({"ko": ko, "idx": [pairs[k][0] for k in idx],
                      "variants": variants, "other": len(remaining)})
    return found
