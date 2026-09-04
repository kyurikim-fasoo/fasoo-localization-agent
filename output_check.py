"""
번역 산출물 검증 — 저장이 끝난 .docx를 원본과 대조한다.

왜 필요한가:
    지금까지 번역 파이프라인은 "LLM이 시킨 대로 했겠지"를 전제로 후처리만
    했다. 그 결과 손상된 마커가 그대로 Word에 찍히고, 버전 문자열이 지워지고,
    하이라이트가 사라져도 아무도 알아채지 못했다. 프롬프트를 아무리 다듬어도
    LLM이 규칙을 어기는 일은 없어지지 않는다. 그러니 **틀린 결과를 코드가
    잡아내는** 층이 먼저 있어야 한다.

이 모듈은 읽기 전용이다. 문서를 고치지 않고 무엇이 어긋났는지만 말한다.
번역 파이프라인 바깥에서 독립적으로 돌 수 있어야 진단 도구로 쓸모가 있다.

사용:
    python output_check.py 원본.docx 번역본.docx
    python output_check.py 원본.docx 번역본.docx --dnt Fireside,Wrapsody
"""
from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

_KOREAN_RE = re.compile(r"[가-힣]")
# v2.1 / v1.7.1 처럼 점이 있는 버전 표기. "v2" 단독은 일반 문장에도 나오므로
# 오탐을 피해 점이 하나 이상 있는 것만 본다.
_VERSION_RE = re.compile(r"\bv\d+(?:\.\d+)+\b", re.IGNORECASE)
# 마커 잔재 — 정상 산출물에는 이 문자들이 있을 수 없다.
_MARKER_CHARS = ("⟦", "⟧")

# 결측값이 문자열로 새어 나온 것. 글로서리/UI 매핑의 빈 셀이 pandas NaN으로
# 넘어와 str(NaN)="nan"이 치환어로 등록되면 본문에 그대로 찍힌다.
# 사람이 읽으면 즉시 이상하지만 마커 검사로는 절대 안 걸린다.
_NULLISH_RE = re.compile(r"(?<![A-Za-z0-9])(nan|none|null|undefined|NaT)"
                         r"(?![A-Za-z0-9])")
# 마커를 걷어낸 자리에서 낱말이 붙어버린 흔적.
#   websiteGo / syncClick / optionsto / toEdit
# 소문자 뒤에 대문자가 오는 자리는 영어에서 거의 항상 오타다. camelCase 식별자
# (appKey, serverId)와 겹치므로 흔한 것은 예외로 둔다.
# 대문자 뒤 소문자는 1자만 있어도 잡아야 한다 — "websiteGo"가 그 형태다.
# 소문자끼리 붙은 "optionsto"는 대소문자 신호가 없어 이 방법으로 못 잡는다.
# 그쪽은 normalize_marker_boundary_spaces()가 애초에 만들지 않도록 막는다.
_GLUED_RE = re.compile(r"[a-z]{2,}[A-Z][a-z]+")
_GLUE_ALLOW = {
    "appKey", "serverId", "userId", "roomId", "javaScript", "iPhone",
    "macOS", "iOS", "openAI", "chatBot", "wrapSody",
}

# 본문에 남아도 되는 한국어 — 회사명·제품명 등. 필요하면 --dnt로 추가한다.
DEFAULT_DNT: Tuple[str, ...] = ()


@dataclass
class Finding:
    level: str            # 오류 | 경고
    title: str
    detail: str = ""

    def __str__(self) -> str:
        icon = {"오류": "✗", "경고": "!"}.get(self.level, "·")
        return f"  {icon} [{self.level}] {self.title}" + (
            f"\n      {self.detail}" if self.detail else ""
        )


@dataclass
class DocStats:
    """문서 하나에서 뽑은 구조 지표. 두 문서를 이걸로 비교한다."""
    name: str
    texts: List[str] = field(default_factory=list)
    # 문단 단위 텍스트. 원본과 번역본은 문단이 1:1로 보존되므로 인덱스만으로
    # 짝이 맞는다 — 용어별 번역 변형을 찾는 데 이 정렬을 쓴다.
    para_texts: List[str] = field(default_factory=list)
    paragraphs: int = 0
    hyperlinks: int = 0
    drawings: int = 0
    breaks: int = 0
    tabs: int = 0
    # 하이라이트는 run 개수로 재면 안 된다. 번역은 한 문단을 더 적은 run으로
    # 다시 쓰므로, 하이라이트가 온전히 살아 있어도 run 수는 크게 준다
    # (실측: 5,161 → 1,131). 보존 여부는 **문단 수**로 봐야 한다.
    highlight_paras: int = 0
    highlight_chars: int = 0
    highlight_colors: Counter = field(default_factory=Counter)
    comments: int = 0
    parts: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.texts)


def _body_parts(zf: zipfile.ZipFile) -> List[str]:
    """
    본문 텍스트가 들어갈 수 있는 모든 파트.

    document.xml만 보면 머리글·바닥글·각주에 남은 한국어나 깨진 마커를
    놓친다. 실제로 번역 순회(iter_all_paragraphs)가 이 영역을 건드리지
    않으므로, 검증기는 반드시 여기까지 봐야 한다.
    """
    keep = []
    for n in zf.namelist():
        if not n.startswith("word/") or not n.endswith(".xml"):
            continue
        base = n[len("word/"):]
        if base == "document.xml" or base.startswith(("header", "footer")) \
                or base in ("footnotes.xml", "endnotes.xml"):
            keep.append(n)
    return sorted(keep)


def collect(path: str) -> DocStats:
    """docx 하나를 열어 구조 지표를 센다."""
    st = DocStats(name=path)
    with zipfile.ZipFile(path) as zf:
        st.parts = _body_parts(zf)
        for part in st.parts:
            root = etree.fromstring(zf.read(part))
            for p in root.iter(W + "p"):
                st.para_texts.append(
                    "".join(t.text or "" for t in p.iter(W + "t")).strip()
                )
                hit = False
                for r in p.iter(W + "r"):
                    hl = r.find(f"{W}rPr/{W}highlight")
                    if hl is None:
                        continue
                    val = hl.get(W + "val")
                    if not val or val == "none":
                        continue
                    hit = True
                    st.highlight_colors[val] += 1
                    st.highlight_chars += sum(
                        len(t.text or "") for t in r.iter(W + "t")
                    )
                if hit:
                    st.highlight_paras += 1
            for el in root.iter():
                tag = el.tag
                if tag == W + "t" and el.text:
                    st.texts.append(el.text)
                elif tag == W + "p":
                    st.paragraphs += 1
                elif tag == W + "hyperlink":
                    st.hyperlinks += 1
                elif tag == W + "drawing":
                    st.drawings += 1
                elif tag == W + "br":
                    st.breaks += 1
                elif tag == W + "tab":
                    # <w:tab/> (run 안의 탭 문자)만 센다. 문단 속성의
                    # <w:tabs><w:tab .../></w:tabs> 는 탭 '정지 위치' 정의라
                    # 본문 문자가 아니다.
                    if el.getparent() is not None and \
                            el.getparent().tag != W + "tabs":
                        st.tabs += 1
        if "word/comments.xml" in zf.namelist():
            croot = etree.fromstring(zf.read("word/comments.xml"))
            st.comments = len(croot.findall(f".//{W}comment"))
    return st


# ──────────────────────────────────────────────────────────────────────
# 용어별 영문 변형 검사
#
# 리포트가 지적한 가장 고치기 어려운 문제 — 같은 "데이터 기반 답변 에이전트"가
#     data-driven response agent / data-based answer agent /
#     data-driven answering agent …
# 다섯 가지로 번역됐다. 문단을 독립적으로 번역하니 당연한 결과다.
#
# 검출은 LLM 없이 된다. 원본과 번역본의 문단이 1:1로 보존되므로(문단 수 검사가
# 이를 보장한다) 인덱스로 짝을 맞출 수 있다. 어떤 한국어 용어가 든 문단들을
# 모아 그 번역문에서 공통으로 나타나는 영어 표현을 찾으면, 그게 곧 그 용어의
# 번역이다. 공통 표현이 일부 문단에만 있다면 나머지는 다르게 번역된 것이다.
# ──────────────────────────────────────────────────────────────────────

# 영어 불용어 — 이것만으로 이뤄진 n-gram은 용어의 번역일 리 없다.
_EN_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "be", "you", "can", "will", "it", "this", "that", "with", "from", "by",
    "at", "as", "if", "not", "click", "select", "enter", "see", "your",
}
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
# 이 비율 미만으로만 공통 표현이 나타나면 표기가 갈린 것으로 본다.
CONSISTENCY_MIN_COVERAGE = 0.7


def _en_ngrams(text: str, lo: int = 2, hi: int = 4):
    """
    2단어 이상만 본다. 1단어를 섞으면 "agent"나 "data-driven"처럼 **모든
    변형에 공통인 조각**이 100% 커버리지로 1등을 차지해 불일치가 통째로
    묻힌다. 실제로 data-driven response agent / data-driven answer agent가
    갈려 있는데도 "agent" 하나 때문에 정상으로 보였다.
    """
    words = _EN_WORD_RE.findall(text.lower())
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            g = words[i:i + n]
            if all(w in _EN_STOP for w in g):
                continue
            yield " ".join(g)


def check_term_consistency(src: DocStats, out: DocStats,
                           min_paras: int = 3,
                           max_terms: int = 40) -> List[Tuple[str, float, List[str]]]:
    """
    한국어 용어별로 영문 표기가 갈렸는지 본다.

    반환: (한국어 용어, 최빈 표기의 등장 비율, 경쟁 표기 상위 목록)
    비율이 낮을수록 표기가 갈렸다는 뜻이다.
    """
    if len(src.para_texts) != len(out.para_texts):
        return []                     # 문단이 어긋나면 짝을 믿을 수 없다
    try:
        from services.catalog import suggest_terms_from_texts
    except Exception:
        return []

    ko_paras = [t for t in src.para_texts if _KOREAN_RE.search(t)]
    cand = suggest_terms_from_texts(ko_paras, limit=max_terms)
    if cand.empty:
        return []

    findings = []
    for term in cand["KO"]:
        idx = [i for i, t in enumerate(src.para_texts) if term in t]
        pool = [out.para_texts[i] for i in idx if out.para_texts[i].strip()]
        if len(pool) < min_paras:
            continue
        # 각 번역문에 어떤 표현이 있는지 (문단당 중복은 한 번만)
        seen_per_para = [set(_en_ngrams(t)) for t in pool]
        tally: Counter = Counter()
        for sset in seen_per_para:
            tally.update(sset)
        if not tally:
            continue
        # 가장 넓게 퍼진 표현을 그 용어의 번역으로 본다. 짧은 조각이 우연히
        # 널리 퍼지므로, 같은 커버리지면 긴 쪽을 택한다.
        best = max(tally.items(), key=lambda kv: (kv[1], len(kv[0].split())))
        # 근거가 너무 얇으면(2개 문단에만 등장) 판단을 보류한다
        if best[1] < 2:
            continue
        coverage = best[1] / len(pool)
        if coverage >= CONSISTENCY_MIN_COVERAGE:
            continue
        rivals = [g for g, n in tally.most_common(6)
                  if len(g.split()) >= 2 and n >= 2]
        findings.append((term, coverage, rivals[:4]))

    findings.sort(key=lambda f: f[1])
    return findings


def _count_check(findings: List[Finding], label: str,
                 a: int, b: int, level: str = "오류") -> None:
    if a != b:
        findings.append(Finding(
            level, f"{label} 개수가 다릅니다",
            f"원본 {a:,}개 → 번역본 {b:,}개 ({b - a:+,})",
        ))


def verify(src_path: str, out_path: str,
           dnt: Sequence[str] = DEFAULT_DNT,
           literals: Sequence[str] = ()) -> Tuple[List[Finding], DocStats, DocStats]:
    """
    원본과 번역본을 대조해 문제 목록을 돌려준다.

    dnt      : 번역본에 한국어로 남아 있어도 되는 표기 (제품명 등)
    literals : 원본에 있었다면 번역본에도 그대로 있어야 하는 영문 리터럴
    """
    src, out = collect(src_path), collect(out_path)
    f: List[Finding] = []

    # 1. 마커 잔재 — 있으면 무조건 오류다. 정상 문서에는 나올 수 없는 문자.
    stray = [t for t in out.texts if any(c in t for c in _MARKER_CHARS)]
    if stray:
        f.append(Finding(
            "오류", f"마커가 문서에 그대로 찍혔습니다 ({len(stray)}곳)",
            " / ".join(t.strip()[:60] for t in stray[:3]),
        ))

    # 2. 번역되지 않은 한국어
    dnt_set = [d for d in dnt if d]
    left = []
    for t in out.texts:
        probe = t
        for d in dnt_set:
            probe = probe.replace(d, "")
        if _KOREAN_RE.search(probe):
            left.append(t.strip())
    if left:
        f.append(Finding(
            "오류", f"한국어가 남아 있습니다 ({len(left)}곳)",
            " / ".join(x[:50] for x in left[:3])
            + ("  ※ 제품명이라면 --dnt 로 제외하세요" if len(left) <= 5 else ""),
        ))

    # 3. 구조 요소 — 번역이 개수를 바꿔서는 안 된다
    _count_check(f, "하이퍼링크", src.hyperlinks, out.hyperlinks)
    _count_check(f, "이미지", src.drawings, out.drawings)
    _count_check(f, "줄바꿈", src.breaks, out.breaks)
    _count_check(f, "탭", src.tabs, out.tabs)
    _count_check(f, "댓글", src.comments, out.comments)

    # 4. 하이라이트 — 문단 단위로 본다 (run 수는 번역이 합쳐서 의미 없음)
    if out.highlight_paras < src.highlight_paras:
        f.append(Finding(
            "오류", "하이라이트가 사라진 문단이 있습니다",
            f"원본 {src.highlight_paras:,}개 문단 → 번역본 "
            f"{out.highlight_paras:,}개 ({out.highlight_paras - src.highlight_paras:+,})",
        ))
    # 색상 종류가 바뀌는 것도 문제 — 개수는 무시하고 어떤 색이 있었는지만
    _sc_colors, _oc_colors = set(src.highlight_colors), set(out.highlight_colors)
    if _sc_colors - _oc_colors:
        f.append(Finding(
            "오류", "하이라이트 색이 통째로 사라졌습니다",
            ", ".join(sorted(_sc_colors - _oc_colors)),
        ))

    # 5. 버전 표기 — 후처리가 v2.1을 삼킨 적이 있어 따로 본다.
    #    "사라졌는가"와 "대소문자가 바뀌었는가"는 심각도가 다르므로 나눠서 본다.
    #    (v2.1 -> V2.1 은 문장 첫 글자 대문자화가 개정 이력 셀까지 건드린 것)
    sv = Counter(m.lower() for m in _VERSION_RE.findall(src.text))
    ov = Counter(m.lower() for m in _VERSION_RE.findall(out.text))
    missing = sorted((sv - ov).elements())
    if missing:
        f.append(Finding(
            "오류", f"버전 표기가 사라졌습니다 ({len(missing)}건)",
            ", ".join(dict.fromkeys(missing)),
        ))
    recased = sorted(
        set(_VERSION_RE.findall(src.text)) - set(_VERSION_RE.findall(out.text))
        - {m for m in _VERSION_RE.findall(src.text) if m.lower() in missing}
    )
    if recased:
        f.append(Finding(
            "경고", f"버전 표기의 대소문자가 바뀌었습니다 ({len(recased)}건)",
            ", ".join(recased) + " — 원문 표기를 그대로 두는 편이 좋습니다",
        ))

    # 6. 보존돼야 할 영문 리터럴 (제품명·약어 등)
    for lit in literals:
        if not lit:
            continue
        a, b = src.text.count(lit), out.text.count(lit)
        if a and b < a:
            f.append(Finding(
                "경고", f"원문 영문 «{lit}» 가 줄었습니다",
                f"원본 {a}회 → 번역본 {b}회",
            ))

    # 7. 문단이 통째로 사라지지 않았는지.
    #    run(w:t) 개수로 보면 안 된다 — 번역은 한 문단을 더 적은 run으로 다시
    #    쓰므로 정상적으로도 크게 준다. 문단(w:p)은 1:1로 보존돼야 한다.
    _count_check(f, "문단", src.paragraphs, out.paragraphs)

    # 8. 내용이 통째로 날아갔는지 — 글자 수가 절반 밑이면 의심한다.
    #    한→영은 보통 늘어나므로 크게 주는 것은 정상이 아니다.
    _sc, _oc = len(src.text), len(out.text)
    if _sc and _oc < _sc * 0.5:
        f.append(Finding(
            "경고", "번역본 글자 수가 원본의 절반 미만입니다",
            f"원본 {_sc:,}자 → 번역본 {_oc:,}자 · 내용 누락일 수 있습니다",
        ))

    # 10. 결측값이 본문에 찍혔는가 — 무조건 차단해야 하는 부류
    _nullish = []
    for t in out.para_texts:
        for m in _NULLISH_RE.finditer(t):
            _nullish.append((m.group(0), t.strip()[:60]))
    if _nullish:
        f.append(Finding(
            "오류", f"결측값이 본문에 찍혔습니다 ({len(_nullish)}곳)",
            " / ".join(f"«{w}» {ctx}" for w, ctx in _nullish[:3])
            + "  ※ 글로서리·UI 매핑의 빈 셀이 문자열로 새어 나온 것입니다",
        ))

    # 11. 마커 자리에서 낱말이 붙었는가
    _glued = []
    for t in out.para_texts:
        for m in _GLUED_RE.finditer(t):
            w = m.group(0)
            if w in _GLUE_ALLOW or w.lower() in {a.lower() for a in _GLUE_ALLOW}:
                continue
            _glued.append((w, t.strip()[:60]))
    if _glued:
        f.append(Finding(
            "오류", f"낱말이 붙어 있습니다 ({len(_glued)}곳)",
            " / ".join(f"«{w}»" for w, _ in _glued[:6])
            + "  ※ 마커 경계에서 공백이 빠진 자리입니다",
        ))

    # 9. 용어별 영문 표기가 갈렸는가 (문단 1:1 정렬이 성립할 때만)
    for term, cov, rivals in check_term_consistency(src, out):
        f.append(Finding(
            "경고", f"«{term}» 의 영문 표기가 갈립니다",
            f"가장 흔한 표현도 {cov * 100:.0f}%에만 등장"
            + (f" · 경쟁 표기: {', '.join(rivals)}" if rivals else ""),
        ))

    return f, src, out


def check_duplicate_targets(pairs: Sequence[Tuple[str, str]]
                            ) -> List[Tuple[str, List[str]]]:
    """
    서로 다른 원문이 같은 영문으로 매핑됐는가.

    "상태"와 "사용 여부"가 둘 다 Status로 등록되면 한 목록 안에
    "Status, and Status"가 나온다. 사람이 목록을 읽어도 구분할 수 없으니
    데이터 단계에서 막아야 한다.
    """
    by_en: Dict[str, List[str]] = {}
    for ko, en in pairs:
        ko, en = (ko or "").strip(), (en or "").strip()
        if not ko or not en:
            continue
        by_en.setdefault(en.lower(), [])
        if ko not in by_en[en.lower()]:
            by_en[en.lower()].append(ko)
    return [(en, kos) for en, kos in sorted(by_en.items()) if len(kos) > 1]


def format_report(findings: List[Finding], src: DocStats, out: DocStats) -> str:
    lines = [
        "=" * 66,
        f"원본   {src.name}",
        f"번역본 {out.name}",
        "=" * 66,
        f"{'항목':<12}{'원본':>10}{'번역본':>10}",
        "-" * 66,
    ]
    for label, a, b in (
        ("문단", src.paragraphs, out.paragraphs),
        ("글자 수", len(src.text), len(out.text)),
        ("하이퍼링크", src.hyperlinks, out.hyperlinks),
        ("이미지", src.drawings, out.drawings),
        ("줄바꿈", src.breaks, out.breaks),
        ("탭", src.tabs, out.tabs),
        ("하이라이트 문단", src.highlight_paras, out.highlight_paras),
        ("하이라이트 글자", src.highlight_chars, out.highlight_chars),
        ("댓글", src.comments, out.comments),
    ):
        mark = "" if a == b else "   ←"
        lines.append(f"{label:<12}{a:>10,}{b:>10,}{mark}")
    lines.append("-" * 66)
    if not findings:
        lines.append("문제 없음.")
    else:
        n_err = sum(1 for x in findings if x.level == "오류")
        lines.append(f"오류 {n_err}건 · 경고 {len(findings) - n_err}건")
        lines.append("")
        lines.extend(str(x) for x in findings)
    lines.append("=" * 66)
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 2
    dnt: List[str] = []
    literals: List[str] = []
    for a in argv:
        if a.startswith("--dnt="):
            dnt = [x.strip() for x in a[len("--dnt="):].split(",")]
        elif a.startswith("--literal="):
            literals = [x.strip() for x in a[len("--literal="):].split(",")]
    findings, src, out = verify(args[0], args[1], dnt=dnt, literals=literals)
    print(format_report(findings, src, out))
    return 1 if any(x.level == "오류" for x in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
