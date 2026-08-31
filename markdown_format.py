"""
마크다운/MDX 읽기·쓰기.

설계 원칙 3가지 — docx 경로에서 검증된 규율을 그대로 옮긴 것:

1. **블록 문법은 LLM에게 주지 않는다.**
   `## `, `- `, `1. `, `> `, `: ` 같은 접두는 유닛 밖(prefix)에 보관하고 안쪽
   텍스트만 번역한다. docx가 XML 구조를 drawing_map/hyperlink_map으로 빼두는
   것과 같은 원리. 모델이 목록 기호를 지우거나 헤딩 레벨을 바꾸는 사고가
   구조적으로 불가능해진다.

2. **AST로 재렌더링하지 않는다.**
   파싱→재출력 방식은 코드펜스·YAML·공백·이스케이프를 전부 정규화해서 문서
   전체를 다시 쓴다. 여기서는 원문 문자열의 **오프셋 구간만 치환**하므로,
   손대지 않은 부분은 바이트 그대로 남는다. `roundtrip_is_identity()`가 이
   성질을 자체 검증한다.

3. **번역 금지 구역은 한국어 판정보다 먼저 봉인한다.**
   `![img](/img/분석실행.png)` 처럼 **경로에 한국어가 든 경우**가 실제로 있다.
   봉인이 늦으면 파일 경로를 번역해버린다.

마커는 translator_engine과 공유한다(아래 상수). tests/test_markdown.py가 두
모듈의 마커 일치를 검증하므로 한쪽만 바뀌면 테스트가 깨진다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── translator_engine과 공유하는 마커 (drift는 테스트가 잡는다) ──────────
B_OPEN = "⟦B⟧"
B_CLOSE = "⟦/B⟧"
I_OPEN = "⟦I⟧"       # 기울임 — 마크다운 전용(Word 경로에는 등장하지 않음)
I_CLOSE = "⟦/I⟧"
D_PREFIX = "⟦D"      # 이미지 — 문장 안에서 위치만 유지
C_PREFIX = "⟦C"      # 리터럴 봉인 — 인라인 코드, 백슬래시 이스케이프, autolink
SUFFIX = "⟧"

_KOREAN_RE = re.compile(r"[가-힣]")

# 번역 대상 front matter 키. 값이 스칼라인 것만 다룬다 — 리스트/중첩 맵은
# 건드리지 않는다(keywords, sidebar_custom_props 등).
FRONT_MATTER_KEYS = ("title", "description", "sidebar_label")

_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
_HEADING_RE = re.compile(r"^(\s{0,3}#{1,6}[ \t]+)(.*?)[ \t]*$")
_EXPLICIT_ID_RE = re.compile(r"\s*\{#[^}]+\}\s*$")
_LIST_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])[ \t]+)(.*)$")
_QUOTE_RE = re.compile(r"^(\s{0,3}(?:>[ \t]?)+)(.*)$")
_DEF_RE = re.compile(r"^(:[ \t]+)(.*)$")          # pandoc definition list
_TABLE_RE = re.compile(r"^\s*\|")
_HTML_RE = re.compile(r"^\s{0,3}<[A-Za-z/!]")
# MDX 전용 — 여러 줄에 걸친 JSX 속성과 import/export 문.
#   <Tabs
#     values={[{label: '설치', ...}]}>
# 처럼 태그가 다음 줄로 이어지면 그 줄들은 문단이 아니라 코드다. 문단으로
# 잡아 번역하면 JSX가 깨진다.
_MDX_STMT_RE = re.compile(r"^\s{0,3}(?:import|export)\s")
_JSX_EXPR_RE = re.compile(r"^\s*[{}]")
_THEMATIC_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_FM_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):[ \t]*(.*)$")


@dataclass
class MdUnit:
    """번역 단위 하나. `start`~`end`가 원문에서 치환될 구간이다."""
    src: str                       # 마크드 텍스트 — LLM이 보는 유일한 것
    start: int
    end: int
    is_heading: bool = False
    kind: str = "para"             # para | heading | list | quote | def | frontmatter
    seals: Dict[str, str] = field(default_factory=dict)
    link_urls: Dict[str, str] = field(default_factory=dict)
    heading_slug: str = ""
    heading_has_id: bool = False
    fm_quote: bool = False         # front matter 값이면 재출력 시 따옴표 필요


# ──────────────────────────────────────────────────────────────────────
# 인라인 봉인 / 복원
# ──────────────────────────────────────────────────────────────────────

def seal_inline(text: str) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    인라인 마크다운을 마커로 바꾼다.

    순서가 중요하다:
      코드 → 이미지 → 링크 → autolink → 이스케이프 → 굵게
    코드를 먼저 봉인해야 `` `**x**` `` 가 굵게로 잡히지 않고, 이미지를 링크보다
    먼저 처리해야 `![alt](p)` 의 `[alt](p)` 부분이 링크로 오인되지 않는다.

    Returns (marked_text, seals, link_urls)
    """
    seals: Dict[str, str] = {}
    link_urls: Dict[str, str] = {}
    counter = {"c": 0, "d": 0, "h": 0}

    def _seal(raw: str, kind: str) -> str:
        i = counter[kind]
        counter[kind] += 1
        token = f"{C_PREFIX if kind == 'c' else D_PREFIX}{i}{SUFFIX}"
        seals[token] = raw
        return token

    # 1) 인라인 코드 — 백틱 개수를 맞춰서 잡는다
    text = re.sub(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)",
                  lambda m: _seal(m.group(0), "c"), text, flags=re.DOTALL)

    # 2) 이미지 — alt는 번역하지 않는다(대부분 파일명이거나 비어 있다)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", lambda m: _seal(m.group(0), "d"), text)

    # 3) 링크 — 텍스트는 번역하고 URL은 봉인
    def _link(m: re.Match) -> str:
        i = counter["h"]
        counter["h"] += 1
        link_urls[str(i)] = m.group(2)
        return f"⟦H{i}⟧{m.group(1)}⟦/H{i}⟧"

    text = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", _link, text)

    # 4) autolink / 원시 URL
    text = re.sub(r"<(?:https?|mailto):[^>]+>", lambda m: _seal(m.group(0), "c"), text)

    # 5) 백슬래시 이스케이프 — `\&` 를 모델이 `&`나 "and"로 바꾸는 걸 막는다
    text = re.sub(r"\\[^\sA-Za-z0-9]", lambda m: _seal(m.group(0), "c"), text)

    # 6) 굵게 — **…** 와 __…__
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"{B_OPEN}{m.group(1)}{B_CLOSE}",
                  text, flags=re.DOTALL)
    text = re.sub(r"(?<![A-Za-z0-9_])__(.+?)__(?![A-Za-z0-9_])",
                  lambda m: f"{B_OPEN}{m.group(1)}{B_CLOSE}", text, flags=re.DOTALL)

    # 7) 기울임 — 굵게를 먼저 마커로 바꿨으므로 남은 * / _ 는 기울임뿐이다.
    #    마커로 감싸두지 않으면 모델이 기호를 떨어뜨려 서식이 사라진다.
    text = re.sub(r"\*(.+?)\*", lambda m: f"{I_OPEN}{m.group(1)}{I_CLOSE}",
                  text, flags=re.DOTALL)
    text = re.sub(r"(?<![A-Za-z0-9_])_(.+?)_(?![A-Za-z0-9_])",
                  lambda m: f"{I_OPEN}{m.group(1)}{I_CLOSE}", text, flags=re.DOTALL)

    return text, seals, link_urls


def unseal_inline(marked: str, seals: Dict[str, str], link_urls: Dict[str, str]) -> str:
    """seal_inline의 역변환 — 마커를 마크다운 문법으로 되돌린다."""
    out = marked

    # 굵게 / 기울임 복원
    out = out.replace(B_OPEN, "**").replace(B_CLOSE, "**")
    out = out.replace(I_OPEN, "*").replace(I_CLOSE, "*")

    # 링크 복원 — ⟦H0⟧텍스트⟦/H0⟧ → [텍스트](url)
    def _unlink(m: re.Match) -> str:
        idx, inner = m.group(1), m.group(2)
        return f"[{inner}]({link_urls.get(idx, '')})"

    out = re.sub(r"⟦H(\d+)⟧(.*?)⟦/H\1⟧", _unlink, out, flags=re.DOTALL)
    # 짝이 깨진 링크 마커는 흔적만 지운다 — 링크는 잃어도 본문은 살린다
    out = re.sub(r"⟦/?H\d+⟧", "", out)

    # 봉인 복원
    for token, raw in seals.items():
        out = out.replace(token, raw)

    return out


# ──────────────────────────────────────────────────────────────────────
# 슬러그 (Docusaurus/github-slugger 호환)
# ──────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """
    헤딩 텍스트 → 앵커 슬러그.

    실제 문서의 `#소스코드-저장소-설정하기` 형태와 일치해야 한다:
    소문자화 → 문장부호 제거 → 공백을 하이픈으로. 한글은 그대로 남는다
    (파이썬 \\w가 한글을 포함).
    """
    s = text.strip().lower()
    s = s.replace(B_OPEN, "").replace(B_CLOSE, "")
    s = re.sub(r"⟦[^⟧]*⟧", "", s)
    s = re.sub(r"[*_`~]", "", s)
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip())
    return s


# ──────────────────────────────────────────────────────────────────────
# 파싱
# ──────────────────────────────────────────────────────────────────────

def _line_spans(text: str) -> List[Tuple[int, int, str]]:
    """(start, end, line) 목록. end는 개행 문자 앞."""
    spans = []
    pos = 0
    for line in text.splitlines():
        spans.append((pos, pos + len(line), line))
        pos += len(line) + 1        # splitlines가 개행을 먹었으므로 +1
    return spans


def _parse_front_matter(text: str, spans: List[Tuple[int, int, str]]) -> Tuple[List[MdUnit], int]:
    """
    YAML front matter에서 번역 대상 값만 유닛으로 뽑는다.

    통째 봉인이 아니라 키 화이트리스트 방식인 이유: `title`/`description`은
    실제로 화면에 보이는 번역 대상이고, `sidebar_position: 1`이나
    `icon: "\\U0001F9E0"` 같은 값은 절대 건드리면 안 되기 때문.

    Returns (units, 본문이 시작하는 줄 인덱스)
    """
    if not spans or spans[0][2].strip() != "---":
        return [], 0

    close = None
    for i in range(1, len(spans)):
        if spans[i][2].strip() in ("---", "..."):
            close = i
            break
    if close is None:
        return [], 0

    units: List[MdUnit] = []
    for i in range(1, close):
        start, end, line = spans[i]
        if line[:1] in (" ", "\t"):
            continue                       # 중첩 값은 건드리지 않는다
        m = _FM_KEY_RE.match(line)
        if not m or m.group(1) not in FRONT_MATTER_KEYS:
            continue
        raw_value = m.group(2)
        if not raw_value or not _KOREAN_RE.search(raw_value):
            continue

        value_start = start + line.index(":") + 1
        lead = len(line[line.index(":") + 1:]) - len(raw_value.lstrip())
        value_start += lead
        value = raw_value.strip()

        quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"
        if quoted:
            value = value[1:-1]

        marked, seals, urls = seal_inline(value)
        units.append(MdUnit(
            src=marked,
            start=value_start,
            end=end,
            kind="frontmatter",
            seals=seals,
            link_urls=urls,
            fm_quote=quoted,    # 원래 인용돼 있었으면 유지
        ))

    return units, close + 1


def _inner_span(line_start: int, line: str, prefix_len: int) -> Tuple[int, int, str]:
    """
    접두를 뺀 '알맹이'의 정확한 구간을 돌려준다.

    앞뒤 공백까지 구간에서 제외해야 치환 후에도 원문의 들여쓰기·후행 공백이
    그대로 남는다 (roundtrip_is_identity가 이걸 검증한다).
    """
    rest = line[prefix_len:]
    lead = len(rest) - len(rest.lstrip())
    inner = rest.strip()
    s = line_start + prefix_len + lead
    return s, s + len(inner), inner


def parse_markdown(text: str) -> List[MdUnit]:
    """마크다운 원문을 번역 단위 목록으로 분해한다."""
    spans = _line_spans(text)
    units, i = _parse_front_matter(text, spans)

    in_fence: Optional[str] = None
    n = len(spans)

    while i < n:
        start, end, line = spans[i]
        stripped = line.strip()

        # 코드펜스 — 열림/닫힘 사이는 통째로 건너뛴다
        fence = _FENCE_RE.match(line)
        if in_fence:
            if fence and fence.group(1)[0] == in_fence[0] and len(fence.group(1)) >= len(in_fence):
                in_fence = None
            i += 1
            continue
        if fence:
            in_fence = fence.group(1)
            i += 1
            continue

        if not stripped or _THEMATIC_RE.match(line):
            i += 1
            continue

        # 표는 v1 미지원 — 손대지 않고 그대로 둔다
        if _TABLE_RE.match(line):
            i += 1
            continue

        # import/export 문과 JSX 표현식 줄
        if _MDX_STMT_RE.match(line) or _JSX_EXPR_RE.match(line):
            i += 1
            continue

        # HTML/JSX 태그 — 태그가 그 줄에서 닫히지 않으면(여러 줄 속성) 닫힐
        # 때까지 건너뛴다. 태그 사이의 본문은 마크다운이므로 계속 번역한다.
        if _HTML_RE.match(line):
            while i < n and not spans[i][2].rstrip().endswith((">", "/>")):
                i += 1
            i += 1
            continue

        # 인용 접두는 벗겨내고 안쪽을 다시 판별한다
        prefix = ""
        body = line
        mq = _QUOTE_RE.match(line)
        if mq:
            prefix = mq.group(1)
            body = mq.group(2)
            if not body.strip():
                i += 1
                continue

        mh = _HEADING_RE.match(body)
        if mh:
            head_prefix = mh.group(1)
            t_start, t_end, htext = _inner_span(start + len(prefix), body, len(head_prefix))
            # 이미 명시적 id가 붙어 있으면 그 부분은 구간 밖으로 밀어낸다
            id_m = _EXPLICIT_ID_RE.search(htext)
            has_id = bool(id_m)
            if id_m:
                htext = htext[: id_m.start()].rstrip()
                t_end = t_start + len(htext)
            marked, seals, urls = seal_inline(htext)
            units.append(MdUnit(
                src=marked,
                start=t_start,
                end=t_end,
                is_heading=True,
                kind="heading",
                seals=seals,
                link_urls=urls,
                heading_slug=slugify(htext),
                heading_has_id=has_id,
            ))
            i += 1
            continue

        ml = _LIST_RE.match(body)
        md_ = None if ml else _DEF_RE.match(body)
        if ml or md_:
            item_prefix = (ml or md_).group(1)
            t_start, t_end, itext = _inner_span(start + len(prefix), body, len(item_prefix))
            marked, seals, urls = seal_inline(itext)
            units.append(MdUnit(
                src=marked,
                start=t_start,
                end=t_end,
                kind="list" if ml else "def",
                seals=seals,
                link_urls=urls,
            ))
            i += 1
            continue

        if prefix:
            # 인용 안의 본문 — 접두가 줄마다 붙으므로 한 줄씩 다룬다
            t_start, t_end, qtext = _inner_span(start + len(prefix), body, 0)
            marked, seals, urls = seal_inline(qtext)
            units.append(MdUnit(
                src=marked, start=t_start, end=t_end, kind="quote",
                seals=seals, link_urls=urls,
            ))
            i += 1
            continue

        # 일반 문단 — 이어지는 평문 줄을 하나로 묶는다. 마크다운에서 문단 안의
        # 개행은 줄바꿈이 아니라 소스 줄바꿈일 뿐이므로, 한 덩어리로 번역하고
        # 한 줄로 되돌린다(하드랩된 문서는 이 지점에서 한 줄로 합쳐진다).
        j = i + 1
        while j < n:
            _, _, nxt = spans[j]
            if not nxt.strip():
                break
            if (_FENCE_RE.match(nxt) or _HEADING_RE.match(nxt) or _LIST_RE.match(nxt)
                    or _QUOTE_RE.match(nxt) or _DEF_RE.match(nxt)
                    or _TABLE_RE.match(nxt) or _HTML_RE.match(nxt)
                    or _THEMATIC_RE.match(nxt) or _MDX_STMT_RE.match(nxt)
                    or _JSX_EXPR_RE.match(nxt)):
                break
            j += 1

        t_start, _, _ = _inner_span(start, line, 0)
        _, t_end, _ = _inner_span(spans[j - 1][0], spans[j - 1][2], 0)
        para_text = " ".join(spans[k][2].strip() for k in range(i, j))
        marked, seals, urls = seal_inline(para_text)
        units.append(MdUnit(
            src=marked, start=t_start, end=t_end, kind="para",
            seals=seals, link_urls=urls,
        ))
        i = j

    return units


# ──────────────────────────────────────────────────────────────────────
# 쓰기
# ──────────────────────────────────────────────────────────────────────

def _quote_yaml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _needs_quote(value: str) -> bool:
    """
    YAML 스칼라로 그냥 둬도 안전한가.

    번역문에 콜론이 들어가는 경우가 실제로 흔하다 — `description: Analysis:
    let's begin.` 은 YAML 파싱 에러다. 반대로 멀쩡한 값까지 전부 인용하면
    원문에 없던 따옴표가 diff로 남으므로 필요할 때만 감싼다.
    """
    if not value:
        return True
    if ": " in value or value.endswith(":") or " #" in value:
        return True
    if value[0] in "-?:,[]{}#&*!|>'\"%@`":
        return True
    if value.strip() != value:
        return True
    return False


def render_unit(unit: MdUnit, translated_marked: str, keep_anchor: bool = True) -> str:
    """번역된 마크드 텍스트를 원문에 다시 넣을 문자열로 만든다."""
    out = unseal_inline(translated_marked, unit.seals, unit.link_urls)
    out = out.strip()

    if unit.kind == "frontmatter":
        return _quote_yaml(out) if (unit.fm_quote or _needs_quote(out)) else out

    if unit.kind == "heading":
        # 원래 슬러그를 명시적 id로 박아둔다 — 다른 문서에서 걸어둔
        # `#소스코드-저장소-설정하기` 같은 앵커가 헤딩 번역 후에도 살아남는다.
        if keep_anchor and not unit.heading_has_id and unit.heading_slug:
            if slugify(out) != unit.heading_slug:
                out = f"{out} {{#{unit.heading_slug}}}"

    return out


def apply_translations(text: str, pairs: List[Tuple[MdUnit, str]],
                       keep_anchor: bool = True) -> str:
    """
    (유닛, 번역결과) 목록을 원문에 반영한다.

    뒤에서 앞으로 치환해야 앞쪽 오프셋이 밀리지 않는다.
    """
    out = text
    for unit, translated in sorted(pairs, key=lambda p: p[0].start, reverse=True):
        replacement = render_unit(unit, translated, keep_anchor=keep_anchor)
        out = out[:unit.start] + replacement + out[unit.end:]
    return out


def roundtrip_is_identity(text: str) -> bool:
    """
    자체 검증: 모든 유닛을 '봉인했다가 그대로 복원'해서 되돌리면 원문과
    같아야 한다. 파서가 구간을 잘못 잡고 있으면 여기서 걸린다.
    """
    units = parse_markdown(text)
    pairs = [(u, u.src) for u in units]
    return apply_translations(text, pairs, keep_anchor=False) == text


# ──────────────────────────────────────────────────────────────────────
# 파일 입출력 — 인코딩과 개행을 원본 그대로 보존
# ──────────────────────────────────────────────────────────────────────

def read_text(path: str) -> Tuple[str, str, str, bool]:
    """
    Returns (text, encoding, newline, had_bom)

    Windows에서 만든 문서라도 Docusaurus 저장소는 LF인 경우가 많다. 감지한
    개행을 그대로 돌려주어 저장 시 재현한다 — 안 그러면 diff가 전 파일로 번진다.
    """
    with open(path, "rb") as f:
        raw = f.read()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    if had_bom:
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("cp949")
        encoding = "cp949"

    crlf = text.count("\r\n")
    newline = "\r\n" if crlf and crlf >= text.count("\n") - crlf else "\n"
    text = text.replace("\r\n", "\n")
    return text, encoding, newline, had_bom


def write_text(path: str, text: str, encoding: str, newline: str, had_bom: bool) -> None:
    data = text.replace("\n", newline).encode(encoding)
    if had_bom:
        data = b"\xef\xbb\xbf" + data
    with open(path, "wb") as f:
        f.write(data)
