import copy
import os
import re
import tempfile
import zipfile
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional, Callable, Any

from docx import Document
from lxml import etree
from openai import OpenAI

# OOXML namespace constants
_W        = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_XML_SPACE = "http://www.w3.org/XML/1998/namespace"


TOTAL_INPUT_TOKENS = 0
TOTAL_CACHED_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_TOKENS = 0

B_OPEN = "⟦B⟧"
B_CLOSE = "⟦/B⟧"
BR_MARKER = "⟦BR⟧"   # soft line break — Word `<w:br/>` element

G_PREFIX = "⟦G"
D_PREFIX = "⟦D"
SUFFIX = "⟧"

KOREAN_RE = re.compile(r"[가-힣]")
PLACEHOLDER_RE = re.compile(r"⟦G\d+⟧")
DRAWING_PH_RE = re.compile(r"⟦D\d+⟧")
H_OPEN_RE    = re.compile(r"⟦H(\d+)⟧")
H_CLOSE_RE   = re.compile(r"⟦/H(\d+)⟧")
HL_OPEN_RE   = re.compile(r"⟦HL:([a-zA-Z]+)⟧")
HL_CLOSE     = "⟦/HL⟧"
BR_MARKER_RE = re.compile(re.escape(BR_MARKER))
ALL_MARKER_RE = re.compile(r"(⟦B⟧|⟦/B⟧|⟦D\d+⟧|⟦H\d+⟧|⟦/H\d+⟧|⟦HL:[a-zA-Z]+⟧|⟦/HL⟧|⟦BR⟧)")
MARKER_SPLIT_RE = re.compile(rf"({re.escape(B_OPEN)}|{re.escape(B_CLOSE)}|⟦G\d+⟧)")

UI_LOWER_WORDS = {
    "name",
    "names",
    "list",
    "lists",
    "detail",
    "details",
    "setting",
    "settings",
    "information",
    "field",
    "fields",
    "filter",
    "filters",
    "menu",
    "menus",
    "tab",
    "tabs",
    "status",
    "type",
    "types",
    "history",
    "option",
    "options",
    "message",
    "messages",
    "group",
    "groups",
    "owner",
    "owners",
    "user",
    "users",
    "rule",
    "rules",
    "policy",
    "policies",
    "log",
    "logs",
    "guideline",
    "guidelines",
    "pattern",
    "patterns",
    "tag",
    "tags",
    "value",
    "values",
    "result",
    "results",
    "data",
    "items",
    "item",
    "dialog",
    "window",
    "button",
    "buttons",
    "criteria",
    "class",
    "level",
}


@dataclass(frozen=True)
class GlossaryEntry:
    ko: str
    en: str
    dnt: bool
    case_sensitive: bool
    product: str
    note: str


def reset_token_counters():
    global TOTAL_INPUT_TOKENS, TOTAL_CACHED_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_TOKENS
    TOTAL_INPUT_TOKENS = 0
    TOTAL_CACHED_INPUT_TOKENS = 0
    TOTAL_OUTPUT_TOKENS = 0
    TOTAL_TOKENS = 0


def contains_korean(text: str) -> bool:
    # Strip drawing placeholders before checking — they are not translatable text
    clean = DRAWING_PH_RE.sub("", text) if text else text
    return bool(clean and KOREAN_RE.search(clean))


def make_marker(prefix: str, i: int) -> str:
    return f"{prefix}{i}{SUFFIX}"


def _to_bool(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"true", "y", "yes", "1"}


def _clean(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def _cap_first_alpha(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


def build_glossary_entries_from_rows(rows: List[dict]) -> List[GlossaryEntry]:
    entries: List[GlossaryEntry] = []

    for r in rows:
        ko = _clean(r.get("KO"))
        en = _clean(r.get("EN"))
        if not ko or not en:
            continue

        entries.append(
            GlossaryEntry(
                ko=ko,
                en=en,
                dnt=_to_bool(r.get("DNT")),
                case_sensitive=_to_bool(r.get("Case-sensitive")),
                product=_clean(r.get("Product")),
                note=_clean(r.get("Note")),
            )
        )

    deduped: List[GlossaryEntry] = []
    seen = set()
    for e in entries:
        key = (e.ko, e.en, e.product, e.note, e.dnt, e.case_sensitive)
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    deduped.sort(key=lambda e: len(e.ko), reverse=True)
    return deduped


def build_pattern_pairs_from_rows(rows: List[dict]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()

    for r in rows:
        ko = _clean(r.get("KO"))
        en = _clean(r.get("EN"))
        if not ko or not en:
            continue

        key = (ko, en)
        if key not in seen:
            seen.add(key)
            pairs.append((ko, en))

    return pairs


def preprocess_with_glossary_placeholders(
    text: str,
    entries: List[GlossaryEntry],
) -> Tuple[str, Dict[str, GlossaryEntry]]:
    """
    Replace glossary KO terms in *text* with ⟦G0⟧ … placeholders.

    Uses a regex that normalises internal whitespace (\s+) so that terms
    like '민감 정보' still match even when the paragraph runs were merged
    with a different number of spaces (or a zero-width space, thin space, etc.).
    Longer KO terms are tried first (entries are pre-sorted by length desc)
    to avoid a short term swallowing part of a longer one.
    """
    out = text
    mapping: Dict[str, GlossaryEntry] = {}
    idx = 0

    for entry in entries:
        if not entry.ko:
            continue

        # Build a pattern that matches the KO term with flexible internal whitespace.
        # Each space in the glossary key becomes \s+ so ' 민감 정보 ' matches
        # '민감정보', '민감  정보', etc.
        escaped_parts = [re.escape(part) for part in entry.ko.split()]
        pattern = r"\s*".join(escaped_parts) if escaped_parts else re.escape(entry.ko)

        # Quick pre-check: at least the first word must be present
        if escaped_parts and not re.search(escaped_parts[0], out):
            continue

        if re.search(pattern, out):
            ph = make_marker(G_PREFIX, idx)
            idx += 1
            out = re.sub(pattern, ph, out)
            mapping[ph] = entry

    return out, mapping


def preprocess_ui_overrides_in_bold(
    text: str,
    ui_overrides: Dict[str, str],
    start_idx: int = 0,
) -> Tuple[str, Dict[str, GlossaryEntry], int]:
    """
    Replace UI text-mapping KO terms with ⟦G#⟧ placeholders **only within
    ⟦B⟧…⟦/B⟧ (bold) segments**.

    UI text mappings are entered by the user to force specific EN wording
    for on-screen labels — which are always rendered as bold in the source
    docx. Applying the same replacement to non-bold occurrences (regular
    prose that happens to contain the same Korean word) produces awkward
    output like "Change the file Retention policy or Delete them", so the
    substitution is confined to bold spans.

    `start_idx` lets callers continue an existing G-index sequence (e.g.
    after a preceding call to `preprocess_with_glossary_placeholders`) so
    the resulting placeholders don't collide.

    Returns (out, mapping, next_idx).
    """
    if not ui_overrides:
        return text, {}, start_idx

    mapping: Dict[str, GlossaryEntry] = {}
    idx = start_idx
    # 긴 term부터 시도 — 짧은 term이 긴 term 안에서 매치되어 부분 치환되는 것 방지
    sorted_terms = sorted(
        [(ko, en) for ko, en in ui_overrides.items() if ko and en],
        key=lambda kv: len(kv[0]),
        reverse=True,
    )

    def _replace_inside_bold(match):
        nonlocal idx
        inner = match.group(1)
        for ko, en in sorted_terms:
            escaped_parts = [re.escape(part) for part in ko.split()]
            pattern = r"\s*".join(escaped_parts) if escaped_parts else re.escape(ko)
            if re.search(pattern, inner):
                ph = make_marker(G_PREFIX, idx)
                idx += 1
                inner = re.sub(pattern, ph, inner)
                mapping[ph] = GlossaryEntry(
                    ko=ko.strip(),
                    en=en.strip(),
                    dnt=False,
                    case_sensitive=True,
                    product="",
                    note="UI text override (bold only)",
                )
        return f"{B_OPEN}{inner}{B_CLOSE}"

    out = re.sub(rf"{re.escape(B_OPEN)}(.+?){re.escape(B_CLOSE)}", _replace_inside_bold, text, flags=re.DOTALL)
    return out, mapping, idx


def _is_at_sentence_or_bold_start(text_before: str) -> bool:
    """
    Return True if the position immediately after text_before is the start of
    a sentence or the start of a bold segment — i.e. the restored term should
    be capitalized.

    Rules:
    - Nothing before → start of text → capitalize.
    - Last non-space character is sentence-ending punctuation (.  !  ?) → capitalize.
    - text_before ends with the bold-open marker ⟦B⟧ (after stripping trailing
      spaces) → we are the first word inside a bold segment → capitalize.
    """
    s = text_before.rstrip()
    if not s:
        return True
    if s[-1] in ".!?":
        return True
    if s.endswith(B_OPEN):
        return True
    return False


def restore_glossary_placeholders(
    text: str,
    mapping: Dict[str, GlossaryEntry],
) -> str:
    """
    Restore each glossary placeholder with position-aware capitalisation.

    - DNT or Case-sensitive entries: always inserted exactly as stored in EN.
    - Normal entries (lowercase EN in glossary):
        • Capitalise the first letter when the placeholder sits at the very
          start of the text, immediately after sentence-ending punctuation,
          or immediately after the bold-open marker ⟦B⟧.
        • Lower-case otherwise.

    This means glossary EN values should always be stored in lowercase.
    The function handles capitalisation automatically based on context.
    """
    out = text

    for ph, entry in mapping.items():
        # Fixed casing: DNT or explicitly case-sensitive terms are never touched.
        if entry.dnt or entry.case_sensitive:
            out = out.replace(ph, entry.en)
            continue

        # Variable casing: process each occurrence independently so we can
        # inspect what precedes that specific occurrence.
        result = ""
        remaining = out
        while True:
            idx = remaining.find(ph)
            if idx == -1:
                result += remaining
                break

            # Everything accumulated so far + text up to this placeholder
            # gives us the "before" context for position detection.
            before = result + remaining[:idx]
            at_start = _is_at_sentence_or_bold_start(before)

            en = entry.en
            if en:
                en = en[0].upper() + en[1:] if at_start else en[0].lower() + en[1:]

            result += remaining[:idx] + en
            remaining = remaining[idx + len(ph):]

        out = result

    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def enforce_case_sensitive_glossary(
    text: str,
    glossary_entries: List[GlossaryEntry],
) -> str:
    """
    Restore the exact casing of DNT and case-sensitive glossary EN values.

    Some downstream steps — heading sentence-case ([_sentence_case_preserving_markers]),
    QA revisions, generic capitalisation helpers — can drift the casing of
    product names like 'Wrapsody eCo' to 'Wrapsody eco'. This function does a
    case-insensitive sweep and rewrites every occurrence with the stored EN
    value, so the user's case_sensitive=True flag actually holds end-to-end.
    Word boundaries are used to avoid partial matches inside other words.
    """
    for entry in glossary_entries:
        if not (entry.dnt or entry.case_sensitive):
            continue
        if not entry.en:
            continue
        pattern = re.compile(rf"\b{re.escape(entry.en)}\b", re.IGNORECASE)
        text = pattern.sub(entry.en, text)
    return text


def _lxml_all_runs(p_elem):
    """Return every w:r in the paragraph in document order,
    including those nested inside w:hyperlink."""
    return p_elem.findall(f".//{{{_W}}}r")


def _run_is_comment_ref(r_elem) -> bool:
    """True for the w:r that holds w:commentReference (the comment bubble icon).
    This run must NEVER be deleted or rebuilt."""
    return r_elem.find(f"{{{_W}}}commentReference") is not None


def _run_highlight_val(r_elem) -> Optional[str]:
    """Return the highlight colour string (e.g. 'yellow') or None."""
    rPr = r_elem.find(f"{{{_W}}}rPr")
    if rPr is None:
        return None
    hl = rPr.find(f"{{{_W}}}highlight")
    return hl.get(f"{{{_W}}}val") if hl is not None else None


def _lxml_run_is_bold(r_elem) -> bool:
    """True when the run carries an active w:b element."""
    rPr = r_elem.find(f"{{{_W}}}rPr")
    if rPr is None:
        return False
    b = rPr.find(f"{{{_W}}}b")
    if b is None:
        return False
    val = b.get(f"{{{_W}}}val")
    return val not in ("false", "0", "False")


def _lxml_text_elem(r_elem):
    """Return the w:t child element of a run, or None."""
    return r_elem.find(f"{{{_W}}}t")


def _run_is_non_text(r_elem) -> bool:
    """True when the run contains a drawing, symbol, or embedded object
    (i.e. an icon) rather than translatable text."""
    return any(
        r_elem.find(f"{{{_W}}}{tag}") is not None
        for tag in ("drawing", "sym", "object", "pict")
    )


def _parse_marked_segments(marked: str) -> List[Tuple]:
    """Split a marked string into typed segments.

    Returns list of (type, text) where type is one of:
      {'bold': bool, 'hl': str|None}  — plain text with formatting
      'drawing'                         — drawing placeholder
      ('h', int)                        — text inside hyperlink <int>
    """
    segments: List[Tuple] = []
    tokens = ALL_MARKER_RE.split(marked)
    bold   = False
    in_hl:    Optional[str] = None
    buf: List[str] = []

    in_hlink: Optional[int] = None

    def _flush() -> None:
        if not buf:
            return
        text = "".join(buf)
        if in_hlink is not None:
            segments.append((("h", in_hlink), text))
        else:
            segments.append(({"bold": bold, "hl": in_hl}, text))
        buf.clear()

    for tok in tokens:
        if tok == B_OPEN:
            _flush(); bold = True
        elif tok == B_CLOSE:
            _flush(); bold = False
        elif DRAWING_PH_RE.fullmatch(tok):
            _flush(); segments.append(("drawing", tok))
        elif H_OPEN_RE.fullmatch(tok):
            _flush(); in_hlink = int(H_OPEN_RE.match(tok).group(1))
        elif H_CLOSE_RE.fullmatch(tok):
            _flush(); in_hlink = None
        elif HL_OPEN_RE.fullmatch(tok):
            _flush(); in_hl = HL_OPEN_RE.match(tok).group(1)
        elif tok == HL_CLOSE:
            _flush(); in_hl = None
        elif tok:
            buf.append(tok)

    _flush()
    return [(b, t) for b, t in segments if t]


def paragraph_to_marked_text(paragraph) -> Tuple[str, Dict, Dict]:
    """
    Extract paragraph text with:
    - ⟦B⟧…⟦/B⟧  bold markers
    - ⟦D0⟧        inline drawing placeholders
    - ⟦H0⟧…⟦/H0⟧ hyperlink span markers (so the LLM can keep link text together)

    Walks *direct children* of w:p so hyperlink nodes are handled as a unit
    rather than mixing their inner runs with regular paragraph runs.

    Returns
    -------
    marked_text   : str
    drawing_map   : {placeholder_str: lxml_run_element}
    hyperlink_map : {index: {'elem': w:hyperlink_element, 'runs': [run_elements]}}
    """
    p_elem = paragraph._p
    parts: List[str] = []
    in_bold = False
    in_hl: Optional[str] = None
    d_idx = 0
    h_idx = 0
    drawing_map: Dict[str, Any] = {}
    hyperlink_map: Dict[int, Any] = {}

    for child in p_elem:
        ctag = child.tag.split("}")[1] if "}" in child.tag else child.tag

        if ctag == "r":
            if _run_is_comment_ref(child):
                continue
            if _run_is_non_text(child):
                if in_bold: parts.append(B_CLOSE); in_bold = False
                if in_hl:   parts.append(HL_CLOSE); in_hl = None
                ph = f"{D_PREFIX}{d_idx}{SUFFIX}"
                parts.append(ph); drawing_map[ph] = child; d_idx += 1
                continue
            # 한 run 안의 자식 요소들을 문서 순서대로 훑는다 — <w:t> 외에
            # <w:br/> (soft line break) 도 만나면 ⟦BR⟧ 마커로 편입시켜, 원문의
            # 줄바꿈 위치가 번역·재쓰기 단계까지 살아남게 한다.
            text_parts: List[str] = []
            for sub in child:
                sub_tag = sub.tag.split("}")[1] if "}" in sub.tag else sub.tag
                if sub_tag == "t":
                    text_parts.append(sub.text or "")
                elif sub_tag == "br":
                    text_parts.append(BR_MARKER)
            if not text_parts:
                continue
            text = "".join(text_parts)
            if not text:
                continue
            is_bold = _lxml_run_is_bold(child)
            hl_val  = _run_highlight_val(child)
            if in_hl is not None and hl_val != in_hl:
                if in_bold:
                    parts.append(B_CLOSE); in_bold = False
                parts.append(HL_CLOSE); in_hl = None
            if hl_val is not None and in_hl is None:
                parts.append(f"⟦HL:{hl_val}⟧"); in_hl = hl_val
            if is_bold and not in_bold:
                parts.append(B_OPEN); in_bold = True
            elif not is_bold and in_bold:
                parts.append(B_CLOSE); in_bold = False
            parts.append(text)

        elif ctag == "hyperlink":
            if in_bold: parts.append(B_CLOSE); in_bold = False
            if in_hl:   parts.append(HL_CLOSE); in_hl = None
            hl_runs = child.findall(f"{{{_W}}}r")
            hl_text = "".join(
                r.find(f"{{{_W}}}t").text or ""
                for r in hl_runs
                if r.find(f"{{{_W}}}t") is not None
            )
            if hl_text:
                parts.append(f"⟦H{h_idx}⟧"); parts.append(hl_text); parts.append(f"⟦/H{h_idx}⟧")
                hyperlink_map[h_idx] = {"elem": child, "runs": hl_runs}
                h_idx += 1

    if in_bold: parts.append(B_CLOSE)
    if in_hl:   parts.append(HL_CLOSE)

    return "".join(parts), drawing_map, hyperlink_map


_BOLD_SEGMENT_RE = re.compile(r"⟦B⟧(.+?)⟦/B⟧", re.DOTALL)
_INNER_MARKER_RE = re.compile(r"⟦[A-Za-z/]+(?::[A-Za-z]+)?\d*⟧")


# 1x1 transparent PNG — sanitize_docx 폴백 placeholder.
# (70 bytes; PIL로 생성하고 verify 통과한 valid PNG)
_FALLBACK_PNG = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
    "0000000D49444154789C6360606060000000050001A5F645400000000049454E44"
    "AE426082"
)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".svg", ".webp", ".emf", ".wmf")


def sanitize_docx(in_path: str) -> str:
    """
    Return a path to a docx whose zip entries all pass CRC-32 verification.

    .docx is a zip container; pipelines like Wrapsody decrypt occasionally
    leave one embedded image with a CRC mismatch, and python-docx then refuses
    to open the whole file. Strategy:

    1. Probe every entry. If all read cleanly → return the original path.
    2. Otherwise rebuild the zip:
       - readable entries → copied as-is
       - broken **image** entries → replaced with a valid placeholder PNG
         (preferring another valid image from the same docx for visual
         consistency, falling back to a 1x1 transparent PNG). The filename
         and zip path stay the same so the rels/content-types references
         that point to it keep working.
       - broken **non-image** entries → re-raised, because dropping or
         emptying word/document.xml or a rels file would silently corrupt
         the document.
    """
    # ── 1차 probe: 어떤 entry가 깨졌는지 + valid image bytes 한 개 확보 ─
    bad_entries: list[str] = []
    valid_image_bytes: Optional[bytes] = None
    try:
        with zipfile.ZipFile(in_path, "r") as z:
            for zinfo in z.infolist():
                try:
                    data = z.read(zinfo.filename)
                    lname = zinfo.filename.lower()
                    if valid_image_bytes is None and lname.endswith(_IMAGE_EXTS):
                        valid_image_bytes = data
                except (zipfile.BadZipFile, zlib.error, RuntimeError):
                    bad_entries.append(zinfo.filename)
    except (zipfile.BadZipFile, zlib.error, RuntimeError):
        # zip 자체가 깨졌으면 손쓸 도리 없음 — 원본 그대로 넘김
        return in_path

    if not bad_entries:
        return in_path

    placeholder = valid_image_bytes or _FALLBACK_PNG

    # 깨진 entry 중 이미지가 아닌 것이 하나라도 있으면 안전하게 복구할 수 없음
    non_image_bad = [n for n in bad_entries if not n.lower().endswith(_IMAGE_EXTS)]
    if non_image_bad:
        raise RuntimeError(
            "docx 내부 파일이 손상되었습니다 (이미지 외 part): "
            + ", ".join(non_image_bad[:3])
            + " — Wrapsody에서 decrypt를 다시 시도해 주세요."
        )

    # ── 새 zip으로 rebuild ─────────────────────────────────────────
    fd, dst = tempfile.mkstemp(suffix=".docx", prefix="repaired_")
    os.close(fd)
    bad_set = set(bad_entries)
    with zipfile.ZipFile(in_path, "r") as zin, \
         zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for zinfo in zin.infolist():
            if zinfo.filename in bad_set:
                # 깨진 이미지 → placeholder bytes로 교체. 이름/경로/확장자 유지.
                zout.writestr(zinfo.filename, placeholder)
            else:
                data = zin.read(zinfo.filename)
                zout.writestr(zinfo, data)
    return dst


def _make_context_excerpt(paragraph_text: str, target: str, around: int = 30) -> str:
    """
    Carve a short window of the source paragraph around `target` and wrap the
    target with 「…」 for visual emphasis. Streamlit's data_editor cells render
    as plain text, so we use a marker character pair rather than markdown.
    """
    idx = paragraph_text.find(target)
    if idx < 0:
        return f"「{target}」"
    start = max(0, idx - around)
    end = min(len(paragraph_text), idx + len(target) + around)
    excerpt = paragraph_text[start:end].strip()
    excerpt = excerpt.replace(target, f"「{target}」", 1)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(paragraph_text) else ""
    return f"{prefix}{excerpt}{suffix}"


def extract_bold_texts_with_context(in_path: str) -> List[Tuple[str, str]]:
    """
    Like [extract_bold_texts], but for each unique bold KO segment also returns
    the sentence-level context in which it first appears, with the target
    surrounded by 「…」 so the user can disambiguate "관리" (메뉴 라벨인지 직책인지)
    while filling out the mapping table.

    Returns a list of (ko, context_excerpt) tuples, preserving first-occurrence order.
    """
    doc = Document(sanitize_docx(in_path))
    seen: set = set()
    result: List[Tuple[str, str]] = []
    for p in iter_all_paragraphs(doc):
        if is_heading_paragraph(p):
            continue
        marked, _, _ = paragraph_to_marked_text(p)
        # 컨텍스트는 plain text 기준 — 모든 마커 제거 후 단락 텍스트
        plain_para = re.sub(r"⟦[^⟧]+⟧", "", marked)
        for match in _BOLD_SEGMENT_RE.finditer(marked):
            text = _INNER_MARKER_RE.sub("", match.group(1)).strip()
            if not text or not contains_korean(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            context = _make_context_excerpt(plain_para, text)
            result.append((text, context))
    return result


def extract_bold_texts(in_path: str) -> List[str]:
    """
    Pull every bold-formatted text segment containing Korean from a docx file.

    Used by the UI text-mapping screen so the user can pre-translate the
    sentences/words that appear inside ⟦B⟧…⟦/B⟧ in the document. Returned
    list is de-duplicated and preserves the order each segment first appears.

    Skips:
    - Heading / Title paragraphs (Heading 1/2/…, Title): these are document
      structure labels, not UI strings, so the user writes their own English
      version separately and they shouldn't pollute the mapping table.
    - Nested markers (drawings, hyperlinks, highlights) inside a bold span:
      stripped — only the human-readable text is returned.
    """
    doc = Document(sanitize_docx(in_path))
    seen: set = set()
    result: List[str] = []
    for p in iter_all_paragraphs(doc):
        if is_heading_paragraph(p):
            continue  # heading/title은 UI 텍스트 매핑 대상이 아님
        marked, _, _ = paragraph_to_marked_text(p)
        for match in _BOLD_SEGMENT_RE.finditer(marked):
            text = _INNER_MARKER_RE.sub("", match.group(1)).strip()
            if not text or not contains_korean(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            result.append(text)
    return result


# Per-run styling tags that must NOT be inherited by rebuilt runs —
# each segment carries its own bold/italic/etc. from the translated markers.
_FORMATTING_TAGS = frozenset(
    # "highlight" is intentionally excluded so that highlighted text keeps
    # its background colour after run rebuilding.
    {"b", "bCs", "i", "iCs", "u", "strike", "dstrike", "rStyle"}
)


def _make_rPr_template(rPr_src):
    """Clone an rPr element keeping only structural properties (font, size,
    colour, spacing…) and stripping per-character styling (bold, italic,
    underline, rStyle) so the template can be safely reused across segments
    with different formatting."""
    if rPr_src is None:
        return None
    tmpl = copy.deepcopy(rPr_src)
    for tag in _FORMATTING_TAGS:
        el = tmpl.find(f"{{{_W}}}{tag}")
        if el is not None:
            tmpl.remove(el)
    hl = tmpl.find(f"{{{_W}}}highlight")
    if hl is not None:
        tmpl.remove(hl)
    return tmpl


def _make_run(rPr_template, text: str, props):
    """Create a fresh w:r with a cloned rPr template, bold flag, and text.

    If `text` contains any ⟦BR⟧ markers they are converted to actual
    <w:br/> soft-line-break elements interleaved with <w:t> chunks — this
    preserves the original document's paragraph-internal line breaks
    across translation.
    """
    is_bold = props.get("bold", False) if isinstance(props, dict) else bool(props)
    hl_val  = props.get("hl",   None)  if isinstance(props, dict) else None
    r = etree.Element(f"{{{_W}}}r")
    if rPr_template is not None:
        rPr_new = copy.deepcopy(rPr_template)
        if is_bold:
            etree.SubElement(rPr_new, f"{{{_W}}}b")
        if hl_val:
            hl_el = etree.SubElement(rPr_new, f"{{{_W}}}highlight")
            hl_el.set(f"{{{_W}}}val", hl_val)
        r.append(rPr_new)
    elif is_bold or hl_val:
        rPr_new = etree.SubElement(r, f"{{{_W}}}rPr")
        if is_bold:
            etree.SubElement(rPr_new, f"{{{_W}}}b")
        if hl_val:
            hl_el = etree.SubElement(rPr_new, f"{{{_W}}}highlight")
            hl_el.set(f"{{{_W}}}val", hl_val)

    # ⟦BR⟧ 마커가 있으면 <w:t> + <w:br/> + <w:t> ... 순으로 여러 요소를 삽입
    if BR_MARKER in text:
        chunks = text.split(BR_MARKER)
        for i, chunk in enumerate(chunks):
            if i > 0:
                etree.SubElement(r, f"{{{_W}}}br")
            if chunk:
                t_elem = etree.SubElement(r, f"{{{_W}}}t")
                t_elem.text = chunk
                if chunk[0] == " " or chunk[-1] == " ":
                    t_elem.set(f"{{{_XML_SPACE}}}space", "preserve")
    else:
        t_elem = etree.SubElement(r, f"{{{_W}}}t")
        t_elem.text = text
        if text and (text[0] == " " or text[-1] == " "):
            t_elem.set(f"{{{_XML_SPACE}}}space", "preserve")
    return r


def _set_run_text(r_elem, text: str, is_bold: bool) -> None:
    """Set w:t text and sync w:b bold on a single run element."""
    t_elem = _lxml_text_elem(r_elem)
    if t_elem is None:
        return
    t_elem.text = text
    if text and (text[0] == " " or text[-1] == " "):
        t_elem.set(f"{{{_XML_SPACE}}}space", "preserve")
    else:
        t_elem.attrib.pop(f"{{{_XML_SPACE}}}space", None)
    rPr = r_elem.find(f"{{{_W}}}rPr")
    if rPr is None:
        if is_bold:
            rPr = etree.SubElement(r_elem, f"{{{_W}}}rPr")
            etree.SubElement(rPr, f"{{{_W}}}b")
            r_elem.insert(0, rPr)
    else:
        b_elem = rPr.find(f"{{{_W}}}b")
        if is_bold and b_elem is None:
            etree.SubElement(rPr, f"{{{_W}}}b")
        elif not is_bold and b_elem is not None:
            rPr.remove(b_elem)


def _write_paragraph_inplace(p_elem, translated_marked: str,
                             drawing_map: Dict, hyperlink_map: Dict) -> None:
    """
    Write translated text back into the paragraph XML — three-path strategy.

    PATH 1 — Drawing paragraph
        Groups runs around drawing elements; first run of each group gets the
        combined translated text for that segment; drawing runs stay untouched.

    PATH 2 — Hyperlink paragraph
        ⟦H0⟧…⟦/H0⟧ text → written into the hyperlink's first inner run.
        Plain text → direct w:r children rebuilt from a clean rPr template,
        inserted at positions that preserve the before/after-hyperlink order.

    PATH 3 — Plain paragraph
        rPr template cloned from first run (stripping per-segment styling).
        All direct w:r removed; rebuilt one per (bold, text) segment.
    """
    all_runs = _lxml_all_runs(p_elem)
    if not all_runs:
        return

    has_drawing   = any(_run_is_non_text(r) for r in all_runs)
    has_hyperlink = bool(hyperlink_map)

    # ── Unified write-back (drawing / hyperlink / highlight / comment-safe) ───
    #
    # All paragraph types use the same strategy:
    # 1. Build run groups separated by drawing runs (direct children only)
    # 2. Each group also tracks which hyperlink nodes belong to it
    # 3. Translated segments for each group are split at hyperlink boundaries
    #    and the resulting text chunks are interleaved with the hyperlink nodes
    # 4. Drawing runs stay in their XML positions; comment-ref runs are never removed

    segs = _parse_marked_segments(translated_marked)

    def _get_rPr_template():
        for child in p_elem:
            ctag = child.tag.split("}")[1] if "}" in child.tag else child.tag
            if ctag == "r" and not _run_is_comment_ref(child) and not _run_is_non_text(child):
                t = _lxml_text_elem(child)
                if t is not None:
                    rPr = child.find(f"{{{_W}}}rPr")
                    if rPr is not None:
                        return _make_rPr_template(rPr)
        return None

    rPr_tmpl = _get_rPr_template()

    def _rebuild_group(group_runs: List, slot_segs: List[Tuple], group_hls: List) -> None:
        if group_runs:
            insert_pos = list(p_elem).index(group_runs[0])
        else:
            insert_pos = len(list(p_elem))

        for r in group_runs:
            p_elem.remove(r)

        for hi, hl_info in hyperlink_map.items():
            if hl_info["elem"] in group_hls:
                hl_runs = hl_info["runs"]
                hl_text = "".join(
                    t for b, t in slot_segs if isinstance(b, tuple) and b == ("h", hi)
                )
                if hl_runs:
                    _set_run_text(hl_runs[0], hl_text, False)
                    for r in hl_runs[1:]:
                        _set_run_text(r, "", False)

        chunks: List[List[Tuple]] = []
        hl_after_chunk: List = []
        cur_chunk: List[Tuple] = []
        for b, t in slot_segs:
            if isinstance(b, tuple) and b[0] == "h":
                hl_elem = hyperlink_map.get(b[1], {}).get("elem")
                chunks.append(cur_chunk)
                hl_after_chunk.append(hl_elem if hl_elem in group_hls else None)
                cur_chunk = []
            else:
                cur_chunk.append((b, t))
        chunks.append(cur_chunk)

        offset = 0
        for ci, chunk in enumerate(chunks):
            for props, text in chunk:
                p_elem.insert(insert_pos + offset, _make_run(rPr_tmpl, text, props))
                offset += 1
            if ci < len(hl_after_chunk) and hl_after_chunk[ci] is not None:
                hl_elem = hl_after_chunk[ci]
                cur_pos = list(p_elem).index(hl_elem)
                target  = insert_pos + offset
                if cur_pos != target:
                    p_elem.remove(hl_elem)
                    p_elem.insert(target if cur_pos > target else target - 1, hl_elem)
                offset += 1

    # Build run groups (separated by drawing runs) from direct children
    direct_children = list(p_elem)
    run_groups: List[List] = []
    hl_groups:  List[List] = []
    cur_runs: List = []
    cur_hls:  List = []
    for child in direct_children:
        ctag = child.tag.split("}")[1] if "}" in child.tag else child.tag
        if ctag == "r" and _run_is_non_text(child):
            run_groups.append(cur_runs); hl_groups.append(cur_hls)
            cur_runs = []; cur_hls = []
        elif ctag == "r" and not _run_is_comment_ref(child) and _lxml_text_elem(child) is not None:
            cur_runs.append(child)
        elif ctag == "hyperlink" and any(info["elem"] is child for info in hyperlink_map.values()):
            cur_hls.append(child)
    run_groups.append(cur_runs); hl_groups.append(cur_hls)

    # Split translated segments at drawing markers → one slot per run group
    text_slots: List[List[Tuple]] = []
    cur_slot: List[Tuple] = []
    for seg in segs:
        if seg[0] == "drawing":
            text_slots.append(cur_slot); cur_slot = []
        else:
            cur_slot.append(seg)
    text_slots.append(cur_slot)

    for gi in range(len(run_groups)):
        _rebuild_group(run_groups[gi], text_slots[gi] if gi < len(text_slots) else [], hl_groups[gi])


def strip_bold_markers(text: str) -> str:
    """Remove ⟦B⟧/⟦/B⟧ bold markers and ⟦HL:colour⟧/⟦/HL⟧ highlight markers."""
    text = text.replace(B_OPEN, "").replace(B_CLOSE, "")
    text = HL_OPEN_RE.sub("", text)
    text = text.replace(HL_CLOSE, "")
    return text


def set_paragraph_text_preserve_style(paragraph, text: str) -> None:
    text = strip_bold_markers(text)
    if paragraph.runs:
        paragraph.runs[0].text = text
        for r in paragraph.runs[1:]:
            r.text = ""
    else:
        paragraph.add_run(text)


def repair_bold_markers(text: str) -> str:
    if not text:
        return text

    # 1. ⟦Brule → ⟦B⟧rule
    text = re.sub(r"⟦B([A-Za-z])", r"⟦B⟧\1", text)

    # 2. ⟦/Brule → ⟦/B⟧rule
    text = re.sub(r"⟦/B([A-Za-z])", r"⟦/B⟧\1", text)

    # 3. ⟦B rule → ⟦B⟧rule
    text = re.sub(r"⟦B\s+", "⟦B⟧", text)

    # 4. ⟦/B rule → ⟦/B⟧rule
    text = re.sub(r"⟦/B\s+", "⟦/B⟧", text)

    # 5. 잘못된 닫힘 제거 (⟧/B⟧ → ⟦/B⟧)
    text = re.sub(r"⟧/B⟧", "⟦/B⟧", text)

    # 6. 개수 맞추기
    opens = text.count("⟦B⟧")
    closes = text.count("⟦/B⟧")

    if opens > closes:
        text += "⟦/B⟧"
    elif closes > opens:
        for _ in range(closes - opens):
            text = text.replace("⟦/B⟧", "", 1)

    return text


def normalize_bold_spaces(text: str) -> str:
    """
    Remove spurious spaces that accumulate at bold-marker boundaries,
    then collapse any remaining double-spaces to single spaces.

    The LLM sometimes outputs "click ⟦B⟧ Rule ⟦/B⟧." or
    "click  ⟦B⟧Rule⟦/B⟧." — both produce double-space artifacts.

    Steps
    -----
    1. Strip whitespace immediately after  ⟦B⟧  → "⟦B⟧ Rule" → "⟦B⟧Rule"
    2. Strip whitespace immediately before ⟦/B⟧ → "Rule ⟦/B⟧" → "Rule⟦/B⟧"
    3. Collapse any remaining consecutive spaces to a single space.
    """
    text = re.sub(r"⟦B⟧\s+", B_OPEN, text)
    text = re.sub(r"\s+⟦/B⟧", B_CLOSE, text)
    text = re.sub(r"  +", " ", text)
    return text


def repair_hl_markers(text: str) -> str:
    """
    Fix common LLM garbling of ⟦HL:colour⟧ … ⟦/HL⟧ highlight markers.

    - ⟦HL: yellow⟧  → ⟦HL:yellow⟧  (space after colon)
    - Unmatched open → append ⟦/HL⟧ at end
    - Unmatched close → remove excess ⟦/HL⟧
    """
    if not text or "⟦HL" not in text:
        return text
    # Fix space after colon
    text = re.sub(r"⟦HL:\s+([a-zA-Z]+)⟧", lambda m: f"⟦HL:{m.group(1)}⟧", text)
    opens  = len(HL_OPEN_RE.findall(text))
    closes = text.count(HL_CLOSE)
    if opens > closes:
        text = text + HL_CLOSE * (opens - closes)
    elif closes > opens:
        for _ in range(closes - opens):
            idx = text.rfind(HL_CLOSE)
            if idx >= 0:
                text = text[:idx] + text[idx + len(HL_CLOSE):]
    return text


def apply_highlight_fallback(translated: str, source_marked: str) -> str:
    """
    Safety net: if the source had ⟦HL:colour⟧ markers but the LLM dropped
    them from the translation, wrap the entire translated text in the dominant
    highlight colour so highlight is never silently lost.
    """
    src_colours = HL_OPEN_RE.findall(source_marked)
    if not src_colours:
        return translated                   # source had no highlight
    if HL_OPEN_RE.search(translated):
        return translated                   # LLM preserved markers — good
    # Fallback: whole-paragraph highlight with the dominant colour
    from collections import Counter
    colour = Counter(src_colours).most_common(1)[0][0]
    return f"⟦HL:{colour}⟧{translated}⟦/HL⟧"


def _split_preserving_markers(text: str) -> List[str]:
    return MARKER_SPLIT_RE.split(text)


def _sentence_case_preserving_markers(text: str) -> str:
    parts = _split_preserving_markers(text)
    out = []
    first_alpha_done = False

    for part in parts:
        if not part:
            out.append(part)
            continue

        if part == B_OPEN or part == B_CLOSE or PLACEHOLDER_RE.fullmatch(part):
            out.append(part)
            continue

        chars = list(part)

        if not first_alpha_done:
            for i, ch in enumerate(chars):
                if ch.isalpha():
                    chars[i] = ch.upper()
                    for j in range(i + 1, len(chars)):
                        if chars[j].isalpha():
                            chars[j] = chars[j].lower()
                    first_alpha_done = True
                    break
            out.append("".join(chars))
        else:
            lowered = []
            for ch in chars:
                lowered.append(ch.lower() if ch.isalpha() else ch)
            out.append("".join(lowered))

    return "".join(out)


def normalize_ui_label_text(text: str) -> str:
    parts = _split_preserving_markers(text)
    out = []
    first_alpha_word_seen = False
    word_re = re.compile(r"[A-Za-z][A-Za-z0-9/-]*")

    for part in parts:
        if not part:
            out.append(part)
            continue

        if part == B_OPEN or part == B_CLOSE or PLACEHOLDER_RE.fullmatch(part):
            out.append(part)
            continue

        def repl(match):
            nonlocal first_alpha_word_seen
            word = match.group(0)
            lower = word.lower()

            if not first_alpha_word_seen:
                first_alpha_word_seen = True
                return word

            if lower in UI_LOWER_WORDS:
                return lower

            return word

        out.append(word_re.sub(repl, part))

    return "".join(out)


def normalize_heading_text(text: str) -> str:
    if not text:
        return text

    s = text.strip()
    s = re.sub(r"[.。]+$", "", s)
    s = _sentence_case_preserving_markers(s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    s = re.sub(r"[.。]+$", "", s)
    return s


def normalize_ui_in_bold_segments(text: str) -> str:
    """
    Normalise each bold segment as a UI label: first alpha word kept as-is
    (already cased correctly by restore step), subsequent words lowercased
    if they are in UI_LOWER_WORDS, and the whole segment has its first alpha
    character capitalised.

    NOTE: Each bold segment is normalised independently, so a term restored
    at the start of a bold segment is already capitalised by
    restore_glossary_placeholders; this function only normalises the
    surrounding plain words inside the segment.
    """
    def repl(match):
        inner = match.group(1)
        inner = normalize_ui_label_text(inner)
        inner = _cap_first_alpha(inner)
        return B_OPEN + inner + B_CLOSE

    return re.sub(
        re.escape(B_OPEN) + r"(.*?)" + re.escape(B_CLOSE),
        repl,
        text,
        flags=re.DOTALL,
    )


def fix_indefinite_articles(text: str) -> str:
    text = re.sub(r"\b([Aa])\s+([aeiouAEIOU])", r"\1n \2", text)
    text = re.sub(r"\b[Aa]n\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])", r"A \1", text)
    return text


def capitalize_bullet_lines(text: str) -> str:
    lines = text.splitlines()
    out_lines = []
    num_prefix_re = re.compile(r"^\s*\d+[\.\)]\s+")
    bullet_prefixes = ("- ", "• ", "∙ ", "* ")

    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue

        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        handled = False
        for bp in bullet_prefixes:
            if stripped.startswith(bp):
                rest = stripped[len(bp) :]
                rest = _cap_first_alpha(rest)
                out_lines.append(indent + bp + rest)
                handled = True
                break
        if handled:
            continue

        m = num_prefix_re.match(stripped)
        if m:
            pre = stripped[: m.end()]
            rest = stripped[m.end() :]
            rest = _cap_first_alpha(rest)
            out_lines.append(indent + pre + rest)
            continue

        out_lines.append(line)

    return "\n".join(out_lines)


def restore_sentence_period(translated: str, source: str) -> str:
    """
    If the source sentence ended with a period and the translation does not
    end with any terminal punctuation, append a period.

    The LLM occasionally drops the trailing period when the sentence ends with
    a glossary placeholder (e.g. 'Enter ⟦B⟧basic information⟦/B⟧' → no '.').
    """
    src_stripped = source.rstrip()
    if not src_stripped.endswith("."):
        return translated          # source had no period — nothing to restore

    tr = translated.rstrip()
    if tr and tr[-1] not in ".!?:;":
        translated = tr + "."
    return translated


_LIST_LINE_RE = re.compile(r"^\s*([-•∙*]|\d+[.)]|[a-zA-Z][.)])\s+")


def normalize_paragraph_breaks(s: str) -> str:
    """
    A translated Word paragraph should stay one continuous line in the docx
    output. LLMs occasionally emit line breaks inside the response for
    prose paragraphs, which show up as rogue soft line breaks in the final
    document. We collapse those into spaces — but preserve line breaks when
    the paragraph is actually a bulleted / numbered list (each line begins
    with a bullet or number marker).
    """
    if not s:
        return s
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip("\n")

    if "\n" not in s:
        return s

    lines = s.split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    # 2번째 라인부터 다수가 list marker로 시작하면 list로 간주 → 원본 유지
    if len(non_empty) >= 2 and sum(bool(_LIST_LINE_RE.match(ln)) for ln in non_empty[1:]) >= 1:
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s

    # 그 외엔 단순 산문 — 모든 line break를 space로 흡수
    return re.sub(r"\s*\n+\s*", " ", s).strip()


def iter_all_paragraphs(doc: Document) -> Iterable:
    for p in doc.paragraphs:
        yield p
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p


def is_heading_paragraph(p) -> bool:
    """
    Recognise heading / title paragraphs across Word locales, custom styles,
    and manually-outlined paragraphs.

    Detection order:
    1. Style name / style_id (English "Heading 1"/"Title", Korean "제목 1"/"표제")
       and its ancestor style chain — so a user-defined style that inherits
       from Heading 1 is caught.
    2. The paragraph's `w:pPr/w:outlineLvl` value. Any explicit outline level
       (0-8) means the author marked this paragraph as a heading via
       Word's Outline setting, even if the applied style isn't a heading style.
    """
    style = getattr(p, "style", None)
    seen_ids: set = set()
    while style is not None:
        sid = (getattr(style, "style_id", None) or "")
        name = (getattr(style, "name", None) or "")
        sid_l = sid.lower()
        name_l = name.lower()

        if sid_l.startswith("heading") or sid_l == "title":
            return True
        if name_l.startswith("heading") or name_l in ("title",):
            return True
        if name.startswith("제목") or name in ("표제",):
            return True

        if sid in seen_ids:
            break
        seen_ids.add(sid)
        style = getattr(style, "base_style", None)

    # outline level 확인 — 스타일과 무관하게 Word "Outline" 설정된 경우
    try:
        p_elem = getattr(p, "_p", None)
        if p_elem is not None:
            outline_nodes = p_elem.findall(f".//{{{_W}}}pPr/{{{_W}}}outlineLvl")
            if outline_nodes:
                val = outline_nodes[0].get(f"{{{_W}}}val")
                # 0-8은 heading level. 9는 body text.
                if val is not None and val.isdigit() and int(val) < 9:
                    return True
    except Exception:
        pass

    return False


def paragraph_has_hyperlink(paragraph) -> bool:
    return paragraph._p.xpath(".//w:hyperlink") != []


def _write_paragraph(p, translated_marked: str,
                     drawing_map: Optional[Dict] = None,
                     hyperlink_map: Optional[Dict] = None) -> None:
    """Write translated text into the paragraph, preserving all XML structure."""
    if is_heading_paragraph(p):
        translated_marked = translated_marked.rstrip()
        if translated_marked.endswith("."):
            translated_marked = translated_marked[:-1].rstrip()

    _write_paragraph_inplace(p._p, translated_marked, drawing_map or {}, hyperlink_map or {})


def normalize_for_scoring(text: str) -> str:
    text = text.replace(B_OPEN, " ").replace(B_CLOSE, " ")
    text = re.sub(r"⟦G\d+⟧", " ", text)
    text = text.replace("~", " ")
    return text


def normalize_colon_label_line(text: str) -> str:
    """
    Normalize 'Label:' patterns so the label reads correctly as a UI element name.

    Handles two structures:
    1. Plain label before colon:  'rule Name:'         -> 'Rule name:'
    2. Bold-close + plain word:   '⟦B⟧Rule⟦/B⟧ Name:' -> '⟦B⟧Rule⟦/B⟧ name:'
       (the plain word after ⟦/B⟧ is a continuation of the bold label and must be
       lowercased when it is a UI_LOWER_WORD)

    The two passes run in order; Pass 2 skips any position already handled by Pass 1
    to prevent re-capitalising words that were just lowercased.
    """
    # Pass 1 — lowercase UI_LOWER_WORDS that sit between ⟦/B⟧ and ':'
    def _lower_after_bold(m):
        return m.group(1) + (m.group(2).lower() if m.group(2).lower() in UI_LOWER_WORDS else m.group(2)) + m.group(3)

    pass1 = re.sub(
        rf"({re.escape(B_CLOSE)}\s+)([A-Za-z][A-Za-z0-9]*)(\s*:)",
        _lower_after_bold,
        text,
    )

    # Pass 2 — normalize plain (non-bold-wrapped) labels before ':'
    # Skip positions that are immediately preceded by ⟦/B⟧ (already handled by Pass 1)
    def repl(m):
        before = pass1[: m.start()]
        if re.search(rf"{re.escape(B_CLOSE)}\s*$", before):
            return m.group(0)   # already handled — leave untouched
        label_norm = normalize_ui_label_text(m.group(1).strip())
        label_norm = _cap_first_alpha(label_norm)
        return f"{label_norm}:"

    return re.sub(r"\b([A-Za-z][A-Za-z0-9 ]{1,50}):", repl, pass1)


def looks_like_heading_text(text: str) -> bool:
    """Heuristic: does this look like a heading rather than a body sentence?

    Korean source sentences are often short (≤ 3 words) even when they are
    step instructions ("기본 정보를 입력합니다."), so we suppress heading
    detection when the text contains Korean characters.  Only apply the
    short-text heuristic to already-translated (English) text.
    """
    s = strip_bold_markers(text).strip()

    if not s:
        return False
    if "\n" in s:
        return False
    if len(s) > 50:
        return False
    if s.endswith(":"):
        return False
    # Korean text: never treat as heading via this heuristic —
    # Korean step instructions are naturally short (≤ 3 words).
    if KOREAN_RE.search(s):
        return False

    words = s.split()
    if len(words) <= 3:
        return True

    return False


def tokenize_koreanish(text: str) -> List[str]:
    norm = normalize_for_scoring(text)
    return re.findall(r"[가-힣A-Za-z0-9]+", norm)


def select_relevant_patterns(
    source_text: str,
    patterns: List[Tuple[str, str]],
    max_pattern: int = 3,
) -> List[Tuple[str, str]]:
    source_tokens = set(tokenize_koreanish(source_text))

    def score_patterns(ko: str) -> int:
        pattern_tokens = set(tokenize_koreanish(ko))
        return len(source_tokens & pattern_tokens)

    scored_patterns = []
    for ko, en in patterns:
        score = score_patterns(ko)
        if score > 0:
            scored_patterns.append((score, ko, en))

    scored_patterns.sort(key=lambda x: (-x[0], -len(x[1])))
    return [(ko, en) for score, ko, en in scored_patterns[:max_pattern]]


def translate_paragraph_with_patterns(
    client: OpenAI,
    source_text: str,
    pattern_examples: List[Tuple[str, str]],
    model: str = "gpt-5.2",
    translation_mode: str = "Manual",
    style_reference: str = "",
) -> str:
    global TOTAL_INPUT_TOKENS, TOTAL_CACHED_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_TOKENS

    pattern_block = (
        "\n".join([f"- {ko} -> {en}" for ko, en in pattern_examples]) or "(none)"
    )

    if translation_mode in {"UI", "UI 텍스트"}:
        style_rules = """
- Prefer short, direct UI-style English.
- Keep labels and instructions concise.
- Avoid unnecessary words.
- Use product-style wording similar to Microsoft or Google UI.
"""
    else:
        style_rules = """
- Prefer natural, clear manual/documentation English.
- Use complete sentences where appropriate.
- Keep the tone professional and concise.
- Use enterprise software documentation style.
"""

    style_ref_block = (
        f"\nExisting English in this document (match tone, vocabulary, sentence structure):\n{style_reference}\n"
        if style_reference else ""
    )

    prompt = f"""
Translate Korean to natural, professional English. Produce a draft, then revise it so a native English speaker would not flag awkwardness, grammar errors, or literal-translation tells.

Rules:
- Preserve markers EXACTLY: ⟦G#⟧, ⟦B⟧, ⟦/B⟧, ⟦D#⟧, ⟦H#⟧/⟦/H#⟧, ⟦HL:colour⟧/⟦/HL⟧, ⟦BR⟧.
- ⟦BR⟧ = soft line break in the source. Emit it at the SAME semantic position in your translation so the paragraph keeps its line break there.
- ⟦G#⟧ placeholders are FIXED glossary terms. Output them BYTE-FOR-BYTE unchanged.
  NEVER translate, paraphrase, expand, or substitute a ⟦G#⟧ placeholder with any word.
- ⟦D#⟧ = inline icon/image — keep it where it naturally fits in the sentence.
- ⟦H#⟧…⟦/H#⟧ = hyperlink span — translate the text inside, keep the markers around it.
- ⟦HL:colour⟧…⟦/HL⟧ = highlighted text — translate the inside and keep the markers
  around the same semantic content in the translation.
- If the source contains existing English words or phrases, leave them EXACTLY as-is.
  Do NOT rephrase, reword, or "improve" any English that is already present.
- Use the pattern examples only as reference guidance.
- Do not copy irrelevant examples.
- Avoid repetition and awkward literal wording.
- Grammar: maintain subject-verb agreement and correct singular/plural agreement.
  Korean does not always mark plurality, but English does. Prefer "all users participating",
  "individual users", "authorized users" over "all user", "individual user", "authorized user".
- Subjects: avoid bare generic singular subjects without articles. "Can user prevent..." is
  ungrammatical — use "Can users prevent..." or supply a concrete subject like
  "Can administrators prevent users from...".
- Naturalize literal renderings into idiomatic English. For example:
    "신개념 협업 플랫폼" → "next-generation collaboration platform" (NOT "new-concept platform")
    "활용 가이드"        → "user guide" or "usage guide" (NOT "utilization guide")
  After your initial draft, re-read each sentence and replace any phrasing that reads as
  Korean-influenced literal English with the way a native speaker would actually write it.
- Spell English words correctly. NEVER transliterate Korean phonetic spellings.
  "프로그램" is "program" (never "logram", "pulogeurem", etc.). Treat any non-word
  English output as a serious error and self-correct.
- Articles: use "a/an/the" appropriately. Korean has no articles, so do not omit them.
- Do not force title case.
- For headings, concise phrase-style English is preferred.
- NEVER merge markers with words. "⟦B⟧rule" is correct; "⟦Brule" is invalid.
- Keep markers as separate tokens.
- Output ONLY the translated text. No explanation, no extra lines.
- ABSOLUTELY DO NOT emit meta labels ("Draft:", "Revised:", "Original:",
  "Before:", "After:", "Version 1:", "v2:", "Option A:", "Improved:",
  "Correction:") or any before/after comparison. Give ONE clean final
  translation, no alternatives, no explanation.
- The output for one input paragraph must be a single line (or a single
  bulleted/numbered list). Do NOT insert soft line breaks in the middle of
  running prose.
{style_rules}{style_ref_block}
Reference pattern examples:
{pattern_block}

Text to translate:
{source_text}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )

    usage = getattr(resp, "usage", None)
    if usage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", 0) or 0

        TOTAL_INPUT_TOKENS += input_tokens
        TOTAL_CACHED_INPUT_TOKENS += cached_tokens
        TOTAL_OUTPUT_TOKENS += output_tokens
        TOTAL_TOKENS += total_tokens

    return resp.output_text.strip()


def translate_remaining_korean(
    client: OpenAI,
    text: str,
    model: str = "gpt-5.2",
) -> str:
    if not contains_korean(text):
        return text

    prompt = f"""
Translate any remaining Korean in the text into natural English.

Rules:
- Preserve markers EXACTLY: ⟦B⟧, ⟦/B⟧, ⟦G#⟧, ⟦BR⟧.
- ⟦G#⟧ placeholders are FIXED terms — output them UNCHANGED. Do NOT translate them.
- Do NOT alter, rephrase, or improve any English that is already present.
- Only translate Korean words/phrases; leave everything else exactly as-is.
- Do not force title case.
- Output ONLY the translated text. No explanation, no extra lines.

Text:
{text}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )

    return resp.output_text.strip()


def extract_doc_style_guide(
    client: OpenAI,
    english_samples: List[str],
    model: str = "gpt-5.2",
) -> str:
    """
    Pre-pass: compress existing English content into a compact style guide.

    Called once per document before any paragraph translation. The returned
    string is reused as the `style_reference` for every per-paragraph LLM call,
    so it must be short (~250 words) — both to save tokens and to maximise
    prompt-cache hits across batches.
    """
    global TOTAL_INPUT_TOKENS, TOTAL_CACHED_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_TOKENS

    if not english_samples:
        return ""

    samples_text = "\n".join(english_samples)

    prompt = f"""Extract a CONCISE style guide from this product document's existing English content. The guide will be reused to keep newly translated Korean paragraphs consistent with the rest of the document.

Cover only what is distinctive (skip generic English advice):
- Specific terminology preferences this product uses (e.g. "Save" vs "Apply", "page" vs "screen")
- Sentence patterns for instructions, headings, and UI labels
- Formality level / tone (UI text vs manual prose)
- Capitalization conventions (sentence case, title case, lowercase UI words)
- Any product-specific phrasing the document repeats

Output as a compact bullet list. No introduction, no examples — bullets only. Maximum 250 words.

Existing English content:
{samples_text}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )

    usage = getattr(resp, "usage", None)
    if usage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
        TOTAL_INPUT_TOKENS += input_tokens
        TOTAL_CACHED_INPUT_TOKENS += cached_tokens
        TOTAL_OUTPUT_TOKENS += output_tokens
        TOTAL_TOKENS += total_tokens

    return resp.output_text.strip()


_QA_HEADER_RE = re.compile(r"\[(\d+)\]")

# 메타 라벨 정규식들 — LLM이 무단으로 "Draft: X Revised: Y" 형태를 반환하는
# 경우를 감지·복구하기 위함. Draft/Revised 뿐 아니라 흔한 대체 표현도 커버.
_META_LABEL_EARLIER = (
    r"Draft|Original|Before|Version\s*1|Ver\s*1|v1|Candidate\s*1|Option\s*A|Initial"
)
_META_LABEL_LATER = (
    r"Revised|Final(?:\s*version)?|After|Corrected|Version\s*2|Ver\s*2|v2|"
    r"Candidate\s*2|Option\s*B|Improved|Better|Updated"
)
_META_LATER_RE = re.compile(rf"\b(?:{_META_LABEL_LATER})\s*[:.\-]", re.IGNORECASE)
_META_EARLIER_LEADING_RE = re.compile(
    rf"^\s*(?:{_META_LABEL_EARLIER})\s*[:.\-]\s*", re.IGNORECASE
)
_META_LABEL_ANY_RE = re.compile(
    rf"\b(?:{_META_LABEL_EARLIER}|{_META_LABEL_LATER})\s*[:.\-]",
    re.IGNORECASE,
)


def strip_meta_version_labels(text: str) -> str:
    """
    Remove "Draft: X ... Revised: Y" style meta-version noise that occasionally
    leaks into pass-1 or pass-2 output despite the prompt forbidding it.

    Strategy:
    1. If a "later-version" label (Revised/Final/After/Improved/...) appears,
       keep only the text AFTER that label — the LLM's own final choice.
    2. If only an "earlier-version" label (Draft/Original/Before/...) appears
       at the start, strip that label.
    3. If neither, return the text unchanged.

    This is a safety net; the prompt still asks the LLM not to emit these,
    but we never trust the LLM to obey.
    """
    if not text:
        return text
    m = _META_LATER_RE.search(text)
    if m:
        text = text[m.end():].strip()
    # After removing later-version prefix there may still be a leading "Draft:"
    # on a residual line — or the text may only have had "Draft:" from the start.
    text = _META_EARLIER_LEADING_RE.sub("", text).strip()
    return text
# 메타 라벨 감지 — 이런 label이 들어있으면 LLM이 프롬프트를 무시하고 다양한
# 버전 비교를 응답에 담은 경우. 안전을 위해 그 revision을 통째로 버림.
_QA_META_LABEL_RE = re.compile(
    r"\b(Draft|Revised|Original|Before|After|Correction|Suggestion|Alternative|"
    r"Option\s+[A-Z]|Version\s*\d|v\d\s*[:.]|Candidate\s*\d)\s*[:.]",
    re.IGNORECASE,
)


def parse_qa_response(text: str) -> Dict[int, str]:
    """
    Parse '[N] revised text' blocks from the QA response.

    Multi-line revisions are supported: each [N] header captures everything up
    to the next [N] header (or end of text). Returns {} when the LLM signals
    no changes (output is empty or 'NONE').

    Safety net: if any [N] block contains meta labels (Draft/Revised/Before/
    After/Option A/…) it means the LLM ignored the prompt and returned
    version-comparison text instead of a clean revision. We skip that block
    so the pass-1 translation is kept as-is.
    """
    revisions: Dict[int, str] = {}
    if not text:
        return revisions
    matches = list(_QA_HEADER_RE.finditer(text))
    if not matches:
        return revisions
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body or body.upper() == "NONE":
            continue
        if _QA_META_LABEL_RE.search(body):
            # LLM이 여러 버전 비교 형식으로 응답 — pass-1 유지 (revision 무시)
            continue
        revisions[idx] = body
    return revisions


def qa_check_batch(
    client: OpenAI,
    items: List[Tuple[int, str, str]],
    style_guide: str,
    glossary_pairs: List[Tuple[str, str]],
    model: str = "gpt-5.2",
) -> Dict[int, str]:
    """
    Pass-2 consistency check on a batch of translated paragraphs.

    Items are (local_index, korean_source, english_translation). The local
    index is what the LLM echoes back in `[N]` headers, so callers should map
    it to their own paragraph identifiers.

    Returns {local_index: revised_translation} containing ONLY items the LLM
    chose to revise. Items not in the result should keep their pass-1
    translation unchanged.
    """
    global TOTAL_INPUT_TOKENS, TOTAL_CACHED_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_TOKENS

    if not items:
        return {}

    glossary_block = (
        "\n".join(f"- {ko} -> {en}" for ko, en in glossary_pairs[:80]) or "(none)"
    )
    pairs_block = "\n\n".join(
        f"[{idx}]\nSOURCE: {src}\nTRANSLATION: {tr}" for idx, src, tr in items
    )

    prompt = f"""You are a translation QA reviewer. Each numbered pair below contains a Korean SOURCE and its English TRANSLATION produced in a first pass.

Revise a translation ONLY when it has a clear problem:
- Terminology inconsistent with the document's existing English style
- Sentence pattern that does not match the document's tone
- Awkward, unnatural, or overly literal English (Korean-influenced phrasing)
  e.g. "new-concept platform" → "next-generation platform", "utilization guide" → "user guide"
- Subject-verb or singular/plural agreement error
  e.g. "all user" → "all users", "individual user" → "individual users", "authorized user" → "authorized users"
- Awkward bare generic singular subject
  e.g. "Can user prevent users" → "Can users prevent" or "Can administrators prevent users"
- Missing article where English requires one (Korean has no articles, so the draft may omit them)
- Misspelling, typo, or non-word — especially Korean phonetic transliteration
  e.g. "logram" → "program", any nonsense English output is wrong
- Missing or duplicated marker, or marker merged with a word
- Glossary term not preserved (see glossary list below) — including the exact casing of case-sensitive terms

Do NOT revise translations that are already acceptable. Do NOT make stylistic preference changes that are not grounded in the style guide. Do NOT shorten or expand for taste. When in doubt, leave it alone.

Strict rules:
- Preserve markers EXACTLY: ⟦B⟧, ⟦/B⟧, ⟦HL:colour⟧, ⟦/HL⟧, ⟦H#⟧, ⟦/H#⟧, ⟦D#⟧, ⟦BR⟧.
- Glossary translations listed below are mandatory — keep those exact English words with their exact casing.
- Do NOT translate or alter any English already present (other than the specific problems above).

Output format:
- For each item that needs revision: a header line `[N]` followed by the revised translation. Nothing else for that item.
- If nothing needs revision, output exactly: NONE
- Do NOT add explanations, comments, or numbering of any other kind.

ABSOLUTELY FORBIDDEN in your output — if you emit ANY of these your revision will be discarded:
- Meta labels of ANY kind: "Draft:", "Revised:", "Original:", "Before:", "After:",
  "Correction:", "Suggestion:", "Alternative:", "Option A/B", "Version 1/2", "v1/v2".
- Any before/after comparison, side-by-side text, or multiple candidate versions.
- Any preamble like "Here is the revised text" or trailing commentary.
- Line breaks WITHIN a single item's revised text unless the source itself had a
  bulleted/numbered list. A single-paragraph source must produce a single-line revision.

Emit ONE clean, final revised sentence per [N] header. Nothing else.

Style guide:
{style_guide or "(no style guide available)"}

Glossary (mandatory translations):
{glossary_block}

Pairs to review:
{pairs_block}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )

    usage = getattr(resp, "usage", None)
    if usage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
        TOTAL_INPUT_TOKENS += input_tokens
        TOTAL_CACHED_INPUT_TOKENS += cached_tokens
        TOTAL_OUTPUT_TOKENS += output_tokens
        TOTAL_TOKENS += total_tokens

    return parse_qa_response(resp.output_text)


def translate_document(
    in_path: str,
    out_path: str,
    glossary_rows: List[dict],
    pattern_rows: List[dict],
    api_key: str,
    enable_cache: bool = True,
    enable_qa: bool = True,
    qa_batch_size: int = 8,
    model: str = "gpt-5.2",
    translation_mode: str = "Manual",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ui_text_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    reset_token_counters()

    glossary_entries = build_glossary_entries_from_rows(glossary_rows)

    # UI 텍스트 매핑은 Glossary와 달리 **⟦B⟧…⟦/B⟧ (bold) 영역 안에만** 적용한다.
    # (Glossary는 본문 어디든 치환. 하지만 UI 라벨은 화면에서 bold이므로
    #  본문에 같은 KO가 나와도 자연 번역을 유지해야 한다.)
    # → glossary_entries에 합치지 않고 별도 dict으로 유지. 각 단락 preprocess
    #   시 preprocess_ui_overrides_in_bold로 별도 치환.
    ui_overrides_clean: Dict[str, str] = {}
    if ui_text_overrides:
        for ko, en in ui_text_overrides.items():
            ko_s = (ko or "").strip()
            en_s = (en or "").strip()
            if ko_s and en_s:
                ui_overrides_clean[ko_s] = en_s
        # 충돌 방지 — glossary에 같은 ko가 있으면 UI 매핑 우선하므로 glossary에서 제거
        if ui_overrides_clean:
            ui_ko_set = set(ui_overrides_clean.keys())
            glossary_entries = [e for e in glossary_entries if e.ko not in ui_ko_set]

    patterns = build_pattern_pairs_from_rows(pattern_rows)
    # QA prompt엔 UI 매핑도 참고용으로 함께 (일관성 유지 위해)
    glossary_pairs_for_qa = [(e.ko, e.en) for e in glossary_entries] + list(ui_overrides_clean.items())

    client = OpenAI(api_key=api_key)
    doc = Document(sanitize_docx(in_path))
    cache: Dict[str, str] = {}

    # ── 문서 내 기존 영문 단락 수집 ────────────────────────────────────
    # Pre-pass(extract_doc_style_guide)에서 한 번만 사용되므로 캡을 넉넉히
    # 잡아도 호출당 비용은 고정. v2.0→v2.1처럼 영문이 풍부한 문서에서
    # 가이드 품질을 끌어올리기 위함.
    english_samples: List[str] = []
    for p in iter_all_paragraphs(doc):
        text = p.text.strip()
        if text and not contains_korean(text) and len(text) > 15:
            english_samples.append(text)
        if sum(len(s) for s in english_samples) >= 3000:
            break

    # ── Pre-pass: doc-specific style guide 추출 (LLM 1회 호출) ─────────
    style_guide = ""
    if english_samples:
        try:
            style_guide = extract_doc_style_guide(client, english_samples, model=model)
        except Exception:
            # 가이드 추출 실패는 치명적이지 않음 — 가이드 없이 진행
            style_guide = ""

    paras: List = []
    marked_texts: List = []

    for p in iter_all_paragraphs(doc):
        marked_ko, drawing_map, hyperlink_map = paragraph_to_marked_text(p)
        if not contains_korean(marked_ko):
            continue
        paras.append(p)
        marked_texts.append((marked_ko, drawing_map, hyperlink_map))

    total_paras = len(paras)

    if total_paras == 0:
        doc.save(out_path)
        return {
            "input_tokens": TOTAL_INPUT_TOKENS,
            "cached_tokens": TOTAL_CACHED_INPUT_TOKENS,
            "output_tokens": TOTAL_OUTPUT_TOKENS,
            "total_tokens": TOTAL_TOKENS,
            "paragraphs_translated": 0,
        }

    # ── 진행률 총량 사전 계산 ─────────────────────────────────────────
    # Pass 2(QA) 대상은 "src 첫 등장이고 heading이 아닌" 단락. Pass 1과
    # 동일한 필터링 규칙을 미리 돌려서 Pass 1/2 합산 진행률을 확정한다.
    qa_estimated = 0
    if enable_qa:
        seen_for_estimate = set()
        for idx_e, (src_e, _, _) in enumerate(marked_texts):
            if src_e in seen_for_estimate:
                continue
            seen_for_estimate.add(src_e)
            if is_heading_paragraph(paras[idx_e]) or looks_like_heading_text(src_e):
                continue
            qa_estimated += 1
    total_work = total_paras + qa_estimated

    # ── Pass 1: 번역 (XML 쓰기는 미루고 메모리에 누적) ─────────────────
    pass1_results: List[Dict] = []

    for idx, p in enumerate(paras):
        src, drawing_map, hyperlink_map = marked_texts[idx]

        # heading 여부를 미리 계산 — UI 매핑 적용 여부와 case-sensitive 재복원
        # 여부를 이 값으로 갈라야 하기 때문.
        is_heading = is_heading_paragraph(p) or looks_like_heading_text(src)

        if enable_cache and src in cache:
            translated = cache[src]
        else:
            # 1) Glossary 치환 (본문 전체)
            gl_pre, gl_map = preprocess_with_glossary_placeholders(src, glossary_entries)
            # 2) UI 텍스트 매핑 치환 — 오직 non-heading에서만.
            #    Heading은 무조건 sentence case로 정규화될 예정이므로 UI 매핑을
            #    적용해봐야 결국 case가 바뀌어 의미가 없고, 오히려 어색해질 수 있다.
            if not is_heading:
                ui_pre, ui_map, _ = preprocess_ui_overrides_in_bold(
                    gl_pre, ui_overrides_clean, start_idx=len(gl_map)
                )
                gl_pre = ui_pre
                if ui_map:
                    gl_map = {**gl_map, **ui_map}

            selected_pattern_examples = select_relevant_patterns(gl_pre, patterns)

            translated = translate_paragraph_with_patterns(
                client=client,
                source_text=gl_pre,
                pattern_examples=selected_pattern_examples,
                model=model,
                translation_mode=translation_mode,
                style_reference=style_guide,
            )

            translated = translated.strip()

            # 1) marker 복구
            translated = repair_bold_markers(translated)
            translated = repair_hl_markers(translated)

            # 1-b) bold 경계 공백 제거
            translated = normalize_bold_spaces(translated)

            # 1-c) HL 마커 누락 폴백
            translated = apply_highlight_fallback(translated, src)

            # 2) glossary 복원 (위치 기반 대소문자)
            translated = restore_glossary_placeholders(translated, gl_map or {})

            # 3) colon label normalize
            translated = normalize_colon_label_line(translated)

            # 4) 남은 한국어 fallback 번역
            if contains_korean(translated):
                translated = translate_remaining_korean(client, translated, model=model)
                translated = repair_bold_markers(translated)
                translated = repair_hl_markers(translated)
                translated = normalize_bold_spaces(translated)
                translated = apply_highlight_fallback(translated, src)
                translated = restore_glossary_placeholders(translated, gl_map or {})
                translated = normalize_colon_label_line(translated)

            # 5) heading / 일반 문단 후처리 (is_heading은 위에서 계산됨)
            if is_heading:
                translated = normalize_heading_text(translated)
                translated = normalize_ui_label_text(translated)
                translated = _cap_first_alpha(translated)
                translated = translated.rstrip()
                if translated.endswith("."):
                    translated = translated[:-1]
            else:
                translated = normalize_ui_in_bold_segments(translated)
                translated = _cap_first_alpha(translated)

            # 6) 마지막 품질 보정
            translated = fix_indefinite_articles(translated)
            translated = capitalize_bullet_lines(translated)
            translated = restore_sentence_period(translated, src)
            translated = normalize_paragraph_breaks(translated)
            # pass-1이 "Draft: X ... Revised: Y" 형태를 뱉는 경우 방어
            translated = strip_meta_version_labels(translated)

            # 7) case-sensitive/DNT glossary 용어의 정확한 대소문자 복원.
            #    Heading에서는 절대 호출하지 않는다 — heading은 항상 sentence
            #    case가 원칙이므로 "Wrapsody eCo" 같은 case-sensitive 용어도
            #    heading에서는 "Wrapsody eco"로 자연스럽게 두는 게 맞다.
            if not is_heading:
                translated = enforce_case_sensitive_glossary(translated, glossary_entries)

            if enable_cache:
                cache[src] = translated

        pass1_results.append({
            "p": p,
            "src": src,
            "translated": translated,
            "drawing_map": drawing_map,
            "hyperlink_map": hyperlink_map,
        })

        if progress_callback:
            progress_callback(idx + 1, total_work)

    # ── Pass 2: batch QA (일관성 검사) ───────────────────────────────
    # src별 그룹핑으로 동일 원문은 한 번만 QA → 결과를 모든 사본에 적용.
    if enable_qa:
        src_groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in pass1_results:
            src_groups[r["src"]].append(r)

        qa_items: List[Dict] = []
        for src, group in src_groups.items():
            rep = group[0]
            if is_heading_paragraph(rep["p"]) or looks_like_heading_text(src):
                continue
            qa_items.append({
                "src": src,
                "translated": rep["translated"],
                "group": group,
            })

        qa_done = 0
        for batch_start in range(0, len(qa_items), qa_batch_size):
            batch = qa_items[batch_start:batch_start + qa_batch_size]
            batch_input = [
                (i, item["src"], item["translated"])
                for i, item in enumerate(batch)
            ]

            try:
                revisions = qa_check_batch(
                    client=client,
                    items=batch_input,
                    style_guide=style_guide,
                    glossary_pairs=glossary_pairs_for_qa,
                    model=model,
                )
            except Exception:
                # QA 실패 시 batch 전체 skip — Pass 1 결과 유지
                revisions = {}

            for i, item in enumerate(batch):
                if i not in revisions:
                    continue
                revised = revisions[i].strip()
                if not revised:
                    continue  # 안전장치: 빈 응답으로 덮어쓰지 않음
                # Pass 2 응답에도 동일한 marker 후처리를 적용
                revised = repair_bold_markers(revised)
                revised = repair_hl_markers(revised)
                revised = normalize_bold_spaces(revised)
                revised = apply_highlight_fallback(revised, item["src"])
                # QA 응답에서도 line break/meta label 정리 (pass-1과 동일 수준으로)
                revised = normalize_paragraph_breaks(revised)
                revised = strip_meta_version_labels(revised)
                # QA가 case-sensitive 용어를 흐트러뜨리는 경우도 복원
                revised = enforce_case_sensitive_glossary(revised, glossary_entries)
                for r in item["group"]:
                    r["translated"] = revised
                    if enable_cache:
                        cache[item["src"]] = revised

            qa_done += len(batch)
            if progress_callback:
                progress_callback(total_paras + qa_done, total_work)

    # ── Final: 모든 결과를 한 번에 XML로 쓰기 ─────────────────────────
    for r in pass1_results:
        _write_paragraph(r["p"], r["translated"], r["drawing_map"], r["hyperlink_map"])

    doc.save(out_path)

    return {
        "input_tokens": TOTAL_INPUT_TOKENS,
        "cached_tokens": TOTAL_CACHED_INPUT_TOKENS,
        "output_tokens": TOTAL_OUTPUT_TOKENS,
        "total_tokens": TOTAL_TOKENS,
        "paragraphs_translated": total_paras,
    }