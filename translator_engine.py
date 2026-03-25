import re
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

G_PREFIX = "⟦G"
D_PREFIX = "⟦D"
SUFFIX = "⟧"

KOREAN_RE = re.compile(r"[가-힣]")
PLACEHOLDER_RE = re.compile(r"⟦G\d+⟧")
DRAWING_PH_RE = re.compile(r"⟦D\d+⟧")
ALL_MARKER_RE = re.compile(r"(⟦B⟧|⟦/B⟧|⟦D\d+⟧)")
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
    out = text
    mapping: Dict[str, GlossaryEntry] = {}
    idx = 0

    for entry in entries:
        if entry.ko and entry.ko in out:
            ph = make_marker(G_PREFIX, idx)
            idx += 1
            out = out.replace(entry.ko, ph)
            mapping[ph] = entry

    return out, mapping


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


def _lxml_all_runs(p_elem):
    """Return every w:r in the paragraph in document order,
    including those nested inside w:hyperlink."""
    return p_elem.findall(f".//{{{_W}}}r")


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
    """Split a marked string into (is_bold_or_drawing_sentinel, text) pairs.

    Drawing placeholders ⟦D0⟧ are returned as ('drawing', '⟦D0⟧') tuples so
    the write-back step can skip them when assigning text to runs.
    """
    segments: List[Tuple] = []
    tokens = ALL_MARKER_RE.split(marked)
    bold = False
    buf: List[str] = []

    for tok in tokens:
        if tok == B_OPEN:
            if buf:
                segments.append((False, "".join(buf)))
                buf = []
            bold = True
        elif tok == B_CLOSE:
            if buf:
                segments.append((True, "".join(buf)))
                buf = []
            bold = False
        elif DRAWING_PH_RE.fullmatch(tok):
            if buf:
                segments.append((bold, "".join(buf)))
                buf = []
            segments.append(("drawing", tok))
        elif tok:
            buf.append(tok)

    if buf:
        segments.append((bold, "".join(buf)))

    return [(b, t) for b, t in segments if t]


def paragraph_to_marked_text(paragraph) -> Tuple[str, Dict[str, Any]]:
    """
    Extract paragraph text with ⟦B⟧…⟦/B⟧ bold markers AND ⟦D0⟧ drawing
    placeholders, reading all w:r elements via lxml (including those inside
    w:hyperlink nodes).

    Returns
    -------
    marked_text : str
        Text with bold and drawing markers ready for the LLM.
    drawing_map : dict
        Mapping of placeholder string (e.g. '⟦D0⟧') -> lxml run element,
        used only for reference; the runs stay in place in the XML tree.
    """
    parts: List[str] = []
    in_bold = False
    d_idx = 0
    drawing_map: Dict[str, Any] = {}

    for r in _lxml_all_runs(paragraph._p):
        if _run_is_non_text(r):
            # Close bold if open, then emit a drawing placeholder
            if in_bold:
                parts.append(B_CLOSE)
                in_bold = False
            ph = f"{D_PREFIX}{d_idx}{SUFFIX}"
            parts.append(ph)
            drawing_map[ph] = r
            d_idx += 1
            continue

        t_elem = _lxml_text_elem(r)
        if t_elem is None:
            continue

        text = t_elem.text or ""
        if not text:
            continue

        is_bold = _lxml_run_is_bold(r)
        if is_bold and not in_bold:
            parts.append(B_OPEN)
            in_bold = True
        elif not is_bold and in_bold:
            parts.append(B_CLOSE)
            in_bold = False

        parts.append(text)

    if in_bold:
        parts.append(B_CLOSE)

    return "".join(parts), drawing_map


def _write_paragraph_inplace(p_elem, translated_marked: str, drawing_map: Dict) -> None:
    """
    Write translated text back into the paragraph XML in-place.

    Rules
    -----
    * Only w:t content is touched — w:drawing, w:sym, w:hyperlink wrappers,
      rPr colour/font settings, etc. are ALL preserved exactly as-is.
    * Bold (w:b) is updated to match the translated_marked bold markers.
    * Drawing placeholders ⟦D#⟧ in translated_marked are ignored during
      text distribution (the drawing runs stay in their original XML position).
    * If there are more translated text segments than text runs, excess text
      is appended to the last run.
    * Runs with no matching translated segment have their w:t cleared.
    """
    segments = _parse_marked_segments(translated_marked)
    # Only text segments drive the run-filling loop; drawing placeholders are skipped
    text_segments = [(b, t) for b, t in segments if b != "drawing"]

    text_runs = [r for r in _lxml_all_runs(p_elem) if _lxml_text_elem(r) is not None]

    if not text_runs:
        return

    n_runs = len(text_runs)
    seg_idx = 0

    for run_idx, r in enumerate(text_runs):
        t_elem = _lxml_text_elem(r)
        rPr   = r.find(f"{{{_W}}}rPr")

        if seg_idx >= len(text_segments):
            t_elem.text = ""
            t_elem.attrib.pop(f"{{{_XML_SPACE}}}space", None)
            continue

        if run_idx == n_runs - 1:
            # Last run: absorb all remaining text segments
            text      = "".join(t for _, t in text_segments[seg_idx:])
            is_bold_s = text_segments[seg_idx][0]
        else:
            is_bold_s, text = text_segments[seg_idx]
            seg_idx += 1

        t_elem.text = text

        # Preserve leading/trailing whitespace
        if text and (text[0] == " " or text[-1] == " "):
            t_elem.set(f"{{{_XML_SPACE}}}space", "preserve")
        else:
            t_elem.attrib.pop(f"{{{_XML_SPACE}}}space", None)

        # Update bold
        if rPr is None:
            if is_bold_s:
                rPr = etree.SubElement(r, f"{{{_W}}}rPr")
                etree.SubElement(rPr, f"{{{_W}}}b")
                r.insert(0, rPr)
        else:
            b_elem = rPr.find(f"{{{_W}}}b")
            if is_bold_s and b_elem is None:
                etree.SubElement(rPr, f"{{{_W}}}b")
            elif not is_bold_s and b_elem is not None:
                rPr.remove(b_elem)


def strip_bold_markers(text: str) -> str:
    return text.replace(B_OPEN, "").replace(B_CLOSE, "")


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


def normalize_paragraph_breaks(s: str) -> str:
    if not s:
        return s
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip("\n")
    return s


def iter_all_paragraphs(doc: Document) -> Iterable:
    for p in doc.paragraphs:
        yield p
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p


def is_heading_paragraph(p) -> bool:
    if not p.style or not getattr(p.style, "name", None):
        return False
    name = (p.style.name or "").lower()
    return name.startswith("heading") or name in ("title",)


def paragraph_has_hyperlink(paragraph) -> bool:
    return paragraph._p.xpath(".//w:hyperlink") != []


def _write_paragraph(p, translated_marked: str, drawing_map: Optional[Dict] = None) -> None:
    """Write translated text into the paragraph, preserving all XML structure.

    Always uses in-place w:t replacement so that hyperlink wrappers, drawing
    runs, rPr colour/font attributes, and every other non-text element survives
    unchanged.  Only w:t content and w:b bold flags are updated.
    """
    if is_heading_paragraph(p):
        # Headings must never end with a period.
        translated_marked = translated_marked.rstrip()
        if translated_marked.endswith("."):
            translated_marked = translated_marked[:-1].rstrip()

    _write_paragraph_inplace(p._p, translated_marked, drawing_map or {})


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
    s = strip_bold_markers(text).strip()

    if not s:
        return False
    if "\n" in s:
        return False
    if len(s) > 50:
        return False
    if s.endswith(":"):
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

    prompt = f"""
Translate Korean to natural, professional English.

Rules:
- Preserve markers EXACTLY: ⟦G#⟧, ⟦B⟧, ⟦/B⟧, ⟦D#⟧.
- ⟦G#⟧ = fixed glossary term — do NOT translate.
- ⟦D#⟧ = inline icon/image — keep it exactly where it naturally fits in the translated sentence.
- Use the pattern examples only as reference guidance.
- Do not copy irrelevant examples.
- Avoid repetition and awkward literal wording.
- Do not force title case.
- For headings, concise phrase-style English is preferred.
- NEVER merge markers with words. "⟦B⟧rule" is correct; "⟦Brule" is invalid.
- Keep markers as separate tokens.
- Output ONLY the translated text. No explanation, no extra lines.
{style_rules}
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
- Preserve markers EXACTLY: ⟦B⟧, ⟦/B⟧.
- Do not change text that is already good English.
- Only translate the remaining Korean parts.
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


def translate_document(
    in_path: str,
    out_path: str,
    glossary_rows: List[dict],
    pattern_rows: List[dict],
    api_key: str,
    enable_cache: bool = True,
    model: str = "gpt-5.2",
    translation_mode: str = "Manual",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    reset_token_counters()

    glossary_entries = build_glossary_entries_from_rows(glossary_rows)
    patterns = build_pattern_pairs_from_rows(pattern_rows)

    client = OpenAI(api_key=api_key)
    doc = Document(in_path)
    cache: Dict[str, str] = {}

    paras: List = []
    marked_texts: List[str] = []

    for p in iter_all_paragraphs(doc):
        marked_ko, drawing_map = paragraph_to_marked_text(p)
        if not contains_korean(marked_ko):
            continue
        paras.append(p)
        marked_texts.append((marked_ko, drawing_map))

    total_paras = len(paras)

    if total_paras == 0:
        doc.save(out_path)
        return {
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "paragraphs_translated": 0,
        }

    for idx, p in enumerate(paras):
        src, drawing_map = marked_texts[idx]

        if enable_cache and src in cache:
            translated = cache[src]
            _write_paragraph(p, translated, drawing_map)
            if progress_callback:
                progress_callback(idx + 1, total_paras)
            continue

        gl_pre, gl_map = preprocess_with_glossary_placeholders(src, glossary_entries)
        selected_pattern_examples = select_relevant_patterns(gl_pre, patterns)

        translated = translate_paragraph_with_patterns(
            client=client,
            source_text=gl_pre,
            pattern_examples=selected_pattern_examples,
            model=model,
            translation_mode=translation_mode,
        )

        translated = translated.strip()

        # 1) marker 복구
        translated = repair_bold_markers(translated)

        # 2) glossary 복원 (위치 기반 대소문자 적용)
        translated = restore_glossary_placeholders(translated, gl_map or {})

        # 3) colon label normalize
        translated = normalize_colon_label_line(translated)

        # 4) 남은 한국어 fallback 번역
        if contains_korean(translated):
            translated = translate_remaining_korean(client, translated, model=model)
            translated = repair_bold_markers(translated)
            # fallback 번역 후에도 glossary 용어가 한국어로 남아 있을 수 있으므로 재복원
            translated = restore_glossary_placeholders(translated, gl_map or {})
            translated = normalize_colon_label_line(translated)

        # 5) heading 판단: source + translated 둘 다 사용
        is_heading = (
            is_heading_paragraph(p)
            or looks_like_heading_text(src)
            or looks_like_heading_text(translated)
        )

        # 6) heading / 일반 문단 후처리
        if is_heading:
            translated = normalize_heading_text(translated)
            translated = normalize_ui_label_text(translated)
            translated = _cap_first_alpha(translated)
            translated = translated.rstrip()
            if translated.endswith("."):
                translated = translated[:-1]
        else:
            translated = normalize_ui_in_bold_segments(translated)

        # 7) 마지막 품질 보정
        translated = fix_indefinite_articles(translated)
        translated = capitalize_bullet_lines(translated)
        translated = normalize_paragraph_breaks(translated)

        if enable_cache:
            cache[src] = translated

        _write_paragraph(p, translated, drawing_map)

        if progress_callback:
            progress_callback(idx + 1, total_paras)

    doc.save(out_path)

    return {
        "input_tokens": TOTAL_INPUT_TOKENS,
        "cached_tokens": TOTAL_CACHED_INPUT_TOKENS,
        "output_tokens": TOTAL_OUTPUT_TOKENS,
        "total_tokens": TOTAL_TOKENS,
        "paragraphs_translated": total_paras,
    }