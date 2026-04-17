import copy
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
H_OPEN_RE    = re.compile(r"⟦H(\d+)⟧")
H_CLOSE_RE   = re.compile(r"⟦/H(\d+)⟧")
HL_OPEN_RE   = re.compile(r"⟦HL:([a-zA-Z]+)⟧")
HL_CLOSE     = "⟦/HL⟧"
ALL_MARKER_RE = re.compile(r"(⟦B⟧|⟦/B⟧|⟦D\d+⟧|⟦H\d+⟧|⟦/H\d+⟧|⟦HL:[a-zA-Z]+⟧|⟦/HL⟧)")
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
            t_elem = _lxml_text_elem(child)
            if t_elem is None:
                continue
            text = t_elem.text or ""
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
    """Create a fresh w:r with a cloned rPr template, bold flag, and text."""
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

    # ── PATH 1: Drawing ────────────────────────────────────────────────────
    if has_drawing:
        segs = _parse_marked_segments(translated_marked)
        text_slots: List[List[Tuple]] = []
        current: List[Tuple] = []
        for seg in segs:
            if seg[0] == "drawing":
                text_slots.append(current)
                current = []
            else:
                current.append(seg)
        text_slots.append(current)

        # Walk DIRECT children of p_elem to build run groups.
        # Hyperlink nodes are NOT split — their inner runs are handled separately.
        # Drawing runs act as group separators; everything else (commentRange*, pPr,
        # hyperlinks) is left in place.
        direct_children   = list(p_elem)
        drawing_positions: List[int] = []   # child indices of drawing runs
        run_groups: List[List]        = []
        cur_direct: List              = []

        for child_idx, child in enumerate(direct_children):
            ctag = child.tag.split("}")[1] if "}" in child.tag else child.tag
            if ctag == "r" and _run_is_non_text(child):
                run_groups.append(cur_direct)
                drawing_positions.append(child_idx)
                cur_direct = []
            elif (
                ctag == "r"
                and not _run_is_comment_ref(child)
                and _lxml_text_elem(child) is not None
            ):
                cur_direct.append(child)
            # hyperlink / commentRange* / pPr: skip (not text, not drawing)
        run_groups.append(cur_direct)

        # rPr template from the first available non-drawing text run
        rPr_tmpl = None
        for group in run_groups:
            for r in group:
                c = r.find(f"{{{_W}}}rPr")
                if c is not None:
                    rPr_tmpl = _make_rPr_template(c)
                    break
            if rPr_tmpl is not None:
                break

        # Rebuild each run group from its translated text slot
        for gi, group in enumerate(run_groups):
            slot = text_slots[gi] if gi < len(text_slots) else []

            # Determine insertion position in p_elem
            if group:
                insert_pos = list(p_elem).index(group[0])
            elif gi > 0 and drawing_positions:
                prev_drawing = direct_children[drawing_positions[gi - 1]]
                insert_pos   = list(p_elem).index(prev_drawing) + 1
            else:
                pPr        = p_elem.find(f"{{{_W}}}pPr")
                insert_pos = (list(p_elem).index(pPr) + 1) if pPr is not None else 0

            # Remove only the direct text runs in this group (not hyperlink inner runs)
            for r in group:
                p_elem.remove(r)

            # Rebuild: one new run per non-hyperlink text segment
            text_segs = [
                (b, t) for b, t in slot
                if not (isinstance(b, tuple) and b[0] == "h")
            ]
            for j, (props, text) in enumerate(text_segs):
                p_elem.insert(insert_pos + j, _make_run(rPr_tmpl, text, props))

        # Sync hyperlink inner-run text (same as PATH 2)
        for hi, hl_info in hyperlink_map.items():
            hl_runs = hl_info["runs"]
            hl_text = "".join(
                t for b, t in segs if isinstance(b, tuple) and b == ("h", hi)
            )
            if hl_runs:
                _set_run_text(hl_runs[0], hl_text, False)
                for r in hl_runs[1:]:
                    _set_run_text(r, "", False)
        return

    # ── PATH 2: Hyperlink ──────────────────────────────────────────────────
    if has_hyperlink:
        segs = _parse_marked_segments(translated_marked)

        # rPr template from the first non-commentRef direct-child run
        rPr_tmpl = None
        for r in p_elem.findall(f"{{{_W}}}r"):
            if _run_is_comment_ref(r):
                continue
            c = r.find(f"{{{_W}}}rPr")
            if c is not None:
                rPr_tmpl = _make_rPr_template(c)
                break

        # Write hyperlink text → first inner run of each hyperlink, clear rest
        for hi, hl_info in hyperlink_map.items():
            hl_runs = hl_info["runs"]
            hl_text = "".join(
                t for b, t in segs if isinstance(b, tuple) and b == ("h", hi)
            )
            if hl_runs:
                _set_run_text(hl_runs[0], hl_text, False)
                for r in hl_runs[1:]:
                    _set_run_text(r, "", False)

        # Split plain segs at hyperlink boundaries
        groups: List[List[Tuple]] = []
        cur: List[Tuple] = []
        for b, t in segs:
            if isinstance(b, tuple) and b[0] == "h":
                groups.append(cur)
                cur = []
            elif b != "drawing":
                cur.append((b, t))
        groups.append(cur)
        # groups[0] = before hl[0], groups[1] = between hl[0] and hl[1], …

        # Collect hyperlink elements in document order
        hl_elems = [
            child for child in p_elem
            if (child.tag.split("}")[1] if "}" in child.tag else child.tag) == "hyperlink"
            and any(info["elem"] is child for info in hyperlink_map.values())
        ]

        # Remove only regular (non-commentRef) direct runs
        direct_runs  = p_elem.findall(f"{{{_W}}}r")
        cmt_ref_runs = [r for r in direct_runs if _run_is_comment_ref(r)]
        for r in [r for r in direct_runs if not _run_is_comment_ref(r)]:
            p_elem.remove(r)

        # Insert rebuilt runs before/after each hyperlink element
        for gi, group_segs in enumerate(groups):
            if gi < len(hl_elems):
                hl_elem = hl_elems[gi]
                hl_pos  = list(p_elem).index(hl_elem)
                for j, (props, text) in enumerate(group_segs):
                    p_elem.insert(hl_pos + j, _make_run(rPr_tmpl, text, props))
                    hl_pos += 1
            else:
                for props, text in group_segs:
                    if cmt_ref_runs:
                        cmt_pos = list(p_elem).index(cmt_ref_runs[0])
                        p_elem.insert(cmt_pos, _make_run(rPr_tmpl, text, props))
                    else:
                        p_elem.append(_make_run(rPr_tmpl, text, props))
        return

    # ── PATH 3: Plain paragraph ────────────────────────────────────────────
    direct_runs  = p_elem.findall(f"{{{_W}}}r")
    cmt_ref_runs = [r for r in direct_runs if _run_is_comment_ref(r)]
    regular_runs = [r for r in direct_runs if not _run_is_comment_ref(r)]

    rPr_tmpl = None
    for r in regular_runs:
        c = r.find(f"{{{_W}}}rPr")
        if c is not None:
            rPr_tmpl = _make_rPr_template(c)
            break

    for r in regular_runs:
        p_elem.remove(r)

    segs = _parse_marked_segments(translated_marked)
    text_segs = [(b, t) for b, t in segs if b != "drawing"]

    for i, (props, text) in enumerate(text_segs):
        r = _make_run(rPr_tmpl, text, props)
        if cmt_ref_runs:
            cmt_pos = list(p_elem).index(cmt_ref_runs[0])
            p_elem.insert(cmt_pos + i, r)
        else:
            pPr     = p_elem.find(f"{{{_W}}}pPr")
            ins_idx = (list(p_elem).index(pPr) + 1) if pPr is not None else 0
            p_elem.insert(ins_idx + i, r)


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
Translate Korean to natural, professional English.

Rules:
- Preserve markers EXACTLY: ⟦G#⟧, ⟦B⟧, ⟦/B⟧, ⟦D#⟧, ⟦H#⟧/⟦/H#⟧, ⟦HL:colour⟧/⟦/HL⟧.
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
- Do not force title case.
- For headings, concise phrase-style English is preferred.
- NEVER merge markers with words. "⟦B⟧rule" is correct; "⟦Brule" is invalid.
- Keep markers as separate tokens.
- Output ONLY the translated text. No explanation, no extra lines.
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
- Preserve markers EXACTLY: ⟦B⟧, ⟦/B⟧, ⟦G#⟧.
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

    # ── 문서 내 기존 영문 단락 추출 (톤앤매너 참조용) ─────────────────────
    english_samples: List[str] = []
    for p in iter_all_paragraphs(doc):
        text = p.text.strip()
        # 한국어 없고, 충분히 긴 (> 15자) 영문 단락만 수집
        if text and not contains_korean(text) and len(text) > 15:
            english_samples.append(text)
        if sum(len(s) for s in english_samples) >= 600:
            break

    style_reference = ""
    if english_samples:
        style_reference = "\n".join(f"  {s}" for s in english_samples[:6])

    paras: List = []
    marked_texts: List[str] = []

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
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "paragraphs_translated": 0,
        }

    for idx, p in enumerate(paras):
        src, drawing_map, hyperlink_map = marked_texts[idx]

        if enable_cache and src in cache:
            translated = cache[src]
            _write_paragraph(p, translated, drawing_map, hyperlink_map)
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
            style_reference=style_reference,
        )

        translated = translated.strip()

        # 1) marker 복구
        translated = repair_bold_markers(translated)
        translated = repair_hl_markers(translated)

        # 1-b) bold 경계 공백 제거 (LLM이 ⟦B⟧ Rule ⟦/B⟧처럼 마커 안에 공백을 넣는 경우)
        translated = normalize_bold_spaces(translated)

        # 1-c) LLM이 HL 마커를 누락한 경우 폴백 처리
        translated = apply_highlight_fallback(translated, src)

        # 2) glossary 복원 (위치 기반 대소문자 적용)
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
            # fallback 번역 후에도 glossary 용어가 한국어로 남아 있을 수 있으므로 재복원
            translated = restore_glossary_placeholders(translated, gl_map or {})
            translated = normalize_colon_label_line(translated)

        # 5) heading 판단: 스타일 기반 + 소스 텍스트 기반 (번역문은 사용하지 않음)
        # looks_like_heading_text는 한국어 소스에서는 False를 반환하므로
        # 실제 Heading 스타일이 없는 단락은 heading으로 처리하지 않음.
        is_heading = (
            is_heading_paragraph(p)
            or looks_like_heading_text(src)
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
            # Cap first alpha in the whole sentence (catches leading bold terms)
            translated = _cap_first_alpha(translated)

        # 7) 마지막 품질 보정
        translated = fix_indefinite_articles(translated)
        translated = capitalize_bullet_lines(translated)
        translated = restore_sentence_period(translated, src)
        translated = normalize_paragraph_breaks(translated)

        if enable_cache:
            cache[src] = translated

        _write_paragraph(p, translated, drawing_map, hyperlink_map)

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