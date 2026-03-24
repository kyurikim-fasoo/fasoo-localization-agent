import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional, Callable, Any

from docx import Document
from openai import OpenAI


TOTAL_INPUT_TOKENS = 0
TOTAL_CACHED_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_TOKENS = 0

B_OPEN = "⟦B⟧"
B_CLOSE = "⟦/B⟧"

G_PREFIX = "⟦G"
SUFFIX = "⟧"

KOREAN_RE = re.compile(r"[가-힣]")


def reset_token_counters():
    global TOTAL_INPUT_TOKENS, TOTAL_CACHED_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_TOKENS
    TOTAL_INPUT_TOKENS = 0
    TOTAL_CACHED_INPUT_TOKENS = 0
    TOTAL_OUTPUT_TOKENS = 0
    TOTAL_TOKENS = 0


def contains_korean(text: str) -> bool:
    return bool(text and KOREAN_RE.search(text))


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


@dataclass(frozen=True)
class GlossaryEntry:
    ko: str
    en: str
    dnt: bool
    case_sensitive: bool
    product: str
    note: str


def dedupe_glossary_entries(entries: List[GlossaryEntry]) -> List[GlossaryEntry]:
    seen = set()
    result = []
    for e in entries:
        key = (e.ko, e.en, e.product, e.category, e.note)
        if key not in seen:
            seen.add(key)
            result.append(e)
    result.sort(key=lambda e: len(e.ko), reverse=True)
    return result


def build_glossary_entries_from_rows(rows: List[dict]) -> List[GlossaryEntry]:
    entries = []

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

    return dedupe_glossary_entries(entries)


def build_pattern_pairs_from_rows(rows: List[dict]) -> List[Tuple[str, str]]:
    pairs = []
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


def preprocess_with_glossary_placeholders(text: str, entries: List[GlossaryEntry]) -> Tuple[str, Dict[str, GlossaryEntry]]:
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


def restore_glossary_placeholders(text: str, mapping: Dict[str, GlossaryEntry]) -> str:
    out = text

    for ph, entry in mapping.items():
        out = out.replace(ph, entry.en)

    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def paragraph_to_marked_text(paragraph) -> str:
    parts = []
    in_bold = False

    for run in paragraph.runs:
        t = run.text or ""
        if not t:
            continue

        is_bold = bool(run.bold)
        if is_bold and not in_bold:
            parts.append(B_OPEN)
            in_bold = True
        if (not is_bold) and in_bold:
            parts.append(B_CLOSE)
            in_bold = False

        parts.append(t)

    if in_bold:
        parts.append(B_CLOSE)

    return "".join(parts)


def clear_paragraph_runs(paragraph) -> None:
    p = paragraph._p
    for r in list(paragraph.runs):
        p.remove(r._r)


def marked_text_to_runs(paragraph, marked_text: str) -> None:
    clear_paragraph_runs(paragraph)

    tokens = re.split(r"(⟦B⟧|⟦/B⟧)", marked_text)
    bold = False

    for tok in tokens:
        if tok is None or tok == "":
            continue
        if tok == B_OPEN:
            bold = True
            continue
        if tok == B_CLOSE:
            bold = False
            continue

        run = paragraph.add_run(tok)
        if bold:
            run.bold = True


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


def _cap_first_alpha(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s

def repair_bold_markers(text: str) -> str:
    if not text:
        return text

    # ⟦Brule -> ⟦B⟧rule
    text = re.sub(r"⟦B(?!⟧)", f"{B_OPEN}", text)

    # ⟦/Brule -> ⟦/B⟧rule 같은 비정상 케이스 방지
    text = re.sub(r"⟦/B(?!⟧)", f"{B_CLOSE}", text)

    # 혹시 닫는 태그가 아예 사라졌는데 여는 태그만 남은 경우 최소 복구
    open_count = text.count(B_OPEN)
    close_count = text.count(B_CLOSE)
    if open_count > close_count:
        text += B_CLOSE
    elif close_count > open_count:
        # 여는 태그 없이 닫는 태그만 많은 경우는 닫는 태그 제거
        diff = close_count - open_count
        for _ in range(diff):
            text = text.replace(B_CLOSE, "", 1)

    return text    

def capitalize_bold_segments(text: str) -> str:
    def repl(match):
        inner = match.group(1)
        return B_OPEN + _cap_first_alpha(inner) + B_CLOSE

    return re.sub(
        re.escape(B_OPEN) + r"(.*?)" + re.escape(B_CLOSE),
        repl,
        text,
        flags=re.DOTALL,
    )

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
        indent = line[:len(line) - len(stripped)]

        handled = False
        for bp in bullet_prefixes:
            if stripped.startswith(bp):
                rest = stripped[len(bp):]
                rest = _cap_first_alpha(rest)
                out_lines.append(indent + bp + rest)
                handled = True
                break
        if handled:
            continue

        m = num_prefix_re.match(stripped)
        if m:
            pre = stripped[:m.end()]
            rest = stripped[m.end():]
            rest = _cap_first_alpha(rest)
            out_lines.append(indent + pre + rest)
            continue

        out_lines.append(indent + _cap_first_alpha(stripped))

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

def normalize_heading_text(text: str) -> str:
    if not text:
        return text

    # marker 제거 전용 보기용 텍스트가 아니라, marker 포함 상태에서 처리
    # bold marker는 그대로 두고 문자만 정리
    s = text.strip()

    # 끝 마침표 제거
    s = re.sub(r"[.。]+$", "", s)

    # 문장 전체를 sentence case로
    # 일단 plain text 기준으로 처리
    parts = re.split(rf"({re.escape(B_OPEN)}|{re.escape(B_CLOSE)})", s)

    normalized_parts = []
    first_text_done = False

    for part in parts:
        if part in (B_OPEN, B_CLOSE):
            normalized_parts.append(part)
            continue

        if not part:
            normalized_parts.append(part)
            continue

        lower = part[:1].upper() + part[1:].lower() if not first_text_done else part.lower()
        normalized_parts.append(lower)
        if part.strip():
            first_text_done = True

    s = "".join(normalized_parts)

    # 이중 공백 정리
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s

def paragraph_has_hyperlink(paragraph) -> bool:
    return paragraph._p.xpath(".//w:hyperlink") != []


def _write_paragraph(p, translated_marked: str) -> None:
    if is_heading_paragraph(p) and not paragraph_has_hyperlink(p):
        marked_text_to_runs(p, translated_marked)
        return

    if paragraph_has_hyperlink(p):
        set_paragraph_text_preserve_style(p, translated_marked)
        return

    marked_text_to_runs(p, translated_marked)


def normalize_for_scoring(text: str) -> str:
    text = text.replace(B_OPEN, " ").replace(B_CLOSE, " ")
    text = re.sub(r"⟦G\d+⟧", " ", text)
    text = text.replace("~", " ")
    return text


def tokenize_koreanish(text: str) -> List[str]:
    norm = normalize_for_scoring(text)
    return re.findall(r"[가-힣A-Za-z0-9]+", norm)


def select_relevant_patterns(
    source_text: str,
    patterns: List[Tuple[str, str]],
    max_pattern: int = 3,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
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

    selected_patterns = [(ko, en) for score, ko, en in scored_patterns[:max_pattern]]

    return selected_patterns


def translate_paragraph_with_patterns(
    client: OpenAI,
    source_text: str,
    pattern_examples: List[Tuple[str, str]],
    model: str = "gpt-5.2",
    translation_mode: str = "Manual",
) -> str:
    global TOTAL_INPUT_TOKENS, TOTAL_CACHED_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_TOKENS

    pattern_block = "\n".join([f"- {ko} -> {en}" for ko, en in pattern_examples]) or "(none)"

    if translation_mode == "UI":
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
- Preserve markers EXACTLY: ⟦G#⟧, ⟦B⟧, ⟦/B⟧.
- Treat glossary placeholders as fixed terms.
- Use the pattern examples only as reference guidance.
- Do not copy irrelevant examples.
- Avoid repetition and awkward literal wording.
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
        marked_ko = paragraph_to_marked_text(p)
        if not contains_korean(marked_ko):
            continue
        paras.append(p)
        marked_texts.append(marked_ko)

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
        src = marked_texts[idx]

        if enable_cache and src in cache:
            translated = cache[src]
            _write_paragraph(p, translated)
            if progress_callback:
                progress_callback(idx + 1, total_paras)
            continue

        gl_pre, gl_map = preprocess_with_glossary_placeholders(src, glossary_entries)

        selected_pattern_examples = select_relevant_patterns(
            gl_pre,
            patterns,
        )

        translated = translate_paragraph_with_patterns(
            client=client,
            source_text=gl_pre,
            pattern_examples=selected_pattern_examples,
            model=model,
            translation_mode=translation_mode,
        )

        translated = translated.strip()
        translated = repair_bold_markers(translated)
        translated = restore_glossary_placeholders(translated, gl_map or {})
        translated = capitalize_bold_segments(translated)
        translated = capitalize_bullet_lines(translated)

        if is_heading_paragraph(p):
            translated = normalize_heading_text(translated)

        translated = normalize_paragraph_breaks(translated)

        if enable_cache:
            cache[src] = translated

        if is_heading_paragraph(p):
            translated = normalize_heading_text(translated)

        _write_paragraph(p, translated)

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