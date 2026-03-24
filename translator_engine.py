import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional, Callable, Any

from docx import Document
from openai import OpenAI


B_OPEN = "⟦B⟧"
B_CLOSE = "⟦/B⟧"

G_PREFIX = "⟦G"
SUFFIX = "⟧"

KOREAN_RE = re.compile(r"[가-힣]")
PLACEHOLDER_RE = re.compile(r"⟦G\d+⟧")


# ----------------------------
# Glossary
# ----------------------------
@dataclass(frozen=True)
class GlossaryEntry:
    ko: str
    en: str
    dnt: bool
    case_sensitive: bool


def build_glossary_entries(rows: List[dict]) -> List[GlossaryEntry]:
    entries = []

    for r in rows:
        ko = str(r.get("KO", "")).strip()
        en = str(r.get("EN", "")).strip()
        if not ko or not en:
            continue

        entries.append(
            GlossaryEntry(
                ko=ko,
                en=en,
                dnt=str(r.get("DNT", "")).lower() in ["true", "y", "yes"],
                case_sensitive=str(r.get("Case-sensitive", "")).lower() in ["true", "y", "yes"],
            )
        )

    # 긴 것 먼저
    entries.sort(key=lambda x: len(x.ko), reverse=True)
    return entries


# ----------------------------
# Marker handling
# ----------------------------
def paragraph_to_marked_text(p) -> str:
    parts = []
    in_bold = False

    for run in p.runs:
        text = run.text or ""
        if not text:
            continue

        if run.bold and not in_bold:
            parts.append(B_OPEN)
            in_bold = True
        elif not run.bold and in_bold:
            parts.append(B_CLOSE)
            in_bold = False

        parts.append(text)

    if in_bold:
        parts.append(B_CLOSE)

    return "".join(parts)


def marked_text_to_runs(p, text: str):
    p.clear()

    tokens = re.split(r"(⟦B⟧|⟦/B⟧)", text)
    bold = False

    for tok in tokens:
        if tok == B_OPEN:
            bold = True
            continue
        if tok == B_CLOSE:
            bold = False
            continue

        run = p.add_run(tok)
        run.bold = bold


def repair_bold(text: str) -> str:
    text = re.sub(r"⟦\s*B\s*", B_OPEN, text)
    text = re.sub(r"⟦\s*/\s*B\s*", B_CLOSE, text)
    return text


# ----------------------------
# Glossary replace
# ----------------------------
def apply_glossary(text: str, entries: List[GlossaryEntry]):
    mapping = {}
    idx = 0

    for e in entries:
        if e.ko in text:
            key = f"{G_PREFIX}{idx}{SUFFIX}"
            text = text.replace(e.ko, key)
            mapping[key] = e.en
            idx += 1

    return text, mapping


def restore_glossary(text: str, mapping: Dict[str, str]):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


# ----------------------------
# Heading / UI normalize
# ----------------------------
UI_LOWER = {
    "name", "list", "details", "settings", "information",
    "field", "filter", "menu", "status", "type",
    "history", "option", "message", "group", "user",
    "rule", "policy", "log"
}


def sentence_case(text: str):
    first = True
    result = []

    for part in re.split(r"(⟦G\d+⟧)", text):
        if PLACEHOLDER_RE.fullmatch(part):
            result.append(part)
            continue

        chars = list(part)

        for i, ch in enumerate(chars):
            if ch.isalpha():
                if first:
                    chars[i] = ch.upper()
                    first = False
                else:
                    chars[i] = ch.lower()
                break

        result.append("".join(chars))

    return "".join(result)


def normalize_ui(text: str):
    def repl(m):
        word = m.group(0)
        if word.lower() in UI_LOWER:
            return word.lower()
        return word

    return re.sub(r"[A-Za-z]+", repl, text)


def normalize_heading(text: str):
    text = sentence_case(text)
    text = normalize_ui(text)
    text = re.sub(r"[.]+$", "", text)
    return text.strip()


# ----------------------------
# Core translate
# ----------------------------
def translate_paragraph(client, text, model):
    prompt = f"""
Translate Korean to natural English.

Rules:
- Keep markers ⟦G#⟧, ⟦B⟧ intact
- Do NOT add unnecessary punctuation
- Prefer concise UI-style wording

{text}
"""

    res = client.responses.create(
        model=model,
        input=prompt,
    )

    return res.output_text.strip()


# ----------------------------
# Main
# ----------------------------
def translate_document(
    in_path,
    out_path,
    glossary_rows,
    api_key,
    model="gpt-5.2"
):
    client = OpenAI(api_key=api_key)
    doc = Document(in_path)

    glossary = build_glossary_entries(glossary_rows)

    for p in doc.paragraphs:
        src = paragraph_to_marked_text(p)

        if not KOREAN_RE.search(src):
            continue

        text, mapping = apply_glossary(src, glossary)

        translated = translate_paragraph(client, text, model)
        translated = repair_bold(translated)

        # 헤딩 처리
        if p.style and "Heading" in p.style.name:
            translated = normalize_heading(translated)

        translated = restore_glossary(translated, mapping)

        marked_text_to_runs(p, translated)

    doc.save(out_path)