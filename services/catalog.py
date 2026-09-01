"""
i18n 카탈로그 → 글로서리 / UI 라벨 후보 추출.

전제:
    제품의 다국어 메시지 카탈로그는 key로 정렬돼 있다. 즉 "어느 한국어가 어느
    영어에 대응하는가"라는 정렬 문제가 애초에 없다. 그래서 이 모듈은 LLM을
    쓰지 않는다 — join·집계·분류만으로 후보가 나온다. 판단이 필요한 지점
    (1:N 충돌 대표형 선택, 용어성 판정)은 사람이 화면에서 고른다.

지원 포맷:
    고객사마다 파일 생김새가 다르므로 아래를 자동 감지한다.
      A. 배열 + 언어 필드   [{"key":"a.b", "en":"Save"}]
      B. 배열 + 양쪽 언어   [{"key":"a.b", "en":"Save", "ko":"저장"}]
      C. 평탄 객체          {"a.b": "저장"}          (i18next/vue-i18n)
      D. 중첩 객체          {"a": {"b": "저장"}}     → 경로를 key로 평탄화
      E. 파일 1개 / 2개     위 어느 것이든

    어느 컬럼이 한국어인지는 파일명이 아니라 **한글 비율**로 판정한다.
    파일명에 ko/en 표시가 없어도 동작하게 하기 위함.
"""
from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_KOREAN_RE = re.compile(r"[가-힣]")
# key로 쓰일 법한 필드명 (우선순위 순)
_KEY_FIELD_HINTS = ("key", "id", "msgid", "code", "name", "property")
# 값 필드가 아님이 분명한 것 — 메타데이터
_META_FIELDS = ("comment", "description", "context", "note", "updated", "created")

LABEL_MAX_KO_CHARS = 25
LABEL_MAX_EN_WORDS = 5

# `{0}` `%s` `{{name}}` — 값이 끼워지는 메시지 템플릿. 용어가 아니라 패턴이다.
_PLACEHOLDER_RE = re.compile(r"\{\d+\}|\{\{|%[sd]\b")
# 한국어 종결어미 — 마침표가 없어도 문장인 경우를 잡는다.
#   강한 표지는 위치를 가리지 않고 찾는다. 실제 카탈로그에는
#   "시간을 입력하세요. (단위: 초)" 처럼 **문장이 끝난 뒤 괄호가 붙는** 항목이
#   흔해서, 끝만 봐서는 놓친다. UI 라벨에 '습니다/하세요'가 들어갈 일은 없다.
_SENT_STRONG_RE = re.compile(r"(습니다|합니다|입니다|하세요|하십시오|십시오)")
#   약한 표지는 다른 단어 안에 우연히 들어갈 수 있어 끝에서만 본다.
_SENT_TAIL_RE = re.compile(r"(한다|된다|세요|어요|아요|겠다|하라|해라)$")
# 조사로 끝나면 용어가 아니라 문장 조각이다 ("…취약점 검출부터").
_JOSA_TAIL_RE = re.compile(r"(부터|까지|에서|으로|에게|보다|처럼|만큼|이며|하며)$")


def _ko_core(ko: str) -> str:
    """끝의 문장부호·괄호·따옴표를 털어낸 몸통 — 종결어미 판정용."""
    return ko.strip().rstrip(").]:;,\"'”’ ").strip()


# ──────────────────────────────────────────────────────────────────────
# 파싱
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ParsedFile:
    """파일 하나를 '컬럼 여러 개'로 정규화한 결과."""
    name: str
    shape: str                              # array | flat | nested
    key_field: str
    columns: Dict[str, Dict[str, str]]      # 컬럼명 -> {key: text}
    n_keys: int
    duplicate_keys: int = 0

    def describe(self) -> str:
        cols = ", ".join(self.columns)
        shape_ko = {"array": "배열형", "flat": "평탄 객체", "nested": "중첩 객체"}[self.shape]
        dup = f" · 중복키 {self.duplicate_keys:,}" if self.duplicate_keys else ""
        return f"{shape_ko} · key={self.key_field} · 컬럼[{cols}] · {self.n_keys:,}개{dup}"


def _flatten(obj: Any, prefix: str = "") -> Dict[str, str]:
    """중첩 객체를 'a.b.c' 경로 key로 평탄화. 문자열 값만 남긴다."""
    out: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, str):
        out[prefix] = obj
    return out


def _pick_key_field(rows: List[dict]) -> Optional[str]:
    """배열형에서 key 역할을 하는 필드 찾기."""
    fields = [f for f in rows[0] if isinstance(rows[0].get(f), (str, int))]
    for hint in _KEY_FIELD_HINTS:
        for f in fields:
            if f.lower() == hint:
                return f
    # 힌트가 없으면 '값이 거의 다 고유하고 한글이 없는' 필드를 고른다
    best, best_score = None, 0.0
    for f in fields:
        vals = [str(r.get(f, "")) for r in rows if r.get(f)]
        if not vals:
            continue
        uniq = len(set(vals)) / len(vals)
        head = vals[:500]
        ko = sum(1 for v in head if _KOREAN_RE.search(v)) / len(head)
        score = uniq * (1 - ko)
        if score > best_score:
            best, best_score = f, score
    return best if best_score > 0.8 else None


def parse_json(name: str, data: Any) -> ParsedFile:
    """JSON 한 덩어리를 ParsedFile로. 형태를 인식하지 못하면 ValueError."""
    if isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
        if not rows:
            raise ValueError("배열이지만 객체가 없습니다.")
        key_field = _pick_key_field(rows)
        if key_field is None:
            raise ValueError("key 역할을 하는 필드를 찾지 못했습니다.")
        value_fields = [
            f for f in rows[0]
            if f != key_field
            and isinstance(rows[0].get(f), str)
            and f.lower() not in _META_FIELDS
        ]
        if not value_fields:
            raise ValueError("텍스트 값 필드가 없습니다.")
        columns: Dict[str, Dict[str, str]] = {f: {} for f in value_fields}
        seen: Counter = Counter()
        for r in rows:
            k = str(r.get(key_field, "")).strip()
            if not k:
                continue
            seen[k] += 1
            for f in value_fields:
                v = r.get(f)
                if isinstance(v, str) and v.strip():
                    columns[f][k] = v
        dupes = sum(1 for c in seen.values() if c > 1)
        return ParsedFile(name, "array", key_field, columns, len(seen), dupes)

    if isinstance(data, dict):
        flat = _flatten(data)
        if not flat:
            raise ValueError("문자열 값을 찾지 못했습니다.")
        # 실제로 계층이 있었는지로 판단한다. 평탄 객체도 key에 점이 들어가므로
        # ("a.save") 결과 key만 봐서는 구분되지 않는다.
        nested = any(isinstance(v, (dict, list)) for v in data.values())
        return ParsedFile(
            name, "nested" if nested else "flat", "(경로)", {name: flat}, len(flat)
        )

    raise ValueError("최상위가 배열도 객체도 아닙니다.")


# ──────────────────────────────────────────────────────────────────────
# 언어 판정
# ──────────────────────────────────────────────────────────────────────

def korean_ratio(values: List[str], sample: int = 800) -> float:
    vals = [v for v in values if v.strip()][:sample]
    if not vals:
        return 0.0
    return sum(1 for v in vals if _KOREAN_RE.search(v)) / len(vals)


@dataclass
class LanguagePick:
    ko: Dict[str, str]
    en: Dict[str, str]
    ko_label: str
    en_label: str
    ratios: List[Tuple[str, float]] = field(default_factory=list)


def pick_languages(files: List[ParsedFile]) -> LanguagePick:
    """
    모든 (파일, 컬럼) 중에서 한국어 쪽과 영어 쪽을 고른다.

    파일명이 아니라 한글 비율로 판정한다 — 고객사 파일에 ko/en 표시가 없을 수
    있기 때문. 한국어는 비율이 가장 높은 컬럼, 영어는 가장 낮은 컬럼.
    """
    cand: List[Tuple[str, Dict[str, str], float]] = []
    for pf in files:
        for col, mapping in pf.columns.items():
            label = f"{pf.name}:{col}" if len(pf.columns) > 1 else pf.name
            cand.append((label, mapping, korean_ratio(list(mapping.values()))))
    if len(cand) < 2:
        raise ValueError(
            "언어 컬럼이 2개 이상 필요합니다. 파일을 2개 올리거나, "
            "한국어와 영어가 함께 든 파일을 올려주세요."
        )

    cand.sort(key=lambda c: c[2], reverse=True)
    ko_label, ko_map, ko_r = cand[0]
    en_label, en_map, en_r = cand[-1]
    if ko_r < 0.2:
        raise ValueError("한국어로 보이는 컬럼이 없습니다.")
    if en_r > 0.5:
        raise ValueError("영어로 보이는 컬럼이 없습니다.")
    return LanguagePick(
        ko=ko_map, en=en_map, ko_label=ko_label, en_label=en_label,
        ratios=[(lbl, r) for lbl, _, r in cand],
    )


# ──────────────────────────────────────────────────────────────────────
# 분석
# ──────────────────────────────────────────────────────────────────────

def is_label(ko: str, en: str) -> bool:
    """
    UI 라벨인가, 아니면 문장(=패턴)인가.

    영어 쪽 종결부호만 보면 안 된다. 실제 카탈로그에는
    "array의 초기화 여부를 검사하지 않습니다." ↔ "Skip array initialization checking"
    처럼 **한국어만 문장인** 항목이 흔하다. 한국어 종결어미와 플레이스홀더도 본다.
    """
    if _PLACEHOLDER_RE.search(en) or _PLACEHOLDER_RE.search(ko):
        return False
    if re.search(r"[.!?]$", en.strip()) or re.search(r"[.!?]$", ko.strip()):
        return False
    if _SENT_STRONG_RE.search(ko) or _SENT_TAIL_RE.search(_ko_core(ko)):
        return False
    return len(ko) <= LABEL_MAX_KO_CHARS and len(en.split()) <= LABEL_MAX_EN_WORDS


def _looks_like_term(ko: str, en: str) -> bool:
    """
    글로서리에 넣을 만한 '용어'인가.

    '확인'·'저장' 같은 짧은 동작어는 문맥마다 영어가 달라지는데, 글로서리에
    올리면 본문 어디서나 치환돼 사고가 난다. 두 어절 이상이거나 충분히 긴
    복합어만 후보로 올린다.
    """
    if not _KOREAN_RE.search(ko):
        return False
    if _JOSA_TAIL_RE.search(_ko_core(ko)):
        return False
    return len(ko.split()) >= 2 or len(ko) >= 5


@dataclass
class ExtractResult:
    labels: pd.DataFrame
    terms: pd.DataFrame
    patterns: pd.DataFrame
    stats: Dict[str, int]


def analyze(pick: LanguagePick,
            existing_terms: Optional[Dict[str, set]] = None) -> ExtractResult:
    """
    KO/EN 맵을 라벨·용어·패턴 후보로 가른다.

    existing_terms: {ko: {en_lower, ...}} — 기존 글로서리. 신규/충돌/동일 판정용.
    """
    existing_terms = existing_terms or {}
    keys = set(pick.ko) & set(pick.en)

    agg: Dict[str, Counter] = defaultdict(Counter)      # ko -> EN 후보 빈도
    ns: Dict[str, set] = defaultdict(set)               # ko -> key 네임스페이스
    sentences: List[Tuple[str, str, str]] = []

    for k in keys:
        ko, en = pick.ko[k].strip(), pick.en[k].strip()
        if not ko or not en:
            continue
        if is_label(ko, en):
            agg[ko][en] += 1
            ns[ko].add(k.split(".")[0])
        else:
            sentences.append((ko, en, k))

    label_rows = []
    for ko, cands in agg.items():
        rep = cands.most_common(1)[0][0]        # 최빈 표기를 대표로
        distinct = {c.lower() for c in cands}   # 대소문자만 다른 건 같은 표기로 본다
        status = "신규"
        if ko in existing_terms:
            status = "동일" if rep.lower() in existing_terms[ko] else "충돌(기존)"
        label_rows.append({
            "KO": ko,
            "EN": rep,
            "후보수": len(distinct),
            "EN 후보": " / ".join(sorted(cands)) if len(distinct) > 1 else "",
            "문맥(key)": " ".join(sorted(ns[ko])[:3]),
            "출현": int(sum(cands.values())),
            "기존대조": status,
            "DNT": ko == rep and not bool(_KOREAN_RE.search(ko)),
        })

    labels = pd.DataFrame(label_rows)
    if not labels.empty:
        labels = labels.sort_values(
            ["후보수", "출현"], ascending=[False, False]
        ).reset_index(drop=True)
        terms = labels[
            (labels["후보수"] == 1)
            & labels.apply(lambda r: _looks_like_term(r["KO"], r["EN"]), axis=1)
        ].copy()
        terms = terms.sort_values(
            "KO", key=lambda s: s.str.len(), ascending=False
        ).reset_index(drop=True)
    else:
        terms = labels.copy()

    # 문장형은 같은 KO가 반복되면 한 번만
    seen_sent: set = set()
    pat_rows = []
    for ko, en, k in sentences:
        if ko in seen_sent:
            continue
        seen_sent.add(ko)
        pat_rows.append({"KO": ko, "EN": en, "문맥(key)": k.split(".")[0]})
    patterns = pd.DataFrame(pat_rows)

    stats = {
        "공통키": len(keys),
        "라벨 고유KO": len(agg),
        "충돌": int((labels["후보수"] > 1).sum()) if not labels.empty else 0,
        "용어후보": len(terms),
        "패턴후보": len(patterns),
        "DNT후보": int(labels["DNT"].sum()) if not labels.empty else 0,
        "신규": int((labels["기존대조"] == "신규").sum()) if not labels.empty else 0,
        "기존충돌": int((labels["기존대조"] == "충돌(기존)").sum()) if not labels.empty else 0,
    }
    return ExtractResult(labels=labels, terms=terms, patterns=patterns, stats=stats)


def existing_terms_index(terms_df: pd.DataFrame) -> Dict[str, set]:
    """load_terms() 결과를 {ko: {en_lower}} 인덱스로."""
    idx: Dict[str, set] = defaultdict(set)
    for _, r in terms_df.iterrows():
        ko = str(r.get("KO", "")).strip()
        en = str(r.get("EN", "")).strip().lower()
        if ko and en:
            idx[ko].add(en)
    return dict(idx)


# ──────────────────────────────────────────────────────────────────────
# 내보내기
# ──────────────────────────────────────────────────────────────────────

def to_excel(terms: pd.DataFrame, patterns: pd.DataFrame, product: str = "ALL") -> bytes:
    """
    기존 마스터 엑셀과 **같은 시트/컬럼 구조**로 내보낸다.

    그래야 [Glossary 관리]의 기존 업로드 경로로 그대로 되먹일 수 있다.
    """
    empty = pd.Series(dtype=str)
    t = pd.DataFrame({
        "KO": terms.get("KO", empty),
        "EN": terms.get("EN", empty),
        "Product": product,
        "DNT": terms.get("DNT", pd.Series(dtype=bool)),
        "Case-sensitive": False,
        "Note": terms.get("문맥(key)", empty),
    })
    p = pd.DataFrame({
        "KO": patterns.get("KO", empty),
        "EN": patterns.get("EN", empty),
        "Note": patterns.get("문맥(key)", empty),
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        t.to_excel(writer, sheet_name="glossary", index=False)
        p.to_excel(writer, sheet_name="pattern", index=False)
    return buf.getvalue()
