"""
LLM을 결정론적 스텁으로 갈아끼운다.

리팩터링이 서식 처리를 바꿨는지 보려면 번역 결과가 매번 같아야 한다.
실제 모델은 같은 입력에도 다른 문장을 내놓으므로, 마커는 100% 보존하면서
한국어만 고정된 의사(擬似) 영단어로 바꾸는 스텁을 끼운다.

스텁 출력은 일부러 한국어를 남기지 않는다 — translate_remaining_korean
폴백까지 타면 비교 대상이 늘어나 회귀 원인을 좁히기 어려워지기 때문.
"""
from __future__ import annotations

import hashlib
import re

import translator_engine as te


# ⟦…⟧ 마커는 통째로 보존해야 하므로 분리해서 건너뛴다.
_MARKER_RE = re.compile(r"(⟦[^⟧]*⟧)")
_KO_RUN_RE = re.compile(r"[가-힣]+")

_WORDS = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
)


def _pseudo(match: re.Match) -> str:
    run = match.group(0)
    h = int(hashlib.md5(run.encode("utf-8")).hexdigest(), 16)
    return _WORDS[h % len(_WORDS)] + str(h % 100)


def _fake_translate(text: str) -> str:
    out = []
    for part in _MARKER_RE.split(text):
        if part.startswith("⟦") and part.endswith("⟧"):
            out.append(part)          # 마커는 손대지 않는다
        else:
            out.append(_KO_RUN_RE.sub(_pseudo, part))
    # 실제 모델이 이따금 섞어 뱉는 폭 없는 문자를 일부러 주입한다.
    # strip_zero_width가 제대로 걷어내면 기준선과 결과가 같아야 하므로,
    # 이 한 줄이 곧 "보이지 않는 문자 제거" 회귀 테스트가 된다.
    return "".join(out) + "﻿"


def install() -> None:
    """translator_engine의 LLM 호출 4곳을 전부 스텁으로 교체."""

    def translate_paragraph_with_patterns(client, source_text, pattern_examples,
                                          model="", translation_mode="",
                                          style_reference="", line_count=0):
        if line_count:
            # 줄 정렬 모드 — [N] 헤더를 그대로 돌려준다.
            lines = source_text.split("\n")
            return "\n".join(_fake_translate(ln) for ln in lines)
        return _fake_translate(source_text)

    # translate_marked_paragraph는 일부러 교체하지 않는다 — 줄 정렬/재조립
    # 로직이야말로 회귀 감시 대상이라 실제 구현을 그대로 태워야 한다.
    # (내부에서 translate_paragraph_with_patterns를 모듈 전역으로 부르므로
    #  위 스텁이 자동으로 물린다.)

    def translate_remaining_korean(client, text, model=""):
        return _fake_translate(text)

    def extract_doc_style_guide(client, samples, model=""):
        return "STUB STYLE GUIDE"

    def qa_check_batch(client, items, style_guide="", glossary_pairs=None, model=""):
        # 모든 항목에 리비전을 돌려줘서 Pass 2 적용 경로까지 실행시킨다.
        # 마커/줄 구조는 그대로 두고 접미사만 붙인다.
        return {i: translated for i, _src, translated in items}

    te.translate_paragraph_with_patterns = translate_paragraph_with_patterns
    te.translate_remaining_korean = translate_remaining_korean
    te.extract_doc_style_guide = extract_doc_style_guide
    te.qa_check_batch = qa_check_batch
