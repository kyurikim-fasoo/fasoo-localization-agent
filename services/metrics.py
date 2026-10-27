"""
대시보드 지표 — translation_logs와 글로서리 테이블에서 집계한다.

원칙: **근거를 댈 수 없는 숫자는 만들지 않는다.**
    "번역가 대비 절감 시간" 같은 환산은 하지 않는다. 번역가 생산성은
    하루 1,500~3,000단어로 출처마다 다르고 실무자 경험칙일 뿐이라,
    어떤 값을 넣든 "그 숫자 근거가 뭐냐"에 답할 수 없다. 대신 측정한
    값만 보여준다 — 처리량, 실제 처리 시간, 공식 단가로 계산한 API 비용.

비용 단가:
    OpenAI 공식 가격표의 gpt-5.2 Standard 단가 (1M 토큰당, USD).
    https://developers.openai.com/api/docs/pricing — 2026-10-27 확인.
    번역 엔진의 모든 호출이 gpt-5.2이므로 단가 하나로 계산한다. 모델이나
    가격이 바뀌면 여기만 고친다.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from db.schema import db_session, init_db

PRICE_MODEL = "gpt-5.2"
PRICE_SOURCE_URL = "https://developers.openai.com/api/docs/pricing"
PRICE_CHECKED = "2026-10-27"
PRICE_PER_1M_INPUT = 1.75
PRICE_PER_1M_CACHED_INPUT = 0.175
PRICE_PER_1M_OUTPUT = 14.00


def api_cost_usd(input_tokens, cached_tokens, output_tokens) -> float:
    """
    공식 단가로 계산한 API 비용.

    Responses API의 input_tokens는 캐시된 토큰을 **포함**한 값이다. 캐시분은
    1/10 단가이므로 빼서 따로 계산한다.
    """
    inp = int(input_tokens or 0)
    cached = min(int(cached_tokens or 0), inp)
    out = int(output_tokens or 0)
    return ((inp - cached) * PRICE_PER_1M_INPUT
            + cached * PRICE_PER_1M_CACHED_INPUT
            + out * PRICE_PER_1M_OUTPUT) / 1_000_000


_LOG_COLUMNS = [
    "created_at", "user", "product", "input_tokens", "cached_tokens",
    "output_tokens", "paragraphs_translated", "elapsed_seconds",
    "applied_terms", "applied_count", "verify_errors", "verify_warnings",
]


def load_logs(since: Optional[str] = None) -> pd.DataFrame:
    """
    대시보드용 로그 전체. since(ISO 날짜)를 주면 그 이후만.

    파생 컬럼: cost_usd, month(YYYY-MM).
    """
    init_db()
    sql = f"SELECT {', '.join(_LOG_COLUMNS)} FROM translation_logs"
    params: list = []
    if since:
        sql += " WHERE created_at >= ?"
        params.append(since)
    with db_session() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
    df = pd.DataFrame(rows, columns=_LOG_COLUMNS)
    df["product"] = df["product"].fillna("").replace("", "(미지정)")
    df["cost_usd"] = [
        api_cost_usd(i, c, o) for i, c, o in
        zip(df["input_tokens"], df["cached_tokens"], df["output_tokens"])
    ]
    df["month"] = df["created_at"].str[:7]
    return df


def glossary_counts() -> dict:
    """현재 등재된 용어·패턴 수 (Team / Personal)."""
    init_db()
    out = {}
    with db_session() as conn:
        for table in ("terms", "patterns"):
            team, personal = conn.execute(
                f"SELECT SUM(owner = ''), SUM(owner != '') FROM {table}"
            ).fetchone()
            out[table] = {"team": int(team or 0), "personal": int(personal or 0)}
    return out


def measured_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    품질·시간 지표가 기록된 로그만. 기록을 시작하기 전 로그(NULL)는 뺀다 —
    섞으면 평균이 0 쪽으로 끌려간다.
    """
    return df[df["elapsed_seconds"].notna()]
