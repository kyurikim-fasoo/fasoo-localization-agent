import io
import json
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from translator_engine import translate_document
from learned_terms import (
    detect_inconsistencies,
    load_learned_terms,
    save_learned_term,
)


load_dotenv()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
COST_PER_1K_INPUT_TOKENS = float(
    st.secrets.get("MODEL_COST_PER_1K_INPUT_TOKENS", os.getenv("MODEL_COST_PER_1K_INPUT_TOKENS", "0.005"))
)
COST_PER_1K_OUTPUT_TOKENS = float(
    st.secrets.get("MODEL_COST_PER_1K_OUTPUT_TOKENS", os.getenv("MODEL_COST_PER_1K_OUTPUT_TOKENS", "0.015"))
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CONFIG_PATH = BASE_DIR / "product_config.json"

# Master Terminology 엑셀에서 추출한 TSV를 저장하는 디렉터리
TERMINOLOGY_DIR = BASE_DIR / "terminology"
TERMINOLOGY_DIR.mkdir(exist_ok=True)

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Config / file helpers
# ─────────────────────────────────────────────

def load_product_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_tsv_flexible(path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-16", "utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read TSV: {path}. Last error: {last_error}")


def read_uploaded_tsv_flexible(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    encodings_to_try = ["utf-16", "utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None
    for enc in encodings_to_try:
        try:
            text = raw.decode(enc)
            df = pd.read_csv(StringIO(text), sep="\t")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read uploaded TSV: {uploaded_file.name}. Last error: {last_error}")


def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    save_path = save_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    cost = (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS) + \
           (output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)
    return round(cost, 4)


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out


# ─────────────────────────────────────────────
# Master Terminology 엑셀 파싱
# ─────────────────────────────────────────────

_GLOSSARY_TYPES  = {"용어", "glossary", "term", "terminology"}
_PATTERN_TYPES   = {"패턴", "pattern"}
_ALL_PRODUCT_VALUES = {"all", "공통", "common", ""}


def _normalize_product(name: str) -> str:
    return str(name).strip().lower()


def _product_matches(cell_value: str, selected_product: str) -> bool:
    v = _normalize_product(cell_value)
    if v in _ALL_PRODUCT_VALUES:
        return True
    return v == _normalize_product(selected_product)


def parse_master_terminology_xlsx(
    file_bytes: bytes,
    selected_product: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Master Terminology 엑셀을 파싱해 선택 제품 + 공통 행만 추출합니다.

    반환: (glossary_df, pattern_df, warnings)
    """
    warnings: list[str] = []

    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"엑셀 파일을 열 수 없습니다: {e}")

    # 'Terminology' 또는 '용어' 포함 시트 우선, 없으면 첫 번째 시트
    sheet_name = xl.sheet_names[0]
    for name in xl.sheet_names:
        if "terminolog" in name.lower() or "용어" in name:
            sheet_name = name
            break

    try:
        raw = xl.parse(sheet_name, dtype=str).fillna("")
    except Exception as e:
        raise ValueError(f"시트 '{sheet_name}' 파싱 오류: {e}")

    raw.columns = [str(c).strip() for c in raw.columns]

    # 필수 열 확인
    missing = {"KO", "EN"} - set(raw.columns)
    if missing:
        raise ValueError(
            f"엑셀에 필수 열이 없습니다: {', '.join(sorted(missing))}. "
            f"현재 열: {', '.join(raw.columns.tolist())}"
        )

    # Product 열 필터링
    if "Product" in raw.columns:
        mask = raw["Product"].apply(lambda v: _product_matches(str(v), selected_product))
        filtered = raw[mask].copy()
        excluded = len(raw) - len(filtered)
        if excluded:
            warnings.append(
                f"Product 필터링: 전체 {len(raw)}행 중 {excluded}행 제외 → "
                f"{len(filtered)}행 사용 ({selected_product} + 공통)"
            )
    else:
        filtered = raw.copy()
        warnings.append("Product 열이 없어 모든 행을 사용합니다.")

    # KO / EN 빈 행 제거
    before = len(filtered)
    filtered = filtered[
        filtered["KO"].str.strip().astype(bool) &
        filtered["EN"].str.strip().astype(bool)
    ].copy()
    dropped = before - len(filtered)
    if dropped:
        warnings.append(f"KO 또는 EN이 비어있는 {dropped}행 제거")

    # Type 열로 용어 / 패턴 분리
    if "Type" in filtered.columns:
        is_glossary = filtered["Type"].str.strip().str.lower().isin(_GLOSSARY_TYPES)
        is_pattern  = filtered["Type"].str.strip().str.lower().isin(_PATTERN_TYPES)
        glossary_raw = filtered[is_glossary].copy()
        pattern_raw  = filtered[is_pattern].copy()
        other_count  = len(filtered) - len(glossary_raw) - len(pattern_raw)
        if other_count:
            other = filtered[~is_glossary & ~is_pattern].copy()
            glossary_raw = pd.concat([glossary_raw, other], ignore_index=True)
            warnings.append(f"Type이 불분명한 {other_count}행은 용어로 처리했습니다.")
    else:
        glossary_raw = filtered.copy()
        pattern_raw  = pd.DataFrame(columns=filtered.columns)
        warnings.append("Type 열이 없어 모든 행을 용어로 처리합니다.")

    def _build_glossary(df: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_columns(df, ["KO", "EN", "DNT", "Case-sensitive", "Product", "Note"])
        df["File"] = sheet_name
        return df[["KO", "EN", "File", "Product", "DNT", "Case-sensitive", "Note"]]

    def _build_pattern(df: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_columns(df, ["KO", "EN", "Note"])
        df["File"] = sheet_name
        return df[["KO", "EN", "File", "Note"]]

    empty_glossary = pd.DataFrame(columns=["KO", "EN", "File", "Product", "DNT", "Case-sensitive", "Note"])
    empty_pattern  = pd.DataFrame(columns=["KO", "EN", "File", "Note"])

    g_df = prepare_glossary_editor_df(
        _build_glossary(glossary_raw) if not glossary_raw.empty else empty_glossary
    )
    p_df = prepare_pattern_editor_df(
        _build_pattern(pattern_raw) if not pattern_raw.empty else empty_pattern
    )

    return g_df, p_df, warnings


def _df_to_tsv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(sep="\t", index=False, encoding="utf-8-sig").encode("utf-8-sig")


def save_terminology_as_default(
    product: str,
    glossary_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    config: dict,
) -> tuple[bool, str]:
    """
    용어/패턴 DataFrame을 해당 제품의 기본 TSV로 저장하고
    product_config.json을 업데이트합니다.
    """
    try:
        safe_product  = product.strip().replace(" ", "_")
        glossary_path = TERMINOLOGY_DIR / f"{safe_product}_glossary.tsv"
        pattern_path  = TERMINOLOGY_DIR / f"{safe_product}_patterns.tsv"

        g_save = glossary_df.drop(columns=["적용"], errors="ignore")
        p_save = pattern_df.drop(columns=["적용"],  errors="ignore")

        glossary_path.write_bytes(_df_to_tsv_bytes(g_save))
        pattern_path.write_bytes(_df_to_tsv_bytes(p_save))

        rel_glossary = str(glossary_path.relative_to(BASE_DIR))
        rel_pattern  = str(pattern_path.relative_to(BASE_DIR))

        if product not in config:
            config[product] = {}
        config[product]["default_glossaries"] = [rel_glossary]
        config[product]["default_patterns"]   = [rel_pattern]

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        return True, (
            f"저장 완료 — 용어 {len(g_save)}개, 패턴 {len(p_save)}개를 "
            f"'{product}' 제품의 기본값으로 설정했습니다."
        )
    except Exception as e:
        return False, f"저장 실패: {e}"


# ─────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────

def prepare_glossary_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    out = df.copy()
    expected_cols = ["적용", "KO", "EN", "File", "Product", "DNT", "Case-sensitive", "Note"]

    for col in expected_cols:
        if col not in out.columns:
            out[col] = True if col == "적용" else ""

    if "적용" in out.columns:
        out["적용"] = out["적용"].fillna(True).astype(bool)

    out = out[expected_cols]

    for col in expected_cols:
        if col != "적용":
            out[col] = out[col].fillna("").astype(str)

    return out


def prepare_pattern_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    out = df.copy()
    expected_cols = ["적용", "KO", "EN", "File", "Note"]

    for col in expected_cols:
        if col not in out.columns:
            out[col] = True if col == "적용" else ""

    if "적용" in out.columns:
        out["적용"] = out["적용"].fillna(True).astype(bool)

    out = out[expected_cols]

    for col in expected_cols:
        if col != "적용":
            out[col] = out[col].fillna("").astype(str)

    return out


def load_glossary_table(paths: list[str]) -> pd.DataFrame:
    frames = []
    for rel_path in paths:
        path = BASE_DIR / rel_path
        if not path.exists():
            continue
        df = read_tsv_flexible(path)
        df = _ensure_columns(df, ["KO", "EN", "DNT", "Case-sensitive", "Product", "Note"])
        df["File"] = path.name
        df = df[["KO", "EN", "File", "Product", "DNT", "Case-sensitive", "Note"]]
        frames.append(df)

    if not frames:
        return prepare_glossary_editor_df(
            pd.DataFrame(columns=["KO", "EN", "File", "Product", "DNT", "Case-sensitive"])
        )

    return prepare_glossary_editor_df(pd.concat(frames, ignore_index=True))


def load_pattern_table(paths: list[str]) -> pd.DataFrame:
    frames = []
    for rel_path in paths:
        path = BASE_DIR / rel_path
        if not path.exists():
            continue
        df = read_tsv_flexible(path)
        df = _ensure_columns(df, ["KO", "EN", "Note"])
        df["File"] = path.name
        df = df[["KO", "EN", "File", "Note"]]
        frames.append(df)

    if not frames:
        return prepare_pattern_editor_df(pd.DataFrame(columns=["KO", "EN", "File", "Note"]))

    return prepare_pattern_editor_df(pd.concat(frames, ignore_index=True))


def build_product_tables(product_name: str, config: dict):
    product_info = config[product_name]
    glossary_df  = load_glossary_table(product_info.get("default_glossaries", []))
    pattern_df   = load_pattern_table(product_info.get("default_patterns", []))
    return glossary_df, pattern_df


def merge_glossary_upload(current_df: pd.DataFrame, uploaded_file) -> pd.DataFrame:
    uploaded_df = read_uploaded_tsv_flexible(uploaded_file)
    uploaded_df = _ensure_columns(uploaded_df, ["KO", "EN", "DNT", "Case-sensitive", "Product", "Note"])
    uploaded_df["File"] = uploaded_file.name
    uploaded_df = uploaded_df[["KO", "EN", "File", "Product", "DNT", "Case-sensitive", "Note"]]
    return prepare_glossary_editor_df(pd.concat([current_df, uploaded_df], ignore_index=True))


def merge_pattern_upload(current_df: pd.DataFrame, uploaded_file) -> pd.DataFrame:
    uploaded_df = read_uploaded_tsv_flexible(uploaded_file)
    uploaded_df = _ensure_columns(uploaded_df, ["KO", "EN", "Note"])
    uploaded_df["File"] = uploaded_file.name
    uploaded_df = uploaded_df[["KO", "EN", "File", "Note"]]
    return prepare_pattern_editor_df(pd.concat([current_df, uploaded_df], ignore_index=True))


def make_default_output_filename(product: str, source_name: str) -> str:
    source_stem  = Path(source_name).stem
    safe_product = product.strip().replace(" ", "_")
    return f"{source_stem}_{safe_product}_en.docx"


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

def init_session_state():
    defaults = {
        "step": 1,
        "selected_product": None,
        "translation_mode": "매뉴얼",
        "enable_cache": True,
        "glossary_df": None,
        "pattern_df": prepare_pattern_editor_df(None),
        "base_glossary_df": None,
        "base_pattern_df": prepare_pattern_editor_df(None),
        "last_result": None,
        "last_output_path": None,
        "last_output_filename": None,
        "glossary_editor_key": 0,
        "pattern_editor_key": 0,
        "enable_consistency_pass": False,
        # Master Terminology 관련
        "xlsx_parse_result": None,   # (g_df, p_df, warns) 또는 None
        "xlsx_filename": None,
        # 학습 제안 관련
        "pending_suggestions": [],   # detect_inconsistencies 결과
        "suggestion_choices": {},    # {ko: selected_en}
        "suggestions_saved": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_translation_result():
    st.session_state.last_result = None
    st.session_state.last_output_path = None
    st.session_state.last_output_filename = None


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────

def render_summary_pills(product: str, mode: str, cache: bool, consistency: bool = False):
    cache_text       = "켜짐" if cache else "꺼짐"
    consistency_text = "켜짐" if consistency else "꺼짐"
    st.markdown(
        f"""
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin: 0 0 16px 0;">
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px;
                         background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>제품</strong> {product}
            </span>
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px;
                         background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>텍스트 유형</strong> {mode}
            </span>
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px;
                         background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>중복 재사용</strong> {cache_text}
            </span>
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px;
                         background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>일관성 재검토</strong> {consistency_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Page config & global styles
# ─────────────────────────────────────────────

st.set_page_config(page_title="Fasoo Localization Agent", layout="wide")
init_session_state()

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stFileUploader"] section {
        min-height: 260px;
        border-radius: 16px;
        border: 2px dashed #c8d1dc;
        background: transparent;
        padding: 0;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 240px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    [data-testid="stFileUploaderDropzone"] div {
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: #57606a;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #8c959f;
    }

    [data-testid="stButton"] button {
        min-height: 46px;
        font-weight: 600;
        border-radius: 12px;
    }

    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stDownloadButton"] button[kind="primary"] {
        min-height: 54px;
        font-weight: 700;
        border-radius: 14px;
        font-size: 16px;
    }

    [data-testid="stDownloadButton"] button {
        min-height: 54px;
        font-weight: 700;
        border-radius: 14px;
        font-size: 16px;
    }

    div[data-testid="stDataEditor"] {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fasoo Localization Agent")
st.markdown("국문 문서를 영문으로 로컬라이즈합니다.")

config = load_product_config()
products = list(config.keys())

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()


# ─────────────────────────────────────────────
# Step 1 — 기본 정보
# ─────────────────────────────────────────────

if st.session_state.step == 1:
    st.subheader("Step 1. 기본 정보")
    st.markdown("번역할 텍스트 유형과 제품을 선택하세요.")

    default_product_index = 0
    if st.session_state.selected_product in products:
        default_product_index = products.index(st.session_state.selected_product)

    selected_product = st.selectbox("제품", products, index=default_product_index)

    translation_mode = st.radio(
        "텍스트 유형",
        options=["UI 텍스트", "매뉴얼"],
        index=0 if st.session_state.translation_mode == "UI 텍스트" else 1,
        horizontal=True,
    )

    enable_cache = st.checkbox(
        "중복된 문장은 재사용해 속도와 비용을 줄입니다.",
        value=st.session_state.enable_cache,
    )

    enable_consistency_pass = st.checkbox(
        "번역 완료 후 일관성 재검토 패스를 실행합니다. (추가 비용 발생)",
        value=st.session_state.enable_consistency_pass,
        help=(
            "번역이 끝난 뒤 LLM이 전체 번역을 한 번 더 검토해 "
            "같은 표현이 다르게 번역된 것을 통일합니다. "
            "단락 수에 비례해 추가 API 호출이 발생합니다."
        ),
    )

    st.markdown("---")

    if st.button("다음", use_container_width=True):
        st.session_state.selected_product = selected_product
        st.session_state.translation_mode = translation_mode
        st.session_state.enable_cache     = enable_cache
        st.session_state.enable_consistency_pass = enable_consistency_pass

        glossary_df, pattern_df = build_product_tables(selected_product, config)

        st.session_state.glossary_df      = glossary_df.copy()
        st.session_state.pattern_df       = pattern_df.copy()
        st.session_state.base_glossary_df = glossary_df.copy()
        st.session_state.base_pattern_df  = pattern_df.copy()

        # 새 제품 선택 시 이전 엑셀 파싱 결과 초기화
        st.session_state.xlsx_parse_result = None
        st.session_state.xlsx_filename     = None

        st.session_state.glossary_editor_key += 1
        st.session_state.pattern_editor_key  += 1

        reset_translation_result()
        st.session_state.step = 2
        st.rerun()


# ─────────────────────────────────────────────
# Step 2 — 용어 및 패턴 선택
# ─────────────────────────────────────────────

elif st.session_state.step == 2:
    st.subheader("Step 2. 용어 및 패턴 선택")
    st.markdown("적용하지 않을 항목은 체크박스를 해제하세요. 필요시 새 항목을 추가하거나 수정, 삭제할 수 있습니다.")
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
        st.session_state.enable_consistency_pass,
    )

    # ── Master Terminology 엑셀 업로드 ───────────────────────────────────
    with st.expander("📊 Master Terminology 엑셀에서 자동 업데이트", expanded=False):
        st.markdown(
            f"Master Terminology 엑셀을 업로드하면 "
            f"**Product 열이 `{st.session_state.selected_product}` 또는 공통(ALL)**인 항목만 "
            f"추출해 용어·패턴 목록을 교체합니다."
        )
        st.caption(
            "필수 열: `KO`, `EN` | 선택 열: `Product`, `Type`(용어/패턴 구분), "
            "`DNT`, `Case-sensitive`, `Note`"
        )

        uploaded_xlsx = st.file_uploader(
            "엑셀 파일 업로드 (.xlsx)",
            type=["xlsx", "xls"],
            key="master_terminology_xlsx",
            label_visibility="collapsed",
        )

        if uploaded_xlsx is not None:
            # 새 파일이거나 파일명이 달라진 경우에만 재파싱
            if st.session_state.xlsx_filename != uploaded_xlsx.name:
                try:
                    with st.spinner("파일 파싱 중..."):
                        g_df, p_df, warns = parse_master_terminology_xlsx(
                            uploaded_xlsx.getvalue(),
                            st.session_state.selected_product,
                        )
                    st.session_state.xlsx_parse_result = (g_df, p_df, warns)
                    st.session_state.xlsx_filename     = uploaded_xlsx.name
                except ValueError as e:
                    st.error(str(e))
                    st.session_state.xlsx_parse_result = None
                    st.session_state.xlsx_filename     = None

            # 파싱 결과 표시
            if st.session_state.xlsx_parse_result is not None:
                g_df, p_df, warns = st.session_state.xlsx_parse_result

                for w in warns:
                    st.info(w, icon="ℹ️")

                col_g, col_p = st.columns(2)
                col_g.metric("추출된 용어", f"{len(g_df)}개")
                col_p.metric("추출된 패턴", f"{len(p_df)}개")

                st.markdown("")
                col_apply, col_save = st.columns(2)

                with col_apply:
                    if st.button(
                        "↩ 용어·패턴 목록에 적용",
                        use_container_width=True,
                        help="아래 용어/패턴 테이블을 엑셀 추출 결과로 교체합니다.",
                    ):
                        st.session_state.glossary_df = g_df.copy()
                        st.session_state.pattern_df  = p_df.copy()
                        st.session_state.glossary_editor_key += 1
                        st.session_state.pattern_editor_key  += 1
                        st.success("용어·패턴 목록이 업데이트되었습니다.")
                        st.rerun()

                with col_save:
                    if st.button(
                        "💾 기본값으로 저장",
                        use_container_width=True,
                        type="primary",
                        help=(
                            f"추출된 용어·패턴을 '{st.session_state.selected_product}' 제품의 "
                            f"기본 TSV 파일로 저장하고 설정을 업데이트합니다."
                        ),
                    ):
                        ok, msg = save_terminology_as_default(
                            st.session_state.selected_product,
                            g_df,
                            p_df,
                            config,
                        )
                        if ok:
                            st.session_state.glossary_df      = g_df.copy()
                            st.session_state.pattern_df       = p_df.copy()
                            st.session_state.base_glossary_df = g_df.copy()
                            st.session_state.base_pattern_df  = p_df.copy()
                            st.session_state.glossary_editor_key += 1
                            st.session_state.pattern_editor_key  += 1
                            config.update(load_product_config())
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

    st.markdown("")

    tab1, tab2 = st.tabs(["용어", "패턴"])

    # ── 용어 탭 ──────────────────────────────
    with tab1:
        st.caption("선택한 용어는 항상 동일하게 번역합니다.")

        with st.expander("TSV 업로드", expanded=False):
            uploaded_glossary_tsv = st.file_uploader(
                "TSV 업로드",
                type=["tsv"],
                key="uploaded_glossary_tsv",
                label_visibility="collapsed",
            )
            if uploaded_glossary_tsv is not None:
                try:
                    st.session_state.glossary_df = merge_glossary_upload(
                        st.session_state.glossary_df,
                        uploaded_glossary_tsv,
                    )
                    st.success(f"{uploaded_glossary_tsv.name}을(를) 용어에 추가했습니다.")
                except Exception as e:
                    st.error(f"업로드 오류: {e}")

        _, col_reset = st.columns([6, 2])
        with col_reset:
            if st.button("초기 설정으로 복원", key="reset_glossary", use_container_width=True):
                st.session_state.glossary_df = prepare_glossary_editor_df(
                    st.session_state.base_glossary_df.copy()
                )
                st.rerun()

        st.session_state.glossary_df = prepare_glossary_editor_df(st.session_state.glossary_df)

        edited_glossary_df = st.data_editor(
            st.session_state.glossary_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["File"],
            column_config={
                "적용":           st.column_config.CheckboxColumn("적용", default=True),
                "KO":             st.column_config.TextColumn("KO"),
                "EN":             st.column_config.TextColumn("EN"),
                "File":           st.column_config.TextColumn("File"),
                "Product":        st.column_config.TextColumn("Product"),
                "DNT":            st.column_config.TextColumn("DNT"),
                "Case-sensitive": st.column_config.TextColumn("Case-sensitive"),
                "Note":           st.column_config.TextColumn("Note"),
            },
            key="glossary_editor_widget",
        )
        st.session_state.glossary_df = prepare_glossary_editor_df(edited_glossary_df)

    # ── 패턴 탭 ──────────────────────────────
    with tab2:
        st.caption("비슷한 패턴이 나오면 아래를 참고해 번역합니다.")

        with st.expander("TSV 업로드", expanded=False):
            uploaded_pattern_tsv = st.file_uploader(
                "TSV 업로드",
                type=["tsv"],
                key="uploaded_pattern_tsv",
                label_visibility="collapsed",
            )
            if uploaded_pattern_tsv is not None:
                try:
                    st.session_state.pattern_df = merge_pattern_upload(
                        st.session_state.pattern_df,
                        uploaded_pattern_tsv,
                    )
                    st.success(f"{uploaded_pattern_tsv.name}을(를) 패턴에 추가했습니다.")
                except Exception as e:
                    st.error(f"업로드 오류: {e}")

        _, col_reset = st.columns([6, 2])
        with col_reset:
            if st.button("초기 설정으로 복원", key="reset_pattern", use_container_width=True):
                st.session_state.pattern_df = prepare_pattern_editor_df(
                    st.session_state.base_pattern_df.copy()
                )
                st.rerun()

        st.session_state.pattern_df = prepare_pattern_editor_df(st.session_state.pattern_df)

        edited_pattern_df = st.data_editor(
            st.session_state.pattern_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["File"],
            column_config={
                "적용":  st.column_config.CheckboxColumn("적용", default=True),
                "KO":    st.column_config.TextColumn("KO"),
                "EN":    st.column_config.TextColumn("EN"),
                "File":  st.column_config.TextColumn("File"),
                "Note":  st.column_config.TextColumn("Note"),
            },
            key="pattern_editor_widget",
        )
        st.session_state.pattern_df = prepare_pattern_editor_df(edited_pattern_df)

    # ── 이전 / 다음 버튼 (탭 바깥) ───────────
    st.markdown("---")
    col_back, col_next = st.columns(2)

    with col_back:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

    with col_next:
        if st.button("다음", use_container_width=True):
            reset_translation_result()
            st.session_state.step = 3
            st.rerun()


# ─────────────────────────────────────────────
# Step 3 — 파일 업로드 & 번역
# ─────────────────────────────────────────────

elif st.session_state.step == 3:
    st.subheader("Step 3. 업로드")
    st.markdown("로컬라이즈할 Word 파일을 업로드하거나 끌어서 놓으세요.")
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
        st.session_state.enable_consistency_pass,
    )

    glossary_rows = (
        st.session_state.glossary_df[st.session_state.glossary_df["적용"] == True]
        .drop(columns=["적용"])
        .to_dict("records")
    )
    pattern_rows = (
        st.session_state.pattern_df[st.session_state.pattern_df["적용"] == True]
        .drop(columns=["적용"])
        .to_dict("records")
    )

    uploaded_docx = st.file_uploader(
        "업로드 또는 끌어서 놓기",
        type=["docx"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    col_back, col_translate = st.columns(2)

    with col_back:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    with col_translate:
        translate_clicked = st.button(
            "번역 시작",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_docx is None),
        )

    if translate_clicked:
        if uploaded_docx is None:
            st.error("Word 파일을 업로드하세요.")
        else:
            input_path = save_uploaded_file(uploaded_docx, UPLOAD_DIR)
            output_filename = make_default_output_filename(
                st.session_state.selected_product,
                uploaded_docx.name,
            )
            output_path = OUTPUT_DIR / output_filename

            progress_bar = st.progress(0)
            status_text  = st.empty()

            def update_progress(done, total):
                progress = int((done / total) * 100) if total else 0
                progress_bar.progress(progress)
                status_text.text(f"번역 중... {done}/{total} 문단")

            try:
                # 학습된 표현 로드
                learned = load_learned_terms(st.session_state.selected_product)

                result = translate_document(
                    in_path=str(input_path),
                    out_path=str(output_path),
                    glossary_rows=glossary_rows,
                    pattern_rows=pattern_rows,
                    api_key=OPENAI_API_KEY,
                    enable_cache=st.session_state.enable_cache,
                    translation_mode=st.session_state.translation_mode,
                    progress_callback=update_progress,
                    learned_terms=learned,
                    enable_consistency_pass=st.session_state.enable_consistency_pass,
                )

                # 번역 쌍에서 불일치 표현 감지
                pairs = result.get("translation_pairs", [])
                suggestions = detect_inconsistencies(
                    pairs,
                    min_occurrences=2,
                    existing_learned=learned,
                )

                st.session_state.pending_suggestions = suggestions
                st.session_state.suggestion_choices  = {
                    s["ko"]: s["candidates"][0][0]   # 최다 빈도를 기본 선택
                    for s in suggestions
                }
                st.session_state.suggestions_saved  = False
                st.session_state.last_result        = result
                st.session_state.last_output_path   = str(output_path)
                st.session_state.last_output_filename = output_filename
                st.session_state.step = 4
                st.rerun()

            except Exception as e:
                st.error(f"오류: {e}")


# ─────────────────────────────────────────────
# Step 4 — 결과 & 다운로드
# ─────────────────────────────────────────────

elif st.session_state.step == 4:
    st.subheader("Step 4. 다운로드")

    result          = st.session_state.last_result
    output_path     = st.session_state.last_output_path
    output_filename = st.session_state.last_output_filename

    if not result or not output_path:
        st.error("번역 결과를 찾을 수 없습니다.")
        st.session_state.step = 1
        st.rerun()

    estimated_cost = estimate_cost_usd(
        result.get("input_tokens", 0),
        result.get("output_tokens", 0),
    )

    st.success("로컬라이즈가 완료되었습니다.")
    st.write("### 결과")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("입력 토큰", f"{result.get('input_tokens', 0):,}")
    col_b.metric("출력 토큰", f"{result.get('output_tokens', 0):,}")
    col_c.metric("예상 비용", f"${estimated_cost}")

    consistency_map = result.get("consistency_map", {})
    if consistency_map:
        with st.expander(f"✅ 일관성 재검토 완료 — {len(consistency_map)}개 표현 통일", expanded=False):
            st.caption("아래 표현들이 문서 전체에 통일되어 적용되었습니다.")
            for ko, en in consistency_map.items():
                st.markdown(f"- `{ko}` → **{en}**")

    st.markdown("")

    with open(output_path, "rb") as f:
        st.download_button(
            label="문서 다운로드",
            data=f,
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
            use_container_width=True,
        )

    # ── 학습 제안 섹션 ────────────────────────────────────────────────────
    suggestions = st.session_state.get("pending_suggestions", [])
    if suggestions and not st.session_state.get("suggestions_saved", False):
        st.markdown("---")
        st.markdown(
            f"### 💡 학습 가능한 표현 발견 ({len(suggestions)}건)",
        )
        st.caption(
            "이번 번역에서 같은 국문 표현이 다르게 번역되었습니다. "
            "원하는 표현을 선택하면 다음 번역부터 일관되게 적용됩니다."
        )

        choices = st.session_state.get("suggestion_choices", {})

        for s in suggestions:
            ko = s["ko"]
            candidates = s["candidates"]   # [('click', 5), ('press', 2)]

            with st.container():
                st.markdown(f"**`{ko}`** — 총 {s['total']}회 사용")
                option_labels = [f"{en}  ({cnt}회)" for en, cnt in candidates]
                option_labels.append("직접 입력")

                current_en = choices.get(ko, candidates[0][0])
                try:
                    default_idx = [en for en, _ in candidates].index(current_en)
                except ValueError:
                    default_idx = len(candidates)  # "직접 입력"

                selected = st.radio(
                    f"_{ko}_",
                    options=option_labels,
                    index=default_idx,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"sug_{ko}",
                )

                if selected == "직접 입력":
                    custom = st.text_input(
                        "직접 입력",
                        value=current_en if current_en not in [en for en, _ in candidates] else "",
                        key=f"sug_custom_{ko}",
                        label_visibility="collapsed",
                        placeholder="EN 표현 입력",
                    )
                    choices[ko] = custom.strip() if custom.strip() else candidates[0][0]
                else:
                    chosen_en = selected.split("  (")[0]
                    choices[ko] = chosen_en

                st.session_state.suggestion_choices = choices
                st.markdown("")

        col_save, col_skip = st.columns(2)
        with col_save:
            if st.button("선택한 표현 학습에 추가", type="primary", use_container_width=True):
                product = st.session_state.selected_product
                source_doc = st.session_state.get("last_output_filename", "")
                for s in suggestions:
                    ko = s["ko"]
                    chosen_en = choices.get(ko, "")
                    if not chosen_en:
                        continue
                    rejected = [en for en, _ in s["candidates"] if en != chosen_en]
                    save_learned_term(
                        product=product,
                        ko=ko,
                        en=chosen_en,
                        source_doc=source_doc,
                        rejected_alternatives=rejected,
                    )
                st.session_state.suggestions_saved = True
                st.session_state.pending_suggestions = []
                st.success(f"{len(suggestions)}개 표현을 학습했습니다. 다음 번역부터 자동 적용됩니다.")
                st.rerun()

        with col_skip:
            if st.button("이번엔 건너뛰기", use_container_width=True):
                st.session_state.suggestions_saved = True
                st.session_state.pending_suggestions = []
                st.rerun()

    elif st.session_state.get("suggestions_saved", False):
        st.markdown("---")
        st.info("학습된 표현은 **📚 학습된 표현 관리** 페이지에서 확인·수정할 수 있습니다.", icon="✅")

    st.markdown("---")
    col_prev, col_restart = st.columns(2)

    with col_prev:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

    with col_restart:
        if st.button("처음으로 돌아가기", use_container_width=True):
            reset_translation_result()
            st.session_state.step = 1
            st.rerun()