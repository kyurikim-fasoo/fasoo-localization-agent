import json
import os
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from translator_engine import translate_document


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

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Config / file helpers
# ─────────────────────────────────────────────

def load_product_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_uploaded_table_flexible(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded glossary/pattern table from xlsx/xls or tsv/txt.

    Detection order: file extension first (most reliable for Excel), then
    fall back to text decoding for everything else. Encoding probing covers
    Korean/Windows defaults that users commonly save TSV as.
    """
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.getvalue()

    # Excel: detect by extension. openpyxl handles .xlsx; .xls needs xlrd
    # but pandas will surface a clear error if xlrd is missing.
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        from io import BytesIO
        df = pd.read_excel(BytesIO(raw))
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # TSV / TXT: tab-separated with flexible encoding
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
    raise RuntimeError(f"Failed to read uploaded file: {uploaded_file.name}. Last error: {last_error}")


def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    save_path = save_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Input/output 토큰 비용을 분리해 계산합니다."""
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


def build_product_tables(product_name: str, config: dict):
    """
    Return empty glossary/pattern tables. Default TSV loading was removed —
    users must upload their own Excel/TSV in Step 2. `product_name` is kept in
    the signature so callers stay product-aware, but no longer triggers any
    file IO here.
    """
    glossary_df = prepare_glossary_editor_df(
        pd.DataFrame(columns=["KO", "EN", "File", "Product", "DNT", "Case-sensitive", "Note"])
    )
    pattern_df = prepare_pattern_editor_df(
        pd.DataFrame(columns=["KO", "EN", "File", "Note"])
    )
    return glossary_df, pattern_df


def _append_df_as_glossary(current_df: pd.DataFrame, new_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    uploaded_df = _ensure_columns(new_df.copy(), ["KO", "EN", "DNT", "Case-sensitive", "Product", "Note"])
    uploaded_df["File"] = source_name
    uploaded_df = uploaded_df[["KO", "EN", "File", "Product", "DNT", "Case-sensitive", "Note"]]
    return prepare_glossary_editor_df(pd.concat([current_df, uploaded_df], ignore_index=True))


def _append_df_as_pattern(current_df: pd.DataFrame, new_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    uploaded_df = _ensure_columns(new_df.copy(), ["KO", "EN", "Note"])
    uploaded_df["File"] = source_name
    uploaded_df = uploaded_df[["KO", "EN", "File", "Note"]]
    return prepare_pattern_editor_df(pd.concat([current_df, uploaded_df], ignore_index=True))


def _filter_by_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Keep rows whose Product cell matches the selected product or 'all'.

    - No `Product` column → return df untouched (e.g. pattern tables that don't carry product info).
    - Comma/semicolon/slash-separated lists are split, so a cell like "fss, wrapsody"
      matches when either 'fss' or 'wrapsody' is selected.
    - Blank cells are excluded — callers wanting blanks to count as 'all'
      should pre-fill the column.
    """
    if df is None or df.empty or "Product" not in df.columns:
        return df
    p = (product or "").strip().lower()
    if not p:
        return df

    def keep(val):
        s = str(val).strip().lower()
        if not s or s == "nan":
            return False
        parts = re.split(r"[,;/]\s*", s)
        return any(x.strip() in ("all", p) for x in parts)

    return df[df["Product"].apply(keep)].copy()


def _classify_table(sheet_name: str, df: pd.DataFrame) -> str:
    """
    Decide whether a table is a glossary or a pattern table.

    Order of evidence:
    1. Sheet/file name substring match (case-insensitive) — explicit user intent.
    2. Column presence — DNT or Case-sensitive uniquely identify glossary tables.
    3. Default to pattern when only KO/EN are present.

    Returns "glossary" | "pattern" | "unknown".
    """
    lname = (sheet_name or "").lower()
    if any(k in lname for k in ["glossary", "용어", "사전"]):
        return "glossary"
    if any(k in lname for k in ["pattern", "패턴"]):
        return "pattern"
    cols = {str(c).strip().lower() for c in df.columns}
    if "dnt" in cols or "case-sensitive" in cols:
        return "glossary"
    if "ko" in cols and "en" in cols:
        return "pattern"
    return "unknown"


def route_uploaded_workbook(
    uploaded_file,
    glossary_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    product_filter: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str, int]]]:
    """
    Route an uploaded Excel workbook (or TSV/TXT) into glossary and/or pattern tables.

    Excel workbooks are processed sheet-by-sheet; each sheet is classified via
    [_classify_table] and appended to the matching dataframe. TSV/TXT files
    contain a single table and are classified by column presence.

    When `product_filter` is set, rows are filtered via [_filter_by_product]
    before being appended — only rows whose Product cell matches the selected
    product (or 'all') survive. Tables without a Product column are not filtered.

    Returns (new_glossary_df, new_pattern_df, applied) where `applied` is a list
    of (target_tab, source_label, kept_row_count) tuples for status display.
    """
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.getvalue()
    applied: list[tuple[str, str, int]] = []

    if name.endswith((".xlsx", ".xlsm", ".xls")):
        from io import BytesIO
        sheets = pd.read_excel(BytesIO(raw), sheet_name=None)
        for sheet_name, sheet_df in sheets.items():
            sheet_df.columns = [str(c).strip() for c in sheet_df.columns]
            target = _classify_table(sheet_name, sheet_df)
            filtered_df = _filter_by_product(sheet_df, product_filter)
            src_label = f"{uploaded_file.name} [{sheet_name}]"
            if target == "glossary":
                glossary_df = _append_df_as_glossary(glossary_df, filtered_df, src_label)
                applied.append(("용어", sheet_name, len(filtered_df)))
            elif target == "pattern":
                pattern_df = _append_df_as_pattern(pattern_df, filtered_df, src_label)
                applied.append(("패턴", sheet_name, len(filtered_df)))
        return glossary_df, pattern_df, applied

    # TSV / TXT
    single_df = read_uploaded_table_flexible(uploaded_file)
    target = _classify_table("", single_df)
    filtered_df = _filter_by_product(single_df, product_filter)
    if target == "glossary":
        glossary_df = _append_df_as_glossary(glossary_df, filtered_df, uploaded_file.name)
        applied.append(("용어", uploaded_file.name, len(filtered_df)))
    else:
        # 컬럼만으로 pattern/unknown인 경우 모두 pattern으로 처리
        pattern_df = _append_df_as_pattern(pattern_df, filtered_df, uploaded_file.name)
        applied.append(("패턴", uploaded_file.name, len(filtered_df)))
    return glossary_df, pattern_df, applied


def make_default_output_filename(product: str, source_name: str) -> str:
    source_stem = Path(source_name).stem
    safe_product = product.strip().replace(" ", "_")
    return f"{source_stem}_{safe_product}_en.docx"


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

def init_session_state():
    """모든 session_state 초기값을 한 곳에서 관리합니다."""
    defaults = {
        "step": 1,
        "selected_product": None,
        "translation_mode": "매뉴얼",
        "enable_cache": True,
        "enable_qa": True,
        "glossary_df": None,
        "pattern_df": prepare_pattern_editor_df(None),   # ← 전역 중복 초기화 제거
        "last_result": None,
        "last_output_path": None,
        "last_output_filename": None,
        "glossary_editor_key": 0,
        "pattern_editor_key": 0,
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

def render_summary_pills(product: str, mode: str, cache: bool, qa: bool = True):
    cache_text = "사용" if cache else "사용 안 함"
    qa_text = "사용" if qa else "사용 안 함"
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
                <strong>캐시</strong> {cache_text}
            </span>
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px;
                         background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>QA</strong> {qa_text}
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

    enable_qa = st.checkbox(
        "2차 일관성 검사(QA)를 수행해 문서 내 영문 톤·용어와 일치하도록 보정합니다.",
        value=st.session_state.enable_qa,
    )

    st.markdown("---")

    if st.button("다음", use_container_width=True):
        st.session_state.selected_product = selected_product
        st.session_state.translation_mode = translation_mode
        st.session_state.enable_cache = enable_cache
        st.session_state.enable_qa = enable_qa

        glossary_df, pattern_df = build_product_tables(selected_product, config)

        st.session_state.glossary_df = glossary_df.copy()
        st.session_state.pattern_df = pattern_df.copy()

        st.session_state.glossary_editor_key += 1
        st.session_state.pattern_editor_key += 1

        reset_translation_result()
        st.session_state.step = 2
        st.rerun()


# ─────────────────────────────────────────────
# Step 2 — 용어 및 패턴
# ─────────────────────────────────────────────

elif st.session_state.step == 2:
    st.subheader("Step 2. 용어 및 패턴 업로드")
    st.markdown("Excel(또는 TSV) 파일을 업로드해서 용어집과 패턴을 채우세요. 적용하지 않을 항목은 체크박스를 해제하고, 필요하면 직접 추가/수정/삭제할 수 있습니다.")
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
        st.session_state.enable_qa,
    )

    # ── 통합 파일 업로드 (Excel 다중 시트 자동 라우팅 + product 필터링) ─
    with st.expander("파일 업로드 (Excel 다중 시트 / TSV)", expanded=True):
        st.caption(
            f"Excel 파일을 올리면 시트별로 용어/패턴 탭에 자동 분배됩니다. "
            f"시트명에 'glossary'/'용어' 또는 'pattern'/'패턴'을 포함하거나, "
            f"컬럼에 'DNT'/'Case-sensitive'가 있으면 용어로 인식합니다.\n\n"
            f"**Product 필터**: 현재 선택된 제품 `{st.session_state.selected_product}` "
            f"또는 `all`로 표기된 항목만 통과합니다 "
            f"(Product 컬럼이 없는 시트는 필터링 없이 전부 적용)."
        )
        uploaded_workbook = st.file_uploader(
            "파일 업로드",
            type=["xlsx", "xlsm", "xls", "tsv", "txt"],
            key="uploaded_workbook",
            label_visibility="collapsed",
        )
        if uploaded_workbook is not None:
            try:
                (
                    st.session_state.glossary_df,
                    st.session_state.pattern_df,
                    applied,
                ) = route_uploaded_workbook(
                    uploaded_workbook,
                    st.session_state.glossary_df,
                    st.session_state.pattern_df,
                    product_filter=st.session_state.selected_product or "",
                )
                if applied:
                    lines = "\n".join(
                        f"- **{target}** ← `{label}` ({rows}건)"
                        for target, label, rows in applied
                    )
                    st.success(f"{uploaded_workbook.name} 적용 완료:\n{lines}")
                else:
                    st.warning("적용된 시트가 없습니다. 시트명 또는 컬럼을 확인하세요.")
            except Exception as e:
                st.error(f"업로드 오류: {e}")

    tab1, tab2 = st.tabs(["용어", "패턴"])

    # ── 용어 탭 ──────────────────────────────
    with tab1:
        st.caption("선택한 용어는 항상 동일하게 번역합니다.")

        _, col_reset = st.columns([6, 2])
        with col_reset:
            if st.button("비우기", key="reset_glossary", use_container_width=True):
                st.session_state.glossary_df = prepare_glossary_editor_df(
                    pd.DataFrame(columns=["KO", "EN", "File", "Product", "DNT", "Case-sensitive", "Note"])
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
                "적용": st.column_config.CheckboxColumn("적용", default=True),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "File": st.column_config.TextColumn("File"),
                "Product": st.column_config.TextColumn("Product"),
                "DNT": st.column_config.TextColumn("DNT"),
                "Case-sensitive": st.column_config.TextColumn("Case-sensitive"),
                "Note": st.column_config.TextColumn("Note"),
            },
            key="glossary_editor_widget",
        )
        st.session_state.glossary_df = prepare_glossary_editor_df(edited_glossary_df)

    # ── 패턴 탭 ──────────────────────────────
    with tab2:
        st.caption("비슷한 패턴이 나오면 아래를 참고해 번역합니다.")

        _, col_reset = st.columns([6, 2])
        with col_reset:
            if st.button("비우기", key="reset_pattern", use_container_width=True):
                st.session_state.pattern_df = prepare_pattern_editor_df(
                    pd.DataFrame(columns=["KO", "EN", "File", "Note"])
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
                "적용": st.column_config.CheckboxColumn("적용", default=True),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "File": st.column_config.TextColumn("File"),
                "Note": st.column_config.TextColumn("Note"),
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
        st.session_state.enable_qa,
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
            disabled=(uploaded_docx is None),   # 파일 없으면 비활성화
        )

    if translate_clicked:
        # 파일 필수 체크 (disabled로 대부분 막히지만 안전장치)
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
            status_text = st.empty()

            label = "번역/QA 진행 중" if st.session_state.enable_qa else "번역 중"

            def update_progress(done, total):
                progress = int((done / total) * 100) if total else 0
                progress_bar.progress(progress)
                status_text.text(f"{label}... {done}/{total}")

            try:
                result = translate_document(
                    in_path=str(input_path),
                    out_path=str(output_path),
                    glossary_rows=glossary_rows,
                    pattern_rows=pattern_rows,
                    api_key=OPENAI_API_KEY,
                    enable_cache=st.session_state.enable_cache,
                    enable_qa=st.session_state.enable_qa,
                    translation_mode=st.session_state.translation_mode,
                    progress_callback=update_progress,
                )

                st.session_state.last_result = result
                st.session_state.last_output_path = str(output_path)
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

    result = st.session_state.last_result
    output_path = st.session_state.last_output_path
    output_filename = st.session_state.last_output_filename

    if not result or not output_path:
        st.error("번역 결과를 찾을 수 없습니다.")
        st.session_state.step = 1
        st.rerun()

    # input/output 토큰 분리 비용 계산
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