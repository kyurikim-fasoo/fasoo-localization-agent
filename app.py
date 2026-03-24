import json
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from translator_engine import translate_document


load_dotenv()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
COST_PER_1K_TOKENS = float(
    st.secrets.get("MODEL_COST_PER_1K_TOKENS", os.getenv("MODEL_COST_PER_1K_TOKENS", "0.01"))
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CONFIG_PATH = BASE_DIR / "product_config.json"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


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


def save_uploaded_file(uploaded_file, save_dir: Path):
    save_path = save_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def estimate_cost_usd(total_tokens: int, rate_per_1k_tokens: float = COST_PER_1K_TOKENS) -> float:
    return round((total_tokens / 1000) * rate_per_1k_tokens, 4)


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out


def prepare_glossary_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    out = df.copy()

    if "적용" not in out.columns:
        out["적용"] = True
    else:
        out["적용"] = out["적용"].fillna(True).astype(bool)

    expected_cols = [
        "적용",
        "KO",
        "EN",
        "File",
        "Product",
        "DNT",
        "Case-sensitive",
        "Note",
    ]

    for col in expected_cols:
        if col not in out.columns:
            if col == "적용":
                out[col] = True
            else:
                out[col] = ""

    out = out[expected_cols]

    for col in expected_cols:
        if col != "적용":
            out[col] = out[col].fillna("").astype(str)

    return out

if "pattern_df" not in st.session_state or st.session_state.pattern_df is None:
    st.session_state.pattern_df = pd.DataFrame(
        columns=["적용", "KO", "EN", "File", "Note"]
    )

if "base_pattern_df" not in st.session_state or st.session_state.base_pattern_df is None:
    st.session_state.base_pattern_df = pd.DataFrame(
        columns=["적용", "KO", "EN", "File", "Note"]
    )

def prepare_pattern_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    out = df.copy()

    if "적용" not in out.columns:
        out["적용"] = True
    else:
        out["적용"] = out["적용"].fillna(True).astype(bool)

    expected_cols = [
        "적용",
        "KO",
        "EN",
        "File",
        "Note",
    ]

    for col in expected_cols:
        if col not in out.columns:
            if col == "적용":
                out[col] = True
            else:
                out[col] = ""

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
        df = _ensure_columns(
            df,
            ["KO", "EN", "DNT", "Case-sensitive", "Product", "Note"],
        )
        df["File"] = path.name

        df = df[
            [
                "KO",
                "EN",
                "File",
                "Product",
                "DNT",
                "Case-sensitive",
                "Note",
            ]
        ]
        frames.append(df)

    if not frames:
        return prepare_glossary_editor_df(
            pd.DataFrame(
                columns=[
                    "KO",
                    "EN",
                    "File",
                    "Product",
                    "DNT",
                    "Case-sensitive",
                ]
            )
        )

    result = pd.concat(frames, ignore_index=True)
    return prepare_glossary_editor_df(result)


def load_pattern_table(paths: list[str], pattern_type: str) -> pd.DataFrame:
    frames = []

    for rel_path in paths:
        path = BASE_DIR / rel_path
        if not path.exists():
            continue

        df = read_tsv_flexible(path)
        df = _ensure_columns(df, ["KO", "EN"])
        df["File"] = path.name
        df["Pattern Type"] = pattern_type

        df = df[["KO", "EN", "File", "Pattern Type"]]
        frames.append(df)

    if not frames:
        return prepare_pattern_editor_df(
            pd.DataFrame(columns=["KO", "EN", "File", "Pattern Type"])
        )

    result = pd.concat(frames, ignore_index=True)
    return prepare_pattern_editor_df(result)


def build_product_tables(product_name: str, config: dict):
    product_info = config[product_name]

    glossary_paths = product_info.get("default_glossaries", [])
    pattern_paths = product_info.get("default_patterns", [])

    glossary_df = load_glossary_table(glossary_paths)
    pattern_df = load_pattern_table(pattern_paths, "Pattern")

    return glossary_df, pattern_df


def merge_glossary_upload(current_df: pd.DataFrame, uploaded_file) -> pd.DataFrame:
    uploaded_df = read_uploaded_tsv_flexible(uploaded_file)
    uploaded_df = _ensure_columns(
        uploaded_df,
        ["KO", "EN", "DNT", "Case-sensitive", "Product", "Note"],
    )
    uploaded_df["File"] = uploaded_file.name

    uploaded_df = uploaded_df[
        [
            "KO",
            "EN",
            "File",
            "Product",
            "DNT",
            "Case-sensitive",
            "Note",
        ]
    ]

    merged = pd.concat([current_df, uploaded_df], ignore_index=True)
    return prepare_glossary_editor_df(merged)


def merge_pattern_upload(current_df: pd.DataFrame, uploaded_file, pattern_type: str) -> pd.DataFrame:
    uploaded_df = read_uploaded_tsv_flexible(uploaded_file)
    uploaded_df = _ensure_columns(uploaded_df, ["KO", "EN"])
    uploaded_df["File"] = uploaded_file.name
    uploaded_df["Pattern Type"] = pattern_type

    uploaded_df = uploaded_df[["KO", "EN", "File", "Pattern Type"]]

    merged = pd.concat([current_df, uploaded_df], ignore_index=True)
    return prepare_pattern_editor_df(merged)


def init_session_state():
    defaults = {
        "step": 1,
        "selected_product": None,
        "translation_mode": "매뉴얼",
        "enable_cache": True,
        "glossary_df": None,
        "pattern_df": None,
        "base_glossary_df": None,
        "base_pattern_df": None,
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


def make_default_output_filename(product: str, source_name: str) -> str:
    source_stem = Path(source_name).stem
    safe_product = product.strip().replace(" ", "_")
    return f"{source_stem}_{safe_product}_en.docx"


def render_summary_pills(product: str, mode: str, cache: bool):
    cache_text = "켜짐" if cache else "꺼짐"
    st.markdown(
        f"""
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin: 0 0 16px 0;">
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px; background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>제품</strong> {product}
            </span>
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px; background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>텍스트 유형</strong> {mode}
            </span>
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px; background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>중복 재사용</strong> {cache_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

# ---------------------------------
# Step 1
# ---------------------------------
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

    st.markdown("---")

    if st.button("다음", use_container_width=True):
        st.session_state.selected_product = selected_product
        st.session_state.translation_mode = translation_mode
        st.session_state.enable_cache = enable_cache

        glossary_df, pattern_df = build_product_tables(selected_product, config)

        st.session_state.glossary_df = glossary_df.copy()
        st.session_state.pattern_df = pattern_df.copy()

        st.session_state.base_glossary_df = glossary_df.copy()
        st.session_state.base_pattern_df = pattern_df.copy()

        st.session_state.glossary_editor_key += 1
        st.session_state.pattern_editor_key += 1

        reset_translation_result()
        st.session_state.step = 2
        st.rerun()

# ---------------------------------
# Step 2
# ---------------------------------

elif st.session_state.step == 2:
    st.subheader("Step 2. 용어 및 패턴 선택")
    st.markdown("적용하지 않을 항목은 체크박스를 해제하세요. 필요시 새 항목을 추가하거나 수정, 삭제할 수 있습니다.")
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
    )

    tab1, tab2 = st.tabs(["용어", "패턴"])

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
                    st.session_state.glossary_df = prepare_glossary_editor_df(
                        st.session_state.glossary_df
                    )
                    st.success(f"{uploaded_glossary_tsv.name}을(를) 용어에 추가했습니다.")
                except Exception as e:
                    st.error(f"업로드 오류: {e}")

        top_left, top_right = st.columns([6, 2])
        with top_right:
            if st.button("초기 설정으로 복원", key="reset_glossary", use_container_width=True):
                st.session_state.glossary_df = prepare_glossary_editor_df(
                    st.session_state.base_glossary_df.copy()
                )
                st.rerun()

        st.session_state.glossary_df = prepare_glossary_editor_df(
            st.session_state.glossary_df
        )

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
                        "Pattern",
                    )
                    st.session_state.pattern_df = prepare_pattern_editor_df(
                        st.session_state.pattern_df
                    )
                    st.success(f"{uploaded_pattern_tsv.name}을(를) 패턴에 추가했습니다.")
                except Exception as e:
                    st.error(f"업로드 오류: {e}")

        top_left, top_right = st.columns([6, 2])
        with top_right:
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
                "적용": st.column_config.CheckboxColumn("적용", default=True),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "File": st.column_config.TextColumn("File"),
                "Note": st.column_config.TextColumn("Note"),
            },
            key="pattern_editor_widget",
        )

        st.session_state.pattern_df = prepare_pattern_editor_df(edited_pattern_df)

        col_back, col_next = st.columns([1, 1])

        with col_back:
            if st.button("이전", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

        with col_next:
            if st.button("다음", use_container_width=True):
                if len(st.session_state.glossary_df) == 0:
                    st.error("적어도 하나의 항목은 남겨 두어야 합니다.")
                else:
                    reset_translation_result()
                    st.session_state.step = 3
                    st.rerun()

# ---------------------------------
# Step 3
# ---------------------------------
elif st.session_state.step == 3:
    st.subheader("Step 3. 업로드")
    st.markdown("로컬라이즈할 Word 파일을 업로드하거나 끌어서 놓으세요.")
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
    )

    glossary_rows = (
    st.session_state.glossary_df[
        st.session_state.glossary_df["적용"] == True
    ]
    .drop(columns=["적용"])
    .to_dict("records")
    )
    pattern_rows = (
        st.session_state.pattern_df[
            st.session_state.pattern_df["적용"] == True
        ]
        .drop(columns=["적용"])
        .to_dict("records")
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:20px;
            font-weight:600;
            color:#57606a;
            margin-bottom:12px;
        ">
            파일을 끌어다 놓거나 클릭해 업로드하세요
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_docx = st.file_uploader(
        "업로드 또는 끌어서 놓기",
        type=["docx"],
        label_visibility="collapsed",
    )

    col_back, col_translate = st.columns([1, 1])

    with col_back:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    with col_translate:
        translate_clicked = st.button("번역 시작", type="primary", use_container_width=True)

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
            status_text = st.empty()

            def update_progress(done, total):
                progress = int((done / total) * 100) if total else 0
                progress_bar.progress(progress)
                status_text.text(f"번역 중... {done}/{total} 문단")

            try:
                result = translate_document(
                    in_path=str(input_path),
                    out_path=str(output_path),
                    glossary_rows=glossary_rows,
                    pattern_rows=pattern_rows,
                    api_key=OPENAI_API_KEY,
                    enable_cache=st.session_state.enable_cache,
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

# ---------------------------------
# Step 4
# ---------------------------------
elif st.session_state.step == 4:
    st.subheader("Step 4. 다운로드")

    result = st.session_state.last_result
    output_path = st.session_state.last_output_path
    output_filename = st.session_state.last_output_filename

    if not result or not output_path:
        st.error("번역 결과를 찾을 수 없습니다.")
        st.session_state.step = 1
        st.rerun()

    estimated_cost = estimate_cost_usd(result["total_tokens"])

    st.success("번역이 완료되었습니다.")
    st.write("### 결과")
    st.write(f"**토큰 사용량:** {result['total_tokens']:,} (약 ${estimated_cost})")

    with open(output_path, "rb") as f:
        st.download_button(
            label="번역 파일 다운로드",
            data=f,
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
            use_container_width=True,
        )

    col_prev, col_restart = st.columns([1, 1])

    with col_prev:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

    with col_restart:
        if st.button("다시 시작", use_container_width=True):
            reset_translation_result()
            st.session_state.step = 1
            st.rerun()