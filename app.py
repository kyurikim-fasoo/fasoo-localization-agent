import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from translator_engine import translate_document


load_dotenv()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
COST_PER_1K_TOKENS = float(st.secrets.get("MODEL_COST_PER_1K_TOKENS", os.getenv("MODEL_COST_PER_1K_TOKENS", "0.01")))

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


def estimate_cost_usd(total_tokens: int, rate_per_1k_tokens: float = COST_PER_1K_TOKENS) -> float:
    return round((total_tokens / 1000) * rate_per_1k_tokens, 4)


def save_uploaded_file(uploaded_file, save_dir: Path):
    save_path = save_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
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
            ["KO", "EN", "Def_KO", "DNT", "Case-sensitive", "Category", "Product", "Note"],
        )
        df["File"] = path.name
        df["Selected"] = True

        df = df[
            [
                "Selected",
                "KO",
                "EN",
                "File",
                "Product",
                "Category",
                "DNT",
                "Case-sensitive",
                "Note",
                "Def_KO",
            ]
        ]
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Selected",
                "KO",
                "EN",
                "File",
                "Product",
                "Category",
                "DNT",
                "Case-sensitive",
                "Note",
                "Def_KO",
            ]
        )

    return pd.concat(frames, ignore_index=True)


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
        df["Selected"] = True

        df = df[["Selected", "KO", "EN", "File", "Pattern Type"]]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Selected", "KO", "EN", "File", "Pattern Type"])

    return pd.concat(frames, ignore_index=True)


def build_product_tables(product_name: str, config: dict):
    product_info = config[product_name]

    glossary_paths = product_info.get("default_glossaries", [])
    sentence_paths = product_info.get("default_sentence_patterns", [])
    phrase_paths = product_info.get("default_phrase_patterns", [])

    glossary_df = load_glossary_table(glossary_paths)
    sentence_df = load_pattern_table(sentence_paths, "Sentence")
    phrase_df = load_pattern_table(phrase_paths, "Phrase")

    return glossary_df, sentence_df, phrase_df


def init_session_state():
    defaults = {
        "step": 1,
        "selected_product": None,
        "translation_mode": "Manual",
        "enable_cache": True,
        "glossary_df": None,
        "sentence_df": None,
        "phrase_df": None,
        "last_result": None,
        "last_output_path": None,
        "last_output_filename": None,
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

    /* 업로드 영역 - 배경 제거 + 점선 박스 */
    [data-testid="stFileUploader"] section {
        min-height: 260px;
        border-radius: 16px;
        border: 2px dashed #c8d1dc;
        background: transparent;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* 내부 드롭존 */
    [data-testid="stFileUploaderDropzone"] {
        min-height: 240px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* 내부 텍스트 스타일 */
    [data-testid="stFileUploaderDropzone"] div {
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: #57606a;
    }

    /* hover 시 살짝 강조 */
    [data-testid="stFileUploader"] section:hover {
        border-color: #8c959f;
    }

    /* 기본 버튼 */
    [data-testid="stButton"] button {
        min-height: 46px;
        font-weight: 600;
        border-radius: 12px;
    }

    /* 강조 버튼 */
    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stDownloadButton"] button[kind="primary"] {
        min-height: 54px;
        font-weight: 700;
        border-radius: 14px;
        font-size: 16px;
    }

    /* 다운로드 버튼도 크게 */
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
st.markdown("국문 문서를 영문으로 번역합니다.")

config = load_product_config()
products = list(config.keys())

if not OPENAI_API_KEY:
    st.error("`.env` 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

# ---------------------------------
# Step 1
# ---------------------------------
if st.session_state.step == 1:
    st.subheader("Step 1. 기본 설정")
    st.markdown("번역할 텍스트 유형과 제품을 선택하세요.")

    default_product_index = 0
    if st.session_state.selected_product in products:
        default_product_index = products.index(st.session_state.selected_product)

    translation_mode = st.radio(
        "텍스트 유형",
        options=["UI 텍스트", "가이드"],
        index=0 if st.session_state.translation_mode == "UI" else 1,
        horizontal=True,
    )

    selected_product = st.selectbox(
        "제품",
        products,
        index=default_product_index,
    )

    enable_cache = st.checkbox(
        "중복된 문장은 이전 번역을 재사용해 속도와 비용을 줄입니다.",
        value=st.session_state.enable_cache,
    )

    st.markdown("---")

    if st.button("다음", use_container_width=True):
        st.session_state.selected_product = selected_product
        st.session_state.translation_mode = translation_mode
        st.session_state.enable_cache = enable_cache

        glossary_df, sentence_df, phrase_df = build_product_tables(selected_product, config)
        st.session_state.glossary_df = glossary_df
        st.session_state.sentence_df = sentence_df
        st.session_state.phrase_df = phrase_df

        reset_translation_result()
        st.session_state.step = 2
        st.rerun()

# ---------------------------------
# Step 2
# ---------------------------------
elif st.session_state.step == 2:
    st.subheader("Step 2. 용어 및 번역 스타일 선택")
    st.markdown("번역에 적용할 용어와 표현 방식을 선택합니다.")
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
    )

    tab1, tab2, tab3 = st.tabs(["Glossaries", "Phrase patterns", "Sentence patterns"])

    with tab1:
        st.caption("선택한 용어는 항상 동일하게 번역됩니다. 적용하지 않을 항목은 선택을 해제하세요.")
        edited_glossary_df = st.data_editor(
            st.session_state.glossary_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=[
                "KO",
                "EN",
                "File",
                "Product",
                "Category",
                "DNT",
                "Case-sensitive",
                "Note",
                "Def_KO",
            ],
            column_config={
                "Selected": st.column_config.CheckboxColumn("Selected"),
                "KO": st.column_config.TextColumn("KO", width="large"),
                "EN": st.column_config.TextColumn("EN", width="large"),
                "File": st.column_config.TextColumn("File", width="small"),
                "Product": st.column_config.TextColumn("Product", width="small"),
                "Category": st.column_config.TextColumn("Category", width="small"),
                "DNT": st.column_config.TextColumn("DNT", width="small"),
                "Case-sensitive": st.column_config.TextColumn("Case-sensitive", width="small"),
                "Note": st.column_config.TextColumn("Note", width="medium"),
                "Def_KO": st.column_config.TextColumn("Def_KO", width="medium"),
            },
            key="glossary_editor",
        )
        st.session_state.glossary_df = edited_glossary_df

    with tab2:
        st.caption("비슷한 표현이 나오면 아래 패턴을 참고해 번역합니다. 적용하지 않을 항목은 선택을 해제하세요.")
        edited_phrase_df = st.data_editor(
            st.session_state.phrase_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["KO", "EN", "File", "Pattern Type"],
            column_config={
                "Selected": st.column_config.CheckboxColumn("Selected"),
                "KO": st.column_config.TextColumn("KO", width="large"),
                "EN": st.column_config.TextColumn("EN", width="large"),
                "File": st.column_config.TextColumn("File", width="small"),
                "Pattern Type": st.column_config.TextColumn("Pattern Type", width="small"),
            },
            key="phrase_editor",
        )
        st.session_state.phrase_df = edited_phrase_df

    with tab3:
        st.caption("비슷한 문장이 나오면 아래 예시를 참고해 번역합니다. 적용하지 않을 항목은 선택을 해제하세요.")
        edited_sentence_df = st.data_editor(
            st.session_state.sentence_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["KO", "EN", "File", "Pattern Type"],
            column_config={
                "Selected": st.column_config.CheckboxColumn("Selected"),
                "KO": st.column_config.TextColumn("KO", width="large"),
                "EN": st.column_config.TextColumn("EN", width="large"),
                "File": st.column_config.TextColumn("File", width="small"),
                "Pattern Type": st.column_config.TextColumn("Pattern Type", width="small"),
            },
            key="sentence_editor",
        )
        st.session_state.sentence_df = edited_sentence_df

    col_back, col_next = st.columns([1, 1])

    with col_back:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

    with col_next:
        if st.button("다음", use_container_width=True):
            selected_glossary_count = int(st.session_state.glossary_df["Selected"].sum()) if not st.session_state.glossary_df.empty else 0
            if selected_glossary_count == 0:
                st.error("적어도 하나의 glossary 항목은 선택해야 합니다.")
            else:
                reset_translation_result()
                st.session_state.step = 3
                st.rerun()

# ---------------------------------
# Step 3
# ---------------------------------
elif st.session_state.step == 3:
    st.subheader("Step 3. 업로드")
    st.markdown("번역할 Word 파일을 업로드하거나 끌어서 놓으세요.")
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
    )

    selected_glossary_rows = [
        row for row in st.session_state.glossary_df.to_dict("records")
        if bool(row.get("Selected")) is True
    ]
    selected_sentence_rows = [
        row for row in st.session_state.sentence_df.to_dict("records")
        if bool(row.get("Selected")) is True
    ]
    selected_phrase_rows = [
        row for row in st.session_state.phrase_df.to_dict("records")
        if bool(row.get("Selected")) is True
    ]

    st.markdown(
        """
        <div style="margin-bottom: 10px; color: #57606a; font-size: 14px;">
            파일을 끌어다 놓거나 클릭해 업로드하세요.
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
                    glossary_rows=selected_glossary_rows,
                    sentence_pattern_rows=selected_sentence_rows,
                    phrase_pattern_rows=selected_phrase_rows,
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