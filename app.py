import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from db.schema import DB_PATH, init_db
from services.glossary import (
    load_patterns,
    load_terms,
    replace_patterns_from_excel,
    replace_terms_from_excel,
    save_patterns_from_dataframe,
    save_terms_from_dataframe,
)
from translator_engine import translate_document


load_dotenv()


def _secret_or_env(key: str, default: str = "") -> str:
    """Read from Streamlit secrets if available, otherwise fall back to env var.

    `st.secrets[...]` raises StreamlitSecretNotFoundError when no secrets.toml
    exists, which means `.get(...)` cannot be relied on. Wrapping it lets the
    .env path work cleanly without a secrets file.
    """
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


OPENAI_API_KEY = _secret_or_env("OPENAI_API_KEY", "")
COST_PER_1K_INPUT_TOKENS = float(_secret_or_env("MODEL_COST_PER_1K_INPUT_TOKENS", "0.005"))
COST_PER_1K_OUTPUT_TOKENS = float(_secret_or_env("MODEL_COST_PER_1K_OUTPUT_TOKENS", "0.015"))

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


def make_default_output_filename(product: str, source_name: str) -> str:
    source_stem = Path(source_name).stem
    safe_product = product.strip().replace(" ", "_")
    return f"{source_stem}_{safe_product}_en.docx"


# ─────────────────────────────────────────────
# Daily DB backup
# ─────────────────────────────────────────────

def run_daily_backup_if_due() -> Optional[Path]:
    """
    Copy data/glossary.db into data/backups/glossary-YYYY-MM-DD.db once per day.

    Cheap to call on every Streamlit run — exits immediately if today's backup
    already exists. Returns the backup path when a new copy is written.

    Phase 2 todo: also copy this file to OneDrive / NAS by setting
    `BACKUP_MIRROR_DIR` env var.
    """
    import shutil
    from datetime import date

    if not DB_PATH.exists():
        return None

    backup_dir = DB_PATH.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    today_backup = backup_dir / f"glossary-{date.today().isoformat()}.db"

    if today_backup.exists():
        return None

    shutil.copy2(DB_PATH, today_backup)

    # 30일 넘은 백업은 자동 정리
    cutoff = date.today().toordinal() - 30
    for old in backup_dir.glob("glossary-*.db"):
        try:
            ymd = old.stem.replace("glossary-", "")
            from datetime import datetime as _dt
            if _dt.fromisoformat(ymd).date().toordinal() < cutoff:
                old.unlink()
        except Exception:
            continue

    mirror = os.getenv("BACKUP_MIRROR_DIR")
    if mirror:
        try:
            mirror_dir = Path(mirror)
            mirror_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(today_backup, mirror_dir / today_backup.name)
        except Exception:
            pass

    return today_backup


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
init_db()
run_daily_backup_if_due()

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

        st.session_state.glossary_editor_key += 1
        st.session_state.pattern_editor_key += 1

        reset_translation_result()
        st.session_state.step = 2
        st.rerun()


# ─────────────────────────────────────────────
# Step 2 — 용어 및 패턴
# ─────────────────────────────────────────────

elif st.session_state.step == 2:
    st.subheader("Step 2. 용어 및 패턴")
    st.markdown(
        "글로서리는 DB에 영구 저장됩니다. 화면에서 추가/수정/삭제하면 즉시 저장돼요. "
        "Wrapsody에서 최신 master 엑셀을 받으면 아래 '교체 임포트'로 통째로 덮어쓰세요."
    )
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
        st.session_state.enable_qa,
    )

    # ── Master Excel 교체 임포트 (옵션 A — Wrapsody 워크플로) ──────────
    with st.expander("Master Excel 교체 임포트 — 기존 데이터 통째로 덮어쓰기", expanded=False):
        st.caption(
            f"매번 Wrapsody에서 decrypt한 master 엑셀을 올리면 DB가 그 시점 스냅샷으로 새로 채워집니다. "
            f"시트명에 'glossary'/'용어'를 포함하면 용어로, 'pattern'/'패턴'을 포함하면 패턴으로 인식합니다.\n\n"
            f"**Product 필터**: 현재 선택된 제품 `{st.session_state.selected_product}` 또는 `all/ALL`만 통과합니다."
        )
        uploaded_workbook = st.file_uploader(
            "Master Excel 파일",
            type=["xlsx", "xlsm", "xls"],
            key="uploaded_workbook",
            label_visibility="collapsed",
        )
        if uploaded_workbook is not None:
            try:
                from io import BytesIO
                sheets = pd.read_excel(BytesIO(uploaded_workbook.getvalue()), sheet_name=None)
                results = []
                # glossary 시트
                gpick = None
                for sname, sdf in sheets.items():
                    if any(k in sname.lower() for k in ["glossary", "용어", "사전"]):
                        gpick = (sname, sdf); break
                if gpick is None and sheets:
                    gpick = list(sheets.items())[0]
                if gpick:
                    sname, sdf = gpick
                    n = replace_terms_from_excel(
                        sdf,
                        source_file=f"{uploaded_workbook.name} [{sname}]",
                        product_filter=st.session_state.selected_product or None,
                    )
                    results.append(("용어", sname, n))
                # pattern 시트
                ppick = None
                for sname, sdf in sheets.items():
                    if any(k in sname.lower() for k in ["pattern", "패턴"]):
                        ppick = (sname, sdf); break
                if ppick is None and len(sheets) >= 2:
                    ppick = list(sheets.items())[1]
                if ppick:
                    sname, sdf = ppick
                    n = replace_patterns_from_excel(
                        sdf, source_file=f"{uploaded_workbook.name} [{sname}]"
                    )
                    results.append(("패턴", sname, n))
                if results:
                    lines = "\n".join(f"- **{t}** ← `{s}` ({n}건)" for t, s, n in results)
                    st.success(f"{uploaded_workbook.name} 교체 임포트 완료:\n{lines}")
                    st.rerun()
                else:
                    st.warning("임포트할 시트를 찾지 못했습니다.")
            except Exception as e:
                st.error(f"임포트 오류: {e}")

    tab1, tab2 = st.tabs(["용어", "패턴"])

    # ── 용어 탭 ──────────────────────────────
    with tab1:
        col_search, col_count = st.columns([4, 1])
        with col_search:
            terms_search = st.text_input(
                "검색 (KO/EN/Note)",
                key="terms_search",
                placeholder="예: Wrapsody, rule, 정책 …",
                label_visibility="collapsed",
            )

        terms_df = load_terms(
            product=st.session_state.selected_product,
            search=terms_search,
        )
        # 화면에 보였던 id 집합 — autosave 시 "이 안에서만 삭제 허용"하는 안전장치.
        # 현재 product/검색 필터로 안 보이는 row가 실수로 삭제되는 사고 방지.
        _terms_view_ids = {int(v) for v in terms_df["id"].dropna().tolist()}
        with col_count:
            st.caption(f"{len(terms_df)}개")

        edited_terms = st.data_editor(
            terms_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["id", "File", "Status"],
            column_config={
                "적용": st.column_config.CheckboxColumn("적용", default=True),
                "id": st.column_config.NumberColumn("ID", help="자동 부여 (편집 불가)"),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "Product": st.column_config.TextColumn("Product", help="ALL 또는 제품명 (예: fss)"),
                "DNT": st.column_config.CheckboxColumn("DNT"),
                "Case-sensitive": st.column_config.CheckboxColumn("Case-sensitive"),
                "Note": st.column_config.TextColumn("Note"),
                "Status": st.column_config.TextColumn("Status"),
                "File": st.column_config.TextColumn("출처"),
            },
            key=f"terms_editor_{st.session_state.glossary_editor_key}",
        )

        # 변경 감지 → 자동 저장. view_ids로 "화면에 보였던 row 안에서만" 삭제 허용.
        if not edited_terms.equals(terms_df):
            try:
                counts = save_terms_from_dataframe(edited_terms, view_ids=_terms_view_ids)
                summary = f"추가 {counts['inserted']} / 수정 {counts['updated']} / 삭제 {counts['deleted']}"
                st.toast(f"용어 저장됨 — {summary}", icon="💾")
                st.rerun()
            except Exception as e:
                st.error(f"저장 오류: {e}")

    # ── 패턴 탭 ──────────────────────────────
    with tab2:
        col_search, col_count = st.columns([4, 1])
        with col_search:
            patterns_search = st.text_input(
                "검색 (KO/EN/Note)",
                key="patterns_search",
                placeholder="예: 클릭, click, 화면 …",
                label_visibility="collapsed",
            )

        patterns_df = load_patterns(search=patterns_search)
        _patterns_view_ids = {int(v) for v in patterns_df["id"].dropna().tolist()}
        with col_count:
            st.caption(f"{len(patterns_df)}개")

        edited_patterns = st.data_editor(
            patterns_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["id", "File", "Status"],
            column_config={
                "적용": st.column_config.CheckboxColumn("적용", default=True),
                "id": st.column_config.NumberColumn("ID"),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "Note": st.column_config.TextColumn("Note"),
                "Status": st.column_config.TextColumn("Status"),
                "File": st.column_config.TextColumn("출처"),
            },
            key=f"patterns_editor_{st.session_state.pattern_editor_key}",
        )

        if not edited_patterns.equals(patterns_df):
            try:
                counts = save_patterns_from_dataframe(edited_patterns, view_ids=_patterns_view_ids)
                summary = f"추가 {counts['inserted']} / 수정 {counts['updated']} / 삭제 {counts['deleted']}"
                st.toast(f"패턴 저장됨 — {summary}", icon="💾")
                st.rerun()
            except Exception as e:
                st.error(f"저장 오류: {e}")

    # ── 이전 / 다음 버튼 ───────────
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

    # DB에서 직접 로드 — Step 2에서 편집한 결과가 즉시 반영됨
    _terms_df = load_terms(product=st.session_state.selected_product)
    _patterns_df = load_patterns()

    glossary_rows = (
        _terms_df[_terms_df["적용"] == True]
        .drop(columns=["적용", "id", "Status"])
        .to_dict("records")
    )
    pattern_rows = (
        _patterns_df[_patterns_df["적용"] == True]
        .drop(columns=["적용", "id", "Status"])
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