from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from db.schema import DB_PATH, init_db
from services.glossary import (
    export_to_excel,
    load_patterns,
    load_terms,
    replace_patterns_from_excel,
    replace_terms_from_excel,
    save_patterns_from_dataframe,
    save_terms_from_dataframe,
)
from services.users import add_user, list_users
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
        "app_mode": "번역 실행",
        "selected_product": None,
        "translation_mode": "매뉴얼",
        "enable_cache": True,
        "enable_qa": True,
        "current_user": "",
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

    /* ── 사이드바 폭 + 영역 축소 (기본 ~244px → 170px) ───────────── */
    section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 170px;
        min-width: 170px;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 170px !important;
        min-width: 170px !important;
    }

    /* ── 사이드바 메뉴 버튼: Wrapsody 스타일 ──────────────────── */
    /* 비활성: 테두리 없음, 투명 배경, 회색 글자 */
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        text-align: left;
        justify-content: flex-start;
        font-size: 15px;
        font-weight: 500;
        padding-left: 12px;
        padding-right: 12px;
        border: none !important;
        background: transparent !important;
        color: #555 !important;
        border-radius: 8px !important;
        min-height: 38px !important;
        box-shadow: none !important;
    }
    /* 호버: 옅은 회색 배경 */
    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover:not(:disabled) {
        background: #F0F2F5 !important;
        color: #222 !important;
    }
    /* 활성(primary): 연한 녹색 배경 + 진한 녹색 글자 + 굵게 */
    section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        background: #E8F5E9 !important;
        color: #2E7D32 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
        background: #DCEDC8 !important;
        color: #1B5E20 !important;
    }
    /* 비활성화(disabled): 더 옅은 회색 */
    section[data-testid="stSidebar"] [data-testid="stButton"] button:disabled {
        color: #BBB !important;
        background: transparent !important;
    }

    /* ── 사용자 popover trigger: 원형 이니셜 아이콘 ─────────────── */
    [data-testid="stPopover"] > div:first-child > button {
        width: 36px !important;
        height: 36px !important;
        min-height: 36px !important;
        border-radius: 50% !important;
        background: #4CAF50 !important;
        color: #fff !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
    }
    [data-testid="stPopover"] > div:first-child > button:hover {
        background: #43A047 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

config = load_product_config()
products = list(config.keys())

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────
# 로그인 페이지 — current_user가 비어있으면 다른 화면 일체 안 보임
# ──────────────────────────────────────────────────────────────────────
if not st.session_state.current_user:
    # 사이드바 숨기기 (CSS)
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;}</style>",
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("# Fasoo Localization Agent")
        st.caption("국문 문서를 영문으로 로컬라이즈합니다.")
        st.markdown(" ")
        st.markdown("### 👋 시작하기")

        registered = list_users()

        if registered:
            st.caption("기존 사용자")
            options = ["(선택)"] + registered
            chosen = st.selectbox(
                "기존 사용자",
                options,
                label_visibility="collapsed",
            )
            if st.button(
                "로그인",
                type="primary",
                use_container_width=True,
                disabled=(chosen == "(선택)"),
            ):
                st.session_state.current_user = chosen
                st.session_state.app_mode = "번역 실행"
                st.rerun()
            st.markdown(" ")
            st.markdown("**또는 새 사용자 등록**")
        else:
            st.caption("아직 등록된 사용자가 없습니다. 첫 사용자를 등록해주세요.")

        new_name = st.text_input(
            "이름",
            placeholder="이름을 입력하세요",
            label_visibility="collapsed",
            key="login_new_name",
        )
        if st.button(
            "등록 후 시작",
            use_container_width=True,
            type="primary" if not registered else "secondary",
            disabled=(not new_name.strip()),
        ):
            add_user(new_name.strip())
            st.session_state.current_user = new_name.strip()
            st.session_state.app_mode = "번역 실행"
            st.rerun()

        st.markdown(" ")
        st.caption(
            "💡 Team 용어는 모두가 함께 관리하고, "
            "내 용어(Personal)는 본인에게만 보이며 본인만 수정할 수 있습니다."
        )
    st.stop()


# ──────────────────────────────────────────────────────────────────────
# 로그인 후 — 사이드바 nav + 메인 헤더 + 메뉴별 콘텐츠
# ──────────────────────────────────────────────────────────────────────

# ── 사이드바: 메뉴 nav (헤더 라벨 없음, 테두리 없는 Wrapsody 스타일) ────
with st.sidebar:
    if st.button(
        "번역 실행",
        use_container_width=True,
        type="primary" if st.session_state.app_mode == "번역 실행" else "secondary",
        key="nav_translate",
    ):
        st.session_state.app_mode = "번역 실행"
        st.rerun()

    if st.button(
        "Glossary 관리",
        use_container_width=True,
        type="primary" if st.session_state.app_mode == "Glossary 관리" else "secondary",
        key="nav_glossary",
    ):
        st.session_state.app_mode = "Glossary 관리"
        st.rerun()

    st.button("로그 (준비 중)", use_container_width=True, disabled=True, key="nav_logs")

# ── 메인 헤더 ─────────────────────────────────────────────────────────
# 우측 상단: 원형 이니셜 아이콘 (popover trigger). 클릭하면 사용자 이름 +
# 로그아웃 버튼이 드롭다운으로 뜬다. 모든 모드 공통.
# "Fasoo Localization Agent" 큰 타이틀은 번역 실행 모드에서만 노출 —
# Glossary 관리에선 페이지 자체 subheader가 있으니 중복 제거.
def _render_user_menu() -> None:
    name = st.session_state.current_user or "?"
    initial = name[0].upper()
    with st.popover(initial, use_container_width=False):
        st.markdown(f"**👤 {name}**")
        st.divider()
        if st.button("로그아웃", use_container_width=True, key="logout_btn"):
            st.session_state.current_user = ""
            st.rerun()


if st.session_state.app_mode == "번역 실행":
    col_title, col_user = st.columns([9, 1])
    with col_title:
        st.title("Fasoo Localization Agent")
        st.caption("국문 문서를 영문으로 로컬라이즈합니다.")
    with col_user:
        st.markdown(" ")
        _render_user_menu()
else:
    # Glossary 모드: 큰 타이틀 없음. 사용자 아이콘만 우측 상단에.
    _spacer, col_user = st.columns([9, 1])
    with col_user:
        _render_user_menu()


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
# Glossary 관리 페이지 (번역 워크플로와 독립)
# ─────────────────────────────────────────────

if st.session_state.app_mode == "Glossary 관리":
    st.subheader("Glossary 관리")
    st.markdown(
        "Team 용어/패턴은 모두가 함께 관리하고, 내 용어/패턴은 본인에게만 보이며 본인만 수정할 수 있습니다. "
        "추가/수정/삭제 후 **변경사항 저장** 버튼을 눌러야 반영됩니다."
    )

    # 제품 필터 — 글로서리 페이지는 번역 워크플로의 selected_product에 묶이지 않음.
    product_options = ["전체"] + products
    _default_p = st.session_state.selected_product or "전체"
    try:
        _default_idx = product_options.index(_default_p)
    except ValueError:
        _default_idx = 0
    glossary_product_choice = st.selectbox(
        "제품 필터",
        product_options,
        index=_default_idx,
        help="특정 제품 + ALL 공통 항목만 보거나, 전체 항목을 봅니다.",
    )
    active_product = None if glossary_product_choice == "전체" else glossary_product_choice

    # ── 다운로드 (백업 / Wrapsody 재업로드용) ─────────────────────────
    with st.expander("📥 Glossary 다운로드 (.xlsx)", expanded=False):
        st.caption(
            "현재 사용자가 볼 수 있는 용어 + 패턴을 엑셀로 다운받아 "
            "**Wrapsody에 다시 업로드(암호화 관리)하거나 로컬에 백업**할 수 있어요."
        )
        col_dl_scope, col_dl_btn = st.columns([1, 1])
        with col_dl_scope:
            _dl_scope_label = st.selectbox(
                "범위",
                ["전체 (Team + 내 것)", "Team만", "내 것만"],
                key="dl_scope_glossary",
            )
        _dl_scope = {
            "전체 (Team + 내 것)": "all",
            "Team만": "team",
            "내 것만": "mine",
        }[_dl_scope_label]
        from datetime import date as _date
        _dl_filename = f"glossary_{_date.today().isoformat()}_{st.session_state.current_user}.xlsx"
        try:
            _dl_bytes = export_to_excel(
                current_user=st.session_state.current_user,
                scope_filter=_dl_scope,
            )
            with col_dl_btn:
                st.write(" ")
                st.download_button(
                    "엑셀로 다운로드",
                    data=_dl_bytes,
                    file_name=_dl_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"다운로드 준비 오류: {e}")

    # ── Master Excel 교체 임포트 (Team-only) ─────────────────────────
    with st.expander("최신 Master Glossary 엑셀 불러오기 (Team 영역만 교체)", expanded=False):
        st.caption(
            f"팀에서 관리하는 최신 Master Glossary 엑셀 파일을 올리면 "
            f"**Team 영역만 새로 채웁니다.** 모든 사용자의 개인(Personal) 용어는 그대로 유지됩니다.\n\n"
            f"현재 필터: 제품 **{active_product or '전체'}** 항목과 모든 제품 공통(`ALL`) 항목만 가져옵니다."
        )
        uploaded_workbook = st.file_uploader(
            "Master Glossary 엑셀",
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
                        product_filter=active_product,
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
        col_search, col_scope, col_count = st.columns([3, 1, 1])
        with col_search:
            terms_search = st.text_input(
                "검색 (KO/EN/Note)",
                key="terms_search",
                placeholder="예: Wrapsody, rule, 정책 …",
                label_visibility="collapsed",
            )
        with col_scope:
            terms_scope_label = st.selectbox(
                "Scope",
                ["전체", "Team만", "내 용어만"],
                key="terms_scope_filter",
                label_visibility="collapsed",
            )
        terms_scope = {"전체": "all", "Team만": "team", "내 용어만": "mine"}[terms_scope_label]

        terms_df = load_terms(
            product=active_product,
            search=terms_search,
            current_user=st.session_state.current_user,
            scope_filter=terms_scope,
        )
        _terms_view_ids = {int(v) for v in terms_df["id"].dropna().tolist()}
        with col_count:
            st.caption(f"{len(terms_df)}개")

        edited_terms = st.data_editor(
            terms_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["id", "File", "Status"],
            column_order=["적용", "Scope", "KO", "EN", "Product", "DNT", "Case-sensitive", "Note"],
            column_config={
                "적용": st.column_config.CheckboxColumn("적용", default=True),
                "id": None,
                "Scope": st.column_config.SelectboxColumn(
                    "Scope",
                    options=["Team", st.session_state.current_user],
                    default=st.session_state.current_user,
                    help="Team = 모두 공유 / 내 이름 = 본인만 보이는 개인 용어",
                ),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "Product": st.column_config.TextColumn("Product", help="ALL 또는 제품명 (예: AI-R DLP)"),
                "DNT": st.column_config.CheckboxColumn("DNT", help="번역하지 말고 원문 그대로 유지"),
                "Case-sensitive": st.column_config.CheckboxColumn("Case-sensitive", help="대소문자를 정확히 보존"),
                "Note": st.column_config.TextColumn("Note"),
                "Status": None,
                "File": None,
            },
            key=f"terms_editor_{st.session_state.glossary_editor_key}",
        )

        # 변경사항이 있으면 저장 버튼 활성화. 명시적으로 클릭해야만 DB 반영.
        terms_changed = not edited_terms.equals(terms_df)
        col_save, col_status = st.columns([1, 4])
        with col_save:
            save_clicked = st.button(
                "변경사항 저장",
                key="save_terms",
                disabled=not terms_changed,
                type="primary" if terms_changed else "secondary",
                use_container_width=True,
            )
        with col_status:
            if terms_changed:
                st.caption("⚠️ 변경됨 — 저장하지 않으면 다른 화면 이동 시 사라집니다.")
            else:
                st.caption("✓ 모든 변경사항 저장됨")

        if save_clicked:
            try:
                counts = save_terms_from_dataframe(
                    edited_terms,
                    view_ids=_terms_view_ids,
                    current_user=st.session_state.current_user,
                )
                summary = f"추가 {counts['inserted']} / 수정 {counts['updated']} / 삭제 {counts['deleted']}"
                if counts.get("denied", 0) > 0:
                    summary += f" / 권한 없어 거부 {counts['denied']}"
                st.toast(f"용어 저장됨 — {summary}", icon="💾")
                st.rerun()
            except Exception as e:
                st.error(f"저장 오류: {e}")

    # ── 패턴 탭 ──────────────────────────────
    with tab2:
        col_search, col_scope, col_count = st.columns([3, 1, 1])
        with col_search:
            patterns_search = st.text_input(
                "검색 (KO/EN/Note)",
                key="patterns_search",
                placeholder="예: 클릭, click, 화면 …",
                label_visibility="collapsed",
            )
        with col_scope:
            patterns_scope_label = st.selectbox(
                "Scope",
                ["전체", "Team만", "내 패턴만"],
                key="patterns_scope_filter",
                label_visibility="collapsed",
            )
        patterns_scope = {"전체": "all", "Team만": "team", "내 패턴만": "mine"}[patterns_scope_label]

        patterns_df = load_patterns(
            search=patterns_search,
            current_user=st.session_state.current_user,
            scope_filter=patterns_scope,
        )
        _patterns_view_ids = {int(v) for v in patterns_df["id"].dropna().tolist()}
        with col_count:
            st.caption(f"{len(patterns_df)}개")

        edited_patterns = st.data_editor(
            patterns_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            disabled=["id", "File", "Status"],
            column_order=["적용", "Scope", "KO", "EN", "Note"],
            column_config={
                "적용": st.column_config.CheckboxColumn("적용", default=True),
                "id": None,
                "Scope": st.column_config.SelectboxColumn(
                    "Scope",
                    options=["Team", st.session_state.current_user],
                    default=st.session_state.current_user,
                    help="Team = 모두 공유 / 내 이름 = 본인만 보이는 개인 패턴",
                ),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "Note": st.column_config.TextColumn("Note"),
                "Status": None,
                "File": None,
            },
            key=f"patterns_editor_{st.session_state.pattern_editor_key}",
        )

        patterns_changed = not edited_patterns.equals(patterns_df)
        col_save_p, col_status_p = st.columns([1, 4])
        with col_save_p:
            save_p_clicked = st.button(
                "변경사항 저장",
                key="save_patterns",
                disabled=not patterns_changed,
                type="primary" if patterns_changed else "secondary",
                use_container_width=True,
            )
        with col_status_p:
            if patterns_changed:
                st.caption("⚠️ 변경됨 — 저장하지 않으면 다른 화면 이동 시 사라집니다.")
            else:
                st.caption("✓ 모든 변경사항 저장됨")

        if save_p_clicked:
            try:
                counts = save_patterns_from_dataframe(
                    edited_patterns,
                    view_ids=_patterns_view_ids,
                    current_user=st.session_state.current_user,
                )
                summary = f"추가 {counts['inserted']} / 수정 {counts['updated']} / 삭제 {counts['deleted']}"
                if counts.get("denied", 0) > 0:
                    summary += f" / 권한 없어 거부 {counts['denied']}"
                st.toast(f"패턴 저장됨 — {summary}", icon="💾")
                st.rerun()
            except Exception as e:
                st.error(f"저장 오류: {e}")

    # Glossary 관리 페이지는 워크플로가 아니므로 여기서 페이지 렌더링 끝.
    st.stop()


# ─────────────────────────────────────────────
# Step 2 — 파일 업로드 & 번역 (번역 실행 모드)
# ─────────────────────────────────────────────

if st.session_state.step == 2:
    st.subheader("Step 2. 업로드")
    st.markdown("로컬라이즈할 Word 파일을 업로드하거나 끌어서 놓으세요.")
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
        st.session_state.enable_qa,
    )

    # DB에서 직접 로드 — Team + 본인 Personal 합쳐서 사용
    _terms_df = load_terms(
        product=st.session_state.selected_product,
        current_user=st.session_state.current_user,
    )
    _patterns_df = load_patterns(current_user=st.session_state.current_user)

    glossary_rows = (
        _terms_df[_terms_df["적용"] == True]
        .drop(columns=["적용", "id", "Status", "Scope"])
        .to_dict("records")
    )
    pattern_rows = (
        _patterns_df[_patterns_df["적용"] == True]
        .drop(columns=["적용", "id", "Status", "Scope"])
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
            st.session_state.step = 1
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
                st.session_state.step = 3
                st.rerun()

            except Exception as e:
                st.error(f"오류: {e}")


# ─────────────────────────────────────────────
# Step 3 — 결과 & 다운로드
# ─────────────────────────────────────────────

elif st.session_state.step == 3:
    st.subheader("Step 3. 다운로드")

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
            st.session_state.step = 2
            st.rerun()

    with col_restart:
        if st.button("처음으로 돌아가기", use_container_width=True):
            reset_translation_result()
            st.session_state.step = 1
            st.rerun()