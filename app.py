from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from db.schema import DB_PATH, init_db
from services.glossary import (
    build_patterns_preview_df,
    build_terms_preview_df,
    export_to_excel,
    load_patterns,
    load_terms,
    replace_patterns_from_excel,
    replace_terms_from_excel,
    save_patterns_from_dataframe,
    save_terms_from_dataframe,
)
from services.translation_logs import (
    create_log,
    delete_log,
    find_logs_by_source_file,
    find_related_logs,
    get_log,
    list_logs,
    update_note,
)
from services import catalog
from services.jobs import get_job, start_job
from services.users import add_user, list_users
from translator_engine import MARKDOWN_EXTENSIONS, extract_bold_terms


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

# GitHub 자동 백업 (Streamlit Cloud ephemeral 회피).
# secrets에 둔 값을 환경변수로 export — services/sync.py가 os.getenv로 읽음.
for _k in ("GITHUB_TOKEN", "GITHUB_REPO", "GITHUB_BRANCH"):
    _v = _secret_or_env(_k, "")
    if _v:
        os.environ[_k] = _v

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
    # 확장자는 원본을 따라간다 — .mdx 문서를 .docx로 내보내면 안 되므로.
    source_path = Path(source_name)
    suffix = source_path.suffix or ".docx"
    safe_product = product.strip().replace(" ", "_")
    return f"{source_path.stem}_{safe_product}_en{suffix}"


# 다운로드 버튼의 Content-Type. 모르는 확장자는 브라우저가 알아서 처리하도록
# 범용 바이너리로 둔다.
_MIME_BY_SUFFIX = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mdx": "text/markdown",
}


def mime_for(filename: str) -> str:
    return _MIME_BY_SUFFIX.get(Path(filename).suffix.lower(), "application/octet-stream")


# ─────────────────────────────────────────────
# 번역 진행 화면
#
# 번역을 스크립트 실행에 인라인으로 두면, 웹소켓이 한 번 끊겼다 붙는
# 것만으로 Streamlit이 스크립트를 처음부터 다시 돌린다. 그러면 세션에
# 남아있는 translating_now 플래그 때문에 번역이 1번 문단부터 재시작되고,
# 화면상 진행률이 0으로 되감긴다 (긴 문서일수록 끊길 확률이 높아 영영
# 끝나지 않는 루프가 된다). 실제 작업은 services/jobs.py의 데몬 스레드가
# 들고 있고, 여기서는 진행 상황만 읽어 그린다 — rerun이 몇 번 나든
# 이미 돌고 있는 잡에 다시 붙기만 하면 된다.
#
# ⚠️ 잡 레지스트리를 이 파일에 두면 안 된다. Streamlit은 메인 스크립트를
#    매 실행마다 새 모듈 네임스페이스에 exec 하므로 app.py의 모듈 전역은
#    rerun 한 번이면 초기화된다. 자세한 내용은 services/jobs.py 참고.
# ─────────────────────────────────────────────

def render_translation_progress() -> None:
    """번역 중 화면 — 진행률만 그리고, 끝나면 로그를 남기고 Step 3으로."""
    job = get_job(st.session_state.get("translate_job_id"))

    def _clear_job_state() -> None:
        st.session_state.translating_now = False
        st.session_state.pop("translate_job_id", None)

    if job is None:
        # 잡이 사라졌다 = 앱 프로세스가 재시작됐다(데몬 스레드는 함께 죽는다).
        # 단순 rerun으로는 여기 오지 않는다 — 레지스트리가 모듈 전역이므로.
        _clear_job_state()
        st.session_state.pop("pending_translate", None)
        st.error("앱이 재시작되어 번역이 중단되었습니다. 파일을 다시 올리고 시도해주세요.")
        if st.button("돌아가기", type="primary"):
            st.rerun()
        return

    label = "번역/QA 진행 중" if st.session_state.enable_qa else "번역 중"
    done, total = job["done"], job["total"]

    st.progress(int(done / total * 100) if total else 0)
    st.caption(f"{label}... {done}/{total}" if total else f"{label}... 문서 분석 중")
    st.caption("⏳ 연결이 잠깐 끊겨도 번역은 서버에서 계속 진행됩니다.")

    if job["status"] == "running":
        # 폴링 — 이 rerun은 위 st.stop() 덕분에 진행률 영역만 다시 그린다.
        time.sleep(1.0)
        st.rerun()
        return

    # ── 종료 처리 ────────────────────────────────────────────────
    _clear_job_state()
    pending = st.session_state.pop("pending_translate", None) or {}
    params = job["params"]

    if job["status"] == "error":
        st.error(f"오류: {job['error']}")
        if st.button("돌아가기", type="primary"):
            st.rerun()
        return

    result = job["result"]
    st.session_state.last_result = result
    st.session_state.last_output_path = params["out_path"]
    st.session_state.last_output_filename = params["output_filename"]

    # 번역 로그 자동 저장
    try:
        st.session_state.last_log_id = create_log(
            user=st.session_state.current_user,
            product=st.session_state.selected_product or "",
            translation_mode=st.session_state.translation_mode or "",
            source_file=pending.get("uploaded_name") or Path(params["in_path"]).name,
            output_file=params["output_filename"],
            ui_text_overrides=params["ui_overrides"] or {},
            metrics=result,
            note="",
        )
    except Exception as log_err:
        st.warning(f"로그 저장 중 경고: {log_err}")

    # 임시 상태 정리
    st.session_state.pop("ui_text_mapping_rows", None)
    st.session_state.pop("ui_text_source_sig", None)
    st.session_state.pop("ui_text_input_path", None)

    st.session_state.step = 3
    st.rerun()


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
        "app_mode": "Localize",
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

    /* type="primary"가 아닌 일반 download_button은 일반 button과 동일 높이 */
    [data-testid="stDownloadButton"] button:not([kind="primary"]) {
        min-height: 46px;
        font-weight: 600;
        border-radius: 12px;
        font-size: 14px;
    }

    div[data-testid="stDataEditor"] {
        width: 100%;
    }

    /* ── 메인 영역: 상단 여백 + 풀폭 사용 ──────────────────────────
       기본 streamlit은 max-width를 약 730~960px로 제한해서 와이드 화면에서
       좌우 여백이 크게 남는다. 데이터 테이블 위주의 화면이라 풀폭으로 풀어
       가로 공간을 활용한다. */
    .block-container {
        padding-top: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 100% !important;
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
        background: #E8F5E9 !important;
        color: #2E7D32 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
    }
    [data-testid="stPopover"] > div:first-child > button:hover {
        background: #DCEDC8 !important;
    }

    /* ── popover 패널 안의 로그아웃 버튼: 빨간 글자, 호버 시 옅은 빨강 ── */
    div[data-baseweb="popover"] [data-testid="stButton"] button {
        color: #D32F2F !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 12px !important;
        background: transparent !important;
        border: none !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="popover"] [data-testid="stButton"] button:hover {
        background: #FFEBEE !important;
        color: #B71C1C !important;
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
                st.session_state.app_mode = "Localize"
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
            st.session_state.app_mode = "Localize"
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

# ── 페이지 이탈 가드 — unsaved 변경이 있으면 모달로 한 번 더 확인 ───────
# 사용 패턴:
#   사이드바 nav 또는 "이전" 같은 페이지 이동 버튼은 직접 app_mode/step을
#   바꾸지 말고 _try_navigate(target)를 호출. 변경 사항이 있으면 모달이
#   뜨고, 모달의 "예"를 눌러야만 실제로 이동한다.
def _has_unsaved_changes() -> bool:
    """현재 페이지에 저장 안 된 변경이 있는지."""
    mode = st.session_state.get("app_mode")
    # Localize Step 2: UI 텍스트 매핑에 EN 입력이 있으면 unsaved
    if mode == "Localize" and st.session_state.get("step") == 2:
        rows = st.session_state.get("ui_text_mapping_rows", [])
        if any(str(r.get("EN (입력)") or "").strip() for r in rows):
            return True
    # Glossary 관리: staged_master 또는 표 편집 dirty 플래그
    if mode == "Glossary 관리":
        if "staged_master" in st.session_state:
            return True
        if st.session_state.get("glossary_table_dirty"):
            return True
    return False


def _clear_unsaved_state() -> None:
    """사용자가 '예, 버리고 이동'을 선택했을 때 임시 상태를 정리."""
    for k in (
        "ui_text_mapping_rows", "ui_text_source_sig", "ui_text_input_path",
        "ui_text_preload_counts", "staged_master", "glossary_table_dirty",
    ):
        st.session_state.pop(k, None)


def _apply_nav_target(target: dict) -> None:
    """모달 통과 또는 unsaved 없음 → 실제 이동."""
    if "app_mode" in target:
        st.session_state.app_mode = target["app_mode"]
    if "step" in target:
        st.session_state.step = target["step"]


def _try_navigate(target: dict) -> None:
    """페이지 이동 트리거. unsaved 있으면 모달 큐, 없으면 즉시 이동."""
    if _has_unsaved_changes():
        st.session_state.pending_nav_target = target
    else:
        _apply_nav_target(target)
    st.rerun()


@st.dialog("저장하지 않은 변경사항이 있어요")
def _confirm_nav_dialog():
    st.warning("저장하지 않은 입력/편집이 사라집니다.", icon="⚠️")
    st.write("정말 다른 화면으로 이동하시겠어요?")
    col_keep, col_leave = st.columns(2)
    # 왼쪽 — 계속 작성 (안전한 default, 흰색/secondary)
    if col_keep.button("계속 작성", use_container_width=True, key="confirm_nav_keep"):
        st.session_state.pop("pending_nav_target", None)
        st.rerun()
    # 오른쪽 — 저장하지 않고 나가기 (위험한 액션, 빨간색/primary)
    if col_leave.button(
        "저장하지 않고 나가기",
        use_container_width=True,
        type="primary",
        key="confirm_nav_leave",
    ):
        target = st.session_state.pop("pending_nav_target", None) or {}
        _clear_unsaved_state()
        _apply_nav_target(target)
        st.rerun()


# 페이지 렌더링 시작 전, 모달 큐가 있으면 띄움
if st.session_state.get("pending_nav_target"):
    _confirm_nav_dialog()


# ── 사이드바: 메뉴 nav (헤더 라벨 없음, 테두리 없는 Wrapsody 스타일) ────
with st.sidebar:
    if st.button(
        "Localize",
        use_container_width=True,
        type="primary" if st.session_state.app_mode == "Localize" else "secondary",
        key="nav_translate",
    ):
        _try_navigate({"app_mode": "Localize"})

    if st.button(
        "Glossary 관리",
        use_container_width=True,
        type="primary" if st.session_state.app_mode == "Glossary 관리" else "secondary",
        key="nav_glossary",
    ):
        _try_navigate({"app_mode": "Glossary 관리"})

    if st.button(
        "Glossary 추출",
        use_container_width=True,
        type="primary" if st.session_state.app_mode == "Glossary 추출" else "secondary",
        key="nav_extract",
    ):
        _try_navigate({"app_mode": "Glossary 추출"})

    if st.button(
        "로그",
        use_container_width=True,
        type="primary" if st.session_state.app_mode == "로그" else "secondary",
        key="nav_logs",
    ):
        _try_navigate({"app_mode": "로그"})

# ── 메인 헤더 ─────────────────────────────────────────────────────────
# 우측 상단: 원형 이니셜 아이콘 (popover trigger). 클릭하면 사용자 이름 +
# 로그아웃 버튼이 드롭다운으로 뜬다. 모든 모드 공통.
# "Fasoo Localization Agent" 큰 타이틀은 번역 실행 모드에서만 노출 —
# Glossary 관리에선 페이지 자체 subheader가 있으니 중복 제거.
def _render_user_menu() -> None:
    name = st.session_state.current_user or "?"
    initial = name[0].upper()
    with st.popover(initial, use_container_width=False):
        # 큰 동그라미 + 이름 + 역할 — Fireside/Wrapsody 스타일
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:14px;
                        padding:6px 4px 14px 4px;min-width:220px;'>
              <span style='display:inline-flex;align-items:center;justify-content:center;
                           width:52px;height:52px;border-radius:50%;
                           background:#E8F5E9;color:#2E7D32;
                           font-weight:700;font-size:20px;'>{initial}</span>
              <div style='line-height:1.35;'>
                <div style='font-weight:700;font-size:15px;color:#222;'>{name}</div>
                <div style='font-size:12px;color:#888;'>Translator</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("로그아웃", use_container_width=True, key="logout_btn"):
            st.session_state.current_user = ""
            st.rerun()


if st.session_state.app_mode == "Localize":
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
        st.markdown(" ")
        _render_user_menu()


# ─────────────────────────────────────────────
# Step 1 — 기본 정보 (번역 실행 모드에서만 노출)
# ─────────────────────────────────────────────

if st.session_state.app_mode == "Localize" and st.session_state.step == 1:
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
    st.caption(
        f"Team 항목은 모두가 함께 보고 편집합니다. "
        f"Scope을 본인 이름(**{st.session_state.current_user}**)으로 둔 항목은 본인에게만 보이고 본인만 수정할 수 있습니다. "
        f"여기서 추가/수정/삭제한 결과는 번역 시 자동으로 적용됩니다."
    )

    # 제품 필터는 사용 안 함 — 전체 항목을 항상 표시.
    active_product = None

    # ── 상단 액션 버튼: 최신 엑셀 불러오기 (옵션) / 백업 다운로드 ──────
    # 둘 다 옵션. 안 눌러도 표는 정상 표시.
    if "show_master_upload" not in st.session_state:
        st.session_state.show_master_upload = False

    from datetime import date as _date
    _dl_filename = f"glossary_{_date.today().isoformat()}_{st.session_state.current_user}.xlsx"
    try:
        _dl_bytes = export_to_excel(
            current_user=st.session_state.current_user,
            scope_filter="all",
        )
        _dl_ok = True
    except Exception:
        _dl_bytes = b""
        _dl_ok = False

    col_act_upload, col_act_download, _col_spacer = st.columns([2, 2, 6])
    with col_act_upload:
        if st.button("📤  최신 엑셀 불러오기", use_container_width=True, key="toggle_master_upload"):
            st.session_state.show_master_upload = not st.session_state.show_master_upload
            st.rerun()
    with col_act_download:
        st.download_button(
            "📥  백업 다운로드",
            data=_dl_bytes,
            file_name=_dl_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            disabled=not _dl_ok,
            key="download_glossary_btn",
        )

    # ── 최신 엑셀 불러오기 영역 (토글) ─────────────────────────────────
    if st.session_state.show_master_upload:
        with st.container(border=True):
            st.markdown("**최신 Master Glossary 엑셀 불러오기**")
            st.caption(
                "팀에서 관리하는 마스터 엑셀을 올리면 **Team 영역만** 새로 채웁니다. "
                "모든 사용자의 Personal 항목은 그대로 유지됩니다."
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
                    gpick = None
                    for sname, sdf in sheets.items():
                        if any(k in sname.lower() for k in ["glossary", "용어", "사전"]):
                            gpick = (sname, sdf); break
                    if gpick is None and sheets:
                        gpick = list(sheets.items())[0]
                    ppick = None
                    for sname, sdf in sheets.items():
                        if any(k in sname.lower() for k in ["pattern", "패턴"]):
                            ppick = (sname, sdf); break
                    if ppick is None and len(sheets) >= 2:
                        ppick = list(sheets.items())[1]

                    if gpick is None and ppick is None:
                        st.warning("적용할 시트를 찾지 못했습니다.")
                    else:
                        # DB는 안 건드림 — staged로 보관해 표에 미리보기로 표시.
                        # 사용자가 [저장] 버튼을 명시적으로 눌러야 DB에 반영된다.
                        st.session_state.staged_master = {
                            "source": uploaded_workbook.name,
                            "glossary_sheet": gpick,
                            "pattern_sheet": ppick,
                            "active_product": active_product,
                        }
                        st.session_state.show_master_upload = False
                        st.rerun()
                except Exception as e:
                    st.error(f"임포트 오류: {e}")
            if st.button("닫기", key="close_master_upload"):
                st.session_state.show_master_upload = False
                st.rerun()

    # ── Master Excel 미리보기(staged) 알림 ───────────────────────────
    _is_staged = "staged_master" in st.session_state
    if _is_staged:
        _stg = st.session_state.staged_master
        col_warn, col_cancel = st.columns([5, 1])
        with col_warn:
            st.warning(
                f"📥 **{_stg['source']}** 미리보기 중. **[저장]** 을 눌러야 DB에 반영됩니다.",
                icon="⚠️",
            )
        with col_cancel:
            st.markdown(" ")
            if st.button("취소", key="cancel_staged_master", use_container_width=True):
                del st.session_state.staged_master
                st.rerun()

    tab1, tab2 = st.tabs(["용어", "패턴"])

    # ── 용어 탭 ──────────────────────────────
    with tab1:
        if _is_staged and _stg.get("glossary_sheet"):
            # staged 모드: Master Excel glossary 시트를 미리보기 (DB 영향 X)
            _sname, _sdf = _stg["glossary_sheet"]
            terms_df = build_terms_preview_df(
                _sdf,
                source_file=f"{_stg['source']} [{_sname}]",
                product_filter=_stg.get("active_product"),
            )
            _terms_view_ids = set()
            st.caption(f"📥 미리보기 {len(terms_df)}건 (저장 시 Team 영역 전체 교체)")
        else:
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

        # 저장 버튼 활성화 조건:
        # - staged 모드: 항상 활성 (미리보기 데이터를 DB에 commit해야 하므로)
        # - 일반 모드: 표 편집이 있을 때만 활성
        terms_changed = _is_staged or not edited_terms.equals(terms_df)
        # 페이지 이탈 가드용 — 다른 탭/메뉴 이동 시 unsaved 알림 트리거
        if terms_changed:
            st.session_state.glossary_table_dirty = True
        col_status, col_save = st.columns([5, 1])
        with col_status:
            if _is_staged:
                st.caption(f"⚠️ Master Excel 미리보기 — [저장] 시 Team {len(terms_df)}건 전체 교체")
            elif terms_changed:
                st.caption("⚠️ 변경됨 — 저장하지 않으면 사라집니다.")
            else:
                st.caption("✓ 저장됨")
        with col_save:
            save_clicked = st.button(
                "저장",
                key="save_terms",
                disabled=not terms_changed,
                type="primary" if terms_changed else "secondary",
                use_container_width=True,
            )

        if save_clicked:
            try:
                if _is_staged and _stg.get("glossary_sheet"):
                    # staged 모드 commit — Team 영역 전체 교체
                    _sname, _sdf = _stg["glossary_sheet"]
                    n = replace_terms_from_excel(
                        _sdf,
                        source_file=f"{_stg['source']} [{_sname}]",
                        product_filter=_stg.get("active_product"),
                    )
                    # patterns 시트가 있고 아직 staged면 한 번에 처리
                    if _stg.get("pattern_sheet"):
                        _psname, _psdf = _stg["pattern_sheet"]
                        m = replace_patterns_from_excel(
                            _psdf, source_file=f"{_stg['source']} [{_psname}]"
                        )
                        st.toast(f"Master Excel 저장됨 — 용어 {n}건 / 패턴 {m}건", icon="💾")
                    else:
                        st.toast(f"Master Excel 저장됨 — 용어 {n}건", icon="💾")
                    del st.session_state.staged_master
                    st.session_state.pop("glossary_table_dirty", None)
                    st.rerun()
                else:
                    counts = save_terms_from_dataframe(
                        edited_terms,
                        view_ids=_terms_view_ids,
                        current_user=st.session_state.current_user,
                    )
                    summary = f"추가 {counts['inserted']} / 수정 {counts['updated']} / 삭제 {counts['deleted']}"
                    if counts.get("denied", 0) > 0:
                        summary += f" / 권한 없어 거부 {counts['denied']}"
                    st.toast(f"용어 저장됨 — {summary}", icon="💾")
                    st.session_state.pop("glossary_table_dirty", None)
                    st.rerun()
            except Exception as e:
                st.error(f"저장 오류: {e}")

    # ── 패턴 탭 ──────────────────────────────
    with tab2:
        if _is_staged and _stg.get("pattern_sheet"):
            _psname, _psdf = _stg["pattern_sheet"]
            patterns_df = build_patterns_preview_df(
                _psdf,
                source_file=f"{_stg['source']} [{_psname}]",
            )
            _patterns_view_ids = set()
            st.caption(f"📥 미리보기 {len(patterns_df)}건 (저장 시 Team 영역 전체 교체)")
        else:
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

        patterns_changed = _is_staged or not edited_patterns.equals(patterns_df)
        if patterns_changed:
            st.session_state.glossary_table_dirty = True
        col_status_p, col_save_p = st.columns([5, 1])
        with col_status_p:
            if _is_staged:
                st.caption(f"⚠️ Master Excel 미리보기 — [저장] 시 Team {len(patterns_df)}건 전체 교체")
            elif patterns_changed:
                st.caption("⚠️ 변경됨 — 저장하지 않으면 사라집니다.")
            else:
                st.caption("✓ 저장됨")
        with col_save_p:
            save_p_clicked = st.button(
                "저장",
                key="save_patterns",
                disabled=not patterns_changed,
                type="primary" if patterns_changed else "secondary",
                use_container_width=True,
            )

        if save_p_clicked:
            try:
                if _is_staged and _stg.get("pattern_sheet"):
                    _psname, _psdf = _stg["pattern_sheet"]
                    m = replace_patterns_from_excel(
                        _psdf, source_file=f"{_stg['source']} [{_psname}]"
                    )
                    if _stg.get("glossary_sheet"):
                        _sname, _sdf = _stg["glossary_sheet"]
                        n = replace_terms_from_excel(
                            _sdf,
                            source_file=f"{_stg['source']} [{_sname}]",
                            product_filter=_stg.get("active_product"),
                        )
                        st.toast(f"Master Excel 저장됨 — 용어 {n}건 / 패턴 {m}건", icon="💾")
                    else:
                        st.toast(f"Master Excel 저장됨 — 패턴 {m}건", icon="💾")
                    del st.session_state.staged_master
                    st.session_state.pop("glossary_table_dirty", None)
                    st.rerun()
                else:
                    counts = save_patterns_from_dataframe(
                        edited_patterns,
                        view_ids=_patterns_view_ids,
                        current_user=st.session_state.current_user,
                    )
                    summary = f"추가 {counts['inserted']} / 수정 {counts['updated']} / 삭제 {counts['deleted']}"
                    if counts.get("denied", 0) > 0:
                        summary += f" / 권한 없어 거부 {counts['denied']}"
                    st.toast(f"패턴 저장됨 — {summary}", icon="💾")
                    st.session_state.pop("glossary_table_dirty", None)
                    st.rerun()
            except Exception as e:
                st.error(f"저장 오류: {e}")

    # Glossary 관리 페이지는 워크플로가 아니므로 여기서 페이지 렌더링 끝.
    st.stop()


# ─────────────────────────────────────────────
# Glossary 추출 — 고객사 i18n 카탈로그에서 용어/UI 라벨 뽑기
#
# 제품의 다국어 메시지 파일은 key로 정렬돼 있어서, 번역문끼리 짝을 맞추는
# 어려운 단계가 없다. 그래서 이 화면은 LLM을 쓰지 않는다 — 업로드 즉시
# 결정론적으로 후보가 나온다.
# ─────────────────────────────────────────────

if st.session_state.app_mode == "Glossary 추출":
    st.subheader("Glossary 추출")
    st.caption(
        "제품의 다국어 메시지 파일(i18n JSON)을 올리면 **용어집 후보**와 "
        "**UI 라벨 표기**를 자동으로 뽑아냅니다. 한국어/영어 파일을 각각 올리거나, "
        "두 언어가 함께 든 파일 하나를 올리세요."
    )

    uploaded_catalogs = st.file_uploader(
        "JSON 업로드 (최대 2개)",
        type=["json"],
        accept_multiple_files=True,
        key="catalog_uploader",
    )

    if not uploaded_catalogs:
        with st.container(border=True):
            st.markdown("##### 지원하는 파일 형태")
            st.markdown(
                "- `[{\"key\": \"a.b\", \"en\": \"Save\"}]` — 배열 + 언어 필드\n"
                "- `[{\"key\": \"a.b\", \"en\": \"Save\", \"ko\": \"저장\"}]` — 한 파일에 양쪽 언어\n"
                "- `{\"a.b\": \"저장\"}` — 평탄 객체 (i18next · vue-i18n)\n"
                "- `{\"a\": {\"b\": \"저장\"}}` — 중첩 객체\n\n"
                "필드 이름이 `key`/`en`/`ko`가 아니어도 자동으로 찾습니다. "
                "어느 쪽이 한국어인지는 파일명이 아니라 **내용의 한글 비율**로 판단하므로, "
                "파일명에 언어 표시가 없어도 됩니다."
            )
    else:
        _sig = "|".join(f"{f.name}:{f.size}" for f in uploaded_catalogs)
        if st.session_state.get("catalog_sig") != _sig:
            # 6MB짜리 JSON을 rerun마다 다시 파싱하지 않도록 시그니처로 캐시
            try:
                _parsed = []
                for f in uploaded_catalogs:
                    _parsed.append(catalog.parse_json(f.name, json.loads(f.getvalue())))
                _pick = catalog.pick_languages(_parsed)
                _result = catalog.analyze(
                    _pick,
                    catalog.existing_terms_index(
                        load_terms(current_user=st.session_state.current_user)
                    ),
                )
                st.session_state.catalog_parsed = _parsed
                st.session_state.catalog_pick_labels = (_pick.ko_label, _pick.en_label, _pick.ratios)
                st.session_state.catalog_result = _result
                st.session_state.catalog_sig = _sig
                st.session_state.pop("catalog_error", None)
            except Exception as e:
                st.session_state.catalog_error = str(e)
                st.session_state.catalog_sig = _sig
                st.session_state.pop("catalog_result", None)

        if st.session_state.get("catalog_error"):
            st.error(f"파일을 읽지 못했습니다 — {st.session_state.catalog_error}")
        elif st.session_state.get("catalog_result") is not None:
            _result = st.session_state.catalog_result
            _ko_label, _en_label, _ratios = st.session_state.catalog_pick_labels
            _stats = _result.stats

            with st.container(border=True):
                st.markdown("##### 인식 결과")
                for pf in st.session_state.catalog_parsed:
                    st.caption(f"📄 **{pf.name}** — {pf.describe()}")
                st.caption(
                    f"🈚 한국어 = `{_ko_label}` · 🔤 영어 = `{_en_label}` "
                    "(한글 비율로 자동 판정)"
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("대응 쌍", f"{_stats['공통키']:,}")
            c2.metric("용어 후보", f"{_stats['용어후보']:,}")
            c3.metric("UI 라벨", f"{_stats['라벨 고유KO']:,}")
            c4.metric("패턴 후보", f"{_stats['패턴후보']:,}")

            if _stats["충돌"]:
                st.warning(
                    f"같은 한국어에 영어 표기가 여러 개인 항목이 **{_stats['충돌']}건** 있습니다. "
                    "제품 화면마다 표기가 다르다는 뜻이라, 어느 것을 쓸지 골라야 합니다. "
                    "[UI 라벨] 탭 위쪽에 모아뒀습니다.",
                    icon="⚠️",
                )
            if _stats["기존충돌"]:
                st.info(
                    f"기존 글로서리와 영어가 다른 항목이 **{_stats['기존충돌']}건** 있습니다. "
                    "기존 항목이 우선이므로 표의 `기존대조` 열을 확인하세요.",
                    icon="ℹ️",
                )

            _tab_terms, _tab_labels, _tab_patterns = st.tabs(
                [f"용어 후보 ({_stats['용어후보']:,})",
                 f"UI 라벨 ({_stats['라벨 고유KO']:,})",
                 f"패턴 후보 ({_stats['패턴후보']:,})"]
            )

            # 등재 대상 설정 — 탭 공통
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                _extract_product = st.selectbox(
                    "등록할 제품",
                    options=["ALL"] + products,
                    help="기존 용어와 섞이지 않도록 제품을 분리해두면 좋습니다.",
                    key="catalog_product",
                )
            with _rc2:
                _extract_scope = st.selectbox(
                    "공개 범위", options=["Team", "Personal"], key="catalog_scope",
                    help="Team = 모두가 사용 · Personal = 본인만",
                )
            _source_name = " + ".join(pf.name for pf in st.session_state.catalog_parsed)

            # 편집기에 한 번에 올릴 최대 행 수. 수천 행을 체크박스로 뿌리면
            # 브라우저가 버티지 못하므로, 검색으로 좁힌 결과에만 편집기를 준다.
            _EDIT_CAP = 500

            def _render_table(df, cols, key, kind):
                if df.empty:
                    st.info("후보가 없습니다.")
                    return

                q = st.text_input(
                    "검색 (한국어 또는 영어)", key=f"catalog_q_{key}",
                    placeholder="예: 분석", label_visibility="collapsed",
                )
                view = df
                if q.strip():
                    m = df["KO"].str.contains(q, case=False, na=False, regex=False) | \
                        df["EN"].str.contains(q, case=False, na=False, regex=False)
                    view = df[m]

                if "기존대조" in view.columns:
                    if st.checkbox("이미 등록된 항목 제외", value=True,
                                   key=f"catalog_skip_{key}"):
                        view = view[view["기존대조"] != "동일"]

                select_all = st.checkbox(
                    "현재 목록 전체 선택", value=False, key=f"catalog_all_{key}",
                    help="실수로 대량 등재되지 않도록 기본은 해제입니다.",
                )

                capped = view.head(_EDIT_CAP)
                if len(view) > _EDIT_CAP:
                    st.caption(
                        f"{len(view):,}개 중 앞 {_EDIT_CAP}개만 편집할 수 있습니다. "
                        "검색으로 좁혀서 나눠 등재하세요."
                    )
                else:
                    st.caption(f"{len(capped):,}개")

                edit_df = capped[cols].copy()
                edit_df.insert(0, "적용", select_all)

                col_cfg = {c: st.column_config.TextColumn(c, disabled=True) for c in cols}
                col_cfg["적용"] = st.column_config.CheckboxColumn("적용", width="small")
                if "DNT" in cols:
                    col_cfg["DNT"] = st.column_config.CheckboxColumn("DNT", disabled=True)
                if "후보수" in cols:
                    col_cfg["후보수"] = st.column_config.NumberColumn("후보수", disabled=True)
                if "EN 후보" in cols:
                    # 충돌 행은 후보 중에서 고르게 한다 — 어느 화면의 표기인지는
                    # 사람만 알 수 있으므로 자동 선택하지 않는다.
                    _opts = sorted({
                        o for s in capped.get("EN 후보", pd.Series(dtype=str))
                        if s for o in str(s).split(" / ")
                    })
                    if _opts:
                        col_cfg["EN"] = st.column_config.SelectboxColumn(
                            "EN", options=sorted(set(_opts) | set(capped["EN"])),
                            help="후보가 여러 개인 행은 여기서 고르세요.",
                        )

                edited = st.data_editor(
                    edit_df, use_container_width=True, hide_index=True, height=420,
                    column_config=col_cfg, num_rows="fixed",
                    key=f"catalog_edit_{key}_{int(select_all)}_{q.strip()}",
                )
                chosen = edited[edited["적용"] == True]  # noqa: E712

                if "기존대조" in chosen.columns:
                    _n_conflict = int((chosen["기존대조"] == "충돌(기존)").sum())
                    if _n_conflict:
                        st.warning(
                            f"선택한 항목 중 **{_n_conflict}건**은 기존 글로서리와 영어가 "
                            "다릅니다. 그대로 넣으면 같은 한국어에 두 표기가 공존합니다.",
                            icon="⚠️",
                        )

                if st.button(
                    f"선택한 {len(chosen)}개 등재",
                    type="primary", disabled=chosen.empty,
                    key=f"catalog_save_{key}", use_container_width=True,
                ):
                    try:
                        if kind == "pattern":
                            rows = pd.DataFrame({
                                "id": None,
                                "Scope": _extract_scope,
                                "KO": chosen["KO"].values,
                                "EN": chosen["EN"].values,
                                "Note": chosen.get("문맥(key)", ""),
                                "Status": "approved",
                                "File": _source_name,
                            })
                            # view_ids=set() — 삽입만 한다. 기존 행이 삭제될 경로를 차단.
                            counts = save_patterns_from_dataframe(
                                rows, view_ids=set(),
                                current_user=st.session_state.current_user,
                            )
                        else:
                            rows = pd.DataFrame({
                                "id": None,
                                "Scope": _extract_scope,
                                "KO": chosen["KO"].values,
                                "EN": chosen["EN"].values,
                                "Product": _extract_product,
                                "DNT": chosen.get("DNT", False),
                                "Case-sensitive": False,
                                "Note": chosen.get("문맥(key)", ""),
                                "Status": "approved",
                                "File": _source_name,
                            })
                            counts = save_terms_from_dataframe(
                                rows, view_ids=set(),
                                current_user=st.session_state.current_user,
                            )
                        st.success(
                            f"{counts['inserted']}개 등재했습니다. "
                            "[Glossary 관리]에서 확인하세요."
                        )
                        # 방금 넣은 것이 '동일'로 잡히도록 대조표를 다시 만든다
                        st.session_state.pop("catalog_sig", None)
                    except Exception as e:
                        st.error(f"등재 오류: {e}")

            with _tab_terms:
                st.caption(
                    "1:1로 대응되고 용어처럼 보이는 것만 추렸습니다. "
                    "'확인'·'저장' 같은 짧은 동작어는 문맥마다 번역이 달라 제외했습니다."
                )
                _render_table(_result.terms,
                              ["KO", "EN", "문맥(key)", "기존대조", "DNT"],
                              "terms", "term")

            with _tab_labels:
                st.caption(
                    "제품 화면에 실제로 쓰이는 표기입니다. 매뉴얼의 버튼·메뉴 이름을 "
                    "화면과 일치시키는 데 씁니다. `후보수`가 2 이상이면 표기가 갈리는 항목입니다."
                )
                _render_table(_result.labels,
                              ["KO", "EN", "후보수", "EN 후보", "문맥(key)", "기존대조"],
                              "labels", "term")

            with _tab_patterns:
                st.caption("문장형 항목입니다. 용어집이 아니라 번역 예시(패턴)로 씁니다.")
                _render_table(_result.patterns, ["KO", "EN", "문맥(key)"],
                              "patterns", "pattern")

            st.markdown("---")
            st.download_button(
                "전체 후보를 엑셀로 내려받기",
                data=catalog.to_excel(
                    _result.terms, _result.patterns, product=_extract_product
                ),
                file_name=f"glossary_extracted_{_extract_product}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption(
                "내려받은 파일은 **[Glossary 관리]의 마스터 엑셀 업로드**에 그대로 넣을 수 있는 "
                "형식(`glossary` · `pattern` 시트)입니다."
            )

    st.stop()


# ─────────────────────────────────────────────
# 로그 페이지 — 번역 이력 + UI 매핑 reuse + 메모 관리
# ─────────────────────────────────────────────

if st.session_state.app_mode == "로그":
    st.subheader("로그")
    st.caption(
        "팀 전체의 번역 이력입니다. 각 행에는 그때 사용한 UI 텍스트 매핑이 함께 저장되어 있어 "
        "동일 문서 재번역이나 에이전트 테스트 시 매핑을 그대로 불러올 수 있습니다."
    )

    _log_search = st.text_input(
        "검색 (파일명 / 제품 / 메모)",
        key="log_search",
        placeholder="예: Wrapsody, getting-started.docx, 에이전트 테스트 …",
        label_visibility="collapsed",
    )

    # 모든 사용자의 번역 이력 — 팀 작업 추적용. 사용자 컬럼이 표에 있어 누구 것인지 식별 가능.
    _logs_df = list_logs(
        user=None,
        search=_log_search,
        limit=200,
    )
    st.caption(f"{len(_logs_df)}개")

    if _logs_df.empty:
        st.info("아직 번역 이력이 없습니다. **Localize** 메뉴에서 첫 번역을 실행해 보세요.", icon="ℹ️")
    else:
        # 표용 데이터 가공 — 핵심 컬럼만. ID는 selectbox 내부 식별자로만 쓰고
        # 표에서는 숨기고 사용자 이름을 노출한다.
        _list_view = _logs_df[[
            "user", "created_at", "product", "source_file",
            "paragraphs_translated", "total_tokens", "mapping_count", "note",
        ]].copy()
        _list_view.columns = ["사용자", "날짜", "제품", "원본 파일", "단락 수", "토큰", "매핑 수", "메모"]

        st.dataframe(
            _list_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "사용자": st.column_config.TextColumn("사용자", width="small"),
                "날짜": st.column_config.TextColumn("날짜", width="medium"),
                "제품": st.column_config.TextColumn("제품", width="small"),
                "원본 파일": st.column_config.TextColumn("원본 파일"),
                "단락 수": st.column_config.NumberColumn("단락 수", width="small"),
                "토큰": st.column_config.NumberColumn("토큰", width="small"),
                "매핑 수": st.column_config.NumberColumn("매핑 수", width="small"),
                "메모": st.column_config.TextColumn("메모"),
            },
        )

        st.markdown(" ")
        st.markdown("##### 🔎 상세 보기 / 관리")
        _id_options = _logs_df["id"].tolist()
        _id_labels = {
            int(row["id"]): f"{row['created_at']} — {row['user']} — {row['source_file']}"
            for _, row in _logs_df.iterrows()
        }
        _selected_id = st.selectbox(
            "로그 선택",
            options=_id_options,
            format_func=lambda i: _id_labels.get(int(i), str(i)),
            key="log_selected_id",
        )

        if _selected_id is not None:
            _detail = get_log(int(_selected_id))
            if _detail:
                with st.container(border=True):
                    col_d_a, col_d_b, col_d_c = st.columns(3)
                    col_d_a.markdown(f"**날짜**\n\n{_detail['created_at']}")
                    col_d_a.markdown(f"**사용자**\n\n{_detail['user']}")
                    col_d_b.markdown(f"**제품**\n\n{_detail['product'] or '(없음)'}")
                    col_d_b.markdown(f"**유형**\n\n{_detail['translation_mode'] or '(없음)'}")
                    col_d_c.markdown(f"**원본 파일**\n\n{_detail['source_file']}")
                    col_d_c.markdown(f"**결과 파일**\n\n{_detail['output_file']}")

                    st.markdown(" ")
                    col_m_a, col_m_b, col_m_c, col_m_d = st.columns(4)
                    col_m_a.metric("단락 수", f"{_detail['paragraphs_translated']:,}")
                    col_m_b.metric("입력 토큰", f"{_detail['input_tokens']:,}")
                    col_m_c.metric("출력 토큰", f"{_detail['output_tokens']:,}")
                    col_m_d.metric("총 토큰", f"{_detail['total_tokens']:,}")

                    st.markdown("##### UI 텍스트 매핑")
                    _mapping = _detail.get("ui_text_overrides") or {}
                    if _mapping:
                        _map_df = pd.DataFrame(
                            [{"KO (Bold)": k, "EN (입력)": v} for k, v in _mapping.items()]
                        )
                        st.dataframe(_map_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("이 번역에서는 UI 텍스트 매핑을 사용하지 않았습니다.")

                    st.markdown("##### 메모")
                    _note_value = st.text_area(
                        "메모",
                        value=_detail.get("note", ""),
                        key=f"log_note_{_selected_id}",
                        label_visibility="collapsed",
                        height=80,
                    )

                    col_act_save, col_act_load, col_act_del = st.columns(3)
                    with col_act_save:
                        if st.button("메모 저장", use_container_width=True, key=f"log_note_save_{_selected_id}"):
                            try:
                                update_note(int(_selected_id), _note_value.strip())
                                st.toast("메모 저장됨", icon="💾")
                                st.rerun()
                            except Exception as e:
                                st.error(f"메모 저장 오류: {e}")
                    with col_act_load:
                        _can_load = bool(_mapping)
                        if st.button(
                            "이 매핑으로 새 번역",
                            use_container_width=True,
                            disabled=not _can_load,
                            type="primary" if _can_load else "secondary",
                            key=f"log_load_{_selected_id}",
                        ):
                            # 다음 Localize Step 2에서 이 매핑을 빈 EN 칸에 채워주도록 보관
                            st.session_state["preload_ui_mapping"] = dict(_mapping)
                            st.session_state.app_mode = "Localize"
                            st.session_state.step = 1
                            st.toast("매핑 가져왔습니다. Localize → Step 2에서 자동 적용됩니다.", icon="📥")
                            st.rerun()
                    with col_act_del:
                        if st.button(
                            "🗑️ 로그 삭제",
                            use_container_width=True,
                            key=f"log_del_{_selected_id}",
                        ):
                            try:
                                delete_log(int(_selected_id))
                                st.toast("로그 삭제됨", icon="🗑️")
                                st.rerun()
                            except Exception as e:
                                st.error(f"삭제 오류: {e}")

    st.stop()


# ─────────────────────────────────────────────
# Step 2 — 파일 업로드 & 번역 (번역 실행 모드)
# ─────────────────────────────────────────────

if st.session_state.step == 2:
    st.subheader("Step 2. 업로드 & UI 텍스트 매핑")
    st.caption(
        "Word(.docx) 또는 Markdown(.md/.mdx) 문서를 올리면 **볼드(Bold)로 표시된 한국어**를 "
        "자동으로 뽑아 표에 보여드립니다. "
        "그대로 쓰고 싶은 영문 표기가 있으면 EN 칸에 입력하세요. "
        "비워두면 LLM이 알아서 번역하고, 입력한 항목은 그 표기 그대로 사용됩니다. "
        "이 매핑은 자동으로 **[로그] 메뉴에 기록**되어, 나중에 같은 문서를 다시 번역할 때 불러올 수 있어요."
    )
    render_summary_pills(
        st.session_state.selected_product,
        st.session_state.translation_mode,
        st.session_state.enable_cache,
        st.session_state.enable_qa,
    )

    # ── 번역 중이면 진행률만 그리고 끝낸다 ────────────────────────────
    # 여기서 끊어야 (1) 1초마다 도는 폴링 rerun이 아래의 DB 로드/볼드
    # 추출을 반복하지 않고, (2) 번역 중에 살아있는 업로더·data_editor를
    # 건드려 rerun이 나는 사고도 원천 차단된다.
    if st.session_state.get("translating_now"):
        render_translation_progress()
        st.stop()

    # DB에서 직접 로드 — Team + 본인 Personal 합쳐서 사용 (번역 시점)
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
        type=["docx", "md", "markdown", "mdx"],
        label_visibility="collapsed",
    )

    # ── 파일 업로드 후: bold 추출 + 글로서리 자동 매칭 → 매핑 입력 표 ─
    ui_mapping_df = None
    saved_input_path: Optional[Path] = None

    if uploaded_docx is not None:
        # 파일 바뀌면 재추출 (파일명 + size + 컬럼 스키마 버전).
        # 스키마 버전은 ui_text_mapping_rows의 키 구조를 바꿀 때마다 올린다 —
        # 그래야 이전 세션의 row가 새 컬럼과 안 맞을 때 자동 재추출된다.
        file_sig = f"{uploaded_docx.name}::{uploaded_docx.size}::ctx_v1"
        if st.session_state.get("ui_text_source_sig") != file_sig:
            try:
                tmp_path = save_uploaded_file(uploaded_docx, UPLOAD_DIR)
                bold_with_ctx = extract_bold_terms(str(tmp_path))

                # 글로서리 자동 매칭 (KO 일치)
                _glossary_lookup = {
                    r["KO"]: r["EN"]
                    for r in glossary_rows
                    if r.get("KO") and r.get("EN")
                }
                # 로그에서 가져온 매핑 — 글로서리보다 우선
                _preloaded = st.session_state.pop("preload_ui_mapping", None) or {}

                initial_rows = []
                _source_counts = {"글로서리": 0, "로그": 0}
                for ko, ctx in bold_with_ctx:
                    if ko in _preloaded:
                        en = _preloaded[ko]
                        _source_counts["로그"] += 1
                    elif ko in _glossary_lookup:
                        en = _glossary_lookup[ko]
                        _source_counts["글로서리"] += 1
                    else:
                        en = ""
                    initial_rows.append({
                        "KO (Bold)": ko,
                        "EN (입력)": en,
                        "맥락": ctx,
                    })

                st.session_state.ui_text_mapping_rows = initial_rows
                st.session_state.ui_text_source_sig = file_sig
                st.session_state.ui_text_input_path = str(tmp_path)
                st.session_state.ui_text_preload_counts = _source_counts
            except Exception as e:
                st.error(f"문서에서 볼드 텍스트 추출 실패: {e}")

        saved_input_path = Path(st.session_state.ui_text_input_path) if st.session_state.get("ui_text_input_path") else None
        ui_mapping_df = pd.DataFrame(
            st.session_state.get("ui_text_mapping_rows", []),
            columns=["KO (Bold)", "EN (입력)", "맥락"],
        )

        if ui_mapping_df.empty:
            st.info("이 문서에는 볼드 처리된 한국어 텍스트가 없습니다. 그대로 번역을 진행하세요.", icon="ℹ️")
        else:
            # 같은 파일 또는 같은 제품의 이전 로그가 있으면 매핑 재사용 옵션 제공
            _related_logs = find_related_logs(
                source_file=uploaded_docx.name,
                product=st.session_state.selected_product or "",
                limit=15,
            )
            if not _related_logs.empty:
                _file_matches = _related_logs[_related_logs["match_type"] == "file"]
                _product_matches = _related_logs[_related_logs["match_type"] == "product"]

                with st.container(border=True):
                    _headline_bits = []
                    if not _file_matches.empty:
                        _headline_bits.append(f"동일 파일 **{len(_file_matches)}건**")
                    if not _product_matches.empty:
                        _headline_bits.append(
                            f"동일 제품(**{st.session_state.selected_product}**) "
                            f"**{len(_product_matches)}건**"
                        )
                    st.markdown(
                        "📚 **이전 번역의 UI 매핑을 재사용할 수 있어요** — "
                        + " · ".join(_headline_bits)
                        + ". 매핑 개수와 메모를 보고 선택하세요."
                    )

                    _prev_options = {}
                    for _, r in _related_logs.iterrows():
                        _icon = "📄" if r["match_type"] == "file" else "🏷️"
                        _map_count = len(json.loads(r["ui_text_overrides"] or "{}"))
                        _note_bit = f" — {r['note']}" if r.get("note") else ""
                        _prev_options[int(r["id"])] = (
                            f"{_icon} {r['created_at']} — {r['user']} — "
                            f"{r['source_file']} — 매핑 {_map_count}개{_note_bit}"
                        )

                    col_pl_sel, col_pl_btn = st.columns([3, 1])
                    with col_pl_sel:
                        _pl_id = st.selectbox(
                            "가져올 로그",
                            options=list(_prev_options.keys()),
                            format_func=lambda i: _prev_options.get(int(i), str(i)),
                            label_visibility="collapsed",
                            key="prev_log_picker",
                        )
                    with col_pl_btn:
                        if st.button("매핑 불러오기", use_container_width=True, key="prev_log_load"):
                            _ld = get_log(int(_pl_id))
                            _ld_map = (_ld or {}).get("ui_text_overrides") or {}
                            _filled = 0
                            for row in st.session_state.ui_text_mapping_rows:
                                if not row.get("EN (입력)"):
                                    ko = row.get("KO (Bold)")
                                    if ko in _ld_map:
                                        row["EN (입력)"] = _ld_map[ko]
                                        _filled += 1
                            st.toast(f"로그에서 {_filled}개 매핑을 채웠습니다.", icon="📥")
                            st.rerun()
                    st.caption(
                        "📄 = 같은 파일 · 🏷️ = 같은 제품. "
                        "매핑에 있는 KO 중 이 문서의 볼드 KO와 일치하는 것만 자동으로 채워집니다 (기존 입력은 덮어쓰지 않음)."
                    )

            # 자동 매칭 결과 요약
            _counts = st.session_state.get("ui_text_preload_counts", {})
            if _counts.get("글로서리", 0) or _counts.get("로그", 0):
                bits = []
                if _counts.get("로그", 0):
                    bits.append(f"이전 로그에서 **{_counts['로그']}개**")
                if _counts.get("글로서리", 0):
                    bits.append(f"글로서리에서 **{_counts['글로서리']}개**")
                st.caption(f"💡 {' / '.join(bits)} 자동 매칭됨")

            st.markdown(f"##### 📋 UI 텍스트 매핑 ({len(ui_mapping_df)}개)")
            ui_mapping_df = st.data_editor(
                ui_mapping_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                disabled=["KO (Bold)", "맥락"],
                column_order=["KO (Bold)", "EN (입력)", "맥락"],
                column_config={
                    "KO (Bold)": st.column_config.TextColumn("KO (Bold)", width="small"),
                    "EN (입력)": st.column_config.TextColumn(
                        "EN (입력)",
                        help="비워두면 LLM이 번역. 입력하면 이 표기 그대로 사용.",
                        width="medium",
                    ),
                    "맥락": st.column_config.TextColumn(
                        "맥락 (앞뒤 문장)",
                        help="해당 단어가 본문에서 등장한 위치의 앞뒤 문맥. 「대상」 으로 강조 표시.",
                        width="large",
                    ),
                },
                key="ui_text_editor",
            )

    st.markdown("---")
    col_back, col_translate = st.columns(2)

    with col_back:
        if st.button("이전", use_container_width=True):
            _try_navigate({"step": 1})

    with col_translate:
        translate_clicked = st.button(
            "번역 시작",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_docx is None),
        )

    # 클릭 시점에 백그라운드 잡을 띄우고 rerun → 다음 렌더부터는 위쪽
    # translating_now 분기가 진행률만 그린다. 번역 자체는 스크립트 실행과
    # 무관하게 스레드에서 계속 돌기 때문에 rerun이 나도 재시작되지 않는다.
    if translate_clicked:
        if uploaded_docx is None:
            st.error("문서를 업로드하세요.")
        else:
            _ui_overrides_pending = {}
            if ui_mapping_df is not None and not ui_mapping_df.empty:
                for _, row in ui_mapping_df.iterrows():
                    ko = str(row.get("KO (Bold)", "") or "").strip()
                    en = str(row.get("EN (입력)", "") or "").strip()
                    if ko and en:
                        _ui_overrides_pending[ko] = en

            # 업로드 파일 저장은 반드시 메인 스레드에서 — UploadedFile은
            # 스크립트 실행에 묶인 객체라 백그라운드 스레드로 넘기지 않는다.
            if saved_input_path is None or not saved_input_path.exists():
                saved_input_path = save_uploaded_file(uploaded_docx, UPLOAD_DIR)

            _output_filename = make_default_output_filename(
                st.session_state.selected_product,
                uploaded_docx.name,
            )
            st.session_state.translate_job_id = start_job({
                "in_path": str(saved_input_path),
                "out_path": str(OUTPUT_DIR / _output_filename),
                "output_filename": _output_filename,
                "glossary_rows": glossary_rows,
                "pattern_rows": pattern_rows,
                "api_key": OPENAI_API_KEY,
                "enable_cache": st.session_state.enable_cache,
                "enable_qa": st.session_state.enable_qa,
                "translation_mode": st.session_state.translation_mode,
                "ui_overrides": _ui_overrides_pending,
            })
            st.session_state.pending_translate = {"uploaded_name": uploaded_docx.name}
            st.session_state.translating_now = True
            st.rerun()


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
            mime=mime_for(output_filename),
            type="primary",
            use_container_width=True,
        )

    # ── 메모 입력 (이번 번역에 대한 비고) ─────────────────────────────
    _log_id = st.session_state.get("last_log_id")
    if _log_id:
        st.markdown(" ")
        with st.container(border=True):
            st.markdown("##### 📝 메모 (선택)")
            st.caption(
                "이번 번역에 대한 메모를 남겨두면 나중에 '로그' 메뉴에서 검색해 다시 찾기 쉽습니다. "
                "UI 텍스트 매핑도 함께 저장돼 있어 같은 문서를 다시 번역할 때 불러올 수 있어요."
            )
            note_text = st.text_area(
                "메모",
                key="step3_note",
                placeholder="예: 에이전트 테스트 2차 — '저장' 라벨만 'Apply'로 강제 매핑",
                label_visibility="collapsed",
                height=80,
            )
            if st.button("메모 저장", key="save_note_step3"):
                try:
                    update_note(int(_log_id), note_text.strip())
                    st.toast("메모 저장됨", icon="💾")
                except Exception as e:
                    st.error(f"메모 저장 오류: {e}")

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