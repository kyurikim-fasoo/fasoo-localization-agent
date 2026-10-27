from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Optional

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
    update_review_edits,
    delete_log,
    find_logs_by_source_file,
    find_related_logs,
    get_log,
    list_logs,
    update_note,
)
from services.sync import sync_db
from services import batch
from services import catalog
from services import effect_report
from services import history
from services import review
from services import metrics
from services.jobs import batch_progress, get_job, start_job
from services.users import add_user, list_users
import translator_engine
from translator_engine import (MARKDOWN_EXTENSIONS, extract_bold_terms,
                               extract_korean_paragraphs)
from translator_engine import _clean as _cell
# _cell(): DataFrame 셀을 안전하게 문자열로. 빈 셀이 pandas NaN으로
# 돌아오는데 NaN은 참(truthy)이라 `v or ""`를 통과하고 str(NaN)이
# "nan"이 된다. 그 "nan"이 치환어로 등록돼 본문에 찍힌 적이 있다.


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

# GitHub 자동 백업 (Streamlit Cloud ephemeral 회피).
# secrets에 둔 값을 환경변수로 export — services/sync.py가 os.getenv로 읽음.
# GITHUB_BRANCH는 이제 데이터 경로에 쓰이지 않는다(services/sync.py 참고).
# 데이터는 GITHUB_DATA_BRANCH(기본 app-data)로 간다 — 배포 브랜치에 커밋하면
# Cloud가 앱을 재배포해 모든 세션이 날아가기 때문이다.
for _k in ("GITHUB_TOKEN", "GITHUB_REPO", "GITHUB_BRANCH", "GITHUB_DATA_BRANCH"):
    _v = _secret_or_env(_k, "")
    if _v:
        os.environ[_k] = _v

# 부팅 시 데이터 브랜치에서 최신 DB·사용자·제품 목록을 내려받는다.
# 배포본에 딸려 온 파일은 코드 커밋 시점의 오래된 스냅샷이므로, 여기서 덮어야
# 재부팅 전에 등재한 용어와 로그가 살아 있다.
#
# 반드시 DB를 처음 읽기 전에 호출해야 한다. 프로세스당 한 번만 도는 것은
# services/sync.py가 모듈 전역 플래그로 보장한다(rerun마다 재시도하지 않음).
try:
    from services.sync import pull_data_files
    pull_data_files()
except Exception as _sync_err:      # 못 내려받아도 배포본 파일로 계속 돌아야 한다
    print(f"[sync] 부팅 시 데이터 복원 실패: {_sync_err}")

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


def add_product(name: str) -> tuple[bool, str]:
    """
    product_config.json에 제품을 추가한다. (성공여부, 메시지)

    이 파일은 지금 제품 목록 역할만 한다 — default_glossaries/patterns는
    글로서리가 DB로 옮겨간 뒤로 아무도 읽지 않는다. 그래도 기존 항목과
    같은 모양을 유지해 둔다.
    """
    name = (name or "").strip()
    if not name:
        return False, "제품 이름을 입력하세요."
    if len(name) > 40:
        return False, "제품 이름은 40자 이내로 입력하세요."
    cfg = load_product_config()
    if name in cfg:
        return False, f"'{name}' 제품이 이미 있습니다."
    if name.upper() == "ALL":
        return False, "ALL은 전체 제품을 뜻하는 예약어라 제품 이름으로 쓸 수 없습니다."
    cfg[name] = {"default_glossaries": [], "default_patterns": []}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # Cloud에서는 파일만 쓰면 재부팅 때 사라진다. 글로서리 DB와 같은 방식으로
    # GitHub에 올려둬야 다음 콜드스타트에서도 남는다.
    try:
        from services.sync import sync_products
        sync_products(f"product: + {name}")
    except Exception:
        pass          # 동기화 실패가 제품 추가 자체를 막지는 않는다
    return True, f"'{name}' 제품을 추가했습니다."


def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    save_path = save_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def estimate_cost_usd(input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """공식 단가로 계산한 비용. 단가는 services/metrics.py 한 곳에 둔다 —
    대시보드와 Step 3가 서로 다른 금액을 보이면 안 되므로."""
    return round(metrics.api_cost_usd(input_tokens, cached_tokens, output_tokens), 4)


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


def render_result_detail(entry: dict) -> None:
    """
    파일 하나의 결과 상세 — 번역 검토(미리보기·적용 용어·수정)·다운로드·
    결과 검사·메모.

    배치 결과 화면에서 선택된 파일 하나만 그린다. 파일이 10개일 때 이 내용을
    모두 펼치면 화면이 감당이 안 되므로, 요약표는 위에서 한눈에 보여주고
    근거·검증은 고른 파일만 펼친다.
    """
    result = entry.get("result")

    if not result:
        st.error(f"번역하지 못했습니다: {entry['source_name']}")
        if entry.get("error"):
            st.caption(entry["error"])
        st.caption("서비스가 일시적으로 혼잡할 수 있습니다. 이 파일만 다시 업로드해 시도하세요.")
        return

    if not entry.get("available"):
        st.warning(
            f"결과 파일을 찾을 수 없습니다: {entry['source_name']}. "
            "앱이 다시 시작되어 임시 파일이 삭제되었습니다. 다시 번역하세요.",
            icon="⚠️",
        )
        return

    render_review(entry)
    _render_download_bar(entry)

    # ── 산출물 검증 ──────────────────────────────────────────────────
    # 번역 파이프라인은 문단 단위라, 링크·아이콘을 문서에 되꽂은 뒤 만들어지는
    # 문장 오류를 스스로 볼 수 없다. 저장된 문서를 다시 읽어 검사한 결과다.
    _verify = result.get("verification") or []
    if result.get("segments"):
        # 용어 표기 불일치는 검토 화면의 '표기 확인' 카드가 단락까지 짚어 보여준다.
        # 문서 전체 비율만 말하는 이 경고는 할 수 있는 일이 없어 여기서는 뺀다.
        _verify = [v for v in _verify if "영어 표기가 일관되지 않습니다" not in v.get("title", "")]
    if _verify:
        _v_err = [v for v in _verify if v.get("level") == "오류"]
        _n_warn = len(_verify) - len(_v_err)
        with st.expander(
            f"{'⚠️' if _v_err else '✅'} 결과 검사"
            + (f" — 오류 {len(_v_err)}건" if _v_err else " — 오류 없음")
            + (f" · 경고 {_n_warn}건" if _n_warn else ""),
            expanded=bool(_v_err),
        ):
            for _v in _verify:
                _icon = "✗" if _v["level"] == "오류" else "!"
                st.markdown(f"**{_icon} {_v['title']}**")
                if _v.get("detail"):
                    st.caption(_v["detail"])
            st.caption("원본과 번역본을 비교해 서식 손상, 누락, 용어 불일치를 확인했습니다.")
            # 단락에서 찾을 수 있는 문제(한국어 남음·공백 누락·빈 값·서식)는
            # 검토 화면의 '확인 필요'에 모여 있다 — 거기로 바로 보낸다.
            _fk = st.session_state.get(f"review_filter_key::{entry['output_path']}")
            if _fk:
                st.button(
                    "확인 필요 단락 보기", key=f"verify_goto::{entry['output_path']}",
                    on_click=lambda: (st.session_state.__setitem__(_fk, "확인 필요"),
                                      st.session_state.pop(
                                          f"review_focus::{entry['output_path']}", None)),
                )

    # ── 메모 입력 (이번 번역에 대한 비고) ─────────────────────────────
    _log_id = entry.get("log_id")
    if _log_id:
        with st.expander("메모 (선택)"):
            st.caption("메모는 작업 기록에서 검색할 수 있습니다.")
            note_text = st.text_area(
                "메모",
                key=f"step3_note::{_log_id}",
                placeholder="예: 2차 검토용, '저장'을 'Apply'로 지정",
                label_visibility="collapsed",
                height=80,
            )
            if st.button("메모 저장", key=f"save_note::{_log_id}"):
                try:
                    update_note(int(_log_id), note_text.strip())
                    st.toast("메모를 저장했습니다.", icon="✅")
                except Exception as e:
                    st.error(f"메모를 저장하지 못했습니다. ({e})")


def _render_download_bar(entry: dict) -> None:
    """이 문서의 비용과 다운로드 — 검토를 마친 자리에서 바로 받게 한다."""
    m = entry["result"]
    output_path, output_filename = entry["output_path"], entry["output_filename"]
    cost = estimate_cost_usd(m.get("input_tokens", 0), m.get("cached_tokens", 0),
                             m.get("output_tokens", 0))
    st.markdown(" ")
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    c1.markdown(
        '<div class="rv-bar"><span class="rv-bar-k">● 번역 비용</span>'
        f'<span>입력 토큰 <b>{int(m.get("input_tokens", 0) or 0):,}</b></span>'
        f'<span>출력 토큰 <b>{int(m.get("output_tokens", 0) or 0):,}</b></span>'
        f'<span>API 비용 <b>${cost:,.2f}</b></span></div>',
        unsafe_allow_html=True,
    )
    with open(output_path, "rb") as f:
        c2.download_button(
            label="번역 확정 및 다운로드",
            data=f,
            file_name=output_filename,
            mime=mime_for(output_filename),
            type="primary",
            use_container_width=True,
            key=f"dl::{output_path}",
            help=output_filename,
        )


# ─────────────────────────────────────────────
# 번역 검토 (Review Window)
#
# 번역 결과를 문서처럼 미리 보여주고(적용된 용어 강조), 편집 모드에서 단락을
# 고치면 결과 문서를 다시 만든다. 문서를 고치는 경로는
# translator_engine.rebuild_with_edits 하나뿐이다 — 번역 때와 같은 쓰기 경로를
# 타야 서식이 똑같이 보존된다.
# ─────────────────────────────────────────────

_REVIEW_FILTERS = ["확인 필요", "전체", "용어 적용", "AI 번역", "수정함"]

_REVIEW_CSS = """<style>
.rv-stats{display:flex;flex-wrap:wrap;gap:6px 32px;padding:14px 20px;margin:4px 0 6px;
  border-radius:10px;background:rgba(127,127,127,.07);font-size:.88rem}
.rv-stats span{opacity:.75}.rv-stats b{font-size:1.05rem;opacity:1;margin-left:6px}
.rv-stats .rv-blue{color:#1E88E5}.rv-stats .rv-amber{color:#E65100}
.rv-doc{padding:6px 18px 0;font-size:.95rem;line-height:1.7}
.rv-p{margin:0 0 .75em;padding:2px 0 2px 12px;border-left:3px solid transparent}
.rv-p.rv-h{font-weight:700;font-size:1.12rem;margin-top:.5em}
.rv-p.rv-edited{border-left-color:#E53935}
.rv-p.rv-flag{border-left-color:#F9A825;background:rgba(249,168,37,.08)}
.rv-term{background:rgba(30,136,229,.13);color:#1565C0;border-radius:3px;font-weight:600}
.rv-link{color:#1565C0;text-decoration:underline}
.rv-img{font-size:.72rem;border:1px solid rgba(127,127,127,.45);border-radius:3px;padding:0 4px;opacity:.7}
.rv-tag{font-size:.7rem;font-weight:600;border-radius:4px;padding:1px 6px;margin-left:6px;vertical-align:1px}
.rv-tag.rv-t-edit{background:rgba(229,57,53,.12);color:#C62828}
.rv-tag.rv-t-flag{background:rgba(249,168,37,.18);color:#E65100}
.rv-thead{display:flex;justify-content:space-between;align-items:center;font-weight:700}
.rv-badge{background:rgba(30,136,229,.13);color:#1565C0;border-radius:6px;padding:1px 9px;font-size:.78rem}
.rv-legend{font-size:.75rem;opacity:.7;margin:4px 0 8px}
.rv-legend i,.rv-tcard i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:1px}
.rv-auto{background:#1E88E5}.rv-added{background:#E53935}
.rv-tcard{border:1px solid rgba(127,127,127,.22);border-radius:8px;padding:7px 10px;margin-bottom:6px;
  display:flex;justify-content:space-between;gap:8px}
.rv-tko{font-size:.8rem;font-weight:600}.rv-ten{font-size:.78rem;opacity:.65}
.rv-tn{font-size:.72rem;opacity:.6;white-space:nowrap}
.rv-bar{display:flex;flex-wrap:wrap;gap:4px 22px;padding:10px 16px;border-radius:8px;
  background:rgba(127,127,127,.07);font-size:.82rem}
.rv-bar span{opacity:.75}.rv-bar .rv-bar-k{color:#1E88E5;opacity:1;font-weight:600}
.rv-src{font-size:.82rem;font-weight:400;opacity:.6;margin-top:2px}
.rv-v0,.rv-v1,.rv-v2,.rv-v3{border-radius:3px;font-weight:600}
span.rv-v0{background:rgba(230,81,0,.15);color:#BF360C}span.rv-v1{background:rgba(142,36,170,.14);color:#6A1B9A}
span.rv-v2{background:rgba(0,137,123,.15);color:#00695C}span.rv-v3{background:rgba(109,76,65,.15);color:#4E342E}
i.rv-v0{background:#E65100}i.rv-v1{background:#8E24AA}i.rv-v2{background:#00897B}i.rv-v3{background:#6D4C41}
.rv-vrow{display:flex;justify-content:space-between;gap:8px;font-size:.8rem;margin-top:3px}
.rv-vrow i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:1px}
.rv-badge-warn{background:rgba(230,81,0,.14);color:#BF360C}
</style>"""


def _review_glossary_rows() -> list:
    """결과 검사의 '같은 영어로 번역되는 용어' 확인용 — 번역 때와 같은 범위."""
    df = load_terms(product=st.session_state.selected_product,
                    current_user=st.session_state.current_user)
    return df[df["적용"] == True][["KO", "EN"]].to_dict("records")  # noqa: E712


def render_review(entry: dict) -> None:
    result = entry.get("result") or {}
    segments = result.get("segments") or []
    output_path = entry.get("output_path")
    in_path = entry.get("in_path")
    if not segments:
        return

    ekey = f"review_edits::{output_path}"
    vkey = f"review_ver::{output_path}"
    edits: Dict[int, str] = st.session_state.setdefault(ekey, {})
    ver = st.session_state.setdefault(vkey, 0)
    added = st.session_state.get(f"review_added::{output_path}") or []

    rows = []
    for seg in segments:
        out = edits.get(seg["i"], seg["out"])
        fl = review.flags(seg["src"], out)
        terms_en = [t[1] for t in seg.get("terms") or []]
        rows.append({
            "i": seg["i"],
            "상태": ("✏️ 수정함" if seg["i"] in edits
                    else ("⚠️ " + ", ".join(fl) if fl else "")),
            "원문": review.to_display(seg["src"]),
            "번역": review.to_display(out),
            "_out": out, "_src": seg["src"], "_heading": bool(seg.get("heading")), "_fl": fl,
            "_terms": terms_en + [a[1] for a in added],
            "_flag": bool(fl), "_edited": seg["i"] in edits, "_term": bool(terms_en),
        })
    n_flag = sum(r["_flag"] for r in rows)
    n_term = sum(r["_term"] for r in rows)

    # 표기 확인 — 같은 한국어가 여러 영어 표기로 갈린 용어. 수정·용어 추가가
    # 있을 때만 다시 계산한다(문서 전체 용어 후보를 뽑으므로 가볍지 않다).
    var_key = f"review_variants::{output_path}::{ver}::{len(added)}"
    if var_key not in st.session_state:
        try:
            st.session_state[var_key] = review.term_variants(
                [(r["i"], review._ANY_MARKER_RE.sub("", r["원문"]),
                  review._ANY_MARKER_RE.sub("", r["번역"])) for r in rows],
                exclude_ko={t[0] for s in segments for t in s.get("terms") or []}
                | {a[0] for a in added},
            )
        except Exception:
            st.session_state[var_key] = []      # 표기 확인은 보조 기능 — 화면을 막지 않는다
    dismissed = st.session_state.setdefault(f"review_dismiss::{output_path}", set())
    variants = [v for v in st.session_state[var_key] if v["ko"] not in dismissed]
    fkey = f"review_focus::{output_path}"
    focus = next((v for v in variants if v["ko"] == st.session_state.get(fkey)), None)

    st.markdown(_REVIEW_CSS, unsafe_allow_html=True)
    st.markdown("##### 번역 검토 및 보정")
    st.caption("원문과 번역을 단락 단위로 대조합니다. 적용된 용어를 확인하고, 필요하면 직접 수정해 완성하세요.")
    st.markdown(
        '<div class="rv-stats">'
        f'<span>총 단락<b>{len(rows):,}개</b></span>'
        f'<span>용어 적용<b>{n_term:,}개</b></span>'
        f'<span>AI 번역<b class="rv-blue">{len(rows) - n_term:,}개</b></span>'
        + (f'<span>수정<b>{len(edits)}개</b></span>' if edits else "")
        + (f'<span>확인 필요<b class="rv-amber">{n_flag}개</b></span>' if n_flag else "")
        + (f'<span>표기 확인<b class="rv-amber">{len(variants)}개</b></span>' if variants else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    _flash = st.session_state.pop(f"review_flash::{output_path}", None)
    if _flash:
        st.success(_flash, icon="✅")

    options = [f for f in _REVIEW_FILTERS if f != "확인 필요" or n_flag]
    # 아래 결과 검사의 '확인 필요 단락 보기'가 이 필터를 바꿀 수 있게 키를 알려둔다
    st.session_state[f"review_filter_key::{output_path}"] = (
        f"review_filter::{output_path}::{len(options)}" if n_flag else None)
    f1, f2, f3, f4 = st.columns([4.6, 2.6, 1.7, 1.1], vertical_alignment="center")
    with f1:
        # 선택지가 바뀌면(확인 필요가 생기거나 사라지면) 새 위젯으로 그린다.
        # 기본값은 세션 상태로만 준다 — '확인 필요 단락 보기'가 같은 키를 바꾸므로
        # index=를 함께 주면 Streamlit이 두 출처가 겹친다고 경고한다.
        _rkey = f"review_filter::{output_path}::{len(options)}"
        st.session_state.setdefault(_rkey, options[0])
        # 필터를 고르면 표기 확인('단락 보기') 모드를 푼다 — 안 풀면 필터가 먹지 않는다
        mode = st.radio("표시", options, horizontal=True, key=_rkey,
                        label_visibility="collapsed",
                        on_change=lambda: st.session_state.pop(fkey, None))
    with f2:
        q = st.text_input("검색", key=f"review_q::{output_path}",
                          placeholder="원문 또는 번역에서 검색",
                          label_visibility="collapsed")
    with f3:
        show_src = st.toggle("원문 함께 보기", key=f"review_src::{output_path}")
    with f4:
        edit_mode = st.toggle("편집", key=f"review_edit::{output_path}",
                              disabled=not in_path,
                              help="단락을 표에서 직접 고칩니다." if in_path
                              else "원본 파일이 없어 수정할 수 없습니다.")

    if focus:
        # 표기 확인 중에는 그 용어가 나온 단락만 — 필터보다 우선한다
        fi = set(focus["idx"])
        view = [r for r in rows if r["i"] in fi]
        mode = "표기 확인"
        n1, n2 = st.columns([5, 1], vertical_alignment="center")
        n1.info(f"'{focus['ko']}'이(가) 나온 단락 {len(view)}개를 표시합니다. "
                "표기마다 색을 다르게 칠했습니다.", icon="🔎")
        n2.button("전체 보기", use_container_width=True, key=f"review_unfocus::{output_path}",
                  on_click=lambda: st.session_state.pop(fkey, None))
    else:
        view = {
            "확인 필요": [r for r in rows if r["_flag"]],
            "용어 적용": [r for r in rows if r["_term"]],
            "AI 번역": [r for r in rows if not r["_term"]],
            "수정함": [r for r in rows if r["_edited"]],
        }.get(mode, rows)
    if q.strip():
        ql = q.strip().lower()
        view = [r for r in view if ql in r["원문"].lower() or ql in r["번역"].lower()]

    left, right = st.columns([7, 3])
    changed: Dict[int, str] = {}
    with left, st.container(border=True):
        st.markdown(f"**{'단락 편집' if edit_mode else '번역 문서 미리보기 (EN)'}**")
        st.caption(
            "번역 칸을 직접 고친 뒤 **수정 내용 적용**을 선택하세요. "
            "\\*\\*로 감싼 굵게 표시와 ⟦ ⟧ 기호는 링크·이미지 등의 위치이므로 지우지 마세요."
            if edit_mode else
            "적용된 용어는 파란색으로 강조됩니다."
        )
        if mode == "확인 필요" and view:
            _render_flag_guide(view, output_path, edit_mode)
        if not view:
            st.caption("확인이 필요한 단락이 없습니다." if mode == "확인 필요"
                       else "표시할 단락이 없습니다.")
        elif edit_mode:
            vdf = pd.DataFrame(view)[["i", "상태", "원문", "번역"]]
            edited_df = st.data_editor(
                vdf, use_container_width=True, hide_index=True, num_rows="fixed",
                height=min(520, 38 + 35 * len(vdf)),
                disabled=["i", "상태", "원문"],
                column_config={
                    "i": None,
                    "상태": st.column_config.TextColumn("상태", width="small"),
                    "원문": st.column_config.TextColumn("원문", width="large"),
                    "번역": st.column_config.TextColumn("번역", width="large"),
                },
                key=f"review_editor::{output_path}::{ver}::{mode}::{q.strip()}",
            )
            changed = {
                int(a["i"]): str(b["번역"] or "")
                for a, b in zip(vdf.to_dict("records"), edited_df.to_dict("records"))
                if (b["번역"] or "") != a["번역"]
            }
        else:
            paras = []
            vsurf = [v["surfaces"] for v in focus["variants"]] if focus else None
            for r in view:
                cls = ("rv-p" + (" rv-h" if r["_heading"] else "")
                       + (" rv-edited" if r["_edited"] else "")
                       + (" rv-flag" if r["_flag"] and not r["_edited"] else ""))
                tag = ('<span class="rv-tag rv-t-edit">수정함</span>' if r["_edited"]
                       else "".join(f'<span class="rv-tag rv-t-flag">{f}</span>'
                                    for f in r["_fl"]))
                # 확인 필요·표기 확인은 원문과 비교해야 판단할 수 있다 — 원문을 늘 붙인다
                src = (f'<div class="rv-src">{review.to_preview_html(r["_src"], [])}</div>'
                       if show_src or mode in ("확인 필요", "표기 확인") else "")
                paras.append(f'<div class="{cls}">'
                             f'{review.to_preview_html(r["_out"], r["_terms"], vsurf)}{tag}{src}</div>')
            with st.container(height=520, border=False):
                st.markdown(f'<div class="rv-doc">{"".join(paras)}</div>',
                            unsafe_allow_html=True)
        st.caption(f"{entry.get('source_name', '')} · 단락 {len(view):,}/{len(rows):,}개 표시")

        if edit_mode:
            b1, b2 = st.columns([3, 1], vertical_alignment="center")
            b1.caption(f"변경한 단락 {len(changed)}개" if changed else "변경한 단락이 없습니다.")
            if b2.button("수정 내용 적용", type="primary", use_container_width=True,
                         disabled=not changed, key=f"review_apply::{output_path}"):
                _apply_review(entry, segments, rows, changed, edits, ekey, vkey, ver)

    with right:
        if variants:
            with st.container(border=True):
                _render_variant_cards(entry, variants, fkey, dismissed)
        with st.container(border=True):
            _render_applied_terms(entry, added)


# 확인 필요 태그마다 — 무엇이 문제인지, 어떻게 고치는지
_FLAG_GUIDE = {
    "한국어 남음": "번역문에 한국어가 남아 있습니다. 남은 한국어를 영어로 바꾸세요.",
    "빈 값": "nan, null 같은 빈 값이 들어갔습니다. 대개 Glossary나 UI 텍스트 매핑의 빈 칸 "
             "때문입니다. 그 단어를 지우거나 알맞은 표현으로 바꾸세요.",
    "공백 누락": "단어 두 개가 붙어 있습니다(예: websiteGo). 사이에 공백을 넣으세요.",
    "서식 확인": "원문의 굵게·링크·이미지 위치가 번역과 맞지 않습니다. 아래 회색 원문과 비교해 "
                 "\\*\\* 위치를 맞추세요. 그대로 두면 다운로드 문서에서 해당 서식이 빠질 수 있습니다.",
}


def _render_flag_guide(view: list, output_path: str, edit_mode: bool) -> None:
    """'확인 필요' 단락에 붙은 태그를 풀어 설명하고, 고치는 곳으로 보낸다."""
    from collections import Counter
    cnt = Counter(f for r in view for f in r["_fl"])
    if not cnt:
        return
    lines = "\n".join(f"- **{f}** {n}곳 — {_FLAG_GUIDE.get(f, '')}" for f, n in cnt.items())
    with st.container(border=True):
        st.markdown(f"**자동 검사에서 확인이 필요한 단락 {len(view)}개를 찾았습니다.**\n\n{lines}")
        if edit_mode:
            st.caption("아래 표의 번역 칸을 고친 뒤 **수정 내용 적용**을 선택하세요.")
        else:
            st.button("편집해서 고치기", type="primary", key=f"review_fixgo::{output_path}",
                      on_click=lambda: st.session_state.__setitem__(
                          f"review_edit::{output_path}", True))
            st.caption("문제가 아니라고 판단되면 그대로 다운로드해도 됩니다.")


def _render_variant_cards(entry: dict, variants: list, fkey: str, dismissed: set) -> None:
    """
    표기 확인 카드 — 무엇이(용어), 어디서(단락 보기), 어떻게(Glossary 등록·무시)를
    한 자리에 둔다. 문서 안 표기를 일괄로 바꾸지는 않는다: 단수·복수·관사·문장
    구조에 따라 고칠 방법이 달라 자동 치환은 문장을 깨뜨린다.
    """
    import html as _html

    output_path = entry["output_path"]
    st.markdown(
        f'<div class="rv-thead"><span>표기 확인</span>'
        f'<span class="rv-badge rv-badge-warn">{len(variants)}</span></div>'
        '<div class="rv-legend">같은 한국어가 여러 영어 표기로 번역되었습니다.</div>',
        unsafe_allow_html=True,
    )
    for n, v in enumerate(variants):
        ko = v["ko"]
        with st.container(border=True):
            lines = "".join(
                f'<div class="rv-vrow"><span><i class="rv-v{j}"></i>{_html.escape(x["en"])}</span>'
                f'<span class="rv-tn">{len(x["idx"])}곳</span></div>'
                for j, x in enumerate(v["variants"][:4])
            ) + (f'<div class="rv-vrow"><span class="rv-ten">그 밖의 표현</span>'
                 f'<span class="rv-tn">{v["other"]}곳</span></div>' if v["other"] else "")
            st.markdown(f'<div class="rv-tko">{_html.escape(ko)}</div>{lines}',
                        unsafe_allow_html=True)
            # 옆 패널이 좁아 버튼을 나란히 두면 글자가 세로로 꺾인다 — 위아래로 쌓는다
            st.button("단락 보기", key=f"review_vfocus::{output_path}::{n}",
                      width="stretch",
                      on_click=lambda k=ko: st.session_state.__setitem__(fkey, k))
            # 팝오버는 쓰지 않는다 — 사용자 메뉴용 전역 스타일이 모든 팝오버 버튼을
            # 동그란 아바타 모양으로 줄여 버린다. 카드 안에서 폼을 펼친다.
            rkey = f"review_vreg::{output_path}"
            if st.session_state.get(rkey) != ko:
                st.button("Glossary에 등록", key=f"review_vregopen::{output_path}::{n}",
                          width="stretch",
                          on_click=lambda k=ko: st.session_state.__setitem__(rkey, k))
            else:
                en = st.radio("이 표기로 통일", [x["en"] for x in v["variants"]],
                              key=f"review_ven::{output_path}::{n}")
                scope = st.selectbox(
                    "범위", ["Team", "Personal"], key=f"review_vscope::{output_path}::{n}",
                    format_func=lambda s: "Team (모든 사용자)" if s == "Team" else "개인 (본인만)",
                )
                st.caption("다음 번역부터 이 표기로 고정됩니다. 이 문서의 다른 표기는 "
                           "'단락 보기'에서 확인해 직접 고치세요.")
                r1, r2 = st.columns(2)
                r1.button("취소", key=f"review_vregcancel::{output_path}::{n}", width="stretch",
                          on_click=lambda: st.session_state.pop(rkey, None))
                if r2.button("등록", type="primary", width="stretch",
                             key=f"review_vadd::{output_path}::{n}"):
                    st.session_state.pop(rkey, None)
                    _save_review_term(entry, ko, en, scope)
            st.button("무시", key=f"review_vdismiss::{output_path}::{n}", type="tertiary",
                      help="문맥상 다르게 쓰는 게 맞으면 이 문서에서는 표시하지 않습니다.",
                      on_click=lambda k=ko: (dismissed.add(k),
                                             st.session_state.pop(fkey, None)))


def _render_applied_terms(entry: dict, added: list) -> None:
    """적용된 용어 패널 — 무엇을 어떤 영문으로 고정했는지, 근거를 옆에 둔다."""
    import html as _html

    result = entry["result"]
    output_path, output_filename = entry["output_path"], entry["output_filename"]
    applied = result.get("applied") or []
    # 같은 용어가 출처(Glossary/UI 매핑)나 대소문자만 달리 여러 줄로 오면 하나로 합친다
    merged: Dict[tuple, int] = {}
    for a in applied:
        k = (a.get("KO", ""), a.get("EN", ""))
        k = next((m for m in merged if m[0] == k[0] and m[1].lower() == k[1].lower()), k)
        merged[k] = merged.get(k, 0) + int(a.get("적용") or 0)
    st.markdown(
        f'<div class="rv-thead"><span>적용된 용어</span>'
        f'<span class="rv-badge">{len(merged) + len(added)}</span></div>'
        '<div class="rv-legend"><i class="rv-auto"></i>자동 적용 &nbsp; '
        '<i class="rv-added"></i>검토 중 추가</div>',
        unsafe_allow_html=True,
    )
    cards = [(ko, en, "rv-added", "추가") for ko, en in added] + [
        (ko, en, "rv-auto", f"{n}회") for (ko, en), n in merged.items()
    ]
    if cards:
        with st.container(height=360, border=False):
            st.markdown("".join(
                f'<div class="rv-tcard"><div><div class="rv-tko"><i class="{c}"></i>'
                f'{_html.escape(str(ko))}</div><div class="rv-ten">{_html.escape(str(en))}</div></div>'
                f'<div class="rv-tn">{n}</div></div>'
                for ko, en, c, n in cards
            ), unsafe_allow_html=True)
    else:
        st.caption("이 문서에는 Glossary나 UI 텍스트 매핑에 있는 표현이 없습니다.")

    # 적용 지점을 표시한 리포트. 산출물 자체는 건드리지 않는다 —
    # 배포용 문서에 표시가 남으면 안 되므로 별도 파일이다.
    # rerun마다 다시 만들지 않도록 산출물 경로로 캐시한다.
    if applied:
        _rep_key = f"effect_report::{output_path}"
        if _rep_key not in st.session_state:
            try:
                st.session_state[_rep_key] = effect_report.build_html(
                    applied, str(output_path), output_filename,
                    st.session_state.get("selected_product"),
                )
            except Exception as _e:
                st.session_state[_rep_key] = None
                st.caption(f"적용 내역 보고서를 만들지 못했습니다. ({_e})")
        _rep = st.session_state.get(_rep_key)
        if _rep:
            st.download_button(
                "적용 내역 보고서",
                data=_rep.encode("utf-8"),
                file_name=f"{Path(output_filename).stem}_적용내역.html",
                mime="text/html",
                use_container_width=True,
                key=f"dlrep::{output_path}",
                help="적용된 위치를 색으로 표시한 HTML 파일입니다. "
                     "노란색은 Glossary, 보라색은 UI 텍스트 매핑입니다.",
            )

    # 검토 중 발견한 표현을 바로 Glossary에 — 다음 번역부터 같은 수정을 반복하지 않게
    with st.expander("Glossary에 용어 추가"):
        _review_add_term(entry, output_path)


def _apply_review(entry, segments, rows, changed, edits, ekey, vkey, ver) -> None:
    output_path, in_path = entry["output_path"], entry["in_path"]
    ok, bad = review.apply_edits(segments, changed, edits)
    for i, probs in bad.items():
        _src = next(r["원문"] for r in rows if r["i"] == i)
        st.error(f"반영하지 못했습니다: {_src[:40]}… — {probs[0]}")
    if not ok:
        return
    new_edits = {**edits, **ok}
    try:
        verification = translator_engine.rebuild_with_edits(
            in_path, output_path, segments, new_edits, _review_glossary_rows(),
        )
    except FileNotFoundError:
        st.error("원본 파일을 찾을 수 없습니다. 앱이 다시 시작되어 임시 파일이 "
                 "삭제되었습니다. 다시 번역하세요.")
        return
    except Exception as e:
        st.error(f"수정 내용을 적용하지 못했습니다. ({e})")
        return
    st.session_state[ekey] = new_edits
    st.session_state[vkey] = ver + 1
    entry["result"]["verification"] = verification
    # 다운로드용 캐시(적용 내역 보고서·ZIP)는 이전 파일 기준이다
    for k in [k for k in st.session_state
              if k.startswith("batch_zip::") or k == f"effect_report::{output_path}"]:
        del st.session_state[k]
    if entry.get("log_id"):
        try:
            update_review_edits(int(entry["log_id"]), len(new_edits))
        except Exception:
            pass        # 기록 실패가 수정 반영을 막지는 않는다
    st.session_state[f"review_flash::{output_path}"] = (
        f"단락 {len(ok)}개를 수정했습니다. 다운로드 파일에 반영되었습니다."
        + (f" ({len(bad)}개는 반영하지 못했습니다.)" if bad else "")
    )
    st.rerun()


def _review_add_term(entry: dict, output_path: str) -> None:
    # 좁은 옆 패널에 들어가므로 세로로 쌓는다
    _ko = st.text_input("한국어", key=f"review_g_ko::{output_path}")
    _en = st.text_input("영어", key=f"review_g_en::{output_path}")
    _scope = st.selectbox(
        "범위", ["Team", "Personal"], key=f"review_g_scope::{output_path}",
        format_func=lambda v: "Team (모든 사용자)" if v == "Team" else "개인 (본인만)",
    )
    if st.button("추가", use_container_width=True, key=f"review_g_add::{output_path}",
                 disabled=not (_ko.strip() and _en.strip())):
        _save_review_term(entry, _ko.strip(), _en.strip(), _scope)


def _save_review_term(entry: dict, ko: str, en: str, scope: str) -> None:
    """번역 검토에서 Glossary에 용어를 등록한다 — 직접 추가와 표기 확인 카드가 함께 쓴다."""
    output_path = entry["output_path"]
    try:
        _res = save_terms_from_dataframe(
            pd.DataFrame([{
                "id": None,
                "Scope": "Team" if scope == "Team" else st.session_state.current_user,
                "KO": ko, "EN": en,
                "Product": st.session_state.selected_product or "ALL",
                "DNT": False, "Case-sensitive": False,
                "Note": "번역 검토에서 추가", "Status": "approved",
                "File": entry.get("source_name", ""),
            }]),
            view_ids=set(), current_user=st.session_state.current_user,
            source="번역 검토",
        )
    except Exception as e:
        st.error(f"추가하지 못했습니다. ({e})")
        return
    if _res.get("inserted"):
        # 옆 패널에 '검토 중 추가'로 바로 보이게 하고, 미리보기에서도 강조한다.
        # 표기 확인 카드는 고정된 용어를 빼고 다시 계산되므로 저절로 사라진다.
        st.session_state.setdefault(f"review_added::{output_path}", []).append((ko, en))
        st.session_state.pop(f"review_focus::{output_path}", None)
        st.session_state[f"review_flash::{output_path}"] = (
            f"'{ko}'을(를) '{en}'(으)로 Glossary에 등록했습니다. 다음 번역부터 적용됩니다.")
        st.rerun()
    else:
        st.toast("같은 용어가 이미 Glossary에 있습니다.", icon="ℹ️")


# ─────────────────────────────────────────────
# Glossary 변경 이력 (Glossary 관리 > 변경 이력 탭)
# ─────────────────────────────────────────────

@st.dialog("변경 되돌리기")
def _confirm_revert_dialog(batch_id: str, summary: str) -> None:
    st.write(f"다음 변경을 되돌립니다: **{summary}**")
    st.caption("이후에 다시 수정된 항목은 덮어쓰지 않고 건너뜁니다.")
    c1, c2 = st.columns(2)
    if c1.button("취소", use_container_width=True, key="revert_cancel"):
        st.rerun()
    if c2.button("되돌리기", type="primary", use_container_width=True, key="revert_ok"):
        try:
            res = history.revert_batch(batch_id, st.session_state.current_user)
        except Exception as e:
            st.error(f"되돌리지 못했습니다. ({e})")
            return
        msg = f"{res['reverted']}개 항목을 되돌렸습니다."
        if res["conflicts"]:
            msg += f" 이후 다시 수정된 {res['conflicts']}개는 건너뛰었습니다."
        st.session_state["history_flash"] = msg
        # 편집기가 되돌린 값을 다시 읽도록 키를 바꾼다
        st.session_state.glossary_editor_key += 1
        st.session_state.pattern_editor_key += 1
        st.rerun()


def render_glossary_history(blocked: bool = False) -> None:
    """저장 묶음 목록 → 선택한 묶음의 상세 → 되돌리기."""
    user = st.session_state.current_user
    st.caption("Glossary를 저장할 때마다 변경 내용이 기록됩니다. "
               "변경을 선택해 이전 상태로 되돌릴 수 있습니다.")
    _flash = st.session_state.pop("history_flash", None)
    if _flash:
        st.success(_flash, icon="✅")

    batches = history.list_batches(user)
    if batches.empty:
        st.info("변경 기록이 없습니다.", icon="ℹ️")
        return

    def _src(s: str) -> str:
        return "되돌리기" if s.startswith("되돌리기:") else s

    view = batches.copy()
    view["changed_at"] = view["changed_at"].str[:16].str.replace("T", " ")
    view["source"] = view["source"].map(_src)
    view["상태"] = view["reverted"].map(lambda v: "되돌림" if v else "")
    st.dataframe(
        view[["changed_at", "user", "source", "table", "추가", "수정", "삭제", "상태"]],
        use_container_width=True, hide_index=True,
        height=min(300, 38 + 35 * len(view)),
        column_config={
            "changed_at": st.column_config.TextColumn("일시", width="small"),
            "user": st.column_config.TextColumn("사용자", width="small"),
            "source": st.column_config.TextColumn("작업", width="small"),
            "table": st.column_config.TextColumn("대상", width="small"),
            "추가": st.column_config.NumberColumn("추가", width="small"),
            "수정": st.column_config.NumberColumn("수정", width="small"),
            "삭제": st.column_config.NumberColumn("삭제", width="small"),
            "상태": st.column_config.TextColumn("상태", width="small"),
        },
    )

    labels = {}
    for _, b in view.iterrows():
        parts = [f"{k} {b[k]}" for k in ("추가", "수정", "삭제") if b[k]]
        labels[b["batch_id"]] = (f"{b['changed_at']} · {b['user']} · {b['source']} "
                                 f"({', '.join(parts)})")
    pick = st.selectbox("변경 선택", list(labels), format_func=lambda k: labels[k],
                        key="history_pick")
    rows = history.batch_rows(pick, user)
    st.dataframe(rows, use_container_width=True, hide_index=True,
                 height=min(320, 38 + 35 * max(1, len(rows))))

    already = bool(batches.set_index("batch_id").loc[pick, "reverted"])
    if already:
        st.caption("이미 되돌린 변경입니다.")
    elif blocked:
        st.caption("저장하지 않은 변경 내용이 있습니다. 먼저 저장하거나 취소하세요.")
    if st.button("이 변경 되돌리기", key="history_revert",
                 disabled=already or blocked):
        _confirm_revert_dialog(pick, labels[pick])


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

_FILE_STATUS_ICON = {
    "pending": "·", "running": "▶", "finished": "✓", "error": "✗",
}


def _draw_progress_bar(job: dict) -> None:
    """진행률 위젯 한 벌. fragment와 종료 화면이 같은 모양을 쓰도록 분리."""
    files = job["files"]
    n = len(files)
    label = "번역 및 검토 중" if st.session_state.enable_qa else "번역 중"

    st.progress(int(batch_progress(job) * 100))

    cur = files[job["current"]] if 0 <= job["current"] < n else None
    n_done = sum(1 for f in files if f["status"] in ("finished", "error"))
    if n > 1:
        head = f"{label}... 파일 {min(n_done + 1, n)}/{n}"
    else:
        head = f"{label}..."
    if cur and cur["total"]:
        head += f" · 문단 {cur['done']}/{cur['total']}"
    elif cur:
        head += " · 문서 분석 중"
    st.caption(head)

    # 파일이 여러 개면 어디까지 왔는지 목록으로 보여준다. 30분 걸리는
    # 배치에서 "지금 무슨 파일을 하고 있나"는 진행률 숫자보다 중요하다.
    if n > 1:
        lines = []
        for f in files:
            icon = _FILE_STATUS_ICON.get(f["status"], "·")
            tail = ""
            if f["status"] == "running" and f["total"]:
                tail = f" — {f['done']}/{f['total']}"
            elif f["status"] == "error":
                tail = " — 실패"
            lines.append(f"{icon} {f['source_name']}{tail}")
        st.caption("\n".join(lines))

    st.caption("창을 닫거나 연결이 끊겨도 번역은 계속 진행됩니다.")


# 진행률 폴링 — fragment로 이 함수만 1초마다 다시 돌고, 스크립트 본문은
# 건드리지 않는다.
#
# ⚠️ 예전에는 `time.sleep(1.0); st.rerun()`으로 전체 스크립트를 1초마다
#    되돌렸다. 그러면 스크립트가 한순간도 "끝난" 상태가 되지 못한다.
#    Streamlit 프론트엔드는 실행이 끝날 때 앞선 실행의 남은 위젯을
#    정리하므로, 끝나지 않으면 클릭 직전에 그려둔 Step 2 폼이 화면에
#    그대로 남는다. 남은 위젯은 실행 중에 stale(회색·클릭 불가)로 표시되고
#    rerun 경계에서 잠깐 정상으로 돌아오므로 — "번역 시작" 버튼이 1초
#    주기로 비활성/활성을 깜빡인다. 게다가 그 버튼이 실제로 눌리기 때문에
#    번역 잡이 중복으로 떠서 API가 429/503(overloaded)을 뱉었다.
#
#    fragment는 자기 컨테이너만 다시 그리므로 스크립트 실행은 정상적으로
#    끝나고, 폼은 제대로 치워지며, 1초마다 app.py 전체(사이드바·DB 조회)를
#    재실행하던 낭비도 사라진다.
@st.fragment(run_every=1.0)
def _poll_translation_progress() -> None:
    job = get_job(st.session_state.get("translate_job_id"))
    if job is None:
        # 프로세스가 재시작됐다. 종료 처리는 스크립트 본문 몫이다.
        st.rerun(scope="app")
        return

    _draw_progress_bar(job)

    if job["status"] != "running":
        # 로그 저장·step 변경은 fragment 범위를 넘는다 → 전체 rerun 한 번.
        st.rerun(scope="app")


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
        st.error("앱이 다시 시작되어 번역이 중단되었습니다. 파일을 다시 업로드하세요.")
        if st.button("돌아가기", type="primary"):
            st.rerun()
        return

    if job["status"] == "running":
        _poll_translation_progress()
        return

    _draw_progress_bar(job)

    # ── 종료 처리 ────────────────────────────────────────────────
    params = job["params"]
    files = job["files"]
    ok_files = [f for f in files if f["status"] == "finished"]

    if not ok_files:
        # 전부 실패. 여기서 translating_now를 먼저 내리면, 이어지는 rerun
        # (무엇이든 — 위젯 조작, 웹소켓 재연결) 한 번에 이 화면이 사라지고
        # Step 2 폼으로 되돌아간다. 사용자에게는 "눌렀는데 아무 설명 없이
        # 원래 화면"으로만 보인다. 확인을 누를 때까지 잡을 붙잡아 둔다.
        if len(files) == 1:
            st.error(f"번역하지 못했습니다. ({files[0]['error']})")
        else:
            st.error(f"파일 {len(files)}개를 모두 번역하지 못했습니다.")
            for f in files:
                st.error(f"{f['source_name']} — {f['error']}")
        st.caption("서비스가 일시적으로 혼잡할 수 있습니다. 잠시 후 다시 시도하세요.")
        if st.button("돌아가기", type="primary"):
            _clear_job_state()
            st.rerun()
        return

    _clear_job_state()

    # ── 파일별 결과 + 로그 ───────────────────────────────────────
    results = []
    for f in files:
        entry = {
            "source_name": f["source_name"],
            "output_filename": f["output_filename"],
            "output_path": f["out_path"],
            "in_path": f["in_path"],          # 번역 검토에서 문서를 다시 만들 때
            "result": f["result"],
            "error": f["error"],
            "log_id": None,
        }
        if f["status"] == "finished":
            # 번역 로그 자동 저장 — 파일마다 한 건. 로그 화면과 '이전 매핑
            # 불러오기'가 파일 단위로 동작하므로 배치로 묶지 않는다.
            #
            # 다만 GitHub 푸시(sync)는 파일마다 하지 않는다. sync_db는 1MB짜리
            # DB를 통째로 올리는 커밋이고, Streamlit Cloud는 커밋마다 앱을
            # 재배포한다. 파일 5개면 푸시 5번 = 재배포 방아쇠 5번이라,
            # 사용자가 결과 화면을 보기도 전에 세션이 날아간다. 게다가 이
            # 업로드는 메인 스레드를 막으므로 화면 전환도 그만큼 늦어진다.
            try:
                entry["log_id"] = create_log(
                    user=st.session_state.current_user,
                    product=st.session_state.selected_product or "",
                    translation_mode=st.session_state.translation_mode or "",
                    source_file=f["source_name"],
                    output_file=f["output_filename"],
                    ui_text_overrides=params["ui_overrides"] or {},
                    metrics=f["result"],
                    note="",
                    sync=False,
                )
            except Exception as log_err:
                st.warning(f"작업 기록을 저장하지 못했습니다: {f['source_name']} ({log_err})")
        results.append(entry)

    st.session_state.last_results = results

    # 배치 전체에 대해 푸시 한 번.
    _logged = [r for r in results if r["log_id"]]
    if _logged:
        try:
            _ids = ", ".join(f"#{r['log_id']}" for r in _logged)
            _names = ", ".join(r["source_name"] for r in _logged)
            sync_db(f"Log {_ids} ({_names}) by {st.session_state.current_user}")
        except Exception as sync_err:
            # 푸시 실패는 치명적이지 않다 — 로그는 이미 로컬 DB에 들어갔다.
            st.warning(f"작업 기록을 서버에 저장하지 못했습니다. ({sync_err})")

    # Step 2에서 채운 반복 용어를 글로서리에 등재 (체크한 경우에만).
    # 번역이 끝난 뒤에 하는 이유 — 번역이 실패하면 검증되지 않은 표기가
    # 글로서리에 남는다. 성공한 표기만 재사용할 가치가 있다.
    _pending_terms = st.session_state.pop("pending_term_registrations", None)
    if _pending_terms:
        try:
            _saved = save_terms_from_dataframe(
                pd.DataFrame({
                    "id": None, "Scope": "Team",
                    "KO": list(_pending_terms.keys()),
                    "EN": list(_pending_terms.values()),
                    "Product": st.session_state.selected_product or "ALL",
                    "DNT": False, "Case-sensitive": False,
                    "Note": "Localize 반복 용어에서 추가",
                    "Status": "approved",
                    "File": ", ".join(f["source_name"] for f in ok_files),
                }),
                view_ids=set(),
                current_user=st.session_state.current_user,
                source="Localize",
            )
            if _saved.get("inserted"):
                st.toast(f"Glossary에 용어 {_saved['inserted']}개를 추가했습니다.",
                         icon="✅")
        except Exception as term_err:
            st.warning(f"Glossary에 용어를 추가하지 못했습니다. ({term_err})")

    # 임시 상태 정리
    for _k in ("ui_text_mapping_rows", "ui_text_source_sig", "ui_text_input_paths",
               "term_candidate_rows", "last_batch_detail"):
        st.session_state.pop(_k, None)

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
        "app_mode": "대시보드",
        "selected_product": None,
        "translation_mode": "매뉴얼",
        "enable_cache": True,
        "enable_qa": True,
        "current_user": "",
        # 배치 결과 — 파일 하나를 올려도 길이 1인 목록이다. 단일/다중을
        # 분기하지 않으려고 모양을 하나로 통일했다.
        "last_results": [],
        "glossary_editor_key": 0,
        "pattern_editor_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_translation_result():
    st.session_state.last_results = []
    st.session_state.pop("last_batch_detail", None)


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────

def render_summary_pills(product: str, mode: str, cache: bool, qa: bool = True):
    cache_text = "켜짐" if cache else "꺼짐"
    qa_text = "켜짐" if qa else "꺼짐"
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
                <strong>문장 재사용</strong> {cache_text}
            </span>
            <span style="padding:7px 12px; border:1px solid #d0d7de; border-radius:999px;
                         background:#f6f8fa; color:#24292f; font-size:14px; line-height:1.4;">
                <strong>일관성 검토</strong> {qa_text}
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

    /* ── 사이드바 폭 + 영역 축소 (기본 ~244px → 200px) ───────────── */
    section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 200px;
        min-width: 200px;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 200px !important;
        min-width: 200px !important;
    }

    /* ── Glossary 추출: 페이지 이동 버튼 — 아이콘을 크게 ────────── */
    /* Streamlit은 key가 있는 위젯 래퍼에 .st-key-<key> 클래스를 붙인다. */
    .st-key-catalog_prev button, .st-key-catalog_next button {
        font-size: 20px !important;
        line-height: 1.1 !important;
        padding: 2px 0 !important;
        font-weight: 600 !important;
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


def render_sync_banner() -> None:
    """
    데이터가 서버에 올라가지 않고 있으면 그 사실을 화면에 알린다.

    Cloud의 디스크는 휘발성이라, 동기화가 실패하면 화면상 저장은 성공했는데
    다음 재부팅에 조용히 사라진다. 실제로 "Glossary를 저장했는데 다시 실행하니
    없다"가 이것이었다. 조용히 잃는 것이 가장 나쁜 결과이므로 밖으로 드러낸다.
    """
    try:
        from services import sync
    except Exception:
        return
    if not sync.is_enabled():
        return          # 로컬 개발 — 파일이 그대로 남으므로 알릴 것이 없다

    err = sync.last_error()
    if err:
        # 부팅 시 연결 실패와 저장 실패를 가른다. 둘 다 "데이터가 안 남는다"는
        # 결과는 같지만, 연결 실패는 아직 저장한 것이 없는 시점이라 "저장한
        # 내용이 올라가지 않았다"고 하면 사용자가 엉뚱한 곳을 찾는다.
        if sync.last_error_stage() == sync.STAGE_SAVE:
            headline = ("**변경 내용을 서버에 저장하지 못했습니다.** "
                        "앱이 다시 시작되면 변경 내용이 사라집니다.")
        else:
            headline = ("**서버에 연결할 수 없습니다.** 지금 저장하는 내용은 "
                        "앱이 다시 시작되면 사라집니다.")

        hint = sync.error_hint()
        st.warning(
            headline
            + " **Glossary 관리 > 백업 다운로드**로 데이터를 내려받고 "
              "관리자에게 알리세요."
            + (f"\n\n**원인:** {hint}" if hint else "")
            + f"\n\n`{err}`",
            icon="⚠️",
        )
    elif sync.fallback_active():
        st.info(
            "데이터는 안전하게 저장되지만, 저장할 때 앱이 다시 시작되어 "
            "첫 화면으로 돌아갈 수 있습니다. 관리자는 GITHUB_DATA_BRANCH 설정을 확인하세요.",
            icon="ℹ️",
        )


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
        st.caption("한국어 문서를 영어로 로컬라이즈합니다.")
        st.markdown(" ")
        st.markdown("### 로그인")

        registered = list_users()

        if registered:
            st.caption("사용자")
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
                st.session_state.app_mode = "대시보드"
                st.rerun()
            st.markdown(" ")
            st.markdown("**처음 사용하시나요?**")
        else:
            st.caption("등록된 사용자가 없습니다. 이름을 입력해 시작하세요.")

        new_name = st.text_input(
            "이름",
            placeholder="이름을 입력하세요",
            label_visibility="collapsed",
            key="login_new_name",
        )
        if st.button(
            "등록하고 시작",
            use_container_width=True,
            type="primary" if not registered else "secondary",
            disabled=(not new_name.strip()),
        ):
            add_user(new_name.strip())
            st.session_state.current_user = new_name.strip()
            st.session_state.app_mode = "대시보드"
            st.rerun()

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
        if any(_cell(r.get("EN (입력)")) for r in rows):
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
        "ui_text_mapping_rows", "ui_text_source_sig", "ui_text_input_paths",
        "term_candidate_rows",
        "ui_text_preload_counts", "staged_master", "glossary_table_dirty",
    ):
        st.session_state.pop(k, None)


def _apply_nav_target(target: dict) -> None:
    """
    모달 통과 또는 unsaved 없음 → 실제 이동.

    app_mode/step 외의 키는 그대로 session_state에 넣는다. "이 글로서리로
    Localize"처럼 이동하면서 제품·모드를 함께 정해줘야 하는 경우가 있다.
    """
    for k, v in target.items():
        st.session_state[k] = v


def _try_navigate(target: dict) -> None:
    """페이지 이동 트리거. unsaved 있으면 모달 큐, 없으면 즉시 이동."""
    if _has_unsaved_changes():
        st.session_state.pending_nav_target = target
    else:
        _apply_nav_target(target)
    st.rerun()


@st.dialog("변경 내용을 저장하지 않았습니다")
def _confirm_nav_dialog():
    st.write("이 화면을 나가면 저장하지 않은 변경 내용이 사라집니다.")
    col_keep, col_leave = st.columns(2)
    # 왼쪽 — 계속 작성 (안전한 default, 흰색/secondary)
    if col_keep.button("취소", use_container_width=True, key="confirm_nav_keep"):
        st.session_state.pop("pending_nav_target", None)
        st.rerun()
    # 오른쪽 — 저장하지 않고 나가기 (위험한 액션, 빨간색/primary)
    if col_leave.button(
        "저장하지 않고 이동",
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


render_sync_banner()


# ── 사이드바: 메뉴 nav (헤더 라벨 없음, 테두리 없는 Wrapsody 스타일) ────
# (app_mode 값, 표시 라벨, 위젯 key). app_mode 값은 코드 곳곳의 분기 키라
# 그대로 두고, 화면에 보이는 이름만 라벨로 바꾼다.
NAV_ITEMS = [
    ("대시보드", "대시보드", "nav_dashboard"),
    ("Localize", "Localize", "nav_translate"),
    ("Glossary 관리", "Glossary 관리", "nav_glossary"),
    ("Glossary 추출", "Glossary 추출", "nav_extract"),
    ("로그", "작업 기록", "nav_logs"),
]

with st.sidebar:
    for _mode, _label, _key in NAV_ITEMS:
        if st.button(
            _label,
            use_container_width=True,
            type="primary" if st.session_state.app_mode == _mode else "secondary",
            key=_key,
        ):
            _try_navigate({"app_mode": _mode})

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
        st.caption("한국어 문서를 영어로 로컬라이즈합니다.")
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
    st.subheader("1단계: 기본 정보")
    st.caption("제품과 텍스트 유형을 선택하세요.")

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
        "같은 문장 재사용",
        help="문서에 반복되는 문장은 한 번만 번역해 시간과 비용을 줄입니다.",
        value=st.session_state.enable_cache,
    )

    enable_qa = st.checkbox(
        "일관성 검토",
        help="번역 후 한 번 더 검토해 문서 전체의 용어와 어조를 맞춥니다. 시간이 더 걸립니다.",
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
        "여기서 변경한 내용은 다음 번역부터 적용됩니다. Team 항목은 모든 사용자가 사용하고, "
        "범위를 본인 이름으로 지정한 항목은 본인만 보고 수정할 수 있습니다."
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
        if st.button("Excel 가져오기", use_container_width=True, key="toggle_master_upload"):
            st.session_state.show_master_upload = not st.session_state.show_master_upload
            st.rerun()
    with col_act_download:
        st.download_button(
            "백업 다운로드",
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
            st.markdown("**Master Glossary 가져오기**")
            st.caption("Team 항목을 Excel 파일 내용으로 바꿉니다. 개인 항목은 바뀌지 않습니다.")
            uploaded_workbook = st.file_uploader(
                "Master Glossary Excel 파일",
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
                        st.warning("가져올 시트가 없습니다. 시트 이름에 glossary 또는 pattern이 들어가야 합니다.")
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
                    st.error(f"파일을 가져오지 못했습니다. ({e})")
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
                f"**{_stg['source']}** 미리 보기입니다. **저장**을 선택해야 적용됩니다.",
                icon="⚠️",
            )
        with col_cancel:
            st.markdown(" ")
            if st.button("취소", key="cancel_staged_master", use_container_width=True):
                del st.session_state.staged_master
                st.rerun()

    tab1, tab2, tab_hist = st.tabs(["용어", "패턴", "변경 이력"])

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
            st.caption(f"미리 보기 {len(terms_df)}개 · 저장하면 Team 용어 전체가 바뀝니다.")
        else:
            col_search, col_scope, col_count = st.columns([3, 1, 1])
            with col_search:
                terms_search = st.text_input(
                    "검색",
                    key="terms_search",
                    placeholder="한국어, 영어 또는 메모로 검색",
                    label_visibility="collapsed",
                )
            with col_scope:
                terms_scope_label = st.selectbox(
                    "Scope",
                    ["전체", "Team", "내 항목"],
                    key="terms_scope_filter",
                    label_visibility="collapsed",
                )
            terms_scope = {"전체": "all", "Team": "team", "내 항목": "mine"}[terms_scope_label]

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
                "적용": st.column_config.CheckboxColumn("사용", default=True,
                                                        help="선택을 해제하면 번역에 적용하지 않습니다."),
                "id": None,
                "Scope": st.column_config.SelectboxColumn(
                    "범위",
                    options=["Team", st.session_state.current_user],
                    default=st.session_state.current_user,
                    help="Team: 모든 사용자가 사용 · 본인 이름: 본인만 사용",
                ),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "Product": st.column_config.TextColumn("제품", help="전체 제품에 적용하려면 ALL을 입력하세요."),
                "DNT": st.column_config.CheckboxColumn("번역 안 함", help="원문을 그대로 둡니다."),
                "Case-sensitive": st.column_config.CheckboxColumn("대소문자 구분", help="영문 대소문자를 그대로 유지합니다."),
                "Note": st.column_config.TextColumn("메모"),
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
                st.caption(f"저장하면 Team 용어가 {len(terms_df)}개로 바뀝니다.")
            elif terms_changed:
                st.caption("저장하지 않은 변경 내용이 있습니다.")
            else:
                st.caption("모든 변경 내용이 저장되었습니다.")
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
                        current_user=st.session_state.current_user,
                    )
                    # patterns 시트가 있고 아직 staged면 한 번에 처리
                    if _stg.get("pattern_sheet"):
                        _psname, _psdf = _stg["pattern_sheet"]
                        m = replace_patterns_from_excel(
                            _psdf, source_file=f"{_stg['source']} [{_psname}]",
                            current_user=st.session_state.current_user,
                        )
                        st.toast(f"Master Glossary를 적용했습니다. 용어 {n}개, 패턴 {m}개", icon="✅")
                    else:
                        st.toast(f"Master Glossary를 적용했습니다. 용어 {n}개", icon="✅")
                    del st.session_state.staged_master
                    st.session_state.pop("glossary_table_dirty", None)
                    st.rerun()
                else:
                    counts = save_terms_from_dataframe(
                        edited_terms,
                        view_ids=_terms_view_ids,
                        current_user=st.session_state.current_user,
                    )
                    summary = f"추가 {counts['inserted']}개, 수정 {counts['updated']}개, 삭제 {counts['deleted']}개"
                    if counts.get("denied", 0) > 0:
                        summary += f", 권한이 없어 제외 {counts['denied']}개"
                    st.toast(f"저장했습니다. {summary}", icon="✅")
                    st.session_state.pop("glossary_table_dirty", None)
                    st.rerun()
            except Exception as e:
                st.error(f"저장하지 못했습니다. ({e})")

    # ── 패턴 탭 ──────────────────────────────
    with tab2:
        if _is_staged and _stg.get("pattern_sheet"):
            _psname, _psdf = _stg["pattern_sheet"]
            patterns_df = build_patterns_preview_df(
                _psdf,
                source_file=f"{_stg['source']} [{_psname}]",
            )
            _patterns_view_ids = set()
            st.caption(f"미리 보기 {len(patterns_df)}개 · 저장하면 Team 패턴 전체가 바뀝니다.")
        else:
            col_search, col_scope, col_count = st.columns([3, 1, 1])
            with col_search:
                patterns_search = st.text_input(
                    "검색",
                    key="patterns_search",
                    placeholder="한국어, 영어 또는 메모로 검색",
                    label_visibility="collapsed",
                )
            with col_scope:
                patterns_scope_label = st.selectbox(
                    "Scope",
                    ["전체", "Team", "내 항목"],
                    key="patterns_scope_filter",
                    label_visibility="collapsed",
                )
            patterns_scope = {"전체": "all", "Team": "team", "내 항목": "mine"}[patterns_scope_label]

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
                "적용": st.column_config.CheckboxColumn("사용", default=True,
                                                        help="선택을 해제하면 번역에 적용하지 않습니다."),
                "id": None,
                "Scope": st.column_config.SelectboxColumn(
                    "범위",
                    options=["Team", st.session_state.current_user],
                    default=st.session_state.current_user,
                    help="Team: 모든 사용자가 사용 · 본인 이름: 본인만 사용",
                ),
                "KO": st.column_config.TextColumn("KO"),
                "EN": st.column_config.TextColumn("EN"),
                "Note": st.column_config.TextColumn("메모"),
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
                st.caption(f"저장하면 Team 패턴이 {len(patterns_df)}개로 바뀝니다.")
            elif patterns_changed:
                st.caption("저장하지 않은 변경 내용이 있습니다.")
            else:
                st.caption("모든 변경 내용이 저장되었습니다.")
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
                        _psdf, source_file=f"{_stg['source']} [{_psname}]",
                        current_user=st.session_state.current_user,
                    )
                    if _stg.get("glossary_sheet"):
                        _sname, _sdf = _stg["glossary_sheet"]
                        n = replace_terms_from_excel(
                            _sdf,
                            source_file=f"{_stg['source']} [{_sname}]",
                            product_filter=_stg.get("active_product"),
                            current_user=st.session_state.current_user,
                        )
                        st.toast(f"Master Glossary를 적용했습니다. 용어 {n}개, 패턴 {m}개", icon="✅")
                    else:
                        st.toast(f"Master Glossary를 적용했습니다. 패턴 {m}개", icon="✅")
                    del st.session_state.staged_master
                    st.session_state.pop("glossary_table_dirty", None)
                    st.rerun()
                else:
                    counts = save_patterns_from_dataframe(
                        edited_patterns,
                        view_ids=_patterns_view_ids,
                        current_user=st.session_state.current_user,
                    )
                    summary = f"추가 {counts['inserted']}개, 수정 {counts['updated']}개, 삭제 {counts['deleted']}개"
                    if counts.get("denied", 0) > 0:
                        summary += f", 권한이 없어 제외 {counts['denied']}개"
                    st.toast(f"저장했습니다. {summary}", icon="✅")
                    st.session_state.pop("glossary_table_dirty", None)
                    st.rerun()
            except Exception as e:
                st.error(f"저장하지 못했습니다. ({e})")

    # ── 변경 이력 탭 ─────────────────────────────
    with tab_hist:
        render_glossary_history(blocked=bool(terms_changed or patterns_changed))

    # Glossary 관리 페이지는 워크플로가 아니므로 여기서 페이지 렌더링 끝.
    st.stop()


# ─────────────────────────────────────────────
# Glossary 추출 — 고객사 i18n 카탈로그에서 용어/패턴 뽑기
#
# 제품의 다국어 메시지 파일은 key로 정렬돼 있어서 번역문끼리 짝을 맞추는
# 어려운 단계가 없다. 그래서 후보 추출에는 LLM을 쓰지 않는다. LLM은 표기가
# 갈리는 항목을 '표기 불일치'와 '문맥 분기'로 가르는 데만 쓴다.
# ─────────────────────────────────────────────

if st.session_state.app_mode == "Glossary 추출":
    st.subheader("Glossary 추출")
    st.caption("제품의 다국어 리소스나 기존 매뉴얼에서 용어와 패턴을 찾아 Glossary에 추가합니다.")

    uploaded_catalogs = st.file_uploader(
        "다국어 리소스(JSON) 또는 한국어·영어 Word 매뉴얼",
        type=["json", "docx"],
        accept_multiple_files=True,
        key="catalog_uploader",
        help=(
            "**JSON**: 언어별 파일 또는 두 언어가 함께 든 파일 1개\n\n"
            "**Word(.docx)**: 같은 버전의 한국어판과 영어판 2개"
        ),
    )

    # 번역 대상 문서 — 후보를 이 문서에 나오는 것으로 좁힌다.
    #
    # 카탈로그는 제품 전체의 UI 문자열이라 수만 건인데, 한 문서에 실제로
    # 쓰이는 건 100건 남짓이다. 이걸 안 주면 카탈로그 내부 빈도로 자르게
    # 되고, 그 기준은 번역할 문서와 무관하다 — 실제로 105건을 등재해서
    # 문서에 걸린 게 1건뿐이었다.
    uploaded_target = st.file_uploader(
        "번역할 문서 (선택)",
        type=["docx", "md", "markdown", "mdx"],
        key="catalog_target_uploader",
        help="번역할 한국어 문서를 함께 업로드하면 이 문서에 나오는 용어만 표시합니다.",
    )

    if uploaded_target is not None:
        _tsig = f"{uploaded_target.name}::{uploaded_target.size}"
        if st.session_state.get("catalog_target_sig") != _tsig:
            try:
                _tpath = save_uploaded_file(uploaded_target, UPLOAD_DIR)
                st.session_state.catalog_target_texts = \
                    extract_korean_paragraphs(str(_tpath))
                st.session_state.catalog_target_name = uploaded_target.name
                # 굵게 = 화면 라벨. 카탈로그가 못 덮는 라벨을 찾는 데 쓴다.
                st.session_state.catalog_target_labels = [
                    t for t, _ctx in extract_bold_terms(str(_tpath))
                ]
            except Exception as _e:
                st.warning(f"번역할 문서를 읽지 못했습니다. ({_e})")
                st.session_state.catalog_target_texts = None
                st.session_state.catalog_target_name = None
                st.session_state.catalog_target_labels = None
            st.session_state.catalog_target_sig = _tsig
    elif st.session_state.get("catalog_target_sig") is not None:
        # 파일을 내리면 종전 방식으로 되돌린다
        st.session_state.catalog_target_sig = None
        st.session_state.catalog_target_texts = None
        st.session_state.catalog_target_name = None
        st.session_state.catalog_target_labels = None

    _target_texts = st.session_state.get("catalog_target_texts") or None
    if _target_texts:
        st.caption(
            f"**{st.session_state.get('catalog_target_name')}**에 나오는 용어만 "
            f"표시합니다. (단락 {len(_target_texts):,}개)"
        )

    if uploaded_catalogs:
        _sig = "|".join(f"{f.name}:{f.size}" for f in uploaded_catalogs)
        if st.session_state.get("catalog_sig") != _sig:
            # 큰 파일을 rerun마다 다시 파싱하지 않도록 시그니처로 캐시
            try:
                _docx = [f for f in uploaded_catalogs
                         if f.name.lower().endswith(".docx")]
                _json = [f for f in uploaded_catalogs
                         if f.name.lower().endswith(".json")]
                _align = None
                if _docx and _json:
                    raise ValueError(
                        "JSON과 Word 파일은 함께 업로드할 수 없습니다. "
                        "한 가지 형식만 업로드하세요."
                    )
                if _docx:
                    if len(_docx) != 2:
                        raise ValueError(
                            f"같은 버전의 한국어판과 영어판, 2개 파일을 업로드하세요. "
                            f"(현재 {len(_docx)}개)"
                        )
                    _a, _b = _docx
                    _pf_a, _pf_b, _align = catalog.parse_docx_pair(
                        _a.name, _a.getvalue(), _b.name, _b.getvalue()
                    )
                    _parsed = [_pf_a, _pf_b]
                else:
                    _parsed = [catalog.parse_json(f.name, json.loads(f.getvalue()))
                               for f in _json]
                _pick = catalog.pick_languages(_parsed)
                st.session_state.catalog_align = _align
                st.session_state.catalog_parsed = _parsed
                st.session_state.catalog_pick = _pick
                st.session_state.catalog_pick_labels = (
                    _pick.ko_label, _pick.en_label, _pick.ratios
                )
                st.session_state.catalog_sig = _sig
                st.session_state.pop("catalog_result", None)
                st.session_state.pop("catalog_result_key", None)
                st.session_state.pop("catalog_resolved", None)
                st.session_state.pop("catalog_error", None)
                st.session_state.pop("catalog_last_save", None)
            except Exception as e:
                st.session_state.catalog_error = str(e)
                st.session_state.catalog_sig = _sig
                st.session_state.pop("catalog_result", None)
                st.session_state.pop("catalog_align", None)
                st.session_state.pop("catalog_pick", None)

    def _relabel_sections(result, align, pick):
        """
        docx 경로의 문맥(key)를 사람이 읽는 절 제목으로 바꾼다.

        JSON 카탈로그의 key는 'settings.push.enable'처럼 첫 마디가 곧 의미
        있는 네임스페이스지만, 매뉴얼에는 그런 게 없어 's38' 같은 내부
        번호가 그대로 노출된다. 절 제목으로 바꿔야 '어느 절에서 나온
        후보인가'라는 원래 의도가 살아난다. 이 값은 등재 시 Note로도
        저장되므로 나중에 출처를 되짚을 때도 쓰인다.
        """
        if align is None or not align.section_titles:
            return
        _a_is_ko = pick.ko_label == align.name_a
        _titles = {sid: (ta if _a_is_ko else tb)
                   for sid, (ta, tb) in align.section_titles.items()}

        def _fix(v):
            parts = [_titles.get(t, t) for t in str(v or "").split() if t]
            return " · ".join(dict.fromkeys(parts))   # 순서 유지 중복 제거

        for _df in (result.labels, result.terms, result.patterns):
            if _df is not None and not _df.empty and "문맥(key)" in _df.columns:
                _df["문맥(key)"] = _df["문맥(key)"].map(_fix)

    # 상한이 바뀌면 파싱은 그대로 두고 집계만 다시 한다 (0.5초).
    # '전체 보기'는 상한을 사실상 없앤다 — head()/슬라이스에 큰 수를 주면 된다.
    # 등재 대상 제품·공개 범위를 결과 표보다 **먼저** 읽는다 —
    # "이미 등재됨" 판정이 이 제품 기준으로 이뤄져야 하기 때문.
    # 기본값은 Localize에서 고른 제품 (매번 다시 고르지 않게).
    # 등재 대상 제품·공개 범위.
    #
    # **위젯 키가 아니라 일반 키에 담는다.** Streamlit은 어떤 실행에서 위젯이
    # 렌더되지 않으면 그 위젯 키를 session_state에서 지운다. 검수 화면은
    # 등재 대상 카드를 통째로 대체하므로, 그동안 제품 선택이 사라지고 등재
    # 직후에는 기본값(ALL)으로 돌아가 있었다 — "새로 추가한 제품이 Localize에
    # 안 보인다"의 정체가 이것이다.
    #
    # 일반 키는 렌더 여부와 무관하게 남는다. 제품 추가 버튼도 여기에 바로
    # 쓰면 되므로 "다음 실행에 반영" 같은 우회가 필요 없다.
    if "catalog_product_value" not in st.session_state:
        _sp = st.session_state.get("selected_product")
        st.session_state.catalog_product_value = _sp if _sp in products else "ALL"
    st.session_state.setdefault("catalog_scope_value", "Team")
    # 제품을 추가한 직후 입력칸을 비운다. 텍스트 입력은 위젯 키라
    # 위젯이 만들어진 뒤에는 대입할 수 없으므로, 버튼은 깃발만 올리고
    # 위젯보다 앞인 여기서 지운다.
    if st.session_state.pop("catalog_clear_new_product", False):
        st.session_state.catalog_new_product = ""

    _extract_product = st.session_state.catalog_product_value
    if _extract_product not in (["ALL"] + products):
        _extract_product = "ALL"          # 제품이 사라진 경우 방어
        st.session_state.catalog_product_value = "ALL"
    _extract_scope = st.session_state.catalog_scope_value

    _NO_LIMIT = 10 ** 9
    _show_all = st.session_state.get("catalog_show_all", False)
    _t_raw = st.session_state.get("catalog_term_limit", catalog.DEFAULT_TERM_LIMIT)
    _p_raw = st.session_state.get("catalog_pattern_limit", catalog.DEFAULT_PATTERN_LIMIT)
    _t_limit = _NO_LIMIT if _show_all else _t_raw
    _p_limit = _NO_LIMIT if _show_all else _p_raw
    # 제품이 바뀌면 "이미 등재됨" 판정이 달라지므로 다시 분석해야 한다.
    _key = (st.session_state.get("catalog_sig"), _t_limit, _p_limit,
            _extract_product, st.session_state.get("catalog_target_sig"))
    if st.session_state.get("catalog_pick") is not None and \
            st.session_state.get("catalog_result_key") != _key:
        _res = catalog.analyze(
            st.session_state.catalog_pick,
            # 대조는 **등재할 제품** 기준이어야 한다. 예전에는 제품 인자 없이
            # 전 제품을 가져와서, 다른 제품에 같은 KO가 있으면 "이미 등재"로
            # 잘못 뜨고 정작 같은 제품 안의 중복은 놓쳤다.
            catalog.existing_terms_index(
                load_terms(product=_extract_product,
                           current_user=st.session_state.current_user)
            ),
            term_limit=_t_limit,
            pattern_limit=_p_limit,
            existing_patterns=catalog.existing_terms_index(
                load_patterns(current_user=st.session_state.current_user)
            ),
            target_texts=_target_texts,
        )
        _relabel_sections(_res, st.session_state.get("catalog_align"),
                          st.session_state.catalog_pick)
        # '기존대조'(신규/동일/충돌(기존))를 한 글자 아이콘으로. 표에서 폭을
        # 거의 쓰지 않으면서 "이미 등재됐는가"를 즉시 알 수 있게 한다.
        _STATE_ICON = {"신규": "🆕", "동일": "✅", "충돌(기존)": "⚠️"}
        # 행이 없어도 컬럼은 만들어 둔다 — 표가 참조하는 컬럼이
        # 상황에 따라 있다 없다 하면 렌더가 깨진다.
        for _df_ in (_res.terms, _res.patterns):
            if _df_ is not None and "기존대조" in _df_.columns:
                _df_["상태"] = _df_["기존대조"].map(lambda v: _STATE_ICON.get(v, ""))
        st.session_state.catalog_result = _res
        st.session_state.catalog_result_key = _key

    # 분석 결과는 session_state에 남겨둔다 — 다른 메뉴에 다녀와도 유지된다.
    if st.session_state.get("catalog_error"):
        st.error(st.session_state.catalog_error)
    elif st.session_state.get("catalog_result") is None:
        st.markdown("##### 지원하는 파일")
        _t_json, _t_docx = st.tabs(["JSON 리소스", "Word 매뉴얼"])
        with _t_json:
            st.markdown(
                "```json\n"
                '[{"key": "a.b", "en": "Save"}]              // 배열 + 언어 필드\n'
                '[{"key": "a.b", "en": "Save", "ko": "저장"}] // 한 파일에 양쪽 언어\n'
                '{"a.b": "저장"}                              // 평탄 객체\n'
                '{"a": {"b": "저장"}}                         // 중첩 객체\n'
                "```"
            )
            st.caption(
                "필드 이름과 언어는 자동으로 인식합니다."
            )
        with _t_docx:
            st.markdown(
                "**같은 버전의 한국어판과 영어판 .docx 파일 2개**를 업로드하세요. "
                "두 문서의 제목 순서를 기준으로 내용을 맞춘 뒤, 굵게 표시된 UI 라벨과 "
                "제목, 표에서 용어를 찾습니다."
            )
            st.markdown(
                "- 단락 단위로 맞추므로 문장 수가 달라도 됩니다.\n"
                "- 표지, 법적 고지, 연락처는 제외합니다.\n"
                "- 두 문서의 구성이 다른 부분은 제외하고 목록으로 보여 줍니다."
            )
            st.info(
                "제목에는 Word의 **제목 1~4** 스타일이 적용되어 있어야 합니다.",
                icon="ℹ️",
            )
    else:
        # 결과 블록 — 들여쓰기 유지를 위한 컨테이너
        with st.container():
            _result = st.session_state.catalog_result
            _ko_label, _en_label, _ratios = st.session_state.catalog_pick_labels
            _stats = _result.stats
            _conf = catalog.conflict_rows(_result.labels)

            st.caption(
                " · ".join(pf.describe() for pf in st.session_state.catalog_parsed)
                + f"  ·  KO `{_ko_label}` / EN `{_en_label}`"
            )

            # Word 매뉴얼은 '얼마나 잘 맞췄는가'를 먼저 보여준다. 정렬이 무너진
            # 채로 나온 후보를 그대로 등재하면 글로서리가 오염되기 때문이다.
            # 방금 등재한 결과를 결과 화면 맨 위에 알린다. 등재 후 표에서
            # 그 항목이 ✅로 바뀌어 남아 있으므로 "된 건지 안 된 건지"를
            # 다시 확인하러 갈 필요가 없다.
            # pop 하면 안 된다. 이 배너 안에 버튼이 있는데, 그리면서 상태를
            # 지우면 클릭이 처리되는 다음 실행에는 버튼이 존재하지 않아
            # 클릭이 통째로 무시된다. 새 등재나 새 파일 업로드 때 덮어쓰거나
            # 지운다.
            _last = st.session_state.get("catalog_last_save")
            if _last:
                st.success(
                    f"{_last['msg']}를 추가했습니다. "
                    f"(제품: {_last['product']}, 범위: {_last['scope']})"
                    + (f" · 제외 {_last['skipped']}개" if _last.get("skipped") else "")
                    + (f" · 이미 있는 항목 {_last['skipped_db']}개"
                       if _last.get("skipped_db") else ""),
                    icon="✅",
                )
                # 등재하자마자 바로 번역하러 갈 수 있게. 방금 등재한 제품을
                # 들고 넘어가므로 Localize에서 다시 고르지 않아도 된다.
                if st.button(
                    f"{_last['product']} 문서 Localize",
                    type="primary", use_container_width=True,
                    key="catalog_go_localize_banner",
                ):
                    _try_navigate({
                        "app_mode": "Localize", "step": 1,
                        "selected_product": _last["product"],
                    })

            _align = st.session_state.get("catalog_align")
            _align_on = _align is not None
            if _align is not None:
                with st.container(border=True):
                    if _align.ok:
                        st.markdown("##### ✅ 두 매뉴얼의 내용을 맞췄습니다")
                    else:
                        st.markdown("##### ⚠️ 두 매뉴얼의 구성이 많이 다릅니다")
                        st.warning(
                            "버전이 다를 수 있습니다. 후보를 하나씩 확인한 뒤 추가하세요.",
                            icon="⚠️",
                        )
                    _m1, _m2, _m3 = st.columns(3)
                    _m1.metric(
                        "일치하는 제목",
                        f"{_align.heading_matched}/{_align.heading_total}",
                        f"{_align.heading_rate * 100:.0f}%",
                        delta_color="normal" if _align.heading_rate >= 0.8 else "inverse",
                        help="두 문서에서 같은 순서로 대응되는 제목 수"
                    )
                    _m2.metric(
                        "일치하는 구간",
                        f"{_align.sections_matched}/{_align.sections}",
                        f"{_align.section_rate * 100:.0f}%",
                        delta_color="normal" if _align.section_rate >= 0.7 else "inverse",
                        help="제목 아래 단락 수까지 같은 구간 수"
                    )
                    _m3.metric(
                        "UI 라벨",
                        f"{_align.n_label:,}",
                        help=f"굵게 표시된 라벨 {_align.n_label}쌍 "
                             f"(확실 {_align.n_bold_direct}, 추정 {_align.n_bold_cooc}). "
                             f"제목과 본문을 포함하면 "
                             f"{_align.n_heading + _align.n_block + _align.n_label:,}쌍"
                    )
                    if _align.front_dropped:
                        st.caption(
                            f"표지와 법적 고지 등 {_align.front_dropped}개 블록은 제외했습니다."
                        )
                    if _align.mismatched:
                        with st.expander(
                            f"구성이 다른 구간 {len(_align.mismatched)}개"
                        ):
                            st.dataframe(
                                pd.DataFrame(
                                    _align.mismatched,
                                    columns=["한국어 제목", "한국어 단락 수",
                                             "영어 제목", "영어 단락 수"],
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption(
                                "이 구간은 후보에서 제외했습니다. 단락 수가 조금 다르면 "
                                "한쪽 문서에 내용이 빠졌을 수 있습니다."
                            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("찾은 쌍", f"{_stats['공통키']:,}")
            c2.metric("용어", f"{_stats['용어후보']:,}",
                      help=(f"전체 후보 {_stats['용어풀']:,}개"
                            if _show_all else
                            f"전체 {_stats['용어풀']:,}개 중 자주 나오는 순으로 "
                            f"{_stats['용어후보']:,}개"))
            c3.metric("패턴", f"{_stats['패턴후보']:,}",
                      help=f"문장 {_stats['패턴풀']:,}개를 문형별로 묶은 결과")
            c4.metric("표기 충돌", f"{len(_conf):,}", help="같은 한국어에 영어 표기가 여러 개인 항목")
            if _stats.get("문서적중"):
                st.caption(
                    f"리소스의 라벨 {_stats['라벨 고유KO']:,}개 중 "
                    f"{_stats['문서적중']:,}개가 번역할 문서에 나옵니다."
                )

            # ── 입력 자료 진단 ────────────────────────────────────────
            # 후보 목록만 보여주면 "이 데이터가 쓸 만한가"를 알 수 없다.
            # 무엇이 문제고 무엇을 하면 되는지를 글로 알려준다.
            _findings = catalog.diagnose(
                _result,
                st.session_state.get("catalog_target_name"),
                st.session_state.get("catalog_target_labels"),
            )
            if _findings:
                _dsum = catalog.diagnosis_summary(_result, _findings)
                _n_act = sum(1 for f in _findings
                             if f["심각도"] == "조치 필요")
                # 등재가 이 화면의 본 목적이므로 진단은 항상 접어 둔다.
                # 접힌 한 줄에도 등급과 조치 건수는 보이게 한다.
                with st.expander(
                    f"데이터 품질 진단 · {_dsum['등급']}"
                    + (f" · 조치 필요 {_n_act}개" if _n_act else ""),
                    expanded=False,
                ):
                    _mcols = st.columns(len(_dsum["지표"]))
                    for _mc, (_mk, _mv) in zip(_mcols, _dsum["지표"]):
                        _mc.metric(_mk, _mv)
                    st.markdown(_dsum["총평"])
                    st.markdown("---")
                    for _fi, _f in enumerate(_findings):
                        if _fi:
                            st.markdown("")
                        st.markdown(f"**{_f['제목']}**")
                        st.caption(_f["심각도"])
                        _body = _f["현황"]
                        if _f.get("영향"):
                            _body += " " + _f["영향"]
                        st.markdown(_body)
                        st.markdown(f"**권장 조치:** {_f['권고']}")
                        _tbl = _f.get("표")
                        if _tbl is not None and len(_tbl):
                            st.dataframe(_tbl.head(8), use_container_width=True,
                                         hide_index=True)
                            if len(_tbl) > 8:
                                st.caption(
                                    f"외 {len(_tbl) - 8}개. 전체 목록은 진단 보고서에 있습니다."
                                )
                    st.markdown("---")
                    st.download_button(
                        "진단 보고서 다운로드",
                        data=catalog.diagnosis_markdown(
                            _findings,
                            st.session_state.get("catalog_target_name"),
                            _dsum,
                        ).encode("utf-8"),
                        file_name="용어DB_진단결과.md",
                        mime="text/markdown",
                        use_container_width=True,
                        help="Markdown 형식입니다. Word에 붙여 넣어 사용할 수 있습니다.",
                    )

            with st.expander(
                "표시 개수" + (" · 전체 표시 중" if _show_all else "")
            ):
                st.checkbox(
                    "모든 후보 표시",
                    key="catalog_show_all",
                    help="후보가 많으면 검색으로 범위를 좁혀 추가하세요.",
                )
                a1, a2 = st.columns(2)
                a1.number_input(
                    "용어", min_value=10, max_value=1000, step=10,
                    key="catalog_term_limit", value=_t_raw, disabled=_show_all,
                    help="자주 나오는 순으로 표시할 개수"
                )
                a2.number_input(
                    "패턴", min_value=5, max_value=200, step=5,
                    key="catalog_pattern_limit", value=_p_raw, disabled=_show_all,
                    help="비슷한 문장을 묶어 자주 나오는 순으로 표시할 개수"
                )
                if _show_all:
                    st.caption(
                        f"용어 {_stats['용어후보']:,}개와 패턴 {_stats['패턴후보']:,}개를 "
                        "모두 표시합니다. 빈도가 낮은 항목에는 용어가 아닌 것도 있습니다."
                    )

            # 제품·공개 범위 선택은 등재 버튼 바로 위로 옮겼다. 무엇이 어디로
            # 가는지 모른 채 버튼을 누르는 일을 막기 위함.
            _source_name = " + ".join(pf.name for pf in st.session_state.catalog_parsed)

            # 편집기에 한 번에 올릴 최대 행 수. 수천 행을 체크박스로 뿌리면
            # 브라우저가 버티지 못하므로, 검색으로 좁힌 결과에만 편집기를 준다.
            _EDIT_CAP = 500

            def _register(chosen, kind):
                """선택한 행을 글로서리/패턴에 삽입."""
                # 'Personal'을 그대로 넘기면 owner가 문자열 'Personal'이 되어
                # 누구의 '내 항목'에도 잡히지 않고 번역에도 쓰이지 않는다.
                _scope = (st.session_state.current_user
                          if _extract_scope == "Personal" else "Team")
                if kind == "pattern":
                    rows = pd.DataFrame({
                        "id": None, "Scope": _scope,
                        "KO": chosen["KO"].values, "EN": chosen["EN"].values,
                        "Note": chosen.get("문맥(key)", ""),
                        "Status": "approved", "File": _source_name,
                    })
                    # view_ids=set() — 삽입만 한다. 기존 행 삭제 경로를 차단.
                    return save_patterns_from_dataframe(
                        rows, view_ids=set(),
                        current_user=st.session_state.current_user,
                        source="Glossary 추출",
                    )
                rows = pd.DataFrame({
                    "id": None, "Scope": _scope,
                    "KO": chosen["KO"].values, "EN": chosen["EN"].values,
                    "Product": _extract_product,
                    "DNT": chosen.get("DNT", False), "Case-sensitive": False,
                    "Note": chosen.get("문맥(key)", ""),
                    "Status": "approved", "File": _source_name,
                })
                return save_terms_from_dataframe(
                    rows, view_ids=set(),
                    current_user=st.session_state.current_user,
                    source="Glossary 추출",
                )

            _KIND_KO = {"term": "용어", "pattern": "패턴"}

            def _start_review(selections):
                """
                등재 직전 검수 — 고른 항목을 LLM에 보내고 검토 단계로 넘어간다.

                용어와 패턴은 판단 기준이 달라 LLM 호출은 따로 하지만(kind별
                프롬프트), 화면은 하나로 합친다. 두 번 고르고 두 번 확인하는
                수고를 없애기 위함.
                """
                from translator_engine import make_client  # 이 화면에서만 필요

                client = make_client(OPENAI_API_KEY)
                state = {}
                with st.spinner("AI가 검토하는 중..."):
                    for kind, sel_df in selections:
                        if sel_df is None or sel_df.empty:
                            continue
                        try:
                            _sugg = catalog.review_entries(
                                client, list(zip(sel_df["KO"], sel_df["EN"])),
                                kind=kind,
                                contexts=(list(sel_df["문서 용례"])
                                          if "문서 용례" in sel_df.columns
                                          else None),
                            )
                        except Exception as e:
                            st.error(f"검토하지 못했습니다. ({e})")
                            return
                        # 기존 글로서리와 표기가 다른 항목도 검수에서 함께 고르게
                        # 한다. "1건이 다릅니다"라고만 알리면 무엇이 어떻게
                        # 다른지 알 수 없다.
                        _merged = {int(i): v for i, v in _sugg.items()}
                        if "기존대조" in sel_df.columns:
                            for _i, (_, _r) in enumerate(sel_df.iterrows()):
                                if _r.get("기존대조") != "충돌(기존)":
                                    continue
                                _prev = str(_r.get("기존 EN") or "").split(" / ")[0].strip()
                                if not _prev or _prev == _r["EN"]:
                                    continue
                                _merged[_i] = {
                                    "suggest": _prev,
                                    "reason": f"Glossary에는 `{_prev}`(으)로 등록되어 있습니다.",
                                    "labels": ["Glossary 표기", "리소스 표기"],
                                }
                        # 카탈로그 안에서 표기가 갈리는 항목은 반드시 사람이
                        # 고르게 한다. 제안이 없으면 검수 카드 없이 그대로
                        # 등재되어, 최빈 표기가 조용히 확정돼 버린다.
                        if "후보수" in sel_df.columns:
                            for _i, (_, _r) in enumerate(sel_df.iterrows()):
                                try:
                                    _n = int(_r.get("후보수") or 1)
                                except (TypeError, ValueError):
                                    _n = 1
                                if _n <= 1:
                                    continue
                                _cur = dict(_merged.get(_i) or {})
                                _cur.setdefault("suggest", _r["EN"])
                                _cur["reason"] = (
                                    f"리소스에 표기가 {_n}가지 있습니다: "
                                    f"{_r.get('EN 후보', '')}"
                                )
                                _merged[_i] = _cur
                        state[kind] = {
                            "rows": sel_df.to_dict("records"), "sugg": _merged,
                        }
                if not state:
                    return
                st.session_state["catalog_review"] = state
                st.rerun()

            _CUSTOM = "직접 입력"
            _SKIP = "추가 안 함"

            def _review_one(kind, i, r, s):
                """
                제안이 있는 항목 하나 — 제안 / 원본 / 직접 입력 / 등재 안 함.

                제외를 고르면 None을 돌려준다. 검수하다 보면 "이건 아예 용어가
                아니네" 싶은 것이 나오는데, 그때 표로 되돌아가 체크를 풀고 다시
                검수를 돌리는 건 낭비다.
                """
                with st.container(border=True):
                    st.markdown(f"**{r['KO']}**")
                    # 이 말이 번역할 문서에서 실제로 어떻게 쓰이는지 먼저.
                    # '기록'이 이력(History)인지 녹화(Record)인지는 카탈로그만
                    # 봐서는 알 수 없고 — 실제로 History로 잘못 등재돼 세 곳이
                    # 틀렸다 — 이 한 줄이면 바로 갈린다.
                    if r.get("문서 용례"):
                        st.caption(f"문서 예문: {r['문서 용례']}")
                    _opts = [s["suggest"], r["EN"]]
                    _caps = list(s.get("labels", ["AI 제안", "원래 표기"]))[:2]
                    if _opts[0] == _opts[1]:          # 같으면 라디오가 깨진다
                        _opts, _caps = _opts[:1], _caps[:1]
                    # 카탈로그에 표기가 여럿이면 나머지도 선택지로 올린다.
                    for _v in str(r.get("EN 후보") or "").split(" / "):
                        _v = _v.strip()
                        if _v and _v not in _opts:
                            _opts.append(_v)
                            _caps.append("리소스 표기")
                    pick = st.radio(
                        r["KO"],
                        options=_opts + [_CUSTOM, _SKIP],
                        captions=_caps + ["", ""],
                        index=0, horizontal=len(_opts) <= 2,
                        label_visibility="collapsed",
                        key=f"catalog_rev_{kind}_{i}",
                    )
                    if pick == _SKIP:
                        st.caption("이 항목은 추가하지 않습니다.")
                        return None
                    if pick == _CUSTOM:
                        # 원본을 채워두고 거기서 고치게 한다 — 빈 칸에서
                        # 다시 타이핑하는 것보다 훨씬 빠르다.
                        typed = st.text_input(
                            "직접 입력", value=r["EN"],
                            key=f"catalog_rev_txt_{kind}_{i}",
                            label_visibility="collapsed",
                            placeholder="영어 표기 입력",
                        )
                        pick = typed.strip() or r["EN"]   # 비우면 원본 유지
                    if s.get("reason"):
                        st.caption(s["reason"])
                return {**r, "EN": pick}

            def _render_review():
                """용어·패턴 제안을 한 화면에서 승인/반려/수정하고 확정 등재."""
                rv = st.session_state["catalog_review"]
                _total = sum(len(v["rows"]) for v in rv.values())
                _all_sugg = {k: v["sugg"] for k, v in rv.items()}
                _n_sugg = sum(len(v) for v in _all_sugg.values())

                st.markdown(f"##### 추가 전 검토 · {_total}개")
                if _n_sugg:
                    _n_prior = sum(1 for sg in _all_sugg.values()
                                   for v in sg.values() if v.get("labels"))
                    _bits = []
                    if _n_sugg - _n_prior:
                        _bits.append(f"수정 제안 {_n_sugg - _n_prior}개")
                    if _n_prior:
                        _bits.append(f"Glossary와 표기가 다름 {_n_prior}개")
                    st.caption(
                        " · ".join(_bits)
                        + f" · 나머지 {_total - _n_sugg}개는 그대로 추가됩니다."
                    )
                else:
                    st.caption("수정할 항목이 없습니다.")

                finals, changed, skipped = {}, 0, 0
                for kind in ("term", "pattern"):
                    blk = rv.get(kind)
                    if not blk:
                        continue
                    rows, sugg = blk["rows"], blk["sugg"]
                    if len(rv) > 1:
                        st.markdown(
                            f"**{_KIND_KO[kind]} {len(rows)}개**"
                            + (f" · 제안 {len(sugg)}개" if sugg else "")
                        )
                    out = []
                    for i, r in enumerate(rows):
                        s = sugg.get(i)
                        if not s:
                            out.append(r)
                            continue
                        picked = _review_one(kind, i, r, s)
                        if picked is None:      # 등재 안 함
                            skipped += 1
                            continue
                        changed += int(picked["EN"] != r["EN"])
                        out.append(picked)
                    finals[kind] = out

                b1, b2 = st.columns([1, 2])
                if b1.button("취소", key="catalog_rev_cancel",
                             use_container_width=True):
                    st.session_state.pop("catalog_review", None)
                    st.rerun()
                _final_n = sum(len(v) for v in finals.values())
                if skipped:
                    st.caption(f"{skipped}개는 추가하지 않습니다.")
                if b2.button(
                    f"{_final_n}개 추가" + (f" (수정 {changed}개)" if changed else ""),
                    type="primary", key="catalog_rev_ok",
                    use_container_width=True,
                    disabled=_final_n == 0,
                ):
                    try:
                        _msg, _skipped_db = [], 0
                        for kind, out in finals.items():
                            if not out:
                                continue
                            counts = _register(pd.DataFrame(out), kind)
                            _msg.append(f"{_KIND_KO[kind]} {counts['inserted']}개")
                            _skipped_db += counts.get("skipped", 0)
                        st.session_state.pop("catalog_review", None)
                        st.session_state["catalog_last_save"] = {
                            "msg": " · ".join(_msg),
                            "product": _extract_product,
                            "scope": "Team" if _extract_scope == "Team" else "개인",
                            "skipped": skipped,
                            # 저장 계층이 이미 있는 항목을 걸러낸 수.
                            # 화면 판정과 별개로 최종 방어선이 동작한 결과다.
                            "skipped_db": _skipped_db,
                        }
                        # 다시 분석해 방금 등재한 항목이 ✅로 바뀌게 한다.
                        st.session_state.pop("catalog_result_key", None)
                        # 이 실행은 이미 검수 화면을 그린 뒤다. 다시 돌려야
                        # 결과 배너와 갱신된 표(✅ 표시)가 보인다.
                        st.rerun()
                    except Exception as e:
                        st.error(f"추가하지 못했습니다. ({e})")

            def _render_table(df, cols, key, kind, hidden=()):
                """
                후보 표를 그리고 **체크된 행을 돌려준다**.

                등재 버튼은 여기 두지 않는다. 용어와 패턴을 한 번에 검수·등재
                하려면 두 탭의 선택을 모아야 하는데, 탭 안에 버튼이 있으면
                각자 등재하는 흐름이 되기 때문.
                """
                if df.empty:
                    st.caption("후보가 없습니다.")
                    return df.head(0)

                f1, f2, f3 = st.columns([1.2, 3, 1.6])
                with f1:
                    select_all = st.checkbox(
                        "전체 선택", value=False, key=f"catalog_all_{key}",
                        help="표에 표시된 항목을 모두 선택합니다.",
                    )
                with f2:
                    q = st.text_input(
                        "검색", key=f"catalog_q_{key}", placeholder="한국어 또는 영어로 검색",
                        label_visibility="collapsed",
                    )
                with f3:
                    # 기본값을 False로 둔다. True면 등재 직후 그 항목이 표에서
                    # **사라져** 성공했는지 알 수 없다. 보이되 상태로 구분한다.
                    skip_dup = st.checkbox(
                        "추가된 항목 숨기기", value=False, key=f"catalog_skip_{key}",
                        help="Glossary에 같은 표기로 이미 있는 항목(✅)을 숨깁니다.",
                    )

                view = df
                if q.strip():
                    view = df[
                        df["KO"].str.contains(q, case=False, na=False, regex=False)
                        | df["EN"].str.contains(q, case=False, na=False, regex=False)
                    ]
                if skip_dup and "기존대조" in view.columns:
                    view = view[view["기존대조"] != "동일"]

                capped = view.head(_EDIT_CAP)
                st.caption(
                    f"{len(capped):,} / {len(view):,}개 표시"
                    + (" · 검색으로 범위를 좁혀 나누어 추가하세요." if len(view) > _EDIT_CAP else "")
                )

                # 숨김 컬럼도 실어 보낸다 — 화면에는 안 나오지만 등재 직전
                # 충돌 안내와 검수 단계가 이 값을 읽는다.
                _keep = [c for c in list(cols) + list(hidden) if c in capped.columns]
                edit_df = capped[_keep].copy()
                edit_df.insert(0, "적용", select_all)

                col_cfg = {c: st.column_config.TextColumn(c, disabled=True) for c in cols}
                col_cfg["적용"] = st.column_config.CheckboxColumn("", width="small")
                if "문맥(key)" in cols:
                    # 매뉴얼은 key가 없으니 절 제목을, 카탈로그는 네임스페이스를 보여준다
                    col_cfg["문맥(key)"] = st.column_config.TextColumn(
                        "위치" if _align_on else "키", disabled=True,
                        help="이 후보가 나온 문서의 절"
                             if _align_on else "리소스 키의 첫 부분",
                    )
                if "문형 빈도" in cols:
                    col_cfg["문형 빈도"] = st.column_config.NumberColumn(
                        "문형", disabled=True, width="small"
                    )
                if "DNT" in cols:
                    col_cfg["DNT"] = st.column_config.CheckboxColumn("번역 안 함", disabled=True)
                if "상태" in cols:
                    col_cfg["상태"] = st.column_config.TextColumn(
                        "상태", disabled=True, width="small",
                        help="🆕 새 항목 · ✅ Glossary에 있음 · ⚠️ Glossary와 표기가 다름",
                    )
                for _num in ("빈도", "후보수"):
                    if _num in cols:
                        col_cfg[_num] = st.column_config.NumberColumn(
                            _num, disabled=True, width="small"
                        )

                edited = st.data_editor(
                    edit_df, use_container_width=True, hide_index=True, height=380,
                    column_config=col_cfg, num_rows="fixed",
                    column_order=["적용"] + [c for c in cols if c in _keep],
                    key=f"catalog_edit_{key}_{int(select_all)}_{q.strip()}",
                )
                # 선택이 없을 때의 안내는 탭 아래 한 곳에서만 한다 —
                # 탭마다 같은 문구를 반복하면 하단 안내와 겹친다.
                return edited[edited["적용"] == True]  # noqa: E712

            if st.session_state.get("catalog_review") is not None:
                _render_review()
            else:
                # 탭 순서는 [Glossary 관리]와 맞춘다 — 용어, 패턴.
                _tab_terms, _tab_patterns = st.tabs(
                    [f"용어 {_stats['용어후보']:,}", f"패턴 {_stats['패턴후보']:,}"]
                )
                with _tab_terms:
                    # '기존대조'는 표에서 뺐다. 보이는 행은 거의 전부 '신규'라
                    # 정보량이 없기 때문 — '동일'은 위 '등록된 항목 숨기기'가
                    # 걸러내고, '충돌(기존)'은 아래 경고에서 따로 다룬다.
                    # 번역 대상 문서를 준 경우엔 카탈로그 내부 빈도(빈도)가
                    # 아니라 그 문서에서 몇 번 쓰이는지(문서빈도)가 판단 근거다.
                    _tcols = (
                        ["상태", "문서빈도", "KO", "EN", "문서 용례",
                         "문맥(key)", "DNT"]
                        if "문서빈도" in _result.terms.columns
                        else ["상태", "빈도", "KO", "EN", "문맥(key)", "DNT"]
                    )
                    _sel_terms = _render_table(
                        _result.terms, _tcols,
                        "terms", "term", hidden=("기존대조", "기존 EN"),
                    )
                with _tab_patterns:
                    _pcols = ["상태", "문형 빈도", "KO", "EN", "문맥(key)"]
                    if ("문서빈도" in _result.patterns.columns
                            and _result.patterns["문서빈도"].max() > 0):
                        _pcols.insert(1, "문서빈도")
                    _sel_patterns = _render_table(
                        _result.patterns, _pcols,
                        "patterns", "pattern", hidden=("기존대조", "기존 EN"),
                    )

                # ── 두 탭의 선택을 합쳐 한 번에 검수·등재 ──────────────
                _n_t, _n_p = len(_sel_terms), len(_sel_patterns)

                # ── 등재 요약 ────────────────────────────────────────
                # 제품·공개 범위 선택과 "무엇이 몇 건 어디로 가는가"를 버튼
                # 바로 위에 모은다. 예전에는 이 선택지가 표 위쪽에 떨어져
                # 있어서, 무엇이 어디로 가는지 모른 채 버튼을 눌렀다.
                with st.container(border=True):
                    st.markdown("##### 추가할 위치")
                    _dc1, _dc2 = st.columns(2)
                    # key= 를 쓰지 않는다 — 위에서 설명한 이유로 위젯 키는
                    # 화면이 바뀌면 사라진다. index로 현재 값을 보여주고,
                    # 고른 값은 일반 키에 되돌려 담는다.
                    with _dc1:
                        _p_opts = ["ALL"] + products
                        _extract_product = st.selectbox(
                            "제품", options=_p_opts,
                            index=_p_opts.index(_extract_product),
                            help="선택한 제품의 Glossary에 추가합니다. 중복도 이 제품 기준으로 확인합니다.",
                        )
                        st.session_state.catalog_product_value = _extract_product
                    with _dc2:
                        _s_opts = ["Team", "Personal"]
                        _extract_scope = st.selectbox(
                            "범위", options=_s_opts,
                            index=_s_opts.index(_extract_scope),
                            format_func=lambda v: ("Team (모든 사용자)" if v == "Team"
                                                   else "개인 (본인만)"),
                        )
                        st.session_state.catalog_scope_value = _extract_scope

                    # 목록에 없는 제품이면 여기서 바로 만든다. 설정 파일을
                    # 열러 가지 않아도 되게.
                    with st.expander("새 제품 추가"):
                        _np1, _np2 = st.columns([3, 1])
                        _new_name = _np1.text_input(
                            "새 제품 이름", key="catalog_new_product",
                            placeholder="예: Fireside", label_visibility="collapsed",
                        )
                        if _np2.button("추가", use_container_width=True,
                                       key="catalog_add_product"):
                            _ok, _m = add_product(_new_name)
                            if _ok:
                                # 일반 키라 위젯 생성 이후에도 자유롭게 쓴다.
                                st.session_state.catalog_product_value = \
                                    _new_name.strip()
                                st.session_state["catalog_clear_new_product"] = True
                                st.toast(_m, icon="✅")
                                st.rerun()
                            else:
                                st.warning(_m, icon="⚠️")
                        st.caption(
                            "추가한 제품은 Localize 제품 목록에도 표시됩니다."
                        )

                    if _n_t + _n_p:
                        # 고른 것 중 이미 등재된 건 건너뛴다 — 묻지 않는다.
                        _same = _dupe = 0
                        for _sdf in (_sel_terms, _sel_patterns):
                            if "기존대조" in _sdf.columns:
                                _same += int((_sdf["기존대조"] == "동일").sum())
                                _dupe += int((_sdf["기존대조"] == "충돌(기존)").sum())
                        _picked_n = _n_t + _n_p
                        _to_save = _picked_n - _same
                        _picked = " + ".join(
                            f"{_KIND_KO[k]} {n:,}"
                            for k, n in (("term", _n_t), ("pattern", _n_p)) if n
                        )
                        st.markdown(
                            f"선택한 항목: **{_picked}**"
                        )
                        _bits = [f"🆕 새 항목 {_to_save - _dupe:,}"]
                        if _same:
                            _bits.append(f"✅ 이미 있음 {_same:,} (제외)")
                        if _dupe:
                            _bits.append(f"⚠️ 표기 다름 {_dupe:,} (검토에서 선택)")
                        st.caption("   ·   ".join(_bits))

                        if _dupe:
                            with st.expander(f"표기가 다른 항목 {_dupe}개"):
                                st.dataframe(
                                    _sel_terms[
                                        _sel_terms["기존대조"] == "충돌(기존)"
                                    ][["KO", "EN", "기존 EN"]].rename(
                                        columns={"EN": "새 표기",
                                                 "기존 EN": "Glossary 표기"}
                                    ),
                                    use_container_width=True, hide_index=True,
                                )
                                st.caption(
                                    "다음 단계에서 항목별로 사용할 표기를 선택합니다."
                                )

                        # 총계를 밝힌다. 예전에는 "58건 검수 후 등재
                        # (이미 등재 72건 제외)"처럼 나와, 58건 중 72건을
                        # 뺀다는 말로 읽혔다.
                        _label = (
                            f"{_to_save:,}개 검토 후 추가"
                        )
                        if st.button(
                            _label,
                            type="primary", key="catalog_save_all",
                            use_container_width=True,
                            disabled=_to_save == 0,
                        ):
                            def _drop_same(df):
                                if "기존대조" in df.columns:
                                    return df[df["기존대조"] != "동일"]
                                return df
                            _start_review([
                                ("term", _drop_same(_sel_terms)),
                                ("pattern", _drop_same(_sel_patterns)),
                            ])
                    else:
                        # 고른 것이 없어도 버튼을 남긴다. 통째로 사라지면
                        # 기능이 없어진 것처럼 보이고, 무엇을 해야 다음으로
                        # 넘어가는지도 알 수 없다.
                        st.button(
                            "검토 후 추가", type="primary",
                            key="catalog_save_all", use_container_width=True,
                            disabled=True,
                        )
                        st.caption(
                            "표에서 추가할 항목을 선택하세요. 용어와 패턴을 함께 선택할 수 있습니다."
                        )

            # 등재 여부와 상관없이, 지금 고른 제품으로 바로 번역하러 갈 수
            # 있게 한다. 추출만 하고 등재는 나중에 하는 경우도 있다.
            st.markdown("---")
            _gl1, _gl2 = st.columns([3, 1])
            _gl1.caption(f"{_extract_product} Glossary로 문서를 번역할 수 있습니다.")
            if _gl2.button("Localize로 이동", use_container_width=True,
                           key="catalog_go_localize"):
                _try_navigate({
                    "app_mode": "Localize", "step": 1,
                    "selected_product": _extract_product,
                })

            # ── 표기가 갈리는 항목 ────────────────────────────────────
            # 표로 두면 안 된다. Streamlit의 SelectboxColumn은 컬럼 전체에
            # 옵션 목록 하나만 줄 수 있어서, 226행의 후보가 전부 뒤섞인
            # 드롭다운이 되기 때문. 행마다 자기 후보만 보여주려면 카드 형태로
            # 라디오를 그려야 한다.
            if not _conf.empty:
                st.markdown("---")
                st.markdown(f"##### 표기 충돌 {len(_conf):,}개")
                st.caption(
                    "같은 한국어에 영어 표기가 여러 개인 항목입니다. 단순한 표기 차이는 AI가 "
                    "정리하고, 의미가 다른 항목만 직접 선택합니다."
                )

                _resolved = st.session_state.get("catalog_resolved")
                if st.session_state.get("catalog_review_conflicts") is not None:
                    _render_review("conflicts")
                elif not _resolved:
                    if st.button(f"AI로 {len(_conf):,}개 분류", key="catalog_resolve"):
                        from translator_engine import make_client  # 이 화면에서만 필요

                        with st.spinner("분류하는 중..."):
                            try:
                                st.session_state.catalog_resolved = catalog.resolve_conflicts(
                                    make_client(OPENAI_API_KEY), _conf
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"분류하지 못했습니다. ({e})")
                else:
                    _rows = _conf.to_dict("records")
                    _n_split = sum(
                        1 for r in _rows
                        if (_resolved.get(r["KO"]) or {}).get("kind") == "SPLIT"
                    )
                    st.caption(
                        f"단순 표기 차이 {len(_rows) - _n_split:,}개는 AI가 표기를 추천했습니다. "
                        f"의미가 다른 {_n_split:,}개는 직접 선택하세요."
                    )

                    _only_split = st.checkbox(
                        "직접 선택할 항목만 표시", value=True, key="catalog_only_split"
                    )
                    if st.session_state.get("catalog_prev_filter") != _only_split:
                        # 목록이 통째로 바뀌므로 첫 페이지로 되돌린다
                        st.session_state.catalog_conf_page = 1
                        st.session_state.catalog_prev_filter = _only_split
                    _shown = [
                        r for r in _rows
                        if not _only_split
                        or (_resolved.get(r["KO"]) or {}).get("kind") == "SPLIT"
                    ]

                    # 페이지 번호는 위젯이 아니라 순수 상태로 들고 간다 —
                    # 이동 버튼을 목록 **아래**에 두려면 카드를 그리기 전에
                    # 현재 페이지를 알아야 하기 때문.
                    _PAGE = 12
                    _pages = max(1, (len(_shown) + _PAGE - 1) // _PAGE)
                    _page = min(st.session_state.get("catalog_conf_page", 1), _pages)
                    st.session_state.catalog_conf_page = _page
                    _slice = _shown[(_page - 1) * _PAGE: _page * _PAGE]

                    # 용어·패턴 탭과 같은 원칙 — 기본은 아무것도 등재하지 않는다.
                    # 각 카드의 첫 선택지가 '선택 안 함'이고, 후보를 한 번
                    # 클릭해야 등재 대상이 된다.
                    _SKIP = "선택 안 함"
                    _use_reco = st.checkbox(
                        "AI 추천 표기 모두 선택", value=False, key="catalog_use_reco",
                        help="이 페이지의 모든 항목에 AI가 추천한 표기를 선택합니다.",
                    )

                    _picked = {}
                    for _r in _slice:
                        _ko = _r["KO"]
                        _info = _resolved.get(_ko) or {}
                        _cands = [c for c in str(_r.get("EN 후보") or "").split(" / ") if c]
                        if not _cands:
                            _cands = [_r["EN"]]
                        # SPLIT이라도 LLM이 최선의 후보를 함께 돌려준다.
                        # 없으면 최빈 표기(_r["EN"])로 채워, 일괄 선택이 항상 동작하게.
                        _reco = (_info.get("pick") or "").strip() or _r["EN"]
                        _options = [_SKIP] + _cands
                        _idx = (
                            _options.index(_reco)
                            if _use_reco and _reco in _options else 0
                        )
                        if _use_reco and _reco not in _options:
                            _idx = 1 if len(_options) > 1 else 0
                        _badge = "직접 선택" if _info.get("kind") == "SPLIT" else "AI 추천"

                        with st.container(border=True):
                            st.markdown(f"**{_ko}**  ·  :gray[{_badge}]")
                            _picked[_ko] = st.radio(
                                _ko, options=_options, index=_idx, horizontal=True,
                                label_visibility="collapsed",
                                # 일괄 선택을 토글하면 기본값이 다시 적용되도록 키를 바꾼다
                                key=f"catalog_pick_{int(_use_reco)}_{_ko}",
                            )
                            _hint = " · ".join(
                                x for x in (
                                    f"추천: **{_reco}**" if _reco else "",
                                    _info.get("reason", ""),
                                ) if x
                            )
                            if _hint:
                                st.caption(_hint)

                    _picked = {k: v for k, v in _picked.items() if v != _SKIP}

                    # 목록 하단 페이지 이동
                    _nav_l, _nav_c, _nav_r = st.columns([1, 3, 1])
                    with _nav_l:
                        if st.button("◀", key="catalog_prev", disabled=_page <= 1,
                                     use_container_width=True, help="이전 페이지"):
                            st.session_state.catalog_conf_page = _page - 1
                            st.rerun()
                    with _nav_c:
                        st.markdown(
                            f"<div style='text-align:center;padding-top:10px;"
                            f"font-size:15px;color:#555;'>{_page} / {_pages}</div>",
                            unsafe_allow_html=True,
                        )
                    with _nav_r:
                        if st.button("▶", key="catalog_next", disabled=_page >= _pages,
                                     use_container_width=True, help="다음 페이지"):
                            st.session_state.catalog_conf_page = _page + 1
                            st.rerun()

                    if st.button(
                        f"{len(_picked)}개 검토 후 추가", type="primary",
                        key="catalog_save_conf", use_container_width=True,
                        disabled=not _picked,
                    ):
                        _start_review(
                            pd.DataFrame({
                                "KO": list(_picked),
                                "EN": [_picked[k] for k in _picked],
                                "문맥(key)": [
                                    next((r["문맥(key)"] for r in _slice if r["KO"] == k), "")
                                    for k in _picked
                                ],
                            }),
                            "conflicts", "term",
                        )
                    if not _picked:
                        st.caption("항목별로 사용할 표기를 선택하세요.")

            st.markdown("---")
            st.download_button(
                "Excel로 다운로드",
                data=catalog.to_excel(
                    _result.terms, _result.patterns, product=_extract_product
                ),
                file_name=f"glossary_extracted_{_extract_product}.xlsx",
                mime=("application/vnd.openxmlformats-officedocument"
                      ".spreadsheetml.sheet"),
                use_container_width=True,
                help="Glossary 관리의 Excel 가져오기에 그대로 사용할 수 있습니다.",
            )

    st.stop()


# ─────────────────────────────────────────────
# 로그 페이지 — 번역 이력 + UI 매핑 reuse + 메모 관리
# ─────────────────────────────────────────────

if st.session_state.app_mode == "대시보드":
    import altair as alt
    from datetime import date, timedelta

    st.subheader("대시보드")

    # ── 바로 가기 ───────────────────────────────────────────────────
    # 대시보드가 첫 화면이므로, 숫자보다 "다음에 할 일"이 먼저 보이게 한다.
    _SHORTCUTS = [
        ("문서 Localize", "문서를 영어로 번역합니다.",
         {"app_mode": "Localize", "step": 1}, "dash_go_localize", "primary"),
        ("Glossary 관리", "용어와 패턴을 편집합니다.",
         {"app_mode": "Glossary 관리"}, "dash_go_glossary", "secondary"),
        ("Glossary 추출", "새 용어 후보를 찾습니다.",
         {"app_mode": "Glossary 추출"}, "dash_go_extract", "secondary"),
        ("작업 기록", "지난 번역을 확인합니다.",
         {"app_mode": "로그"}, "dash_go_logs", "secondary"),
    ]
    for _col, (_title, _desc, _target, _key, _type) in zip(
        st.columns(len(_SHORTCUTS)), _SHORTCUTS
    ):
        with _col, st.container(border=True):
            st.markdown(f"**{_title}**")
            st.caption(_desc)
            if st.button("시작" if _type == "primary" else "열기",
                         key=_key, type=_type, use_container_width=True):
                _try_navigate(_target)

    # ── 현황 ────────────────────────────────────────────────────────
    st.markdown(" ")
    _sh, _sp = st.columns([3, 2], vertical_alignment="bottom")
    _sh.markdown("**로컬라이즈 현황**")
    with _sp:
        _period = st.segmented_control(
            "기간", ["전체", "최근 30일", "최근 90일", "올해"],
            default="전체", key="dash_period", label_visibility="collapsed",
        ) or "전체"
    _since = {
        "최근 30일": (date.today() - timedelta(days=30)).isoformat(),
        "최근 90일": (date.today() - timedelta(days=90)).isoformat(),
        "올해": date(date.today().year, 1, 1).isoformat(),
    }.get(_period)

    _df = metrics.load_logs(since=_since)
    _gc = metrics.glossary_counts()
    _n_terms = _gc["terms"]["team"] + _gc["terms"]["personal"]

    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric("번역한 문서", f"{len(_df):,}")
    _k2.metric("번역한 단락", f"{int(_df['paragraphs_translated'].sum()):,}")
    _cost = float(_df["cost_usd"].sum())
    _k3.metric(
        "API 비용", f"${_cost:,.2f}",
        help=(f"{metrics.PRICE_MODEL} 공식 단가 기준 (1M 토큰당 입력 "
              f"${metrics.PRICE_PER_1M_INPUT}, 캐시된 입력 ${metrics.PRICE_PER_1M_CACHED_INPUT}, "
              f"출력 ${metrics.PRICE_PER_1M_OUTPUT}, {metrics.PRICE_CHECKED} 확인)"),
    )
    _k4.metric(
        "Glossary 용어", f"{_n_terms:,}",
        help="현재 등록된 용어 수입니다. 기간과 관계없습니다.",
    )
    if len(_df):
        st.caption(
            f"문서당 평균 비용 ${_cost / len(_df):,.2f} · "
            f"단가 출처: [OpenAI API Pricing]({metrics.PRICE_SOURCE_URL})"
        )

    if _df.empty:
        st.info("이 기간에 번역한 문서가 없습니다.", icon="ℹ️")
        st.stop()

    # 처리 시간 · 품질 (기록 시작 이후분만)
    _m = metrics.measured_subset(_df)
    with st.container(border=True):
        st.markdown("**처리 시간 및 품질**")
        if _m.empty:
            st.caption("처리 시간, Glossary 적용, 결과 검사 지표는 다음 번역부터 집계됩니다.")
        else:
            _secs = float(_m["elapsed_seconds"].mean())
            _paras = float(_m["paragraphs_translated"].sum())
            _q1, _q2, _q3, _q4 = st.columns(4)
            _q1.metric("문서당 처리 시간", f"{int(_secs // 60)}분 {int(_secs % 60)}초")
            _q2.metric(
                "단락당 처리 시간",
                f"{float(_m['elapsed_seconds'].sum()) / _paras:.1f}초" if _paras else "—",
            )
            _q3.metric(
                "문서당 Glossary 적용",
                f"{float(_m['applied_count'].mean()):,.0f}회",
                help="Glossary와 UI 텍스트 매핑이 적용된 횟수의 문서당 평균",
            )
            _q4.metric(
                "검사 통과율",
                f"{(_m['verify_errors'] == 0).mean() * 100:.0f}%",
                help="결과 검사에서 오류가 없었던 문서의 비율",
            )
            _rev = _m["review_edits"].fillna(0)
            st.caption(
                f"{_m['created_at'].min()[:10]} 이후 {len(_m)}개 문서 기준"
                + (f" · 번역 검토에서 {int((_rev > 0).sum())}개 문서, "
                   f"단락 {int(_rev.sum())}개 수정" if _rev.sum() else "")
            )

    # 한 차트에 시리즈 하나 — 색은 정체성을 나타낼 일이 없으므로 한 가지만 쓴다.
    _BAR = "#2E7D32"
    _monthly = (
        _df.groupby("month")
        .agg(문서=("created_at", "size"), 단락=("paragraphs_translated", "sum"),
             비용=("cost_usd", "sum"))
        .reset_index()
    )
    _monthly["비용"] = _monthly["비용"].round(2)
    _g = (_df.groupby("product")
          .agg(문서=("created_at", "size"), 단락=("paragraphs_translated", "sum"))
          .reset_index().sort_values("문서", ascending=False))

    # 두 차트를 나란히 — 대시보드 전체가 스크롤 없이 한 화면에 들어오게 한다
    _CH = 200
    _c_month, _c_prod = st.columns(2)
    with _c_month:
        st.markdown("**월별 번역 문서**")
        st.altair_chart(
            alt.Chart(_monthly).mark_bar(color=_BAR, cornerRadiusTopLeft=4,
                                         cornerRadiusTopRight=4, size=28)
            .encode(
                x=alt.X("month:O", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("문서:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                tooltip=[alt.Tooltip("month:O", title="월"), "문서:Q",
                         alt.Tooltip("단락:Q", format=","),
                         alt.Tooltip("비용:Q", format="$,.2f")],
            ).properties(height=_CH),
            use_container_width=True,
        )
    with _c_prod:
        st.markdown("**제품별 번역 문서**")
        # 제품이 늘어도 높이는 고정하고 막대 두께가 줄어든다
        st.altair_chart(
            alt.Chart(_g).mark_bar(color=_BAR, cornerRadiusTopRight=4,
                                   cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("product:N", title=None, sort="-x",
                        scale=alt.Scale(paddingInner=0.35)),
                x=alt.X("문서:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                tooltip=[alt.Tooltip("product:N", title="제품"), "문서:Q",
                         alt.Tooltip("단락:Q", format=",")],
            ).properties(height=_CH),
            use_container_width=True,
        )

    st.stop()


if st.session_state.app_mode == "로그":
    st.subheader("작업 기록")
    st.caption("팀 전체의 번역 기록입니다. 기록에 저장된 UI 텍스트 매핑을 다음 번역에 다시 사용할 수 있습니다.")

    _log_search = st.text_input(
        "검색",
        key="log_search",
        placeholder="파일 이름, 제품 또는 메모로 검색",
        label_visibility="collapsed",
    )

    # 모든 사용자의 번역 이력 — 팀 작업 추적용. 사용자 컬럼이 표에 있어 누구 것인지 식별 가능.
    _logs_df = list_logs(
        user=None,
        search=_log_search,
        limit=200,
    )
    st.caption(f"{len(_logs_df)}개 기록")

    if _logs_df.empty:
        st.info("번역 기록이 없습니다. Localize에서 첫 문서를 번역하세요.", icon="ℹ️")
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
        st.markdown("##### 기록 상세")
        _id_options = _logs_df["id"].tolist()
        _id_labels = {
            int(row["id"]): f"{row['created_at']} — {row['user']} — {row['source_file']}"
            for _, row in _logs_df.iterrows()
        }
        _selected_id = st.selectbox(
            "기록 선택",
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
                    col_d_b.markdown(f"**제품**\n\n{_detail['product'] or '—'}")
                    col_d_b.markdown(f"**텍스트 유형**\n\n{_detail['translation_mode'] or '—'}")
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
                            [{"한국어": k, "영어": v} for k, v in _mapping.items()]
                        )
                        st.dataframe(_map_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("UI 텍스트 매핑을 사용하지 않은 번역입니다.")

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
                                st.toast("메모를 저장했습니다.", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"메모를 저장하지 못했습니다. ({e})")
                    with col_act_load:
                        _can_load = bool(_mapping)
                        if st.button(
                            "이 매핑으로 번역",
                            use_container_width=True,
                            disabled=not _can_load,
                            type="primary" if _can_load else "secondary",
                            key=f"log_load_{_selected_id}",
                        ):
                            # 다음 Localize Step 2에서 이 매핑을 빈 EN 칸에 채워주도록 보관
                            st.session_state["preload_ui_mapping"] = dict(_mapping)
                            st.session_state.app_mode = "Localize"
                            st.session_state.step = 1
                            st.toast("매핑을 가져왔습니다. 2단계에서 자동으로 채워집니다.", icon="✅")
                            st.rerun()
                    with col_act_del:
                        if st.button(
                            "기록 삭제",
                            use_container_width=True,
                            key=f"log_del_{_selected_id}",
                        ):
                            try:
                                delete_log(int(_selected_id))
                                st.toast("기록을 삭제했습니다.", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"삭제하지 못했습니다. ({e})")

    st.stop()


# ─────────────────────────────────────────────
# Step 2 — 파일 업로드 & 번역 (번역 실행 모드)
# ─────────────────────────────────────────────

if st.session_state.step == 2:
    st.subheader("2단계: 문서 업로드")
    st.caption(
        "Word(.docx) 또는 Markdown(.md, .mdx) 문서를 업로드하세요. 여러 파일을 한 번에 "
        "업로드할 수 있습니다. 굵게 표시된 UI 텍스트의 영어 표기를 미리 지정할 수 있습니다."
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

    uploaded_docs = st.file_uploader(
        "문서 업로드",
        type=["docx", "md", "markdown", "mdx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_docs:
        if len(uploaded_docs) > 1:
            st.caption(
                f"파일 {len(uploaded_docs)}개를 차례로 번역합니다: "
                + ", ".join(d.name for d in uploaded_docs)
            )
        else:
            st.caption(uploaded_docs[0].name)

    # ── 파일 업로드 후: bold 추출 + 글로서리 자동 매칭 → 매핑 입력 표 ─
    #
    # 여러 파일을 올리면 표는 **하나로 병합**한다. 파일마다 따로 입력받으면
    # 같은 용어를 여러 번 적어야 하고, 무엇보다 문서 간 표기가 갈린다
    # (이 도구의 존재 이유가 표기 통일이므로 그건 곧 실패다). 어느 파일에서
    # 왔는지는 '출처' 열로 보여주고, 입력한 영문은 배치 전체에 적용한다.
    ui_mapping_df = None
    saved_input_paths: list = []

    if uploaded_docs:
        # 파일 묶음이 바뀌면 재추출 (파일명+size 전체 + 컬럼 스키마 버전).
        # 스키마 버전은 ui_text_mapping_rows의 키 구조를 바꿀 때마다 올린다 —
        # 그래야 이전 세션의 row가 새 컬럼과 안 맞을 때 자동 재추출된다.
        file_sig = "||".join(f"{d.name}::{d.size}" for d in uploaded_docs) + "::batch_v1"
        if st.session_state.get("ui_text_source_sig") != file_sig:
            try:
                # 업로드 파일 저장은 반드시 메인 스레드에서 — UploadedFile은
                # 스크립트 실행에 묶인 객체라 백그라운드로 넘기지 않는다.
                _saved = [
                    (d.name, save_uploaded_file(d, UPLOAD_DIR)) for d in uploaded_docs
                ]

                # 글로서리 자동 매칭 (KO 일치)
                _glossary_lookup = {
                    r["KO"]: r["EN"]
                    for r in glossary_rows
                    if r.get("KO") and r.get("EN")
                }
                # 로그에서 가져온 매핑 — 글로서리보다 우선
                _preloaded = st.session_state.pop("preload_ui_mapping", None) or {}

                # KO 기준으로 합친다 — 같은 라벨은 한 행, 출처 파일은 모아서.
                initial_rows, _source_counts = batch.merge_bold_terms(
                    [(_name, extract_bold_terms(str(_path)))
                     for _name, _path in _saved],
                    glossary_lookup=_glossary_lookup,
                    preloaded=_preloaded,
                )

                st.session_state.ui_text_mapping_rows = initial_rows
                st.session_state.ui_text_source_sig = file_sig
                st.session_state.ui_text_input_paths = [
                    {"source_name": n, "path": str(p)} for n, p in _saved
                ]
                st.session_state.ui_text_preload_counts = _source_counts

                # 반복 용어 → 글로서리 후보.
                #
                # 글로서리는 보통 **이전 버전** 매뉴얼로 만들어 둔다. 그래서
                # 이번 버전에 새로 생긴 용어는 글로서리에 있을 수가 없고,
                # 문단마다 다르게 번역된다 (실제로 "데이터 기반 답변
                # 에이전트"가 다섯 가지로 나왔다). 번역 직전에 이 문서에서
                # 반복되는 용어를 보여주고 영문을 정하게 한다.
                #
                # 여러 파일의 문단을 한 번에 넣는다 — 빈도가 배치 전체 기준이
                # 되어야 "문서 3개에 흩어져 각 2번"처럼 파일별로는 기준에 못
                # 미치지만 실제로는 반복되는 용어가 잡힌다.
                _known = {r.get("KO") for r in glossary_rows if r.get("KO")}
                _known |= {r["KO (Bold)"] for r in initial_rows}
                _all_paras = []
                for _name, _path in _saved:
                    _all_paras.extend(extract_korean_paragraphs(str(_path)))
                st.session_state.term_candidate_rows = (
                    catalog.suggest_terms_from_texts(
                        _all_paras, exclude=_known,
                    ).to_dict("records")
                )
            except Exception as e:
                st.error(f"문서를 읽지 못했습니다. ({e})")

        saved_input_paths = st.session_state.get("ui_text_input_paths") or []
        ui_mapping_df = pd.DataFrame(
            st.session_state.get("ui_text_mapping_rows", []),
            columns=["KO (Bold)", "EN (입력)", "맥락", "출처"],
        )

        if ui_mapping_df.empty:
            st.info("굵게 표시된 UI 텍스트가 없습니다. 바로 번역을 시작할 수 있습니다.", icon="ℹ️")
        else:
            # 같은 파일 또는 같은 제품의 이전 로그가 있으면 매핑 재사용 옵션 제공.
            # 여러 파일이면 첫 파일 이름으로 찾는다 — '동일 제품' 매칭이 어차피
            # 같이 걸리고, 불러온 매핑은 KO 일치로만 채워지므로 배치 전체에
            # 안전하게 적용된다.
            _related_logs = find_related_logs(
                source_file=uploaded_docs[0].name,
                product=st.session_state.selected_product or "",
                limit=15,
            )
            if not _related_logs.empty:
                _file_matches = _related_logs[_related_logs["match_type"] == "file"]
                _product_matches = _related_logs[_related_logs["match_type"] == "product"]

                with st.container(border=True):
                    _headline_bits = []
                    if not _file_matches.empty:
                        _headline_bits.append(f"같은 파일 {len(_file_matches)}개")
                    if not _product_matches.empty:
                        _headline_bits.append(
                            f"같은 제품 {len(_product_matches)}개"
                        )
                    st.markdown(
                        "**이전 번역의 UI 텍스트 매핑을 사용할 수 있습니다.** "
                        + "(" + ", ".join(_headline_bits) + ")"
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
                            "가져올 기록",
                            options=list(_prev_options.keys()),
                            format_func=lambda i: _prev_options.get(int(i), str(i)),
                            label_visibility="collapsed",
                            key="prev_log_picker",
                        )
                    with col_pl_btn:
                        if st.button("매핑 가져오기", use_container_width=True, key="prev_log_load"):
                            _ld = get_log(int(_pl_id))
                            _ld_map = (_ld or {}).get("ui_text_overrides") or {}
                            _filled = 0
                            for row in st.session_state.ui_text_mapping_rows:
                                if not row.get("EN (입력)"):
                                    ko = row.get("KO (Bold)")
                                    if ko in _ld_map:
                                        row["EN (입력)"] = _ld_map[ko]
                                        _filled += 1
                            st.toast(f"매핑 {_filled}개를 채웠습니다.", icon="✅")
                            st.rerun()
                    st.caption(
                        "📄 같은 파일 · 🏷️ 같은 제품. 이 문서에 있는 항목만 채우며, "
                        "이미 입력한 값은 바꾸지 않습니다."
                    )

            # 자동 매칭 결과 요약
            _counts = st.session_state.get("ui_text_preload_counts", {})
            if _counts.get("글로서리", 0) or _counts.get("로그", 0):
                bits = []
                if _counts.get("로그", 0):
                    bits.append(f"작업 기록 {_counts['로그']}개")
                if _counts.get("글로서리", 0):
                    bits.append(f"Glossary {_counts['글로서리']}개")
                st.caption(f"자동으로 채운 항목: {', '.join(bits)}")

            st.markdown(
                f"##### UI 텍스트 매핑 ({len(ui_mapping_df)}개)"
            )
            st.caption(
                "영어 표기를 지정할 항목만 입력하세요. 비워 두면 AI가 번역합니다. "
                "Enter를 누르면 다음 행으로 이동합니다."
            )
            ui_mapping_df = st.data_editor(
                ui_mapping_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                disabled=["KO (Bold)", "맥락", "출처"],
                column_order=(["KO (Bold)", "EN (입력)", "출처", "맥락"]
                              if len(uploaded_docs) > 1
                              else ["KO (Bold)", "EN (입력)", "맥락"]),
                column_config={
                    "KO (Bold)": st.column_config.TextColumn("한국어", width="small"),
                    "출처": st.column_config.TextColumn(
                        "출처 파일", disabled=True, width="small",
                        help="입력한 영어 표기는 업로드한 모든 파일에 적용됩니다.",
                    ),
                    "EN (입력)": st.column_config.TextColumn(
                        "영어",
                        help="입력하면 이 표기를 그대로 사용합니다.",
                        width="medium",
                    ),
                    "맥락": st.column_config.TextColumn(
                        "문맥",
                        help="본문에서 이 텍스트가 나오는 문장입니다.",
                        width="large",
                    ),
                },
                key="ui_text_editor",
            )

        # ── 반복 용어 (글로서리 후보) ────────────────────────────
        # UI 텍스트와 같은 위계로 둔다. 다만 성격이 다르다 —
        # UI 매핑은 **굵은 라벨 안에서만** 치환되고, 이쪽은 글로서리와
        # 똑같이 **본문 어디서든** 치환된다. 그래서 표를 나눈다.
        _term_rows = st.session_state.get("term_candidate_rows") or []
        if _term_rows:
            term_df = pd.DataFrame(
                _term_rows, columns=["KO", "EN (입력)", "빈도", "묶음", "맥락"])
            _n_grouped = int((term_df["묶음"] != "").sum())
            st.markdown(f"##### 반복 용어 ({len(term_df)}개)")
            st.caption(
                "문서에 3번 이상 나오지만 Glossary에 없는 용어입니다. "
                "영어 표기를 지정하면 문서 전체에 일관되게 적용됩니다."
            )
            if _n_grouped:
                st.info(
                    f"**그룹**이 표시된 {_n_grouped}개는 앞말에 따라 영어가 달라질 수 있습니다. "
                    "각각 지정하세요. (예: 대화 내용 내보내기 → export, 사용자 내보내기 → remove)",
                    icon="ℹ️",
                )
            term_df = st.data_editor(
                term_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                disabled=["KO", "빈도", "묶음", "맥락"],
                column_order=["묶음", "빈도", "KO", "EN (입력)", "맥락"],
                column_config={
                    "빈도": st.column_config.NumberColumn(
                        "횟수", disabled=True, width="small",
                        help="문서에 나오는 횟수",
                    ),
                    "KO": st.column_config.TextColumn("한국어", width="medium"),
                    "EN (입력)": st.column_config.TextColumn(
                        "영어",
                        help="입력하면 문서 전체에서 이 표기를 사용합니다.",
                        width="medium",
                    ),
                    "묶음": st.column_config.TextColumn(
                        "그룹", disabled=True, width="small",
                        help="끝말이 같은 용어 그룹. 앞말에 따라 영어가 달라질 수 있습니다.",
                    ),
                    "맥락": st.column_config.TextColumn(
                        "문맥", width="large",
                        help="본문에서 이 용어가 처음 나오는 문장입니다.",
                    ),
                },
                key="term_candidate_editor",
            )
            st.checkbox(
                "번역이 끝나면 입력한 용어를 Glossary에 추가",
                key="term_candidates_register",
            )

    st.markdown("---")
    col_back, col_translate = st.columns(2)

    with col_back:
        if st.button("이전", use_container_width=True):
            _try_navigate({"step": 1})

    with col_translate:
        translate_clicked = st.button(
            f"번역 시작 ({len(uploaded_docs)}개)" if len(uploaded_docs or []) > 1
            else "번역 시작",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_docs,
        )

    # 클릭 시점에 백그라운드 잡을 띄우고 rerun → 다음 렌더부터는 위쪽
    # translating_now 분기가 진행률만 그린다. 번역 자체는 스크립트 실행과
    # 무관하게 스레드에서 계속 돌기 때문에 rerun이 나도 재시작되지 않는다.
    if translate_clicked:
        # 이미 돌고 있는 잡이 있으면 무시. 잡을 두 개 띄우면 같은 문서를
        # 두 번 번역하면서 API rate limit을 스스로 때리게 된다 — 출력 파일
        # 경로도 같아 나중에 끝난 쪽이 앞선 결과를 덮어쓴다.
        if st.session_state.get("translating_now"):
            pass
        elif not uploaded_docs:
            st.error("문서를 업로드하세요.")
        else:
            # 사용자가 채운 반복 용어를 이번 번역의 글로서리에 얹는다.
            # UI 매핑(굵은 라벨 한정)과 달리 본문 전체에 적용돼야 하므로
            # glossary_rows 쪽에 넣는다.
            _term_pairs: Dict[str, str] = {}
            if _term_rows:
                for _, _tr in term_df.iterrows():
                    _tko = _cell(_tr.get("KO"))
                    _ten = _cell(_tr.get("EN (입력)"))
                    if _tko and _ten:
                        _term_pairs[_tko] = _ten
            if _term_pairs:
                _existing_ko = {r.get("KO") for r in glossary_rows}
                glossary_rows = glossary_rows + [
                    {"KO": k, "EN": v, "DNT": False, "Case-sensitive": False}
                    for k, v in _term_pairs.items() if k not in _existing_ko
                ]
                st.session_state["pending_term_registrations"] = (
                    _term_pairs if st.session_state.get("term_candidates_register")
                    else {}
                )

            _ui_overrides_pending = {}
            if ui_mapping_df is not None and not ui_mapping_df.empty:
                for _, row in ui_mapping_df.iterrows():
                    ko = _cell(row.get("KO (Bold)"))
                    en = _cell(row.get("EN (입력)"))
                    if ko and en:
                        _ui_overrides_pending[ko] = en

            # 업로드 파일 저장은 반드시 메인 스레드에서 — UploadedFile은
            # 스크립트 실행에 묶인 객체라 백그라운드 스레드로 넘기지 않는다.
            # 위 추출 단계에서 이미 저장해 뒀으면 그 경로를 재사용한다.
            _saved_by_name = {
                e["source_name"]: Path(e["path"])
                for e in (saved_input_paths or [])
            }
            _out_names = batch.unique_output_names([
                make_default_output_filename(
                    st.session_state.selected_product, _doc.name,
                )
                for _doc in uploaded_docs
            ])
            _job_files = []
            for _doc, _out_name in zip(uploaded_docs, _out_names):
                _in = _saved_by_name.get(_doc.name)
                if _in is None or not _in.exists():
                    _in = save_uploaded_file(_doc, UPLOAD_DIR)
                _job_files.append({
                    "source_name": _doc.name,
                    "in_path": str(_in),
                    "out_path": str(OUTPUT_DIR / _out_name),
                    "output_filename": _out_name,
                })

            st.session_state.translate_job_id = start_job({
                "files": _job_files,
                "glossary_rows": glossary_rows,
                "pattern_rows": pattern_rows,
                "api_key": OPENAI_API_KEY,
                "enable_cache": st.session_state.enable_cache,
                "enable_qa": st.session_state.enable_qa,
                "translation_mode": st.session_state.translation_mode,
                "ui_overrides": _ui_overrides_pending,
            })
            st.session_state.translating_now = True
            st.rerun()


# ─────────────────────────────────────────────
# Step 3 — 결과 & 다운로드
# ─────────────────────────────────────────────

elif st.session_state.step == 3:
    st.subheader("3단계: 번역 검토 및 다운로드")

    results = st.session_state.get("last_results") or []

    # 세션이 비었으면 앱이 재시작된 것이다. 조용히 첫 화면으로 되돌리면
    # 사용자는 "다운로드를 눌렀더니 화면이 튕겼다"로만 인식한다.
    if not results:
        st.warning(
            "앱이 다시 시작되어 번역 결과를 표시할 수 없습니다. "
            "번역 기록은 작업 기록에서 확인할 수 있습니다.",
            icon="⚠️",
        )
        _c1, _c2 = st.columns(2)
        if _c1.button("처음으로", type="primary", use_container_width=True,
                      key="step3_restart"):
            st.session_state.step = 1
            st.rerun()
        if _c2.button("작업 기록 보기", use_container_width=True, key="step3_logs"):
            _try_navigate({"app_mode": "로그"})
        st.stop()

    # 산출물 파일이 사라진 경우 — 클라우드는 재시작하면 임시 파일을 잃는다.
    # 여기서 걸러야 open()에서 예외로 터지지 않는다. 파일별로 따로 본다:
    # 5개 중 1개가 없어져도 나머지 4개는 받을 수 있어야 한다.
    for _r in results:
        _r["available"] = bool(
            _r.get("output_path") and Path(_r["output_path"]).exists()
        )

    ok = [r for r in results if r.get("result") and r["available"]]
    failed = [r for r in results if not r.get("result")]
    gone = [r for r in results if r.get("result") and not r["available"]]

    if not ok:
        st.warning(
            "결과 파일을 찾을 수 없습니다. 앱이 다시 시작되어 임시 파일이 "
            "삭제되었습니다. 다시 번역하세요.",
            icon="⚠️",
        )
        if st.button("처음으로", type="primary", key="step3_restart_missing"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    # ── 배치 요약 ────────────────────────────────────────────────────
    def _cost_of(r: dict) -> float:
        m = r.get("result") or {}
        return estimate_cost_usd(m.get("input_tokens", 0), m.get("cached_tokens", 0),
                                 m.get("output_tokens", 0))

    def _errors_of(r: dict) -> list:
        return [v for v in ((r.get("result") or {}).get("verification") or [])
                if v.get("level") == "오류"]

    _n_verify_err = sum(1 for r in ok if _errors_of(r))

    if failed:
        st.warning(
            f"파일 {len(results)}개 중 {len(ok)}개를 번역했고 {len(failed)}개는 실패했습니다. "
            "아래 표에서 실패 사유를 확인하세요.",
            icon="⚠️",
        )
    elif _n_verify_err:
        st.warning(
            f"번역을 완료했지만 파일 {_n_verify_err}개에서 오류가 발견되었습니다. "
            "배포하기 전에 결과 검사 내용을 확인하세요.",
            icon="⚠️",
        )
    else:
        st.success(
            f"파일 {len(ok)}개를 번역했습니다." if len(ok) > 1
            else "번역을 완료했습니다."
        )
    if gone:
        st.caption(
            "앱이 다시 시작되어 결과 파일을 찾을 수 없는 항목: "
            + ", ".join(r["source_name"] for r in gone)
        )

    # 파일이 여러 개면 합계와 한눈에 보는 표를 먼저. 상세는 아래에서 하나씩.
    # 파일이 하나면 검토 화면의 통계·비용 막대가 같은 내용을 보여준다.
    if len(results) > 1:
        _tot_paras = sum((r.get("result") or {}).get("paragraphs_translated", 0) or 0 for r in ok)
        _tot_tokens = sum((r.get("result") or {}).get("total_tokens", 0) or 0 for r in ok)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("번역한 단락", f"{_tot_paras:,}")
        col_b.metric("토큰", f"{_tot_tokens:,}")
        col_c.metric("API 비용", f"${sum(_cost_of(r) for r in ok):,.2f}")

        _rows = []
        for r in results:
            m = r.get("result") or {}
            _v_err = _errors_of(r)
            _v_all = m.get("verification") or []
            if not r.get("result"):
                _state, _verify_cell = "✗ 실패", "—"
            elif not r["available"]:
                _state, _verify_cell = "⚠️ 파일 없음", "—"
            elif _v_err:
                _state, _verify_cell = "✓ 완료", f"⚠️ 오류 {len(_v_err)}"
            elif _v_all:
                _state, _verify_cell = "✓ 완료", f"! 경고 {len(_v_all)}"
            else:
                _state, _verify_cell = "✓ 완료", "✓"
            _rows.append({
                "파일": r["source_name"],
                "상태": _state,
                "문단": m.get("paragraphs_translated", 0) or 0,
                "토큰": m.get("total_tokens", 0) or 0,
                "비용": f"${_cost_of(r):,.2f}" if r.get("result") else "—",
                "검증": _verify_cell,
                # 실패가 없으면 빈 열이 되므로 아래에서 떼어낸다.
                "사유": (r.get("error") or "")[:120],
            })
        _sdf = pd.DataFrame(_rows)
        if not failed:
            _sdf = _sdf.drop(columns=["사유"])
        st.dataframe(
            _sdf, use_container_width=True, hide_index=True,
            column_config={
                "파일": st.column_config.TextColumn("파일", width="medium"),
                "상태": st.column_config.TextColumn("상태", width="small"),
                "문단": st.column_config.NumberColumn("단락", width="small"),
                "토큰": st.column_config.NumberColumn("토큰", width="small"),
                "비용": st.column_config.TextColumn("비용", width="small"),
                "검증": st.column_config.TextColumn("결과 검사", width="small"),
                "사유": st.column_config.TextColumn("실패 사유", width="large"),
            },
        )

        # ZIP 일괄 다운로드 — 파일마다 버튼을 누르게 하면 10개짜리 배치에서
        # 열 번 눌러야 한다. 산출물 경로로 캐시해 rerun마다 다시 만들지 않는다.
        _zip_key = "batch_zip::" + "|".join(r["output_path"] for r in ok)
        if _zip_key not in st.session_state:
            try:
                _buf = io.BytesIO()
                with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                    for r in ok:
                        _zf.write(r["output_path"], arcname=r["output_filename"])
                st.session_state[_zip_key] = _buf.getvalue()
            except Exception as _e:
                st.session_state[_zip_key] = None
                st.caption(f"ZIP 파일을 만들지 못했습니다. ({_e})")
        _zip_bytes = st.session_state.get(_zip_key)
        if _zip_bytes:
            _zip_name = (
                f"{(st.session_state.selected_product or 'localized')}_en_"
                f"{len(ok)}files.zip"
            )
            st.download_button(
                f"모두 다운로드 (ZIP, {len(ok)}개)",
                data=_zip_bytes,
                file_name=_zip_name,
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

    st.markdown("")

    # ── 파일 선택 → 상세 ─────────────────────────────────────────────
    if len(results) > 1:
        st.markdown("##### 파일별 결과")
        _labels = {
            i: ("✓ " if r.get("result") and r["available"] else "✗ ") + r["source_name"]
            for i, r in enumerate(results)
        }
        _pick = st.selectbox(
            "파일 선택",
            options=list(_labels.keys()),
            format_func=lambda i: _labels[int(i)],
            key="last_batch_detail",
            label_visibility="collapsed",
        )
        detail = results[int(_pick)]
    else:
        detail = results[0]

    render_result_detail(detail)

    st.markdown("---")
    col_prev, col_restart = st.columns(2)

    with col_prev:
        if st.button("이전 단계로", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    with col_restart:
        if st.button("새 문서 번역", use_container_width=True):
            reset_translation_result()
            st.session_state.step = 1
            st.rerun()
