"""
app.py UI 스모크 테스트 — 실제 Streamlit 세션으로 스크립트를 돌려본다.

    python tests/test_app_ui.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

import services.jobs as jobs

APP = str(ROOT / "app.py")
PRODUCT = list(json.loads((ROOT / "product_config.json").read_text(encoding="utf-8")).keys())[0]

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        failures.append(name)


def make_at(**session):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["current_user"] = "SmokeTest"
    at.session_state["app_mode"] = "Localize"
    at.session_state["step"] = 2
    at.session_state["selected_product"] = PRODUCT
    for k, v in session.items():
        at.session_state[k] = v
    return at


def fake_params():
    return {
        "in_path": str(ROOT / "tests" / "fixtures" / "runAnalysis.mdx"),
        "out_path": str(ROOT / "outputs" / "smoke_out.mdx"),
        "output_filename": "smoke_out.mdx",
        "glossary_rows": [], "pattern_rows": [], "api_key": "sk-fake",
        "enable_cache": True, "enable_qa": False,
        "translation_mode": "매뉴얼", "ui_overrides": {},
    }


def wait_done(job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get_job(job_id)
        if job and job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("잡이 끝나지 않음")


print("[1] 부팅 / 로그인 화면")
at = AppTest.from_file(APP, default_timeout=60).run()
check("예외 없음", not at.exception, str(at.exception))

print("[2] 잡 유실 시 안전 복구 (앱 재시작 시나리오)")
at = make_at(translating_now=True, translate_job_id="no-such-job").run()
check("예외 없음", not at.exception, str(at.exception))
check("안내 문구", any("앱이 재시작" in e.value for e in at.error))
check("플래그 정리", at.session_state["translating_now"] is False)
check("업로더 미노출", len(at.get("file_uploader")) == 0)

print("[3] 일반 Step 2 렌더")
at = make_at().run()
check("예외 없음", not at.exception, str(at.exception))
check("업로더 1개", len(at.get("file_uploader")) == 1)
check("번역 시작 버튼", "번역 시작" in [b.label for b in at.button])

print("[4] rerun을 넘어 살아있는 잡을 UI가 찾아냄")
_orig = jobs.translate_document
jobs.translate_document = lambda **kw: (_ for _ in ()).throw(RuntimeError("의도된 실패"))
job_id = jobs.start_job(fake_params())
wait_done(job_id)
at = make_at(translating_now=True, translate_job_id=job_id).run()
msgs = [e.value for e in at.error]
check("잡 조회됨(레지스트리 유지)", not any("앱이 재시작" in m for m in msgs), str(msgs))
check("스레드 예외가 UI로 전달", any("의도된 실패" in m for m in msgs), str(msgs))

print("[5] 진행 중 잡에 계속 붙어 폴링")
gate = threading.Event()


def _blocking(progress_callback=None, **kwargs):
    if progress_callback:
        progress_callback(3, 10)
    gate.wait(timeout=20)
    return {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "paragraphs_translated": 0}


jobs.translate_document = _blocking
job_id = jobs.start_job(fake_params())
for _ in range(100):
    if jobs.get_job(job_id)["total"]:
        break
    time.sleep(0.05)
check("progress_callback 반영", jobs.get_job(job_id)["done"] == 3)
at = make_at(translating_now=True, translate_job_id=job_id)
try:
    at.run(timeout=4)      # 폴링 루프 → 타임아웃이 정상
except Exception:
    pass
check("진행 중 플래그 유지", at.session_state["translating_now"] is True)
gate.set()
jobs.translate_document = _orig

print("[6] Step 3 — .mdx 결과 다운로드 화면")
tmp_out = ROOT / "outputs" / "_ui_probe_FSP_en.mdx"
tmp_out.parent.mkdir(exist_ok=True)
tmp_out.write_text("# hi\n", encoding="utf-8")
at = AppTest.from_file(APP, default_timeout=60)
at.session_state["current_user"] = "SmokeTest"
at.session_state["app_mode"] = "Localize"
at.session_state["step"] = 3
at.session_state["selected_product"] = PRODUCT
at.session_state["last_result"] = {"input_tokens": 10, "output_tokens": 20}
at.session_state["last_output_path"] = str(tmp_out)
at.session_state["last_output_filename"] = tmp_out.name
at.run()
check("예외 없음", not at.exception, str(at.exception))
check("다운로드 버튼 렌더", len(at.get("download_button")) == 1)
tmp_out.unlink(missing_ok=True)

print("[7] Glossary 추출 메뉴")
at = AppTest.from_file(APP, default_timeout=60)
at.session_state["current_user"] = "SmokeTest"
at.session_state["app_mode"] = "Glossary 추출"
at.run()
check("예외 없음", not at.exception, str(at.exception))
check("업로더 렌더", len(at.get("file_uploader")) == 1)
check("사이드바 메뉴 버튼 존재", "Glossary 추출" in [b.label for b in at.button])

print("[8] Glossary 추출 — 분석 결과 화면과 등재 버튼")
import pandas as pd  # noqa: E402
from services import catalog as ct  # noqa: E402

# 용어 후보는 '반복되는 복합어'만 남으므로 같은 표현을 여러 key에 깔아준다
_ko_seed = {f"a.x{i}": "취약점 점검" for i in range(4)}
_en_seed = {f"a.x{i}": "Vulnerability check" for i in range(4)}
_ko_seed["b.y"] = "분석을 시작합니다."
_en_seed["b.y"] = "Analysis started."
_pick = ct.pick_languages([
    ct.parse_json("ko.json", _ko_seed),
    ct.parse_json("en.json", _en_seed),
])
_res = ct.analyze(_pick)


class _FakeParsed:
    name = "ko.json"

    def describe(self):
        return "테스트"


at = AppTest.from_file(APP, default_timeout=60)
at.session_state["current_user"] = "SmokeTest"
at.session_state["app_mode"] = "Glossary 추출"
at.session_state["catalog_result"] = _res
at.session_state["catalog_parsed"] = [_FakeParsed()]
at.session_state["catalog_pick_labels"] = ("ko.json", "en.json", [])
at.session_state["catalog_sig"] = "seeded"
at.run()
check("예외 없음", not at.exception, str(at.exception))
check("업로드 없이도 결과 화면이 뜸", len(at.metric) >= 4,
      f"metric {len(at.metric)}개")
check("탭 2개(용어·패턴) 렌더", len(at.tabs) == 2, f"tabs {len(at.tabs)}개")
_caps = " ".join(str(c.value) for c in at.caption)
check("선택 안내 노출", "체크박스" in _caps, _caps[:120])
check("등재 버튼은 선택 후에만", not any("등재" in b.label for b in at.button),
      str([b.label for b in at.button]))

print("[9] 표기 충돌 — 후보를 클릭으로 고르는 카드 UI")
_conf_pick = ct.pick_languages([
    ct.parse_json("ko.json", {"l.a": "확인", "l.b": "확인", "r.c": "확인"}),
    ct.parse_json("en.json", {"l.a": "OK", "l.b": "OK", "r.c": "Check"}),
])
_conf_res = ct.analyze(_conf_pick)
at = AppTest.from_file(APP, default_timeout=60)
at.session_state["current_user"] = "SmokeTest"
at.session_state["app_mode"] = "Glossary 추출"
at.session_state["catalog_result"] = _conf_res
at.session_state["catalog_parsed"] = [_FakeParsed()]
at.session_state["catalog_pick_labels"] = ("ko.json", "en.json", [])
at.session_state["catalog_sig"] = "seeded"
at.session_state["catalog_resolved"] = {
    "확인": {"kind": "SPLIT", "pick": "", "reason": "화면마다 뜻이 다름"}
}
at.run()
check("예외 없음", not at.exception, str(at.exception))
check("후보가 라디오로 렌더", len(at.radio) >= 1, f"radio {len(at.radio)}개")
if at.radio:
    check("행의 후보만 옵션으로", set(at.radio[0].options) == {"OK", "Check"},
          str(at.radio[0].options))
check("판정 근거 노출", any("화면마다" in str(c.value) for c in at.caption),
      str([str(c.value)[:30] for c in at.caption]))

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
