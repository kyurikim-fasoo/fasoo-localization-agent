"""
번역 잡 레지스트리 — Streamlit rerun을 넘어 살아남는 보관소.

왜 별도 모듈인가 (중요):
    Streamlit은 메인 스크립트를 실행할 때마다 **새 모듈 네임스페이스**를
    만들어 거기에 exec 한다.

        # streamlit/runtime/scriptrunner/script_runner.py
        module = self._new_module("__main__")
        exec(code, module.__dict__)

    따라서 app.py의 모듈 전역은 rerun 한 번이면 초기값으로 되돌아간다.
    잡 레지스트리를 app.py에 두면 첫 폴링 rerun에서 통째로 사라져,
    돌고 있는 번역을 UI가 잃어버린다(스레드는 계속 도는데 화면은 "중단됨").

    반면 import 되는 모듈은 sys.modules에 캐시되므로 한 번만 실행되고,
    프로세스가 사는 동안 전역 상태가 유지된다. 그래서 여기에 둔다.

스레드 규칙:
    작업 스레드에는 ScriptRunContext가 없다. 여기 job dict 외에는 아무것도
    건드리지 않으며, st.* 는 절대 호출하지 않는다.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

from translator_engine import translate_document


# 완료된 잡을 붙잡고 있을 시간. UI가 결과를 읽어갈 시간만 주면 되지만,
# 사용자가 탭을 잠시 떠나 있을 수 있어 넉넉히 잡는다.
_FINISHED_TTL_SEC = 30 * 60

_JOBS: dict = {}
_LOCK = threading.Lock()


def get_job(job_id: Optional[str]) -> Optional[dict]:
    """job_id에 해당하는 잡. 앱 프로세스가 재시작됐으면 None."""
    if not job_id:
        return None
    with _LOCK:
        return _JOBS.get(job_id)


def _purge_stale() -> None:
    """끝난 지 오래된 잡 정리. 호출자가 _LOCK을 쥐고 있어야 한다."""
    now = time.monotonic()
    for jid in [
        jid for jid, j in _JOBS.items()
        if j["status"] != "running" and now - (j["finished_at"] or now) > _FINISHED_TTL_SEC
    ]:
        del _JOBS[jid]


def batch_progress(job: dict) -> float:
    """
    배치 전체 진행률 0.0~1.0.

    **파일 수**를 분모로 쓰고 진행 중인 파일의 문단 비율만 소수점으로 얹는다.
    파일별 총 문단 수는 그 파일을 열어봐야 알 수 있어서(번역 직전에 센다),
    문단 총합을 분모로 삼으면 다음 파일이 시작될 때마다 분모가 커져 막대가
    뒤로 밀린다. 사용자에게는 그게 "진행률이 되감겼다"로 보인다.
    """
    files = job["files"]
    if not files:
        return 0.0
    acc = 0.0
    for f in files:
        if f["status"] in ("finished", "error"):
            acc += 1.0
        elif f["status"] == "running" and f["total"]:
            acc += min(f["done"] / f["total"], 1.0)
    return min(acc / len(files), 1.0)


def start_job(params: dict) -> str:
    """
    여러 문서를 **순차로** 번역하는 잡을 데몬 스레드에서 시작하고 job_id를
    돌려준다.

    순차인 이유 — 문서 하나의 번역이 이미 문단마다 API를 부르는 순차 루프다.
    파일까지 병렬로 돌리면 초당 요청 수가 파일 수만큼 배로 늘어 429/503
    (overloaded)을 스스로 유발한다. 총 시간은 각 파일 시간의 합이 된다.

    params 키:
        files: [{source_name, in_path, out_path, output_filename}, ...]
        glossary_rows, pattern_rows, api_key, enable_cache, enable_qa,
        translation_mode, ui_overrides
    """
    job_id = uuid.uuid4().hex
    files = [
        {
            "source_name": f["source_name"],
            "in_path": f["in_path"],
            "out_path": f["out_path"],
            "output_filename": f["output_filename"],
            "status": "pending",      # pending | running | finished | error
            "done": 0,
            "total": 0,
            "result": None,
            "error": None,
        }
        for f in params["files"]
    ]
    job = {
        "status": "running",          # running | finished
        "files": files,
        "current": 0,                 # 진행 중인 파일의 인덱스
        "finished_at": None,
        "params": params,
    }
    with _LOCK:
        _purge_stale()
        _JOBS[job_id] = job

    def _make_progress(entry: dict):
        # 단순 대입만 — 읽는 쪽은 UI 한 곳뿐이라 락이 필요 없다.
        def _on_progress(done: int, total: int) -> None:
            entry["done"] = done
            entry["total"] = total
        return _on_progress

    def _run() -> None:
        for idx, entry in enumerate(files):
            job["current"] = idx
            entry["status"] = "running"
            try:
                entry["result"] = translate_document(
                    in_path=entry["in_path"],
                    out_path=entry["out_path"],
                    glossary_rows=params["glossary_rows"],
                    pattern_rows=params["pattern_rows"],
                    api_key=params["api_key"],
                    enable_cache=params["enable_cache"],
                    enable_qa=params["enable_qa"],
                    translation_mode=params["translation_mode"],
                    progress_callback=_make_progress(entry),
                    ui_text_overrides=params["ui_overrides"] or None,
                )
                entry["status"] = "finished"
            except Exception as e:
                # 한 파일이 실패해도 나머지는 계속 간다. 5개 중 1개 때문에
                # 나머지 4개를 처음부터 다시 번역하게 만들면 안 된다.
                # 스레드에서 새는 예외는 UI가 볼 수 없으므로 여기서 잡는다.
                entry["error"] = str(e)
                entry["status"] = "error"

        job["status"] = "finished"
        job["finished_at"] = time.monotonic()
        # 용어/패턴 목록은 번역이 끝나면 쓸 일이 없다 — 완료된 잡이
        # 수백 KB씩 붙잡고 있지 않도록 떼어낸다.
        params.pop("glossary_rows", None)
        params.pop("pattern_rows", None)
        params.pop("api_key", None)

    threading.Thread(target=_run, name=f"translate-{job_id}", daemon=True).start()
    return job_id
