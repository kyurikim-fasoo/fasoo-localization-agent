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


def start_job(params: dict) -> str:
    """
    번역을 데몬 스레드에서 시작하고 job_id를 돌려준다.

    params 키: in_path, out_path, output_filename, glossary_rows,
    pattern_rows, api_key, enable_cache, enable_qa, translation_mode,
    ui_overrides.
    """
    job_id = uuid.uuid4().hex
    job = {
        "status": "running",  # running | finished | error
        "done": 0,
        "total": 0,
        "result": None,
        "error": None,
        "finished_at": None,
        "params": params,
    }
    with _LOCK:
        _purge_stale()
        _JOBS[job_id] = job

    def _on_progress(done: int, total: int) -> None:
        # 단순 대입만 — 읽는 쪽은 UI 한 곳뿐이라 락이 필요 없다.
        job["done"] = done
        job["total"] = total

    def _run() -> None:
        try:
            job["result"] = translate_document(
                in_path=params["in_path"],
                out_path=params["out_path"],
                glossary_rows=params["glossary_rows"],
                pattern_rows=params["pattern_rows"],
                api_key=params["api_key"],
                enable_cache=params["enable_cache"],
                enable_qa=params["enable_qa"],
                translation_mode=params["translation_mode"],
                progress_callback=_on_progress,
                ui_text_overrides=params["ui_overrides"] or None,
            )
            job["status"] = "finished"
        except Exception as e:  # 스레드에서 새는 예외는 UI가 볼 수 없다
            job["error"] = str(e)
            job["status"] = "error"
        finally:
            job["finished_at"] = time.monotonic()
            # 용어/패턴 목록은 번역이 끝나면 쓸 일이 없다 — 완료된 잡이
            # 수백 KB씩 붙잡고 있지 않도록 떼어낸다.
            params.pop("glossary_rows", None)
            params.pop("pattern_rows", None)
            params.pop("api_key", None)

    threading.Thread(target=_run, name=f"translate-{job_id}", daemon=True).start()
    return job_id
