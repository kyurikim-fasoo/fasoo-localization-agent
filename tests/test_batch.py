"""
여러 문서 동시 로컬라이즈 — 병합 로직과 배치 잡.

Streamlit의 AppTest는 파일 업로드를 지원하지 않아 업로더 뒤의 코드를 UI로는
검증할 수 없다. 그래서 순수 로직은 services/batch.py로 떼어내 여기서 못 박고,
잡의 순차 실행·부분 실패는 실제 스레드를 돌려 확인한다.

    python tests/test_batch.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services import batch
import services.jobs as jobs

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


print("[1] 볼드 라벨 병합 — KO가 같으면 한 행, 출처는 모은다")
rows, counts = batch.merge_bold_terms(
    [
        ("a.md", [("워크그룹", "…워크그룹을 만들고…"), ("내부 사용자", "…내부 사용자…")]),
        ("b.md", [("워크그룹", "…다른 맥락의 워크그룹…"), ("파일 전송", "…파일 전송…")]),
        ("c.docx", [("워크그룹", "…또 다른…")]),
    ],
    glossary_lookup={"워크그룹": "Workgroup"},
    preloaded={"파일 전송": "File Transfer"},
)
by_ko = {r["KO (Bold)"]: r for r in rows}
check("중복 KO는 한 행", len(rows) == 3, str([r["KO (Bold)"] for r in rows]))
check("출처 파일이 모인다", by_ko["워크그룹"]["출처"] == "a.md, b.md, c.docx",
      by_ko["워크그룹"]["출처"])
check("한 파일만 나온 것은 그 파일만", by_ko["내부 사용자"]["출처"] == "a.md")
check("맥락은 처음 만난 파일 것", "만들고" in by_ko["워크그룹"]["맥락"],
      by_ko["워크그룹"]["맥락"])
check("등장 순서 유지",
      [r["KO (Bold)"] for r in rows] == ["워크그룹", "내부 사용자", "파일 전송"],
      str([r["KO (Bold)"] for r in rows]))

print("\n[2] EN 기본값 우선순위 — 로그 > 글로서리 > 빈칸")
check("글로서리에서 채움", by_ko["워크그룹"]["EN (입력)"] == "Workgroup")
check("로그에서 채움", by_ko["파일 전송"]["EN (입력)"] == "File Transfer")
check("없으면 빈칸", by_ko["내부 사용자"]["EN (입력)"] == "")
check("자동 매칭 건수", counts == {"글로서리": 1, "로그": 1}, str(counts))

# 로그와 글로서리에 같은 KO가 있으면 로그가 이긴다 (단일 파일 때와 동일).
rows2, counts2 = batch.merge_bold_terms(
    [("a.md", [("저장", "…")])],
    glossary_lookup={"저장": "Save"},
    preloaded={"저장": "Apply"},
)
check("로그가 글로서리를 이긴다", rows2[0]["EN (입력)"] == "Apply", rows2[0]["EN (입력)"])
check("로그로 센다", counts2 == {"글로서리": 0, "로그": 1}, str(counts2))

print("\n[3] 빈 입력")
rows3, counts3 = batch.merge_bold_terms([], {}, {})
check("파일이 없으면 빈 목록", rows3 == [] and counts3 == {"글로서리": 0, "로그": 0})
rows4, _ = batch.merge_bold_terms([("a.md", [])], {}, {})
check("볼드가 없으면 빈 목록", rows4 == [])

print("\n[4] 산출물 이름 충돌 회피 — 덮어쓰기 방지")
out = batch.unique_output_names(["a_en.md", "a_en.md", "b_en.md", "a_en.md"])
check("같은 이름에 번호가 붙는다",
      out == ["a_en.md", "a_en_2.md", "b_en.md", "a_en_3.md"], str(out))
check("전부 서로 다르다", len(set(out)) == len(out))
check("확장자 유지", all(o.endswith(".md") for o in out), str(out))
out2 = batch.unique_output_names(["x_en.docx", "x_en_2.docx", "x_en.docx"])
check("이미 _2가 있으면 _3으로",
      out2 == ["x_en.docx", "x_en_2.docx", "x_en_3.docx"], str(out2))

print("\n[5] 배치 잡 — 순차 실행, 한 파일이 실패해도 나머지는 계속")
FIX = str(ROOT / "tests" / "fixtures" / "runAnalysis.mdx")
order = []
overlap = {"max": 0, "now": 0}
_orig = jobs.translate_document


def _fake(in_path=None, out_path=None, progress_callback=None, **kw):
    overlap["now"] += 1
    overlap["max"] = max(overlap["max"], overlap["now"])
    order.append(Path(out_path).name)
    if progress_callback:
        progress_callback(2, 4)
    time.sleep(0.05)
    overlap["now"] -= 1
    if "fail" in Path(out_path).name:
        raise RuntimeError("의도된 실패")
    return {"input_tokens": 1, "cached_tokens": 0, "output_tokens": 2,
            "total_tokens": 3, "paragraphs_translated": 4}


jobs.translate_document = _fake
try:
    files = [
        {"source_name": n, "in_path": FIX,
         "out_path": str(ROOT / "outputs" / f"_batch_{n}"), "output_filename": n}
        for n in ("one.mdx", "fail.mdx", "three.mdx")
    ]
    jid = jobs.start_job({
        "files": files, "glossary_rows": [], "pattern_rows": [], "api_key": "k",
        "enable_cache": True, "enable_qa": False, "translation_mode": "",
        "ui_overrides": {},
    })
    deadline = time.time() + 15
    job = jobs.get_job(jid)
    while job["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
    check("배치가 끝난다", job["status"] == "finished", job["status"])
    check("동시에 한 개만 돈다 (순차)", overlap["max"] == 1, f"최대 {overlap['max']}개")
    check("업로드 순서대로", order == ["_batch_one.mdx", "_batch_fail.mdx",
                                 "_batch_three.mdx"], str(order))
    check("성공/실패가 파일별로 갈린다",
          [f["status"] for f in job["files"]] == ["finished", "error", "finished"],
          str([f["status"] for f in job["files"]]))
    check("실패 사유 보관", job["files"][1]["error"] == "의도된 실패",
          str(job["files"][1]["error"]))
    check("성공 파일은 결과 보관",
          job["files"][0]["result"]["paragraphs_translated"] == 4)
    check("실패해도 진행률은 100%", jobs.batch_progress(job) == 1.0,
          str(jobs.batch_progress(job)))
    check("메모리 정리 — 용어/키는 떼어낸다",
          "glossary_rows" not in job["params"] and "api_key" not in job["params"],
          str(sorted(job["params"].keys())))
finally:
    jobs.translate_document = _orig

print("\n[6] 진행률 — 진행 중 파일은 문단 비율만 얹는다")
mk = lambda st_, d, t: {"status": st_, "done": d, "total": t}
check("아무것도 시작 전 0%",
      jobs.batch_progress({"files": [mk("pending", 0, 0)] * 4}) == 0.0)
check("1/4 완료 = 25%",
      jobs.batch_progress({"files": [mk("finished", 4, 4)] + [mk("pending", 0, 0)] * 3})
      == 0.25)
check("1개 완료 + 1개 절반 = 37.5%",
      jobs.batch_progress({"files": [mk("finished", 4, 4), mk("running", 2, 4)]
                                    + [mk("pending", 0, 0)] * 2}) == 0.375)
check("총 문단을 모르는 동안은 더하지 않는다",
      jobs.batch_progress({"files": [mk("finished", 4, 4), mk("running", 0, 0)]}) == 0.5)
check("파일이 없으면 0%", jobs.batch_progress({"files": []}) == 0.0)

for _p in (ROOT / "outputs").glob("_batch_*"):
    _p.unlink(missing_ok=True)

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
