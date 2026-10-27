"""
Glossary 변경 이력 · 되돌리기 테스트.

되돌리기는 잘못 동작하면 데이터를 지우는 기능이다. 그래서 '되돌린다'보다
'되돌리면 안 되는 때 건드리지 않는다'를 더 많이 본다.

    python tests/test_history.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db.schema as schema

_TMP = Path(tempfile.mkdtemp()) / "glossary.db"
schema.DB_PATH = _TMP

import pandas as pd

import services.sync as sync
sync.sync_db = lambda *a, **k: None          # GitHub 푸시 차단
import services.glossary as g
g.sync_db = lambda *a, **k: None
from services import history

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


def terms():
    return g.load_terms(current_user="kim")


def editor_rows(df):
    """load_terms 결과를 편집기에서 돌아온 모양으로."""
    return df.copy()


def add(ko, en, scope="Team", user="kim", source="Glossary 관리"):
    return g.save_terms_from_dataframe(
        pd.DataFrame([{"id": None, "Scope": scope, "KO": ko, "EN": en,
                       "Product": "ALL", "DNT": False, "Case-sensitive": False,
                       "Note": "", "Status": "approved", "File": ""}]),
        view_ids=set(), current_user=user, source=source,
    )


schema.init_db()

print("[1] 추가·수정·삭제가 이력에 남는다")
add("저장", "Save")
add("취소", "Cancel")
b = history.list_batches("kim")
check("저장 2번 = 묶음 2개", len(b) == 2, str(len(b)))
check("출처 기록", set(b["source"]) == {"Glossary 관리"})

df = terms()
df.loc[df["KO"] == "저장", "EN"] = "Apply"
c = g.save_terms_from_dataframe(df, view_ids=set(df["id"]), current_user="kim")
check("바뀐 행만 '수정'으로 센다", c["updated"] == 1, str(c))
b = history.list_batches("kim")
check("수정 묶음에는 1건만", b.iloc[0]["수정"] == 1 and b.iloc[0]["추가"] == 0, str(b.iloc[0].to_dict()))
rows = history.batch_rows(b.iloc[0]["batch_id"], "kim")
check("이전/이후 표시", (rows.iloc[0]["이전"], rows.iloc[0]["이후"]) == ("Save", "Apply"),
      str(rows.to_dict("records")))
check("바뀐 항목 표시", rows.iloc[0]["바뀐 항목"] == "en", rows.iloc[0]["바뀐 항목"])

print("[2] 수정 되돌리기")
r = history.revert_batch(b.iloc[0]["batch_id"], "kim")
check("1건 되돌림", r["reverted"] == 1 and r["conflicts"] == 0, str(r))
check("원래 값 복원", terms().set_index("KO").loc["저장", "EN"] == "Save")
check("되돌리기도 이력에 남음",
      history.list_batches("kim").iloc[0]["source"].startswith("되돌리기:"))
check("되돌린 묶음 표시", bool(history.list_batches("kim").iloc[1]["reverted"]))

print("[3] 그 뒤에 또 바뀐 행은 건드리지 않는다")
df = terms()
df.loc[df["KO"] == "취소", "EN"] = "Close"
g.save_terms_from_dataframe(df, view_ids=set(df["id"]), current_user="kim")
first = history.list_batches("kim").iloc[0]["batch_id"]
df = terms()
df.loc[df["KO"] == "취소", "EN"] = "Dismiss"
g.save_terms_from_dataframe(df, view_ids=set(df["id"]), current_user="kim")
r = history.revert_batch(first, "kim")
check("충돌로 건너뜀", r == {"reverted": 0, "conflicts": 1, "denied": 0}, str(r))
check("나중 값 유지", terms().set_index("KO").loc["취소", "EN"] == "Dismiss")

print("[4] 삭제 되돌리기 — 같은 id로 다시 들어간다")
df = terms()
old_id = int(df.loc[df["KO"] == "저장", "id"].iloc[0])
g.save_terms_from_dataframe(df[df["KO"] != "저장"], view_ids=set(df["id"]), current_user="kim")
check("삭제됨", "저장" not in set(terms()["KO"]))
r = history.revert_batch(history.list_batches("kim").iloc[0]["batch_id"], "kim")
df = terms()
check("복원됨", r["reverted"] == 1 and "저장" in set(df["KO"]), str(r))
check("같은 id", int(df.loc[df["KO"] == "저장", "id"].iloc[0]) == old_id)

print("[5] 추가 되돌리기 — 지운다. 단, 같은 (KO, EN)이 이미 있으면 복원하지 않는다")
add("닫기", "Close")
b_add = history.list_batches("kim").iloc[0]["batch_id"]
history.revert_batch(b_add, "kim")
check("추가 취소", "닫기" not in set(terms()["KO"]))
add("닫기", "Close")                       # 다시 추가
undo = history.list_batches("kim").iloc[1]["batch_id"]   # 앞선 '되돌리기' 묶음
r = history.revert_batch(undo, "kim")     # 되돌리기의 되돌리기 = 재삽입 시도
check("중복이면 건너뜀", r["conflicts"] == 1 and r["reverted"] == 0, str(r))
check("중복 행이 생기지 않음", (terms()["KO"] == "닫기").sum() == 1)

print("[6] 다른 사람의 개인 항목")
add("내보내기", "Export", scope="lee", user="lee")
check("이력이 보이지 않음",
      not any("내보내기" in str(history.batch_rows(x, "kim")["한국어"].tolist())
              for x in history.list_batches("kim")["batch_id"]))
lee_batch = history.list_batches("lee").iloc[0]["batch_id"]
r = history.revert_batch(lee_batch, "kim")
check("되돌릴 수 없음", r["denied"] == 1 and r["reverted"] == 0, str(r))

print("[7] Excel 교체도 묶음 하나로 되돌릴 수 있다")
before = set(zip(terms()["KO"], terms()["EN"]))
n = g.replace_terms_from_excel(
    pd.DataFrame([{"KO": "열기", "EN": "Open", "Product": "ALL"}]),
    source_file="master.xlsx", current_user="kim",
)
check("교체됨", set(terms()["KO"]) - {"내보내기"} == {"열기"}, str(set(terms()["KO"])))
top = history.list_batches("kim").iloc[0]
check("한 묶음에 삭제+추가", top["source"] == "Excel 가져오기" and top["추가"] == 1
      and top["삭제"] >= 2, str(top.to_dict()))
r = history.revert_batch(top["batch_id"], "kim")
after = set(zip(terms()["KO"], terms()["EN"]))
check("교체 전으로 복원", after == before, f"{after ^ before}")

print("[8] 이력 기록 실패는 저장도 되돌린다 (같은 트랜잭션)")
_orig = history.record
history.record = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    add("실패", "Fail")
except RuntimeError:
    pass
history.record = _orig
check("저장되지 않음", "실패" not in set(terms()["KO"]))

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
