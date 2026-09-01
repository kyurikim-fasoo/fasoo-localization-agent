"""
i18n 카탈로그 추출 테스트.

고객사마다 파일 형태가 다르므로 '어떤 모양이 와도 읽어내는가'가 핵심이다.

    python tests/test_catalog.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from services import catalog as ct

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


print("[1] 형태 감지 — 배열 + 언어 필드 (파일 2개)")
en = [{"key": "a.save", "en": "Save"}, {"key": "a.cancel", "en": "Cancel"}]
ko = [{"key": "a.save", "ko": "저장"}, {"key": "a.cancel", "ko": "취소"}]
pf_en = ct.parse_json("x-en.json", en)
pf_ko = ct.parse_json("x-ko.json", ko)
check("배열형 인식", pf_en.shape == "array" and pf_en.key_field == "key", pf_en.describe())
pick = ct.pick_languages([pf_en, pf_ko])
check("언어 자동 판정", pick.ko["a.save"] == "저장" and pick.en["a.save"] == "Save")

print("[2] 파일명에 언어 힌트가 없어도 내용으로 판정")
pick2 = ct.pick_languages([ct.parse_json("file1.json", ko), ct.parse_json("file2.json", en)])
check("한글 비율로 KO 식별", pick2.ko["a.save"] == "저장", pick2.ko_label)

print("[3] 한 파일에 양쪽 언어")
both = [{"key": "a.save", "en": "Save", "ko": "저장"},
        {"key": "a.cancel", "en": "Cancel", "ko": "취소"}]
pf = ct.parse_json("both.json", both)
check("컬럼 2개 인식", set(pf.columns) == {"en", "ko"}, str(list(pf.columns)))
p3 = ct.pick_languages([pf])
check("단일 파일에서 언어 분리", p3.ko["a.cancel"] == "취소" and p3.en["a.cancel"] == "Cancel")

print("[4] 평탄 객체 / 중첩 객체 (i18next·vue-i18n)")
flat_ko = {"a.save": "저장", "a.cancel": "취소"}
flat_en = {"a.save": "Save", "a.cancel": "Cancel"}
pf_f = ct.parse_json("ko.json", flat_ko)
check("평탄 객체 인식", pf_f.shape == "flat" and pf_f.n_keys == 2, pf_f.describe())
nested_ko = {"a": {"save": "저장", "cancel": "취소"}}
pf_n = ct.parse_json("ko.json", nested_ko)
check("중첩 객체 평탄화", pf_n.shape == "nested" and "a.save" in pf_n.columns["ko.json"],
      str(list(pf_n.columns.values())))
p4 = ct.pick_languages([ct.parse_json("1.json", nested_ko),
                        ct.parse_json("2.json", {"a": {"save": "Save", "cancel": "Cancel"}})])
check("중첩끼리도 대응", p4.ko["a.save"] == "저장" and p4.en["a.save"] == "Save")

print("[5] key 필드명이 달라도 찾는다")
odd = [{"msgid": "a.save", "value": "저장"}, {"msgid": "a.cancel", "value": "취소"}]
pf_o = ct.parse_json("odd.json", odd)
check("msgid를 key로", pf_o.key_field == "msgid", pf_o.key_field)

print("[6] 라벨 / 문장 판정")
cases = [
    ("저장", "Save", True, "짧은 라벨"),
    ("분석 대상 확장자", "Target of file extension", True, "복합 라벨"),
    ("array의 초기화 여부를 검사하지 않습니다.", "Skip array initialization checking",
     False, "한국어만 문장인 경우"),
    ("보안에 취약한 random 함수를 작성하세요", "Write insecure random function",
     False, "종결어미(마침표 없음)"),
    ("하드코딩된 인증 정보가 검출되었습니다: {0}", "Hardcoded credentials detected: {0}",
     False, "플레이스홀더 = 메시지 템플릿"),
    ("분석을 시작합니다.", "Analysis started.", False, "양쪽 다 문장"),
]
for ko_t, en_t, want, why in cases:
    check(f"{why}: {ko_t[:22]!r}", ct.is_label(ko_t, en_t) is want)

print("[7] 1:N 충돌 집계와 대표형 선택")
pick7 = ct.pick_languages([
    ct.parse_json("ko.json", {"l.a": "확인", "l.b": "확인", "r.c": "확인"}),
    ct.parse_json("en.json", {"l.a": "OK", "l.b": "OK", "r.c": "Check"}),
])
r7 = ct.analyze(pick7)
row = r7.labels.iloc[0]
check("후보수 2로 집계", int(row["후보수"]) == 2, str(row["후보수"]))
check("최빈 표기를 대표로", row["EN"] == "OK", row["EN"])
check("문맥 네임스페이스 수집", set(row["문맥(key)"].split()) == {"l", "r"}, row["문맥(key)"])

class _FakeResp:
    def __init__(self, text):
        self.output_text = text


class _FakeClient:
    """responses.create를 흉내내는 스텁."""

    def __init__(self, text):
        self._text = text
        self.calls = 0
        outer = self

        class _R:
            def create(self, **kw):
                outer.calls += 1
                outer.prompt = kw.get("input", "")
                return _FakeResp(outer._text)

        self.responses = _R()


print("[8] 기존 글로서리 대조")
pick8 = ct.pick_languages([
    ct.parse_json("ko.json", {"a": "문서", "b": "취약점 점검"}),
    ct.parse_json("en.json", {"a": "Document", "b": "Vulnerability check"}),
])
r8 = ct.analyze(pick8, existing_terms={"문서": {"file"}})
st_map = dict(zip(r8.labels["KO"], r8.labels["기존대조"]))
check("영어가 다르면 충돌", st_map["문서"] == "충돌(기존)", st_map.get("문서"))
check("없던 건 신규", st_map["취약점 점검"] == "신규", st_map.get("취약점 점검"))
_prev = dict(zip(r8.labels["KO"], r8.labels["기존 EN"]))
check("기존 표기를 원본 그대로 보관", _prev["문서"] == "file", str(_prev))
check("신규는 기존 표기 없음", _prev["취약점 점검"] == "", str(_prev))

print("[8-b] 충돌 판정은 대소문자를 무시")
r8b = ct.analyze(pick8, existing_terms={"문서": {"DOCUMENT"}})
check("대소문자만 다르면 동일",
      dict(zip(r8b.labels["KO"], r8b.labels["기존대조"]))["문서"] == "동일")

print("[8-c] SPLIT도 대체 표기를 받는다")
_split_cli = _FakeClient("[0] SPLIT|OK|화면마다 다름")
_r8c = ct.resolve_conflicts(_split_cli, ct.conflict_rows(r7.labels))
check("SPLIT에도 pick이 채워짐", _r8c["확인"]["pick"] == "OK", str(_r8c))
check("kind는 SPLIT 유지", _r8c["확인"]["kind"] == "SPLIT", str(_r8c))

print("[9] 용어 후보 필터 — '길면 용어'가 아니라 '반복되는 복합어'")
check("조사로 끝나면 제외", not ct._looks_like_term("보안 취약점 검출부터"))
check("두 어절 복합어는 포함", ct._looks_like_term("취약점 점검"))
check("영문만이면 제외", not ct._looks_like_term("SQL"))
check("한 어절은 제외(일반어가 대부분)", not ct._looks_like_term("함수"))
check("너무 길면 제외", not ct._looks_like_term("공격 대상이 될 수 있는 보안 취약점 검출"))

# 빈도가 낮은 표현은 용어로 보지 않는다
pick9 = ct.pick_languages([
    ct.parse_json("ko.json", {f"k{i}": "취약점 점검 결과" for i in range(5)} | {"z": "단발 표현"}),
    ct.parse_json("en.json", {f"k{i}": "Vulnerability check result" for i in range(5)} | {"z": "One off"}),
])
r9 = ct.analyze(pick9)
check("반복되는 복합어는 뽑힘", "취약점 점검 결과" in set(r9.terms["KO"]), str(list(r9.terms["KO"])))
check("빈도 컬럼 존재", "빈도" in r9.terms.columns, str(list(r9.terms.columns)))
check("빈도 내림차순 정렬",
      list(r9.terms["빈도"]) == sorted(r9.terms["빈도"], reverse=True))

print("[9-b] 표기 충돌 목록")
conf9 = ct.conflict_rows(r7.labels)
check("충돌만 추려짐", len(conf9) == 1 and conf9.iloc[0]["KO"] == "확인", str(len(conf9)))

print("[10] 엑셀 내보내기 — 마스터 업로드와 같은 시트/컬럼")
xlsx = ct.to_excel(r8.terms, r8.patterns, product="Sparrow")
sheets = pd.read_excel(__import__("io").BytesIO(xlsx), sheet_name=None)
check("시트 2개", set(sheets) == {"glossary", "pattern"}, str(list(sheets)))
check("glossary 컬럼",
      list(sheets["glossary"].columns) == ["KO", "EN", "Product", "DNT", "Case-sensitive", "Note"],
      str(list(sheets["glossary"].columns)))
check("pattern 컬럼", list(sheets["pattern"].columns) == ["KO", "EN", "Note"],
      str(list(sheets["pattern"].columns)))

print("[11] 오류 안내")
try:
    ct.pick_languages([pf_en])
    check("단일 언어 컬럼이면 에러", False)
except ValueError as e:
    check("단일 언어 컬럼이면 에러", "2개 이상" in str(e), str(e))
try:
    ct.parse_json("x.json", 42)
    check("지원 안 하는 최상위 타입", False)
except ValueError:
    check("지원 안 하는 최상위 타입", True)

print("[12] 실제 카탈로그 (있을 때만)")
real = sorted(ROOT.glob("watchtower-all-*.json"))
if len(real) == 2:
    files = [ct.parse_json(f.name, json.loads(f.read_text(encoding="utf-8"))) for f in real]
    res = ct.analyze(ct.pick_languages(files))
    s = res.stats
    print(f"       {s}")
    check("3만 쌍 이상 대응", s["공통키"] > 30000, str(s["공통키"]))
    check("용어 후보에 문장이 안 섞임",
          not res.terms["KO"].str.contains(r"습니다|하세요|\{0\}", regex=True).any())
    check("충돌 항목이 잡힘", s["충돌"] > 0)
else:
    print("       원본 JSON이 없어 건너뜀")

print("[13] 등재 — app.py가 만드는 행 모양 그대로 저장되는가")
# 실제 glossary.db를 건드리지 않도록 임시 DB로 돌린다.
import tempfile
import db.schema as _schema
_schema.DB_PATH = Path(tempfile.mkdtemp()) / "test.db"
from services.glossary import (  # noqa: E402
    load_terms, save_patterns_from_dataframe, save_terms_from_dataframe,
)

rows = pd.DataFrame({
    "id": None, "Scope": "Team",
    "KO": ["취약점 점검", "무한 재귀 호출"],
    "EN": ["Vulnerability check", "Infinite recursive call"],
    "Product": "Sparrow", "DNT": False, "Case-sensitive": False,
    "Note": "rule", "Status": "approved", "File": "watchtower.json",
})
c = save_terms_from_dataframe(rows, view_ids=set(), current_user="tester")
check("용어 2개 삽입", c["inserted"] == 2, str(c))
got = load_terms(current_user="tester")
check("조회됨", len(got) == 2, str(len(got)))
check("제품 분리 기록", set(got["Product"]) == {"Sparrow"}, str(set(got["Product"])))

# view_ids=set() 계약: 삽입만 하고 기존 행을 지우지 않는다
c2 = save_terms_from_dataframe(rows, view_ids=set(), current_user="tester")
check("삭제가 일어나지 않음", c2["deleted"] == 0, str(c2))

prows = pd.DataFrame({
    "id": None, "Scope": "Team",
    "KO": ["분석을 시작합니다."], "EN": ["Analysis started."],
    "Note": "msg", "Status": "approved", "File": "watchtower.json",
})
cp = save_patterns_from_dataframe(prows, view_ids=set(), current_user="tester")
check("패턴 1개 삽입", cp["inserted"] == 1, str(cp))

print("[14] 개수 상한 — 후보를 쏟아내지 않는다")
_lim_ko = {f"k{i}": f"취약점 점검 결과{i % 7}" for i in range(60)}
_lim_en = {f"k{i}": f"Vulnerability check result{i % 7}" for i in range(60)}
_lim_ko.update({f"s{i}": f"항목{i}을(를) 찾을 수 없습니다." for i in range(40)})
_lim_en.update({f"s{i}": f"Item{i} not found." for i in range(40)})
_lp = ct.pick_languages([ct.parse_json("ko.json", _lim_ko), ct.parse_json("en.json", _lim_en)])
r14 = ct.analyze(_lp, term_limit=3, pattern_limit=2)
check("용어 상한 적용", len(r14.terms) <= 3, str(len(r14.terms)))
check("패턴 상한 적용", len(r14.patterns) <= 2, str(len(r14.patterns)))
check("상한 전 개수도 보고", r14.stats["패턴풀"] > r14.stats["패턴후보"],
      f"{r14.stats['패턴풀']} vs {r14.stats['패턴후보']}")
check("문형 빈도 컬럼", "문형 빈도" in r14.patterns.columns, str(list(r14.patterns.columns)))

# 같은 어미의 문장은 한 덩어리로 묶여 대표만 남는다
check("같은 문형은 하나로", len(r14.patterns) < 40, str(len(r14.patterns)))
check("기본 상한값", (ct.DEFAULT_TERM_LIMIT, ct.DEFAULT_PATTERN_LIMIT) == (100, 30),
      f"{ct.DEFAULT_TERM_LIMIT}/{ct.DEFAULT_PATTERN_LIMIT}")

class _FakeBoom:
    class responses:
        @staticmethod
        def create(**kw):
            raise RuntimeError("network")


print("[15] 등재 전 검수 — 응답 파싱")


_items = [("취약점 점검", "Vulnerability Check"), ("무한 재귀 호출", "Infinite recursive call")]
_cli = _FakeClient("[0] vulnerability check|일반 명사는 소문자")
_rev = ct.review_entries(_cli, _items)
check("제안된 항목만 반환", set(_rev) == {0}, str(_rev))
check("제안 내용", _rev[0]["suggest"] == "vulnerability check", str(_rev[0]))
check("사유 포함", "소문자" in _rev[0]["reason"], str(_rev[0]))

check("NONE이면 빈 결과", ct.review_entries(_FakeClient("NONE"), _items) == {})
check("원본과 같은 제안은 무시",
      ct.review_entries(_FakeClient("[1] Infinite recursive call|동일"), _items) == {})
check("호출 실패해도 죽지 않음", ct.review_entries(_FakeBoom(), _items) == {})

_cli2 = _FakeClient("NONE")
ct.review_entries(_cli2, _items, kind="pattern")
check("패턴은 다른 기준 사용", "translation examples" in _cli2.prompt, _cli2.prompt[:60])

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
