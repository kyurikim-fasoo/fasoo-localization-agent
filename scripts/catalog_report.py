# -*- coding: utf-8 -*-
"""
고객사 i18n 카탈로그 진단 리포트 (CLI).

앱의 [Glossary 추출] 화면에도 같은 리포트를 내려받는 버튼이 있다. 이 스크립트는
파일이 아주 크거나 여러 고객 자료를 한꺼번에 돌릴 때 쓴다. 리포트를 만드는
로직은 services/catalog.py의 report_excel()에 있고 양쪽이 그것을 공유한다.

내용:
  ① 표기 불일치 — 같은 국문에 영문이 여럿인 항목. 제품 UI가 이미 흔들리고
     있다는 뜻이라, 번역 이전에 고객이 알아야 할 정보다.
  ② 문서 적중 — 번역할 문서에 실제로 쓰이는 라벨. 이번 번역에서 고정해야
     할 용어 목록이 곧 이것이다. (--doc 를 준 경우)

사용:
    python scripts/catalog_report.py ko.json en.json
    python scripts/catalog_report.py ko.json en.json --doc runAnalysis.mdx
    python scripts/catalog_report.py ko.json en.json --doc a.mdx -o 리포트.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import catalog as ct

NO_LIMIT = 10 ** 9


def _load(path: Path):
    return ct.parse_json(path.name, json.loads(path.read_text(encoding="utf-8")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="고객사 카탈로그 진단 리포트")
    ap.add_argument("ko_json", type=Path, help="국문 카탈로그 JSON")
    ap.add_argument("en_json", type=Path, help="영문 카탈로그 JSON")
    ap.add_argument("--doc", type=Path, default=None,
                    help="번역 대상 문서 (.mdx/.md/.docx)")
    ap.add_argument("-o", "--out", type=Path, default=Path("catalog_report.xlsx"))
    a = ap.parse_args(argv)

    for p in (a.ko_json, a.en_json, *([a.doc] if a.doc else [])):
        if not p.exists():
            print(f"파일이 없습니다: {p}", file=sys.stderr)
            return 1

    texts = None
    if a.doc is not None:
        from translator_engine import extract_korean_paragraphs
        texts = extract_korean_paragraphs(str(a.doc))

    pick = ct.pick_languages([_load(a.ko_json), _load(a.en_json)])
    res = ct.analyze(pick, term_limit=NO_LIMIT, pattern_limit=NO_LIMIT,
                     target_texts=texts)

    a.out.write_bytes(ct.report_excel(res, a.doc.name if a.doc else None))
    summary, incons, _hits = ct.report_frames(res, a.doc.name if a.doc else None)

    print(f"→ {a.out}")
    for _, r in summary.iterrows():
        print(f"   {r['항목']:<24} {r['값']}")
    if not incons.empty:
        print()
        print("   표기 불일치 상위:")
        for _, r in incons.head(8).iterrows():
            print(f"     {r['KO']:<14} {r['표기 수']}가지  "
                  f"{str(r['영문 표기들'])[:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
