# -*- coding: utf-8 -*-
"""
고객사 i18n 카탈로그 진단 리포트.

번역을 시작하기 전에, 고객이 준 KO/EN 카탈로그 자체를 훑어 두 가지를 낸다.

  ① 표기 불일치 — 같은 국문에 영문이 여러 개인 항목.
     제품 UI가 이미 흔들리고 있다는 뜻이라, 번역 이전에 고객이 알아야 할
     정보다. Sparrow 카탈로그에서는 라벨 5,749건 중 354건(6.2%)이 걸렸다
     ('확인' → CHECK / CHECKED / Check / Checked / Confirm / OK).

  ② 문서 적중 — 번역할 문서에 실제로 쓰이는 라벨.
     카탈로그는 제품 전체라 수만 건이지만 한 문서에 쓰이는 건 100건 남짓
     이다. 이 목록이 곧 "이번 번역에서 고정해야 할 용어"다.

사용:
    python scripts/catalog_report.py ko.json en.json
    python scripts/catalog_report.py ko.json en.json --doc runAnalysis.mdx
    python scripts/catalog_report.py ko.json en.json --doc a.mdx -o 리포트.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from services import catalog as ct


def _load(path: Path):
    return ct.parse_json(path.name, json.loads(path.read_text(encoding="utf-8")))


def build(ko_path: Path, en_path: Path, doc_path: Path | None):
    pick = ct.pick_languages([_load(ko_path), _load(en_path)])

    texts = None
    if doc_path is not None:
        from translator_engine import extract_korean_paragraphs
        texts = extract_korean_paragraphs(str(doc_path))

    res = ct.analyze(pick, term_limit=10 ** 9, pattern_limit=10 ** 9,
                     target_texts=texts)
    labels = res.labels

    # ── ① 표기 불일치 ──────────────────────────────────────────────
    incons = labels[labels["후보수"] > 1].copy()
    incons = incons.rename(columns={"EN 후보": "영문 표기들", "후보수": "표기 수"})
    cols = ["KO", "표기 수", "영문 표기들", "문맥(key)", "출현"]
    if texts is not None:
        cols.insert(1, "문서빈도")
    incons = incons[[c for c in cols if c in incons.columns]]
    incons = incons.sort_values(
        ["문서빈도", "표기 수"] if texts is not None else ["표기 수", "출현"],
        ascending=False,
    ).reset_index(drop=True)

    # ── ② 문서 적중 ────────────────────────────────────────────────
    hits = None
    if texts is not None:
        hits = labels[labels["문서빈도"] > 0].copy()
        hits = hits[hits["KO"].str.len() >= ct.DOC_TERM_MIN_CHARS]
        hits["검수 필요"] = hits["후보수"].map(lambda n: "예 (표기 충돌)" if n > 1 else "")
        hits = hits[["KO", "문서빈도", "EN", "EN 후보", "검수 필요", "문맥(key)"]]
        hits = hits.sort_values("문서빈도", ascending=False).reset_index(drop=True)

    # ── 요약 ───────────────────────────────────────────────────────
    st = res.stats
    rows = [
        ("카탈로그 전체 키", st["공통키"]),
        ("고유 국문 라벨", st["라벨 고유KO"]),
        ("표기가 흔들리는 라벨", len(incons)),
        ("표기 불일치 비율",
         f"{len(incons) / max(st['라벨 고유KO'], 1) * 100:.1f}%"),
        ("문장형(패턴) 원본", st["패턴풀"]),
    ]
    if texts is not None:
        rows += [
            ("번역 대상 문서", doc_path.name),
            ("문서의 한국어 문단", len(texts)),
            ("문서에 나오는 카탈로그 라벨", len(hits)),
            ("그중 표기 충돌", int((hits["검수 필요"] != "").sum())),
        ]
    summary = pd.DataFrame(rows, columns=["항목", "값"])
    return summary, incons, hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="고객사 카탈로그 진단 리포트")
    ap.add_argument("ko_json", type=Path, help="국문 카탈로그 JSON")
    ap.add_argument("en_json", type=Path, help="영문 카탈로그 JSON")
    ap.add_argument("--doc", type=Path, default=None,
                    help="번역 대상 문서 (.mdx/.md/.docx)")
    ap.add_argument("-o", "--out", type=Path, default=Path("catalog_report.xlsx"))
    a = ap.parse_args(argv)

    for p in (a.ko_json, a.en_json, *( [a.doc] if a.doc else [] )):
        if not p.exists():
            print(f"파일이 없습니다: {p}", file=sys.stderr)
            return 1

    summary, incons, hits = build(a.ko_json, a.en_json, a.doc)

    with pd.ExcelWriter(a.out, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="요약", index=False)
        incons.to_excel(w, sheet_name="표기 불일치", index=False)
        if hits is not None:
            hits.to_excel(w, sheet_name="문서 적중 용어", index=False)

    print(f"→ {a.out}")
    for _, r in summary.iterrows():
        print(f"   {r['항목']:<24} {r['값']}")
    print()
    print("   표기 불일치 상위:")
    for _, r in incons.head(8).iterrows():
        print(f"     {r['KO']:<14} {r['표기 수']}가지  {str(r['영문 표기들'])[:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
