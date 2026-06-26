"""
Re-import the master Excel into the glossary database.

This is the Phase-1 / option-A workflow: every time the team's Wrapsody
glossary changes, decrypt the file, save the .xlsx anywhere reachable, and
run this script with the path. The DB becomes an exact mirror of that
snapshot (existing rows are wiped first — "교체 모드").

Usage:
    python -m scripts.import_master_excel "C:\\path\\to\\master.xlsx"
    python -m scripts.import_master_excel "C:\\path\\to\\master.xlsx" --product fss

Sheet detection:
    - The glossary sheet is the first one whose name contains 'glossary' or '용어'.
    - The pattern sheet is the first one whose name contains 'pattern' or '패턴'.
    - Fallback: first sheet → glossary, second sheet → pattern.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# scripts/ 폴더에서 직접 실행해도 상위 모듈을 import할 수 있게 path 보강
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.glossary import (  # noqa: E402
    replace_patterns_from_excel,
    replace_terms_from_excel,
)


def _pick_sheet(sheets: dict[str, pd.DataFrame], keywords: list[str]) -> tuple[str, pd.DataFrame] | None:
    for name, df in sheets.items():
        lname = name.lower()
        if any(k in lname for k in keywords):
            return name, df
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Master Excel → glossary DB importer")
    parser.add_argument("excel_path", help="Path to the master xlsx file")
    parser.add_argument(
        "--product",
        default="",
        help="If set, only rows with Product = this value or 'ALL' are imported (terms only).",
    )
    args = parser.parse_args()

    src = Path(args.excel_path)
    if not src.exists():
        print(f"[ERROR] file not found: {src}", file=sys.stderr)
        return 1

    sheets = pd.read_excel(src, sheet_name=None)
    print(f"[OK] loaded {len(sheets)} sheets from {src.name}: {list(sheets.keys())}")

    # ── glossary ──────────────────────────────────────────────────────
    pick = _pick_sheet(sheets, ["glossary", "용어", "사전"])
    if pick is None and len(sheets) >= 1:
        pick = list(sheets.items())[0]
        print(f"[WARN] no sheet named glossary/용어 — falling back to first sheet '{pick[0]}'")
    if pick is not None:
        name, df = pick
        n = replace_terms_from_excel(df, source_file=f"{src.name} [{name}]", product_filter=args.product or None)
        filt = f" (product='{args.product}'+ALL)" if args.product else ""
        print(f"[OK] terms: imported {n} rows from sheet '{name}'{filt}")

    # ── pattern ───────────────────────────────────────────────────────
    pick = _pick_sheet(sheets, ["pattern", "패턴"])
    if pick is None and len(sheets) >= 2:
        pick = list(sheets.items())[1]
        print(f"[WARN] no sheet named pattern/패턴 — falling back to second sheet '{pick[0]}'")
    if pick is not None:
        name, df = pick
        n = replace_patterns_from_excel(df, source_file=f"{src.name} [{name}]")
        print(f"[OK] patterns: imported {n} rows from sheet '{name}'")

    print("\nDone. DB file: data/glossary.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
