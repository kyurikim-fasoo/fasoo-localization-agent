"""
하우스 스타일(Fasoo Writing Style Guide v2.0) — 기계적 표기 규칙.

이 규칙들은 번역 결과를 **자동으로 고친다**. 그래서 "고쳐야 할 것을 고치는가"
보다 **"건드리면 안 되는 것을 그대로 두는가"**가 더 중요하다. 기술 매뉴얼에서
코드나 버전 문자열을 망가뜨리면, 가이드를 안 지키는 것보다 나쁘다.

[2]가 그 안전망이다. 규칙을 넓히려 할 때 여기부터 돌려 볼 것.

    python tests/test_house_style.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import markdown_format as mf
import translator_engine as te

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


def fix(text: str) -> str:
    return te.apply_house_style_en(text)


def same(name: str, text: str) -> None:
    """그대로 유지돼야 하는 입력."""
    out = fix(text)
    check(name, out == text, f"{text!r} -> {out!r}")


def becomes(name: str, src: str, want: str) -> None:
    out = fix(src)
    check(name, out == want, f"{src!r} -> {out!r} (기대 {want!r})")


print("[1] 가이드가 고치라는 것")
becomes("영문 단위 띄어쓰기", "The maximum size is 3MB.", "The maximum size is 3 MB.")
becomes("bytes 띄어쓰기", "It is 1,024bytes.", "It is 1,024 bytes.")
becomes("영문 축약어 괄호", "USB(Universal Serial Bus)", "USB (Universal Serial Bus)")
becomes("선택 입력 괄호", "Write a comment(optional).", "Write a comment (optional).")
becomes("여러 낱말 괄호", "the config(see note)", "the config (see note)")
becomes("중간점 앞뒤 띄어쓰기", "Edited·Today·2 MB", "Edited · Today · 2 MB")
becomes("기간 물결표 → en dash", "Mar 1 ~ Mar 31, 2024", "Mar 1–Mar 31, 2024")
becomes("ISO 기간 물결표", "2024-03-01 ~ 2024-03-31", "2024-03-01–2024-03-31")
becomes("곡선 따옴표 → 직선", "“from the internet”", '"from the internet"')
becomes("구두점을 따옴표 안으로", 'like "the internet".', 'like "the internet."')
becomes("쉼표도 안으로", 'Type "yes", then press Enter.',
        'Type "yes," then press Enter.')
becomes("다섯 자리 숫자 쉼표", "The limit is 40000 files.",
        "The limit is 40,000 files.")

print("\n[2] ★안전망★ 건드리면 안 되는 것")
same("코드 호출", "print(value)")
same("코드 호출(밑줄 인자)", "translate_document(in_path)")
same("관용 약어", "HTTP(S)")
same("연도 네 자리", "Released in 2024.")
same("네 자리 수량은 손대지 않는다", "Up to 1000 users")
same("페이지 번호", "page 10091")
same("소수점 이하", "The value is 1.06377 units.")
same("버전 문자열", "Version v20250101 build")
same("목록에 없는 단위", "Supports 4K and 1080p output.")
same("대략을 뜻하는 물결표", "Set it to ~100 users.")
same("국문 중간점은 붙여 쓴다", "파일 열람·다운로드·삭제")
same("슬래시는 의도적으로 미적용", "Use the Lock / Unlock button.")
same("IP 주소", "Connect to 192.168.10.20000")
same("마커 토큰", "⟦HL:yellow⟧Warning⟦/HL⟧")
same("번호가 큰 마커 토큰", "⟦X99999⟧ and ⟦C12⟧")

print("\n[3] 마크다운 인라인 코드는 봉인된 채로 지나간다")
marked, seals, urls = mf.seal_inline("Set `max=3MB` in the config(see note).")
restored = mf.unseal_inline(fix(marked), seals, urls)
check("코드 안 3MB 보존", "`max=3MB`" in restored, restored)
check("코드 밖 괄호는 고쳐짐", "config (see note)" in restored, restored)

print("\n[4] 빈 입력·멱등성")
check("빈 문자열", fix("") == "")
_once = fix("Size 3MB, comment(optional), Edited·Today")
check("두 번 적용해도 같다", fix(_once) == _once, f"{_once!r} -> {fix(_once)!r}")

print("\n[5] 프롬프트 규칙 파일")
rules = te.house_style_rules()
check("규칙이 로드된다", bool(rules))
check("HTML 주석은 잘라낸다", "<!--" not in rules)
check("프롬프트에 넣기엔 충분히 짧다", len(rules) < 4000, f"{len(rules)}자")
for keyword in ("present tense", "Title Case", "Oxford comma", "percentage",
                "turn on", "hover over", "tap and hold", "optional"):
    check(f"'{keyword}' 규칙 포함", keyword in rules)
check("국문 전용 규칙은 넣지 않는다",
      "십시오" not in rules and "끼어 놓기" not in rules)

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
