"""
여러 문서를 한 번에 로컬라이즈할 때 필요한 병합 로직.

app.py에 인라인으로 두지 않는 이유: Streamlit의 AppTest는 파일 업로드를
지원하지 않아서, 업로더 뒤에 붙은 코드는 UI 테스트로 검증할 수가 없다.
순수 함수로 떼어내면 여기만 단위 테스트로 못 박을 수 있다.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def merge_bold_terms(
    per_file: Sequence[Tuple[str, Iterable[Tuple[str, str]]]],
    glossary_lookup: Dict[str, str],
    preloaded: Dict[str, str],
) -> Tuple[List[dict], Dict[str, int]]:
    """
    파일별로 뽑은 볼드 라벨을 **KO 기준 하나의 표**로 합친다.

    파일마다 따로 입력받으면 같은 용어를 여러 번 적어야 하고, 무엇보다 문서
    간 표기가 갈린다 — 표기 통일이 이 도구의 존재 이유이므로 그건 곧 실패다.
    그래서 KO가 같으면 한 행으로 합치고, 어느 파일에서 왔는지는 '출처' 열에
    모아 보여준다. 입력한 영문은 배치 전체에 적용된다.

    EN 기본값의 우선순위는 단일 파일 때와 같다: 로그 > 글로서리 > 빈칸.
    맥락은 **처음 만난 파일**의 것을 쓴다(파일 순서 = 업로드 순서).

    Args:
        per_file: [(파일명, [(KO, 맥락), ...]), ...]
        glossary_lookup: KO -> EN (글로서리)
        preloaded: KO -> EN (이전 로그에서 불러온 매핑, 글로서리보다 우선)

    Returns:
        (rows, source_counts)
        rows: [{"KO (Bold)", "EN (입력)", "맥락", "출처"}, ...] — 등장 순서 유지
        source_counts: {"글로서리": n, "로그": n} — 자동 매칭 요약용
    """
    merged: Dict[str, dict] = {}
    files_of: Dict[str, List[str]] = {}
    counts = {"글로서리": 0, "로그": 0}

    for name, terms in per_file:
        for ko, ctx in terms:
            if ko in merged:
                if name not in files_of[ko]:
                    files_of[ko].append(name)
                continue
            if ko in preloaded:
                en = preloaded[ko]
                counts["로그"] += 1
            elif ko in glossary_lookup:
                en = glossary_lookup[ko]
                counts["글로서리"] += 1
            else:
                en = ""
            merged[ko] = {"KO (Bold)": ko, "EN (입력)": en, "맥락": ctx}
            files_of[ko] = [name]

    rows = []
    for ko, row in merged.items():
        row["출처"] = ", ".join(files_of[ko])
        rows.append(row)
    return rows, counts


def unique_output_names(names: Sequence[str]) -> List[str]:
    """
    산출물 이름이 겹치지 않게 만든다.

    서로 다른 폴더의 동명 파일을 함께 올리면 산출물 경로가 같아져, 나중에
    끝난 쪽이 앞선 결과를 조용히 덮어쓴다. 뒤에 _2, _3을 붙여 비켜준다.
    """
    from pathlib import Path

    used: set = set()
    out: List[str] = []
    for name in names:
        final = name
        if final in used:
            stem, suffix = Path(name).stem, Path(name).suffix
            k = 2
            while f"{stem}_{k}{suffix}" in used:
                k += 1
            final = f"{stem}_{k}{suffix}"
        used.add(final)
        out.append(final)
    return out
