"""
GitHub 저장소를 데이터 보관소로 쓰기 — Streamlit Cloud 재부팅을 넘기기 위함.

Cloud의 컨테이너 디스크는 휘발성이다. 재부팅·슬립마다 작업 사본이 git에
체크인된 상태로 되돌아가므로, 글로서리·로그·사용자 데이터를 살려두려면
저장소에 올려둬야 한다. 부팅할 때 다시 내려받아 덮는다.

⚠️ 데이터는 **배포 브랜치가 아닌 별도 브랜치**에 올린다
   (GITHUB_DATA_BRANCH, 기본값 app-data). 이게 핵심이다.

   Streamlit Community Cloud는 배포 브랜치에 커밋이 올라오면 앱을 재배포한다.
   예전에는 데이터도 배포 브랜치(main)에 올렸는데, 그러면 번역 로그 한 건을
   남기는 것만으로 앱이 재시작되고 모든 사용자의 session_state가 날아갔다.
   사용자 눈에는 "Localize가 끝나니까 갑자기 첫 화면으로 돌아간다"로 보였고,
   용어를 등재하거나 사용자를 추가할 때도 같은 일이 벌어졌다.
   재부팅을 견디려고 만든 장치가 정작 재부팅을 일으키고 있었던 셈이다.

   그래서 쓰기·읽기 모두 배포되지 않는 브랜치를 쓴다. 그 브랜치는 코드와
   무관하므로 Cloud가 재배포하지 않고, 데이터는 그대로 살아남는다.

   GITHUB_BRANCH는 더 이상 데이터 경로에 쓰지 않는다 — 그 값이 배포 브랜치로
   설정돼 있으면 문제가 그대로 재현되기 때문이다.

전제:
- GITHUB_TOKEN(repo 스코프 PAT)과 GITHUB_REPO("owner/name")가 Cloud Secrets
  또는 환경변수에 있어야 한다. 없으면 전부 no-op — 로컬 개발은 영향 없다.
- 데이터 브랜치가 없으면 기본 브랜치 HEAD에서 자동으로 만든다(운영 작업 불필요).
- 한 파일 = 한 커밋. 5명이 같은 순간에 저장하면 마지막 저장이 이긴다.
- SQLite는 바이너리라 히스토리가 빨리 커진다. 주기적으로 잘라내거나, 커지면
  외부 DB(Supabase 등)로 옮기는 것을 고려.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import List, Optional

import requests


GITHUB_API = "https://api.github.com"

# 데이터 전용 브랜치의 기본 이름. 배포 브랜치와 절대 같으면 안 된다.
DEFAULT_DATA_BRANCH = "app-data"

# 데이터로 취급하는 파일 — 푸시도 이 목록, 부팅 시 내려받기도 이 목록.
DATA_FILES = ("data/glossary.db", "data/users.json", "product_config.json")

# 프로세스당 한 번만 하면 되는 일들. 모듈 전역에 두는 이유는 jobs.py와 같다 —
# Streamlit은 app.py를 매 실행마다 새 네임스페이스에 exec 하지만, import된
# 모듈은 sys.modules에 캐시되어 프로세스가 사는 동안 상태가 유지된다.
_branch_checked = False
_pulled = False


def _config() -> tuple[str, str, str]:
    """(token, repo, data_branch). 토큰이 비면 sync 비활성."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "").strip()
    branch = os.getenv("GITHUB_DATA_BRANCH", "").strip() or DEFAULT_DATA_BRANCH
    return token, repo, branch


def is_enabled() -> bool:
    token, repo, _ = _config()
    return bool(token and repo)


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _blob_sha(data: bytes) -> str:
    """git이 매기는 blob SHA-1. 원격 sha와 그대로 비교할 수 있다."""
    header = b"blob " + str(len(data)).encode() + bytes([0])
    return hashlib.sha1(header + data).hexdigest()


def _local_path_for(repo_path: str) -> Path:
    """저장소 경로 → 로컬 경로."""
    return Path(__file__).resolve().parent.parent / repo_path


def ensure_data_branch(timeout: int = 15) -> bool:
    """
    데이터 브랜치가 있는지 확인하고, 없으면 기본 브랜치 HEAD에서 만든다.

    운영자가 브랜치를 미리 만들어 두지 않아도 되게 하려는 것. 프로세스당 한 번만
    확인한다.
    """
    global _branch_checked
    token, repo, branch = _config()
    if not (token and repo):
        return False
    if _branch_checked:
        return True

    try:
        h = _headers(token)
        r = requests.get(f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}",
                         headers=h, timeout=timeout)
        if r.status_code == 200:
            _branch_checked = True
            return True
        if r.status_code != 404:
            print(f"[sync] 브랜치 조회 실패: {r.status_code} {r.text[:200]}")
            return False

        # 없으면 기본 브랜치에서 잘라낸다.
        r = requests.get(f"{GITHUB_API}/repos/{repo}", headers=h, timeout=timeout)
        if r.status_code != 200:
            print(f"[sync] 저장소 조회 실패: {r.status_code}")
            return False
        default_branch = r.json().get("default_branch") or "main"
        if default_branch == branch:
            # 이러면 데이터 커밋이 다시 재배포를 유발한다.
            print(f"[sync] 경고: 데이터 브랜치가 기본 브랜치({branch})와 같습니다. "
                  f"GITHUB_DATA_BRANCH를 다른 이름으로 설정하세요.")
            _branch_checked = True
            return True

        r = requests.get(f"{GITHUB_API}/repos/{repo}/git/ref/heads/{default_branch}",
                         headers=h, timeout=timeout)
        if r.status_code != 200:
            print(f"[sync] 기본 브랜치 ref 조회 실패: {r.status_code}")
            return False
        base_sha = r.json()["object"]["sha"]

        r = requests.post(
            f"{GITHUB_API}/repos/{repo}/git/refs", headers=h, timeout=timeout,
            json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        )
        if r.status_code in (200, 201):
            print(f"[sync] 데이터 브랜치 {branch} 생성 ({default_branch} 기준)")
            _branch_checked = True
            return True
        if r.status_code == 422:
            # 다른 프로세스가 막 만들었다 (already exists).
            _branch_checked = True
            return True
        print(f"[sync] 브랜치 생성 실패: {r.status_code} {r.text[:200]}")
        return False
    except Exception as e:
        print(f"[sync] 브랜치 준비 중 예외: {e}")
        return False


def pull_data_files(timeout: int = 20) -> List[str]:
    """
    데이터 브랜치의 파일을 로컬로 내려받는다. 프로세스당 한 번.

    데이터 브랜치가 **권위 있는 사본**이다 — 쓰기가 전부 그쪽으로만 가기
    때문이다. 배포본에 딸려 온 파일은 코드 커밋 시점의 오래된 스냅샷이므로,
    내용이 다르면 원격을 믿는다.

    실패는 삼킨다 — 못 내려받으면 배포본에 든 파일로 계속 돌아야 한다.
    내려받은 저장소 경로 목록을 돌려준다.
    """
    global _pulled
    token, repo, branch = _config()
    if not (token and repo) or _pulled:
        return []
    _pulled = True          # 실패해도 매 rerun마다 재시도하지 않는다

    if not ensure_data_branch(timeout=timeout):
        return []

    h = _headers(token)
    updated: List[str] = []
    for repo_path in DATA_FILES:
        try:
            r = requests.get(f"{GITHUB_API}/repos/{repo}/contents/{repo_path}",
                             headers=h, params={"ref": branch}, timeout=timeout)
            if r.status_code == 404:
                continue    # 아직 올라간 적 없음 — 배포본 파일을 쓴다
            if r.status_code != 200:
                print(f"[sync] GET {repo_path} 실패: {r.status_code}")
                continue
            remote_sha = r.json().get("sha")
            local = _local_path_for(repo_path)
            if local.exists() and remote_sha == _blob_sha(local.read_bytes()):
                continue    # 이미 같다

            # blobs API + raw 미디어 타입. contents API는 1MB를 넘으면 content를
            # 비워서 돌려주는데, glossary.db가 이미 1MB 경계에 붙어 있다.
            rb = requests.get(
                f"{GITHUB_API}/repos/{repo}/git/blobs/{remote_sha}",
                headers={**h, "Accept": "application/vnd.github.raw"},
                timeout=timeout,
            )
            if rb.status_code != 200:
                print(f"[sync] blob {repo_path} 실패: {rb.status_code}")
                continue
            data = rb.content
            if _blob_sha(data) != remote_sha:
                # raw 대신 JSON을 준 경우 — base64로 한 번 더 시도한다.
                try:
                    data = base64.b64decode(rb.json().get("content", ""))
                except Exception:
                    print(f"[sync] blob {repo_path} 본문 해석 실패")
                    continue
                if _blob_sha(data) != remote_sha:
                    print(f"[sync] blob {repo_path} sha 불일치 — 건너뜀")
                    continue

            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_suffix(local.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(local)      # 같은 볼륨 내 원자적 교체
            updated.append(repo_path)
        except Exception as e:
            print(f"[sync] pull {repo_path} 예외: {e}")

    if updated:
        print(f"[sync] {branch}에서 내려받음: {', '.join(updated)}")
    return updated


def push_file_to_github(
    local_path: Path,
    repo_path: str,
    commit_message: str,
    timeout: int = 15,
) -> Optional[str]:
    """
    Upload `local_path` to `repo_path` in the configured GitHub repo.

    Returns the new commit SHA on success, None on failure / disabled.
    Failures are swallowed and printed — calling code must NOT crash on
    a sync error (the user-facing save already succeeded locally).
    """
    token, repo, branch = _config()
    if not (token and repo):
        return None  # sync disabled — no-op
    if not local_path.exists():
        return None

    # 데이터 브랜치가 없으면 만든다 — 없는 브랜치에 PUT하면 422로 떨어진다.
    if not ensure_data_branch(timeout=timeout):
        return None

    api_url = f"{GITHUB_API}/repos/{repo}/contents/{repo_path}"
    headers = _headers(token)

    try:
        # 현재 파일 sha 조회 (update에 필요). 파일이 새로 생기는 경우엔 sha 없음.
        sha: Optional[str] = None
        r = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=timeout)
        if r.status_code == 200:
            sha = r.json().get("sha")
        elif r.status_code not in (404,):
            # 다른 에러는 silently skip — 사용자 흐름 막지 않음
            print(f"[sync] GET {repo_path} failed: {r.status_code} {r.text[:200]}")
            return None

        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")

        body = {
            "message": commit_message,
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            body["sha"] = sha

        r = requests.put(api_url, headers=headers, json=body, timeout=timeout)
        if r.status_code in (200, 201):
            return r.json().get("commit", {}).get("sha")
        print(f"[sync] PUT {repo_path} failed: {r.status_code} {r.text[:200]}")
        return None
    except Exception as e:
        print(f"[sync] exception: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수 — 호출 위치에서 한 줄로 쓰기 쉽게.
# ─────────────────────────────────────────────────────────────────────────────
def sync_db(commit_message: str = "Auto-update glossary DB") -> None:
    """Push the SQLite DB to GitHub. Silently no-op if sync is disabled."""
    from db.schema import DB_PATH
    push_file_to_github(DB_PATH, "data/glossary.db", commit_message)


def sync_products(commit_message: str = "Auto-update product list") -> None:
    """
    product_config.json을 GitHub에 올린다.

    Cloud의 컨테이너 디스크는 재부팅마다 git 체크아웃으로 되돌아간다. 그래서
    화면에서 추가한 제품이 파일에만 쓰이면 다음 재부팅에 사라진다 — 실제로
    "새 제품을 추가했는데 Localize 목록에 없다"는 증상이 이것이었다.
    """
    root = Path(__file__).resolve().parent.parent
    push_file_to_github(root / "product_config.json",
                        "product_config.json", commit_message)


def sync_users(commit_message: str = "Auto-update users list") -> None:
    """Push the users.json registry to GitHub."""
    from services.users import USERS_PATH
    push_file_to_github(USERS_PATH, "data/users.json", commit_message)
