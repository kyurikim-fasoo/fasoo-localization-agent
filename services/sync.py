"""
GitHub 저장소를 데이터 보관소로 쓰기 — Streamlit Cloud 재부팅을 넘기기 위함.

Cloud의 컨테이너 디스크는 휘발성이다. 재부팅·슬립마다 작업 사본이 git에
체크인된 상태로 되돌아가므로, 글로서리·로그·사용자 데이터를 살려두려면
저장소에 올려둬야 한다. 부팅할 때 다시 내려받아 덮는다.

⚠️ 데이터는 **배포 브랜치가 아닌 별도 브랜치**에 올린다
   (GITHUB_DATA_BRANCH, 기본값 app-data). 이게 핵심이다.

   Streamlit Community Cloud는 배포 브랜치에 커밋이 올라오면 앱을 재배포한다.
   예전에는 데이터도 배포 브랜치에 올렸는데, 그러면 번역 로그 한 건을 남기는
   것만으로 앱이 재시작되고 모든 사용자의 session_state가 날아갔다. 사용자
   눈에는 "Localize가 끝나니까 갑자기 첫 화면으로 돌아간다"로 보였다.
   재부팅을 견디려고 만든 장치가 정작 재부팅을 일으키고 있었던 셈이다.

데이터 브랜치를 만들 때의 함정 (실제로 데이터를 잃었던 지점):

   브랜치를 **저장소 기본 브랜치**에서 잘라내면 안 된다. 이 저장소의 기본
   브랜치(master)는 배포 브랜치(main)보다 26일 뒤처져 있었고, 그 상태로
   app-data를 만들면 낡은 glossary.db가 '권위 있는 사본'이 된다. 부팅 pull이
   그걸 내려받아 정상 DB를 덮으면 그동안 등재한 용어가 통째로 사라진다.

   그래서 (1) GITHUB_BRANCH(배포 브랜치)를 기준으로 잘라내고, (2) 그것마저
   확실하지 않으므로 **방금 만든 브랜치에서는 내려받지 않는다** — 대신 지금
   컨테이너에 있는 배포본 파일을 올려 브랜치를 맞춘다. 내려받기는 이전에
   누군가 데이터를 올려둔 브랜치에서만 한다.

   그리고 브랜치를 준비하지 못하면 **조용히 포기하지 않는다.** 예전 구현은
   그때 None을 반환해서, 저장은 성공한 듯 보이는데 아무것도 푸시되지 않고
   다음 재부팅에 전부 사라졌다. 세션이 초기화되는 불편보다 데이터가 사라지는
   것이 훨씬 나쁘므로, 그 경우엔 배포 브랜치로라도 올린다(경고를 남긴다).

전제:
- GITHUB_TOKEN(repo 스코프 PAT)과 GITHUB_REPO("owner/name")가 Cloud Secrets
  또는 환경변수에 있어야 한다. 없으면 전부 no-op — 로컬 개발은 영향 없다.
- GITHUB_BRANCH = 배포 브랜치 이름. 데이터 브랜치의 기준점으로만 쓴다.
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

# 브랜치 준비 결과.
BRANCH_EXISTS = "exists"            # 전에도 있던 브랜치 — 내려받아도 된다
BRANCH_CREATED = "created"          # 이번에 우리가 만들었다 — 내려받으면 안 된다
BRANCH_UNAVAILABLE = "unavailable"  # 준비 실패 — 배포 브랜치로 폴백

# 프로세스당 한 번만 하면 되는 일들. 모듈 전역에 두는 이유는 jobs.py와 같다 —
# Streamlit은 app.py를 매 실행마다 새 네임스페이스에 exec 하지만, import된
# 모듈은 sys.modules에 캐시되어 프로세스가 사는 동안 상태가 유지된다.
_branch_state: str = ""
_pulled = False
_last_error: str = ""
_fallback_used = False


def _config() -> tuple[str, str, str]:
    """(token, repo, data_branch). 토큰이 비면 sync 비활성."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "").strip()
    branch = os.getenv("GITHUB_DATA_BRANCH", "").strip() or DEFAULT_DATA_BRANCH
    return token, repo, branch


def is_enabled() -> bool:
    token, repo, _ = _config()
    return bool(token and repo)


def last_error() -> str:
    """가장 최근 동기화 실패 메시지. 없으면 빈 문자열.

    UI가 이 값을 보고 "저장은 됐지만 서버에 올라가지 않았다"를 알려줄 수 있다.
    조용히 잃는 것이 가장 나쁜 결과이므로 밖으로 내보낸다.
    """
    return _last_error


def fallback_active() -> bool:
    """
    데이터 브랜치를 못 써서 배포 브랜치로 올리고 있는가.

    이 상태에서는 저장이 성공하더라도(= last_error는 비어 있다) 저장할 때마다
    앱이 재배포될 수 있다. 데이터는 지키고 그 사실은 화면에 알린다.
    """
    return _fallback_used


def _fail(msg: str) -> None:
    global _last_error
    _last_error = msg
    print(f"[sync] {msg}")


def _ok() -> None:
    global _last_error
    _last_error = ""


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


def _deploy_branch() -> str:
    """
    배포 브랜치 이름(GITHUB_BRANCH). 데이터 브랜치의 **기준점**으로 쓴다.

    저장소 기본 브랜치를 쓰면 안 된다 — 기본 브랜치가 배포 브랜치보다 한참
    뒤처져 있을 수 있고, 그 낡은 데이터가 권위 있는 사본이 되어버린다.
    """
    return os.getenv("GITHUB_BRANCH", "").strip()


def _repo_default_branch(repo: str, h: dict, timeout: int) -> str:
    r = requests.get(f"{GITHUB_API}/repos/{repo}", headers=h, timeout=timeout)
    if r.status_code != 200:
        return ""
    return r.json().get("default_branch") or ""


def ensure_data_branch(timeout: int = 15) -> str:
    """
    데이터 브랜치를 준비하고 상태를 돌려준다.

    BRANCH_EXISTS / BRANCH_CREATED / BRANCH_UNAVAILABLE 중 하나.
    프로세스당 한 번만 판정한다.
    """
    global _branch_state
    token, repo, branch = _config()
    if not (token and repo):
        return BRANCH_UNAVAILABLE
    if _branch_state:
        return _branch_state

    try:
        h = _headers(token)
        r = requests.get(f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}",
                         headers=h, timeout=timeout)
        if r.status_code == 200:
            _branch_state = BRANCH_EXISTS
            return _branch_state
        if r.status_code != 404:
            _fail(f"데이터 브랜치 조회 실패: {r.status_code} {r.text[:200]}")
            _branch_state = BRANCH_UNAVAILABLE
            return _branch_state

        # 없으면 만든다. 기준은 **배포 브랜치** — 기본 브랜치는 뒤처져 있을 수 있다.
        base = _deploy_branch() or _repo_default_branch(repo, h, timeout)
        if not base:
            _fail("기준 브랜치를 알 수 없어 데이터 브랜치를 만들 수 없습니다. "
                  "GITHUB_BRANCH를 배포 브랜치 이름으로 설정하세요.")
            _branch_state = BRANCH_UNAVAILABLE
            return _branch_state
        if base == branch:
            # 데이터와 배포가 같은 브랜치 = 데이터 커밋마다 재배포된다.
            _fail(f"데이터 브랜치가 배포 브랜치({branch})와 같습니다. "
                  f"GITHUB_DATA_BRANCH를 다른 이름으로 설정하세요.")
            _branch_state = BRANCH_EXISTS   # 동작은 시킨다(기존 방식과 동일)
            return _branch_state

        r = requests.get(f"{GITHUB_API}/repos/{repo}/git/ref/heads/{base}",
                         headers=h, timeout=timeout)
        if r.status_code != 200:
            _fail(f"기준 브랜치({base}) ref 조회 실패: {r.status_code}")
            _branch_state = BRANCH_UNAVAILABLE
            return _branch_state
        base_sha = r.json()["object"]["sha"]

        r = requests.post(
            f"{GITHUB_API}/repos/{repo}/git/refs", headers=h, timeout=timeout,
            json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        )
        if r.status_code in (200, 201):
            print(f"[sync] 데이터 브랜치 {branch} 생성 ({base} 기준)")
            _branch_state = BRANCH_CREATED
            return _branch_state
        if r.status_code == 422:
            # 다른 프로세스가 막 만들었다 (already exists).
            _branch_state = BRANCH_EXISTS
            return _branch_state
        _fail(f"데이터 브랜치 생성 실패: {r.status_code} {r.text[:200]}")
        _branch_state = BRANCH_UNAVAILABLE
        return _branch_state
    except Exception as e:
        _fail(f"데이터 브랜치 준비 중 예외: {e}")
        _branch_state = BRANCH_UNAVAILABLE
        return _branch_state


def _seed_data_branch(timeout: int = 20) -> List[str]:
    """
    방금 만든 데이터 브랜치를 **지금 컨테이너에 있는 배포본**으로 맞춘다.

    잘라낸 기준 브랜치의 데이터가 배포본보다 오래됐을 수 있으므로, 내려받는
    대신 올린다. 여기서 반대로 했다가 26일 묶은 DB로 덮어써 용어를 잃었다.
    """
    seeded = []
    for repo_path in DATA_FILES:
        local = _local_path_for(repo_path)
        if not local.exists():
            continue
        if push_file_to_github(local, repo_path,
                               "seed data branch from deployed snapshot",
                               timeout=timeout):
            seeded.append(repo_path)
    if seeded:
        print(f"[sync] 데이터 브랜치 초기화 — 배포본을 올림: {', '.join(seeded)}")
    return seeded


def pull_data_files(timeout: int = 20) -> List[str]:
    """
    데이터 브랜치의 파일을 로컬로 내려받는다. 프로세스당 한 번.

    **이전부터 있던 브랜치에서만** 내려받는다. 이번에 만든 브랜치는 내용이
    배포본보다 오래됐을 수 있으므로 반대로 배포본을 올린다(_seed_data_branch).

    실패는 삼킨다 — 못 내려받으면 배포본에 든 파일로 계속 돌아야 한다.
    내려받은 저장소 경로 목록을 돌려준다.
    """
    global _pulled
    token, repo, branch = _config()
    if not (token and repo) or _pulled:
        return []
    _pulled = True          # 실패해도 매 rerun마다 재시도하지 않는다

    state = ensure_data_branch(timeout=timeout)
    if state == BRANCH_UNAVAILABLE:
        return []
    if state == BRANCH_CREATED:
        _seed_data_branch(timeout=timeout)
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
                _fail(f"GET {repo_path} 실패: {r.status_code}")
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
                _fail(f"blob {repo_path} 실패: {rb.status_code}")
                continue
            data = rb.content
            if _blob_sha(data) != remote_sha:
                # raw 대신 JSON을 준 경우 — base64로 한 번 더 시도한다.
                try:
                    data = base64.b64decode(rb.json().get("content", ""))
                except Exception:
                    _fail(f"blob {repo_path} 본문 해석 실패")
                    continue
                if _blob_sha(data) != remote_sha:
                    _fail(f"blob {repo_path} sha 불일치 — 건너뜀")
                    continue

            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_suffix(local.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(local)      # 같은 볼륨 내 원자적 교체
            updated.append(repo_path)
        except Exception as e:
            _fail(f"pull {repo_path} 예외: {e}")

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
    `local_path`를 저장소의 `repo_path`로 올린다. 성공 시 커밋 sha.

    데이터 브랜치를 준비하지 못하면 **배포 브랜치로라도 올린다.** 그쪽은 앱을
    재배포시키지만, 조용히 아무것도 올리지 않아 다음 재부팅에 데이터를 잃는
    것보다는 낫다. 실패는 삼키고 last_error()에 남긴다 — 화면의 저장 자체는
    이미 로컬 DB에 성공했으므로 호출자를 막지 않는다.
    """
    token, repo, branch = _config()
    if not (token and repo):
        return None  # sync 비활성 — no-op
    if not local_path.exists():
        return None

    state = ensure_data_branch(timeout=timeout)
    if state == BRANCH_UNAVAILABLE:
        global _fallback_used
        _fallback_used = True
        branch = _deploy_branch() or "main"
        print(f"[sync] 데이터 브랜치를 쓸 수 없어 배포 브랜치({branch})로 올립니다 — "
              f"앱이 재배포될 수 있습니다.")

    api_url = f"{GITHUB_API}/repos/{repo}/contents/{repo_path}"
    headers = _headers(token)

    try:
        # 현재 파일 sha 조회 (update에 필요). 파일이 새로 생기는 경우엔 sha 없음.
        sha: Optional[str] = None
        r = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=timeout)
        if r.status_code == 200:
            sha = r.json().get("sha")
        elif r.status_code not in (404,):
            _fail(f"GET {repo_path} failed: {r.status_code} {r.text[:200]}")
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
            _ok()
            return r.json().get("commit", {}).get("sha")
        _fail(f"PUT {repo_path} failed: {r.status_code} {r.text[:200]}")
        return None
    except Exception as e:
        _fail(f"exception: {e}")
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
