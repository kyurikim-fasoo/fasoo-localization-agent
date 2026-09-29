"""
GitHub 데이터 동기화 — 배포 브랜치와 분리되었는지, 부팅 복원이 맞는지.

이 코드가 틀리면 조용히 데이터를 잃거나(복원 실패) 앱을 계속 재배포시킨다
(배포 브랜치에 커밋). 네트워크는 스텁으로 갈아끼우고 요청 내용을 직접 본다.

⚠️ pull_data_files는 실제 data/glossary.db를 덮어쓰는 함수다. 테스트에서는
   _local_path_for를 임시 폴더로 돌려 실제 파일을 건드리지 않는다.

    python tests/test_sync.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import services.sync as sy

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


class FakeResp:
    def __init__(self, status=200, json_data=None, content=b""):
        self.status_code = status
        self._json = json_data if json_data is not None else {}
        self.content = content
        self.text = str(json_data or "")

    def json(self):
        return self._json


class FakeRequests:
    """URL 패턴별 응답을 미리 정해두고, 오간 요청을 기록한다."""

    def __init__(self, routes):
        # routes: [(method, 부분문자열, FakeResp 또는 callable)]
        # ⚠️ 먼저 맞는 것이 이긴다. "/repos/own/rep"는 모든 URL의 접두사이므로
        #    구체적인 경로(/git/..., /contents/...)를 반드시 앞에 둘 것.
        self.routes = routes
        self.calls = []               # (method, url, kwargs)

    def _dispatch(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for m, frag, resp in self.routes:
            if m == method and frag in url:
                return resp(url, kw) if callable(resp) else resp
        return FakeResp(404, {})

    def get(self, url, **kw):
        return self._dispatch("GET", url, **kw)

    def post(self, url, **kw):
        return self._dispatch("POST", url, **kw)

    def put(self, url, **kw):
        return self._dispatch("PUT", url, **kw)


def reset(token="tok", repo="own/rep", data_branch=None, deploy_branch="main"):
    """모듈 전역 1회성 플래그와 환경변수를 초기화."""
    sy._branch_checked = False
    sy._pulled = False
    os.environ["GITHUB_TOKEN"] = token
    os.environ["GITHUB_REPO"] = repo
    os.environ["GITHUB_BRANCH"] = deploy_branch
    if data_branch is None:
        os.environ.pop("GITHUB_DATA_BRANCH", None)
    else:
        os.environ["GITHUB_DATA_BRANCH"] = data_branch


print("[1] 설정 — 데이터는 배포 브랜치로 가지 않는다")
reset(deploy_branch="main")
_, _, br = sy._config()
check("기본 데이터 브랜치는 app-data", br == "app-data", br)
check("GITHUB_BRANCH(main)를 데이터에 쓰지 않는다", br != "main", br)
reset(data_branch="my-data")
check("GITHUB_DATA_BRANCH로 덮어쓸 수 있다", sy._config()[2] == "my-data")
reset(token="")
check("토큰 없으면 비활성", sy.is_enabled() is False)
reset()
check("토큰·저장소 있으면 활성", sy.is_enabled() is True)

print("\n[2] blob sha — git과 같은 방식으로 계산")
check("빈 파일", sy._blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
check("b'hello'", sy._blob_sha(b"hello") == "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0")

print("\n[3] 데이터 브랜치 자동 생성")
reset()
fake = FakeRequests([
    ("GET", "/git/ref/heads/app-data", FakeResp(404)),
    ("GET", "/git/ref/heads/main", FakeResp(200, {"object": {"sha": "basesha"}})),
    ("POST", "/git/refs", FakeResp(201, {})),
    ("GET", "/repos/own/rep", FakeResp(200, {"default_branch": "main"})),
])
sy.requests = fake
check("생성 성공", sy.ensure_data_branch() is True)
_posts = [c for c in fake.calls if c[0] == "POST"]
check("refs/heads/app-data를 만든다",
      _posts and _posts[0][2]["json"]["ref"] == "refs/heads/app-data",
      str(_posts[0][2]["json"]) if _posts else "POST 없음")
check("기본 브랜치 HEAD에서 잘라낸다",
      _posts and _posts[0][2]["json"]["sha"] == "basesha")

n_before = len(fake.calls)
check("프로세스당 한 번만 확인", sy.ensure_data_branch() is True and len(fake.calls) == n_before,
      f"{len(fake.calls) - n_before}회 추가 호출")

print("\n[4] 이미 있으면 만들지 않는다")
reset()
fake = FakeRequests([("GET", "/git/ref/heads/app-data", FakeResp(200, {}))])
sy.requests = fake
check("성공", sy.ensure_data_branch() is True)
check("POST 없음", not [c for c in fake.calls if c[0] == "POST"])

print("\n[5] 데이터 브랜치가 배포 브랜치와 같으면 경고하고 넘어간다")
reset(data_branch="main")
fake = FakeRequests([
    ("GET", "/git/ref/heads/main", FakeResp(404)),
    ("GET", "/repos/own/rep", FakeResp(200, {"default_branch": "main"})),
])
sy.requests = fake
check("막지는 않는다(기존 동작 유지)", sy.ensure_data_branch() is True)
check("브랜치를 만들려 하지 않는다", not [c for c in fake.calls if c[0] == "POST"])

print("\n[6] 부팅 복원 — 다르면 내려받고, 같으면 건너뛴다")
BODY = b"remote-db-bytes"
SHA = sy._blob_sha(BODY)
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    sy._local_path_for = lambda rp: tmp / rp        # 실제 파일 보호
    (tmp / "data").mkdir()
    (tmp / "data" / "users.json").write_bytes(BODY)          # 이미 같음
    (tmp / "data" / "glossary.db").write_bytes(b"old-local")  # 다름

    reset()
    fake = FakeRequests([
        ("GET", "/git/ref/heads/app-data", FakeResp(200, {})),
        ("GET", "/contents/data/glossary.db", FakeResp(200, {"sha": SHA})),
        ("GET", "/contents/data/users.json", FakeResp(200, {"sha": SHA})),
        ("GET", "/contents/product_config.json", FakeResp(404)),
        ("GET", "/git/blobs/" + SHA, FakeResp(200, {}, content=BODY)),
    ])
    sy.requests = fake
    updated = sy.pull_data_files()
    check("다른 파일만 내려받는다", updated == ["data/glossary.db"], str(updated))
    check("내용이 원격으로 교체됨",
          (tmp / "data" / "glossary.db").read_bytes() == BODY)
    check("같은 파일은 그대로", (tmp / "data" / "users.json").read_bytes() == BODY)
    check("없는 파일(404)은 조용히 건너뛴다",
          not (tmp / "product_config.json").exists())
    check(".part 임시파일을 남기지 않는다",
          not list(tmp.rglob("*.part")), str(list(tmp.rglob("*.part"))))
    check("데이터 브랜치에서만 읽는다",
          all(c[2].get("params", {}).get("ref", "app-data") == "app-data"
              for c in fake.calls if "/contents/" in c[1]))

    n_before = len(fake.calls)
    check("프로세스당 한 번만 복원", sy.pull_data_files() == []
          and len(fake.calls) == n_before)

    print("\n[7] 복원 실패는 삼킨다 — 배포본 파일로 계속 돌아야 한다")
    reset()
    (tmp / "data" / "glossary.db").write_bytes(b"deployed")
    fake = FakeRequests([
        ("GET", "/git/ref/heads/app-data", FakeResp(200, {})),
        ("GET", "/contents/data/glossary.db", FakeResp(200, {"sha": SHA})),
        ("GET", "/git/blobs/" + SHA, FakeResp(500, {})),   # blob 실패
    ])
    sy.requests = fake
    try:
        got = sy.pull_data_files()
        check("예외를 던지지 않는다", True)
        check("아무것도 갱신하지 않는다", got == [], str(got))
        check("기존 파일을 망가뜨리지 않는다",
              (tmp / "data" / "glossary.db").read_bytes() == b"deployed")
    except Exception as e:
        check("예외를 던지지 않는다", False, repr(e))

    print("\n[8] sha가 안 맞는 응답은 쓰지 않는다")
    reset()
    fake = FakeRequests([
        ("GET", "/git/ref/heads/app-data", FakeResp(200, {})),
        ("GET", "/contents/data/glossary.db", FakeResp(200, {"sha": SHA})),
        ("GET", "/git/blobs/" + SHA, FakeResp(200, {}, content=b"corrupted")),
    ])
    sy.requests = fake
    got = sy.pull_data_files()
    check("갱신하지 않는다", got == [], str(got))
    check("파일 유지", (tmp / "data" / "glossary.db").read_bytes() == b"deployed")

print("\n[9] 푸시 — 배포 브랜치가 아니라 데이터 브랜치로 간다")
reset(deploy_branch="main")
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "x.bin"
    f.write_bytes(b"payload")
    fake = FakeRequests([
        ("GET", "/git/ref/heads/app-data", FakeResp(200, {})),
        ("GET", "/contents/data/x.bin", FakeResp(404)),
        ("PUT", "/contents/data/x.bin", FakeResp(201, {"commit": {"sha": "newsha"}})),
    ])
    sy.requests = fake
    sha = sy.push_file_to_github(f, "data/x.bin", "msg")
    check("커밋 sha 반환", sha == "newsha", str(sha))
    _puts = [c for c in fake.calls if c[0] == "PUT"]
    check("branch=app-data로 올린다",
          _puts and _puts[0][2]["json"]["branch"] == "app-data",
          str(_puts[0][2]["json"].get("branch")) if _puts else "PUT 없음")
    check("main으로 올리지 않는다",
          all(c[2].get("json", {}).get("branch") != "main"
              for c in fake.calls if c[0] == "PUT"))

print("\n[10] sync 비활성이면 아무 요청도 하지 않는다")
reset(token="")
fake = FakeRequests([])
sy.requests = fake
check("pull no-op", sy.pull_data_files() == [])
check("ensure no-op", sy.ensure_data_branch() is False)
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "y.bin"
    f.write_bytes(b"z")
    check("push no-op", sy.push_file_to_github(f, "data/y.bin", "m") is None)
check("네트워크 호출 0회", not fake.calls, str(fake.calls))

for _k in ("GITHUB_TOKEN", "GITHUB_REPO", "GITHUB_BRANCH", "GITHUB_DATA_BRANCH"):
    os.environ.pop(_k, None)

print()
if failures:
    print(f"FAILED {len(failures)}건: {failures}")
    raise SystemExit(1)
print("ALL PASS")
