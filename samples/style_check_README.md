# 스타일 가이드 적용 확인용 문서

`style_check_ko.md`를 Localize에 넣고, 나온 영문이 아래와 맞는지 보면 됩니다.

문서는 **틀리기 쉬운 쪽으로 유도하도록** 썼습니다. 예를 들어 "변환이 끝나면
완료 메시지가 표시됩니다"는 영어로 옮길 때 `will be displayed`가 먼저 떠오르는
문장인데, 가이드는 현재형을 쓰라고 합니다. 그런 자리만 골라 모아놨습니다.

체크리스트를 같은 문서에 넣지 않은 이유 — 번역기는 문서 안의 기존 영문을 읽어
그 문서의 스타일 가이드를 따로 만듭니다(`extract_doc_style_guide`). 정답을
문서에 적어두면 그게 섞여 들어가 결과가 오염됩니다.

---

## A. 코드가 자동으로 고치는 것 — 틀리면 버그

이건 모델 판단과 무관하게 **항상** 맞아야 합니다. 하나라도 틀리면 알려주세요.

| 국문 | 영문에서 확인할 것 |
|---|---|
| 최대 크기는 3MB입니다 | `3 MB` — 숫자와 단위 사이 공백 |
| 1,024바이트입니다 | `1,024 bytes` — 공백 |
| USB(범용 직렬 버스) | `USB (Universal Serial Bus)` — 괄호 앞 공백 |
| FSP(Fasoo Smart Print) | `FSP (Fasoo Smart Print)` |
| 의견을 작성합니다(옵션) | `(optional)` 앞에 공백 |
| 파일 열람·다운로드·삭제 | `views · downloads · deletions` — 중간점 앞뒤 공백 |
| 최대 40000개 | `40,000` — 쉼표 |
| '완료됨'인 경우 | `"Completed,"` — 큰따옴표, 마침표·쉼표는 **따옴표 안** |

## B. 모델이 지켜야 하는 것 — 프롬프트 규칙

여기가 이번에 새로 넣은 규칙이 실제로 먹히는지 보는 자리입니다.

**시제 — 전부 현재형이어야 합니다** (가장 중요)

| 국문 | O | X |
|---|---|---|
| 창이 나타납니다 | `the window appears` | `will appear` |
| 완료 메시지가 표시됩니다 | `is displayed` | `will be displayed` |
| 예약된 시간에 변환됩니다 | `are converted at the scheduled time` | `will be converted` |
| 10분 후 삭제됩니다 | `are deleted after 10 minutes` | `will be deleted` |
| 승인하면 메일이 발송됩니다 | `an email is sent` | `will be sent` |

단, **"다음 백업이 실행될 때 해당 파일이 백업됩니다"**는 예외입니다. 동작보다
결과가 나중임을 강조하는 자리라 `will be backed up`이 맞습니다. 가이드에 그대로
나오는 예외 문장이라 일부러 넣었습니다.

**UI 동작 용어**

| 국문 | O | X |
|---|---|---|
| 토글을 켜면 / 끄면 | `turn on` / `turn off` | `enable/disable`, `toggle on/off` |
| 마우스를 올리면 | `hover over` | `move the mouse over` |
| 길게 눌러 | `tap and hold` | `touch and hold` |
| 체크박스를 선택합니다 | `Select the checkbox` | `Click the checkbox` |
| 선택을 해제하면 | `Deselect` / `Clear` | `uncheck` |
| 사용 / 사용 안 함 (토글 옆 상태) | `Enabled` / `Disabled` | — 이건 예외라 맞습니다 |

**그 외**

| 국문 | O | X |
|---|---|---|
| Windows, macOS 또는 Linux | `Windows, macOS, or Linux` | 마지막 쉼표 생략 |
| Word 문서, Excel 파일, 텍스트 파일 | `Word documents, Excel files, and text files` | `TEXT files` |
| 암호화율이 목표보다 높습니다 | `the percentage of encrypted documents` | `encryption rate` |
| 비율은 30%입니다 | `30% of all files are encrypted` | `The ratio ... is 30%` |
| 95%에 도달했습니다 | `reached 95%` | `reached a rate of 95%` |
| 정책을 생성/추가하십시오 | `create or add` | `create/add` |
| 업로드/다운로드 기능 | `upload and download` | `upload/download` |
| 3월 1일부터 3월 31일까지 | `Mar 1–Mar 31, 2024` 또는 `From Mar 1 to Mar 31` | `Mar 1 ~ Mar 31` |
| 2024-03-25 14:30:00 | 그대로 | — |
| 2025년 3월 18일(화) | `Tue, Mar 18, 2025` | — |
| `-` / `N/A` / `알 수 없음` | `-` / `N/A` / `Unknown` — 셋이 서로 섞이지 않아야 함 | — |

## C. 건드리면 안 되는 것 — 하나라도 바뀌면 버그

맨 아래 "참고" 절에 함정을 모아뒀습니다. **전부 원문 그대로** 남아야 합니다.

```
HTTP(S)                      ← HTTP (S) 로 벌어지면 안 됨
4K, 1080p                    ← 4 K, 1080 p 안 됨
print(value)                 ← print (value) 안 됨
translate_document(in_path)  ← 공백 들어가면 안 됨
page 10091                   ← page 10,091 안 됨
1.06377 units                ← 1.06,377 안 됨
192.168.10.20000             ← 쉼표 들어가면 안 됨
v2.1.0                       ← 그대로
2024년 → 2024                ← 2,024 되면 안 됨
최대 1000명                   ← 네 자리는 코드가 손대지 않음. 모델이
                                1,000으로 써도 정상입니다
```

마지막 줄의 백틱 문장(`이 문구는 백틱 안에 있지만...`)은 **번역돼야** 합니다.
백틱 안이라도 한국어 산문이면 번역하도록 돼 있습니다. 백틱 자체는 남아야 합니다.

## D. 애매한 것 — 판단이 필요합니다

**제목(`##`)의 대소문자.** 가이드의 Title Case 규칙은 "UI의 제목, 버튼"이
대상입니다. 번역기는 문서 제목을 sentence case로 정규화합니다
(`normalize_heading_text`). 지금은 `## Conversion policy settings`처럼 나올
텐데, `## Conversion Policy Settings`를 원하시면 말씀해 주세요 — 엔진 동작을
바꿔야 하는 부분이라 임의로 건드리지 않았습니다.

**볼드 UI 라벨.** `**저장**`, `**도메인 추가**`, `**자동 보안 적용**`은 Step 2의
UI 텍스트 매핑 표에 뜹니다. EN 칸을 비워두면 모델이 번역하므로 Title Case가
지켜지는지 볼 수 있고, 직접 입력하면 그 표기가 강제됩니다. **처음에는 비워두고**
돌려보세요.

## 돌리는 법

1. Localize → 제품 선택 → Step 2
2. `samples/style_check_ko.md` 업로드
3. UI 텍스트 매핑 EN 칸은 **비워둔 채로** 번역 시작
4. Step 3에서 받은 `.md`를 열어 A → B → C 순서로 확인

A가 틀리면 코드 버그입니다. B가 틀리면 프롬프트 규칙을 더 강하게 써야 합니다.
C가 틀리면 자동 교정 규칙이 너무 넓은 것이니 바로 알려주세요.
