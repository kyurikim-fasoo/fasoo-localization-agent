<!--
Fasoo Writing Style Guide v2.0에서 **영문 산출물에 적용되는 규칙만** 추린 것.

여기 있는 내용은 번역 문단마다 프롬프트로 전송되므로 짧게 유지한다. 길어지면
비용이 문단 수만큼 곱해지고, 모델이 매번 해당 규칙을 집어낼 가능성도 떨어진다.

여기에 **넣지 말 것**:
  - 국문 원고에만 해당하는 규칙 (문체 '-십시오/-세요', 외래어 표기, 선호
    띄어쓰기, '끌어 놓기'). 이 도구는 KO→EN이라 산출물에 영향이 없다.
  - 기계적으로 고칠 수 있는 표기 규칙 (단위·괄호·중간점 띄어쓰기, 숫자 쉼표,
    따옴표). 그건 apply_house_style_en()이 코드로 강제하므로 여기서 또
    부탁하면 프롬프트만 길어진다.

HTML 주석은 프롬프트로 보내기 전에 잘라낸다(house_style_rules 참고).
-->

- Use the present tense for product behavior, rules, and procedure results, even
  when the result happens later: "the window appears" (not "will appear"), "a
  message is displayed", "documents are converted at the scheduled time".
  Express timing with wording instead: "at the scheduled time", "after N
  minutes", "the next time". Use the future tense ONLY when you must stress that
  the result comes later than the action.
- Use Title Case for UI titles, buttons, menu items, and labels only — never for
  body prose. Lowercase prepositions under 5 letters (at, by, for, from, in, of,
  on, to), but capitalize them as the first or last word ("Applied To", "Shared
  With"). Capitalize the adverb or preposition in a phrasal verb ("Turn On
  Notifications", "Log In", "Start Up").
- Use the Oxford comma: "Windows, macOS, or Linux".
- Prefer "percentage" or "%" over "rate" and "ratio": "the percentage of
  encrypted documents", "reached 95%", "30% of all files are encrypted".
- UI interaction wording:
  - toggles: "turn on" / "turn off" (never "enable/disable", "toggle on/off").
    A toggle's own label omits them ("Auto replies", not "Turn on auto replies").
    Status text beside a toggle keeps "Enabled/Disabled" or
    "Activated/Deactivated".
  - pointer: "hover over" (never "move the mouse over").
  - mobile press-and-hold: "tap and hold" (never "touch and hold").
  - check boxes: "Select", "Deselect", "Clear"; state is "selected/unselected"
    (never "click the checkbox", "checked/unchecked").
- Do not use "/" to mean "and" or "or" — write the word ("upload and download",
  "create or add"). Keep "/" only where UI length forces it.
- Mark optional input with "(optional)" at the end of the sentence.
- File formats stay lowercase as common nouns: "Word documents", "Excel files",
  "text files" (never "TEXT files"). Extensions are lowercase (".docx"), but
  uppercase is fine in filters, drop-downs, and UI titles ("PDF file").
- Dates: "Mar 25, 2024" or "Tue, Mar 18, 2025" for user-facing screens;
  "2024-03-25 14:30:00" for logs and narrow UI. Date ranges use an en dash
  ("Mar 1–Mar 31, 2024") or "from ... to", never "~".
- Keep the tone factual and calm. Describe what happens; do not exaggerate or
  add emotion, including in error and warning messages. One sentence, one idea.
  Prefer verbs over nominalizations.
