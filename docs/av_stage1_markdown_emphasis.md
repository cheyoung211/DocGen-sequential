# AV Stage 1: Markdown 강조 구문 규제 (Fix A)

이 문서는 Notion "SNU / NCIA / Personal Research / Multimodal Agents / **Revision (Additional
verifiers)**" 계획의 **1단계** (A1–A4, Fix A) 산출물이다. [[docs/av_stage0_scope_and_contracts.md]]가
확정한 공통 설계(P1–P4) 위에서, 본문·표 셀·목록·caption에 남는 `**bold**`/`__bold__`가 PDF에 그대로
출력되는 문제를 생성 시점에 검출·변환하고, 변환 불가능한 경우에만 기존 block 복구로 넘긴다.

코드 기준: 브랜치 `additional_verifier`, Stage 0 round 4 commit `34be680` 위에서 작업.

## 범위

- `src/agents/latex_assembler.py`: 공통 탐지·변환 함수(`MarkdownEmphasisMatch`,
  `find_markdown_emphasis_matches`, `convert_markdown_emphasis`), plain-text 필드 규칙
  (`validate_plain_title`/`validate_plain_caption`, 공유 헬퍼 `_reject_plain_text_markup`), 조립 전
  잔여 검사(`_validate_fragment`에 추가), `DEFAULT_ENABLED_ASSEMBLER_VALIDATORS["markdown_emphasis"]`.
- `src/agents/text_agent.py`: `SYSTEM_PROMPT`에 `<Text emphasis>` 규칙 추가, `_validate_one_block`에
  생성 직후 변환·검사 연결, `DEFAULT_ENABLED_VALIDATORS["markdown_emphasis"]`,
  `REMEDIATION_HINTS["markdown_emphasis"/"plain_caption"]`.
- `src/core/adapters.py`: `_GATE_KEY_BY_CODE`에 `markdown_emphasis`/`plain_caption` 매핑 추가.
- `scripts/run_full_pipeline.py` + `src/evaluation/schemas.py`: Stage 0에서 항상 `False`였던
  `AdditionalVerifierRunMetadata.markdown_emphasis_enabled`를 실제 `enabled_validators` 스냅샷에서
  읽어오도록 수정 (Fix A가 실제로 켜졌다는 사실을 반영 — 아래 "Stage 0 메타데이터와의 관계" 참고).
- `test/test_markdown_emphasis.py` (신규, 37 tests): 탐지·변환 단위 테스트, TextAgent/LatexAssembler
  배선 테스트, tectonic을 이용한 실제 compile end-to-end 테스트 1건.
- `test/test_run_full_pipeline_additional_verifier_metadata.py`: 메타데이터 플래그 갱신에 맞춰 기존
  주장(Fix A도 아직 없다는 전제) 수정 + 신규 ablation 테스트 1건.

**범위에서 명시적으로 제외**: Fix B(계산·수식 검증)·Fix C(cross-section dependency) — 2단계 이후.
`run_full_pipeline.py`/LangGraph controller로의 새 실행-경로 연결(§4 I1)도 대상이 아니다 — Fix A는
TextAgent/LatexAssembler가 이미 모든 실제 실행에서 무조건 호출하는 `enabled_validators` 메커니즘에
얹혀 있어서, 이 단계만큼은 새 배선이 필요 없었다 (아래 참고).

## A1. 표현 규칙과 적용 범위

- `text_agent.py::SYSTEM_PROMPT`에 `<Text emphasis>` 절 추가: 본문에서 Markdown 강조 금지,
  `\textbf{...}` 사용 지시. `<Cross-references and titles>` 절도 title/caption에 Markdown 강조까지
  금지하도록 확장.
- 허용 범위: 수식(`$...$`/`\(...\)`/`\[...\]`)과 "코드처럼 보이는" 패턴(`x**2`)은 전혀 건드리지 않는다 —
  탐지 단계에서 애초에 매치되지 않는다(아래 A2).
- 기존 Markdown pipe table/목록 입력은 그대로 유지 — 이번 작업은 "Markdown 금지"가 아니라 "강조 구문만
  규제"다.

## A2. 탐지와 제한된 변환

핵심 함수 전부 `src/agents/latex_assembler.py`. 아래는 **라운드 2 리뷰 반영 후** 최종 설계다 — 이전
버전들의 설계·버그는 "라운드 1 리뷰"/"라운드 2 리뷰" 절에 따로 기록했다.

- `_literal_or_math_mask(text)`: 전체 텍스트(여러 줄 포함) 기준으로 한 번에 계산하는 제외 마스크.
  `$...$`/`\(...\)`/`\[...\]` 수식은 상태를 줄마다 리셋하지 않고 텍스트 전체에 걸쳐 추적하고,
  `\verb<delim>...<delim>`(리터럴)과 `\texttt{...}`(이 프로젝트의 code/식별자 표기 규약) 내부도
  제외한다. 백슬래시로 이스케이프된 `\$`/`\{`/`\}`/`\\`는 (`_is_escaped_literal`) 수식 토글이나
  `\texttt{}` 중괄호 깊이 계산 어느 쪽에도 영향을 주지 않는 리터럴 한 쌍으로 먼저 소비된다 — 라운드 2
  리뷰 수정(아래 참고).
- `find_markdown_emphasis_matches(text, *, apply_literal_math_mask=True)`: 줄 단위로 `*`/`_` 문자의
  **연속된 run**을 스캔한다(위 마스크에 걸리는 위치는 애초에 run에 들어가지 않음). run 길이가 정확히
  2면 평범한 marker 후보, 1이면 이 프로젝트가 지원하지 않는 강조라 무시, **3 이상이면 통째로 하나의
  ambiguous match로 직접 보고하고 marker 짝짓기 풀에는 전혀 들어가지 않는다** — `***bold***` 같은
  복합(bold+italic) marker를 라운드 2 리뷰가 잡아냈다(아래 참고). 길이-2 run들은 marker별로
  **1-depth 스택**(`_find_matches_for_marker`)으로 짝짓는다: 이미 열린 pending이 있으면 무조건 닫고
  (그래서 같은 marker는 절대 자기 자신 안에 중첩될 수 없다), pending이 없을 때는 **여는 쪽 바로 앞
  글자가 영숫자가 아닐 때만** 새로 연다. "닫는 쪽 뒤에 오는 글자"는 더 이상 조건에 넣지 않는다 —
  한국어처럼 공백 없이 조사·어미가 바로 붙는 언어의 정상적인 표현을 "코드처럼 보임"으로 오인하지 않기
  위해서다. 줄 끝까지 안 닫힌 pending은 ambiguous(미종료)로 보고한다. 서로 다른 marker(`**`/`__`)가
  겹치면 — 완전히 포함되는 중첩이든 부분적으로만 겹치든 — 양쪽 다 ambiguous로 표시한다
  (`_flag_cross_marker_overlaps`). span의 inner가 비어 있거나 앞뒤에 공백이 있거나 표 셀 경계
  (이스케이프되지 않은 `|`)를 넘거나 **중괄호 짝이 안 맞으면**(`_has_balanced_braces` — 라운드 2 리뷰
  수정, 아래 참고) 그 pair 자체가 ambiguous다(`_classify_pair`). `apply_literal_math_mask=False`를
  주면 위 마스크를 통째로 건너뛴다 — title/caption처럼 실제 LaTeX 수식을 전혀 통과하지 않는 plain-text
  필드용(`_reject_plain_text_markup`이 사용 — 아래 참고).
- `convert_markdown_emphasis(content, block_id, *, matches=None)`: 안전(ambiguous=False)한 span만
  `\textbf{...}`로 치환하고, 나머지는 원본 그대로 둔 채 위치·이유가 담긴 설명 문자열 목록을 반환한다.
  **idempotent**. `matches`를 미리 넘기면(비용 측정을 이미 호출자가 했을 때) 다시 스캔하지 않는다.
- Dunder/식별자(`__init__`) 보호는 넣지 않았다: "__three__"처럼 공백 없는 한 단어짜리 진짜
  bold("__bold__" 변형)를 깨뜨리는 회귀가 있어서다 — 다만 `\texttt{__init__}`처럼 명시적으로 literal로
  표시된 경우는 위 `_literal_or_math_mask`가 별도로 보호한다.

## A3. 생성·복구·조립 연결

- **생성 직후, 검사 (TextAgent)**: `_validate_one_block`의 markdown_emphasis 체크는 **순수 읡기 전용**
  이다 — `find_markdown_emphasis_matches`로 **남아 있는 모든** span(안전하든 ambiguous든)을 확인하고,
  `content_block`은 절대 건드리지 않는다(라운드 1 리뷰로 뒤집힌 설계). **라운드 2 리뷰 수정**: 예전에는
  ambiguous span만 실패 처리했는데, "변환 가능하다"는 사실이 "지금 이 artifact가 규칙을 충족한다"는
  뜻은 아니다 — 변환 단계를 거치지 않는 직접 호출자(바로 다음 항목의 adapter)에게는 안전-but-미변환
  span도 명백한 위반이다. 지금은 남은 span이 하나라도 있으면
  `SemanticBlockError("markdown_emphasis", ...)`를 내서 기존 attempt-retry 루프로 넘긴다.
- **생성 직후, 변환 (TextAgent)**: 실제 안전-span 변환은 `_validate_blocks_against_layout`에서 **명시적으로,
  검사보다 먼저** 일어난다 — 블록마다 `find_markdown_emphasis_matches`로 한 번 스캔하고 그 결과를
  `convert_markdown_emphasis(..., matches=...)`에 넘겨 이중 스캔 없이 변환한다. 내용이 실제로 바뀐
  경우에만 `content_block.content`를 갱신하지만, **변환 여부와 무관하게 매 블록마다**
  `persist_markdown_emphasis_pass`(`src/agents/latex_assembler.py`, phase=`"generation"`)로
  원본·unified diff(바뀐 경우만)·변환본(바뀐 경우만, 아니면 `None`)·safe/ambiguous 매치 수·
  `rule_version`·transform latency를 `{node_id}.markdown_emphasis_conversion_N.json`에 남긴다 —
  **라운드 2 리뷰 수정**: 예전에는 `converted != original`일 때만 기록이 생겨서, 변환할 게 없는 블록이나
  전부 ambiguous라 안 바뀐 블록의 비용이 전혀 기록되지 않았다(아래 참고). `regenerate_blocks`도 같은
  `_run_generation_loop`를 거치므로 block 복구 경로도 동일하게 적용된다.
- **title/caption**: 둘 다 LaTeX를 전혀 통과하지 않고 `_escape_latex`로만 처리되므로(Markdown 개념이
  없는 escaper), 변환이 아니라 **항상 거부**한다 — `validate_plain_title`은 title, 신규
  `validate_plain_caption`은 caption에 같은 규칙(백슬래시 명령 + Markdown 강조 둘 다 금지)을 적용한다.
  둘 다 같은 `enabled_validators["plain_title"]` 게이트를 공유한다 — `figure_content`가 두 코드를
  공유하는 기존 패턴과 동일. **라운드 2 리뷰 수정**: 공유 헬퍼 `_reject_plain_text_markup`이 내부에서
  `find_markdown_emphasis_matches`를 호출할 때 `apply_literal_math_mask=False`를 넘긴다 — title/
  caption은 진짜 LaTeX 수식을 전혀 거치지 않으므로 bare `$`를 수식 시작으로 "보호"하면 안 된다(아래
  참고).
- **저장 직전**: `_run_generation_loop`이 모든 block 통과 후에만 `.blocks.json`/`.draft.tex`를 쓰므로,
  변환된 `\textbf{...}` 형태가 이미 canonical artifact에 들어간다 — 별도 배선 불필요.
- **조립 전 잔여 검사 (LatexAssembler)**: `_validate_fragment`에 추가한 체크는 **검사만** 하고 절대
  변환하지 않는다. `content`뿐 아니라 block의 `title`/`caption`도 같이 검사하고(라운드 1 리뷰 #5 참고),
  `_render_figure`에서는 planner가 작성한 figure 자체의 caption(`FigureRegistryEntry.caption`/
  `FigurePlacement.caption`)도 같은 규칙으로 거부한다 — 둘 다 `_escape_latex`로만 렌더링되는
  "실제 최종 출력에 쓰이는 필드"이기 때문이다. `DEFAULT_ENABLED_ASSEMBLER_VALIDATORS["markdown_emphasis"]`
  로 독립 토글 가능. **라운드 2 리뷰 수정**: 두 체크 모두 선택적 `sections_dir`/`node_id` 파라미터
  (기본 `None`, 이 클래스의 `event_logger` 파라미터와 동일한 관례)를 받아, 실제 호출 경로
  (`_validate_semantic_blocks`/`_render_section`)에서는 `phase="assembler_residual_check"`/
  `"assembler_figure_caption_check"`로 자신의 비용도 `persist_markdown_emphasis_pass`에 기록한다 —
  예전에는 이 조립 단계의 검사 비용이 TextAgent 쪽과 달리 전혀 기록되지 않았다.
- **Validator adapter**: `src/core/adapters.py::_GATE_KEY_BY_CODE`에 `markdown_emphasis` →
  `markdown_emphasis`, `plain_caption` → `plain_title` 매핑 추가. 쪼개진 구조적/게이트 체크 순서 로직
  (`_short_circuited_before`)은 일반화돼 있어 추가 수정이 필요 없었다 — 라운드 2에서도 마찬가지로
  이 파일은 전혀 바뀌지 않았다(`_validate_one_block`이 순수한 FAIL/PASS 신호만 주면 adapter의 일반화된
  매핑 로직이 알아서 올바르게 동작한다).

## A4. 회귀 및 실제 출력 확인

`test/test_markdown_emphasis.py`(70 tests) + `test/core/test_adapters.py`(adapter 순수성/FAIL 판정
2건):

- 탐지기/변환기 단위 테스트: 본문 bold, 표 셀 bold, 여러 강조 구간, 짝 불일치(미종료 marker), 표 셀
  경계를 넘는 ambiguous span, 서로 다른 marker의 중첩/겹침(둘 다 ambiguous), 기존 `\textbf{...}`
  무변화, `x**2`류 코드 무시(및 뒤에 오는 진짜 bold를 방해하지 않음 확인), 한국어 조사가 닫는 marker
  뒤에 바로 붙는 경우, `\verb`/`\texttt{}`/여러 줄 수식 보존, `$...$`/`\[...\]` 내부 무시, idempotency,
  이스케이프된 `\$`/`\{`/`\}`가 뒤쪽 강조를 가리지 않음, 복합 marker run(`***`/`****`)과 중괄호 불균형
  span이 ambiguous로 처리됨, 위반 메시지가 실제 원본 부분 문자열을 보여줌(marker+inner+marker 이중
  래핑 없이).
- TextAgent 배선: **남은 span이 있으면(안전하든 ambiguous든) 항상 실패**, 이미 변환된 내용은 통과,
  토글 끄면 통과, title/caption 거부(bare `$` 뒤에 오는 강조 포함). `_validate_blocks_against_layout`의
  실제 변환 + provenance 기록이 **항상**(변환 없음/전부 ambiguous라 변환 실패하는 경우 포함) 남음,
  `rule_version`/`phase`/safe·ambiguous 매치 수 필드 확인.
- LatexAssembler 배선: content/title/caption 잔여물 각각 거부, 이미 변환된 내용 통과, 토글 끄면 통과,
  `sections_dir`를 주면 잔여 검사 비용이 영향(성공/실패 모두) 기록됨. 별도 클래스로 figure 자체 caption
  거부/통과 + 비용 기록.
- Adapter: `TextAgentValidatorAdapter.check()`가 markdown_emphasis를 검사해도
  `content_block.content`가 바뀌지 않음(순수성), 안전하지만 미변환인 span은 FAIL, 이미 변환된 내용은
  PASS.
- **E2E (`@unittest.skipUnless(shutil.which("tectonic"), ...)`, 이 환경에서 실제 실행됨)**: 스크립트된
  LLM으로 `TextAgent.process_node` 실제 재시도 루프(1차 시도에서 3개 block 안전 변환, 한국어 조사
  포함 + 1개 block ambiguous로 실패 → 2차 시도에서 그 1개만 재생성) → provenance 파일 3개 이상 생성
  확인 → `LatexAssembler.assemble` → `LatexCompiler.compile_pdf`(tectonic)까지 수행해, 렌더링된
  `.tex`에 `**`/`__`가 전혀 남지 않고 `\textbf{...}`로 바뀌었음과 실제 PDF가 생성됨을 확인.

변경 전/후 전체 테스트 스위트 비교: 기존에 알려진, 이 revision과 무관한 57개 실패
(`test_text_agent_structural_toggle.py` 등 — [[docgen_revision_2026_10_15]] Stage 1 메모 참고)가
**정확히 동일하게** 유지됨을 `git stash`로 전/후 비교해 확인했다(실패한 테스트 목록까지 diff로 동일함을
확인 — 개수만 같은 게 아니다). 통과 테스트는 879 → 934(라운드 1) → 952(라운드 2).

## Stage 0 메타데이터와의 관계

Stage 0은 `AdditionalVerifierRunMetadata.markdown_emphasis_enabled`를 "Fix A가 없으니 항상 False"로
못 박았다. Fix A가 실제로 생겼으므로 그 전제가 깨졌다 — `run_full_pipeline.py::_build_run_result`가
이제 이 필드를 실제 `text_agent_enabled_validators` 스냅샷에서 읽어(`.get("markdown_emphasis",
True)`) 채운다. Fix B/C(`math_verifier_enabled`/`dependency_verifier_enabled`)는 여전히 False로
남는다 — 이 둘은 Fix A와 달리 지금 없는 production `ConstraintProvider`가 필요하다(Stage 0 문서의
"Known gap" 참고). `test_run_full_pipeline_additional_verifier_metadata.py`의 기존 "세 플래그 모두
False" 주장도 이에 맞춰 수정했고, 명시적 ablation(끄기)이 여전히 정확히 반영되는지 보는 테스트를
추가했다.

## Stage 1, 라운드 1 리뷰 (2026-11-03)

최초 구현을 실제 재현 입력으로 검토한 사용자 리뷰가 7건의 실제 버그를 찾았다 ([[docgen_stage_review_rigor]]
패턴에 정확히 들어맞는, 테스트가 못 잡은 종류의 버그들). 모두 코드로 고쳤다:

1. **중첩 강조 변환이 내용을 손상시킴.** `"**foo __bar__ baz**"` → `"\textbf{foo __bar__ baz}baz**"`.
   원인: 겹치는 두 span을 각각 독립적으로 안전하다고 판정한 뒤, 안쪽 span을 먼저 치환하고 바깥쪽 span을
   "원본 텍스트에서 미리 캡처해둔" `inner`/`end` 인덱스로 다시 이어붙였다 — 안쪽 치환으로 문자열 길이가
   이미 바뀐 뒤였으므로 꼬였다. 수정: 중첩은(마커 종류 무관) 항상 ambiguous — 독립적으로 변환하지
   않고 둘 다 block 복구로 넘긴다(`_flag_cross_marker_overlaps`). 이 버그 자체가 "중첩은 멋대로 변환
   하지 말라"는 계획 문구를 코드가 실제로 지키지 않고 있었다는 뜻이었다.
2. **순차 짝짓기 때문에 코드형 패턴이 뒤쪽 진짜 bold의 짝을 흐트러뜨림.** `"x**2 and **important**."`
   에서 아무 것도 변환되지 않았다 — `x**2`의 `**`가 `**important**`의 *여는* `**`와 잘못 짝지어졌다.
   수정: 짝짓기 전에 "열 수 있는지"(앞 글자가 영숫자가 아닌지)를 **개별 occurrence 단위로** 먼저
   판정하고, pending이 없는데 열 수 없는 occurrence는 아예 짝짓기 풀에 들어가지 않는다
   (`_find_matches_for_marker`). `"**foo **bar** baz**"`도 같은 사용자 리포트의 사례였는데, 이건
   "조용히 통과"가 아니라 ambiguous로 보고되도록 고쳤다(수정 전: 양쪽 다 과도하게 넓은 "앞 또는 뒤가
   영숫자면 전부 무시" 규칙에 걸려 둘 다 조용히 버려졌다).
3. **"뒤에 영숫자가 붙으면 무시" 규칙이 한국어 조사·어미를 놓침.** `"**중요**합니다."`가 변환되지
   않았다 — 닫는 `**` 바로 뒤에 오는 글자가 한글(영숫자로 취급됨)이라는 이유만으로 통째로 무시됐다.
   수정: "여는 marker 앞"만 검사하고 "닫는 marker 뒤"는 더 이상 조건에 넣지 않는다 — 공백 없이
   조사/어미가 붙는 언어의 정상적인 표현을 코드 패턴으로 오인하지 않기 위함. `x**2`류 보호는 "열 수
   없음" 판정만으로 여전히 유지된다(#2 참고).
4. **`\verb`/`\texttt{}`/여러 줄 수식이 보존되지 않음.** `\verb|**bold**|`의 리터럴 내용이 바뀌고,
   `\texttt{__init__}`이 `\texttt{\textbf{init}}`으로 깨지고, 여러 줄 `\[ ... \]` 수식은 줄마다 상태가
   리셋돼 중간 줄이 전혀 보호되지 않았다. 수정: 새 `_literal_or_math_mask`가 텍스트 전체(여러 줄
   포함) 기준으로 수식 상태를 추적하고, `\verb`와 `\texttt{}` 인수를 통째로 제외 마스크에 넣는다.
5. **Validator adapter가 검사 중 내용을 바꾸고 옛 버전에 PASS를 붙임.** `_validate_one_block`의
   markdown_emphasis 체크가 `content_block.content`를 직접 변환하는 side effect를 갖고 있었는데,
   `TextAgentValidatorAdapter.check()`는 이 메서드를 직접 감싸는 "순수 검사기"로 문서화돼 있었다 —
   그래서 저장된 artifact 버전(`b1@1`)에 대한 PASS 판정이 실제로는 그 버전이 아니라 몰래 바뀐 내용에
   대한 판정이었다. 수정: 변환을 `_validate_one_block`에서 완전히 빼내 `_validate_blocks_against_
  layout`(실제 생성 경로만 거치는, adapter는 호출하지 않는 지점)으로 옮겼다 — `_validate_one_block`은
   이제 이 메서드의 다른 모든 체크처럼 완전히 순수하다. Controller 전체 연결은 여전히 후속 단계.
6. **caption 검사 범위가 실제 최종 렌더링을 다 못 덮음.** 새 caption 검사는 `ContentBlock.caption`만
   봤는데, 실제 figure caption은 `FigureRegistryEntry`/`FigurePlacement`라는 완전히 다른(planner가
   작성하는) 필드에서 온다. 조립기의 잔여 검사도 `content`만 보고 `title`/`caption`은 아예 안 봤다.
   수정: `_validate_fragment`에 title/caption 검사를 추가하고, `_render_figure`에도 같은 규칙
   (`validate_plain_caption`)을 연결했다.
7. **성공한 자동 변환의 원본·diff·비용 기록이 전혀 없었음.** 1차 시도에서 바로 변환·성공하는 흔한
   경우, 변환된 `.blocks.json`/`.draft.tex`만 남고 원본이나 diff는 어디에도 남지 않았다 —
   `_persist_failed_attempt`는 실패한 시도에만 동작해서 이 경우를 커버하지 못했다. 수정: 새
   `TextAgent._persist_markdown_emphasis_conversion`이 `_validate_blocks_against_layout`에서 실제로
   내용이 바뀐 블록마다 원본·unified diff·변환본·transform latency를
   `{node_id}.markdown_emphasis_conversion_N.json`으로 남긴다.

전체 재검증: 7가지 모두 사용자가 제시한 정확한 재현 문자열로 직접 확인(`convert_markdown_emphasis`에
그대로 입력해 출력 비교), `test/test_markdown_emphasis.py`(53 tests, 전면 재작성) +
`test/core/test_adapters.py`(adapter 순수성 테스트 1건 추가). `git stash`로 전/후 비교해 기존 57개
무관 실패가 동일하게 유지됨을 재확인(통과: 879 → 934).

## Stage 1, 라운드 2 리뷰 (2026-11-03)

라운드 1에서 고친 설계를 다시 실제 재현 입력으로 검토한 사용자 리뷰가 추가로 4건의 실제 버그/누락을
찾았다. 모두 코드로 고쳤다:

1. **[P1] Adapter가 변환되지 않은 금지 강조에 여전히 PASS를 반환함.** `b1@1 = "This is **important**."`
   에 대해 검사 후에도 내용이 안 바뀐 채로(`**important**`가 그대로 남은 채) adapter가 PASS를 반환했다
   — assembler는 당연히 같은 내용을 FAIL시킨다. 원인: `_validate_one_block`이 `ambiguous=True`인
   후보만 오류로 처리했다 — "안전하게 변환될 수 있다"는 사실이 "지금 이 artifact가 규칙을 충족한다"는
   뜻은 아니다. 실제 생성 경로는 검사 전에 이미 변환을 하므로, 순수 검사 함수 입장에서는 남아 있는
   모든 금지 강조(안전하든 ambiguous든)가 FAIL이어야 맞다. 수정: `_validate_one_block`은 이제
   `find_markdown_emphasis_matches`의 결과가 하나라도 있으면 실패한다. 이 입력의 PASS를 정답으로
   고정하던 adapter 테스트도 FAIL로 뒤집었고, 이미 변환된 내용(`\textbf{...}`)은 여전히 PASS임을 보는
   테스트를 새로 추가했다.
2. **[P1] 새 문맥 마스크가 escape를 잘못 처리해 뒤쪽 강조를 놓침.** `"The price is \$5 and
   **important**."`가 생성·조립 검사를 모두 통과하고 별표가 남은 채 출력됐다 — `\$`의 `$`를 (이스케이프
   무시하고) 수식 시작으로 오인해 이후 전체를 제외 마스크에 넣었다. `"\texttt{\{} then **bold**."`도
   같은 근본 원인으로 깨졌다 — `\texttt{}`의 중괄호 깊이 계산이 `\{`(이스케이프된 리터럴 중괄호)를 진짜
   미매칭 group-open으로 세어, 그 뒤 두 번째 실제 닫는 `}`를 못 찾고 줄 끝까지 전부 `\texttt{}` 인수로
   간주했다. plain-text caption(`"Cost $5 **discount**"`)도 같은 이유로 통과했는데, 여기는 애초에 수식
   제외 규칙 자체를 적용하면 안 되는 필드다 — title/caption은 진짜 LaTeX 수식을 전혀 거치지 않고
   `_escape_latex`만 거치므로 `$`도 그냥 일반 문자로 렌더링된다. 수정: `_literal_or_math_mask`에
   `_is_escaped_literal`(백슬래시로 이스케이프된 `$`/`{`/`}`/`\`는 리터럴 한 쌍으로 먼저 소비, 어떤
   상태도 바꾸지 않음)을 추가해 메인 루프와 `\texttt{}` 중괄호 추적 양쪽에 적용. `find_markdown_
   emphasis_matches`에 `apply_literal_math_mask=False` 옵션을 추가하고, `_reject_plain_text_markup`
   (title/caption 공통 경로)이 이를 사용 — 백슬래시 자체는 이미 그 앞에서 통째로 거부되므로(title/
   caption에 LaTeX 명령이 있으면 그 자체로 에러), 수식 마스크가 보호해 줄 게 원래 아무것도 없는
   필드다.
3. **[P2] 자동 변환의 안전 판정이 아직 너무 넓음.** `"***bold***"`가 `"\textbf{*bold}*"`로 변환되고
   검사를 통과했다 — 2글자 창(window) 스캔이 3글자 이상 연속된 marker run을 "`**` + `*bold` + `**` +
   `*`"로 잘못 쪼갰다. `"**text { brace**"`는 `"\textbf{text { brace}"`로 변환된 뒤 조립 중 중괄호
   불균형 오류를 냈다 — inner에 짝 안 맞는 `{`가 그대로 들어가면서 `\textbf{...}`로 감싸면 전체 괄호
   구조가 깨진다. 수정: 같은 marker 문자가 3개 이상 연속되면(run) 통째로 ambiguous match로 바로
   보고하고 marker 짝짓기 풀에는 전혀 넣지 않는다(`find_markdown_emphasis_matches`의 run-스캔 루프).
   `_classify_pair`에 `_has_balanced_braces` 검사를 추가해, inner의 중괄호가 짝이 안 맞으면(이스케이프된
   `\{`/`\}`는 제외하고 계산) 그 pair도 ambiguous로 분류한다. 둘 다 block 복구로 넘어가며, 자동 변환은
   더 이상 시도하지 않는다. (부수적으로, 위반 메시지가 `marker+inner+marker`를 이중으로 이어붙여
   `"***bold***"`류를 `"'*******'"`처럼 알아보기 힘들게 표시하던 것도 실제 원본 부분 문자열을 그대로
   보여주도록 고쳤다.)
4. **[P2] 성공한 자동 변환 외 다른 모든 검사/변환 경로의 비용·버전 기록이 빠짐.** 원본·변환본·diff·
   `transform_latency_s`는 `converted != original`일 때만 기록됐다 — 그래서 (a) 변환할 게 없는 블록,
   (b) 강조가 전부 ambiguous라 안 바뀐 블록, (c) 조립 전 잔여 검사(`LatexAssembler._validate_fragment`/
   `_render_figure`)의 비용이 전혀 기록되지 않았다. 변환 기록에는 어떤 규칙 버전이 그 판정을 내렸는지
   알 수 있는 필드도 없었다. 수정: 공유 함수 `persist_markdown_emphasis_pass`(`latex_assembler.py`)를
   새로 만들어 — (1) `MARKDOWN_EMPHASIS_RULE_VERSION`(현재 3; 히스토리는 코드 주석에 기록)을 매 기록에
   찍고, (2) `_validate_blocks_against_layout`이 **변환 여부와 무관하게 매 블록마다** `phase=
   "generation"`으로 기록하도록 바꾸고(실패로 끝나도 기록은 먼저 남도록 `finally`로 감쌈), (3)
   `_validate_fragment`/`_render_figure`에 선택적 `sections_dir`/`node_id` 파라미터(기본 `None`, 기존
   `event_logger` 관례와 동일 — 직접 호출하는 단위 테스트는 그대로 둠)를 추가해 `_validate_semantic_
   blocks`/`_render_section`이라는 실제 호출 경로에서 `phase="assembler_residual_check"`/
   `"assembler_figure_caption_check"`로 조립 쪽 비용도 기록하게 했다. `safe_match_count`/
   `ambiguous_match_count` 필드도 추가했다.

전체 재검증: 4가지 모두 사용자가 제시한 정확한 재현 입력으로 직접 확인, `test/test_markdown_emphasis.py`
(70 tests, 신규 17건) + `test/core/test_adapters.py`(신규 1건 + 기존 1건 판정 뒤집음). `git stash`로
전/후 비교해 기존 57개 무관 실패가 **실패 목록까지 정확히 동일하게** 유지됨을 재확인(통과: 934 → 952).

## 다음 단계

계획 §4 순서상 다음은 **2단계 (C1–C2, C3의 참조·순서 검사)** — Fix C의 요구사항 매핑과 source 문맥
연결이다.
