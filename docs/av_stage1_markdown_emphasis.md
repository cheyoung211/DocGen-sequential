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

핵심 함수 3개, 전부 `src/agents/latex_assembler.py`:

- `find_markdown_emphasis_matches(text)` (471행): 줄 단위로 `**`/`__` 쌍을 찾아 `MarkdownEmphasisMatch`
  (367행) 목록을 반환한다. 전체 CommonMark 파서가 아니라, 다음 네 가지만 명시적으로 처리한다:
  1. `$...$`/`\(...\)`/`\[...\]` 내부에서 시작하는 delimiter는 애초에 후보에서 제외 (`_find_math_spans`
     재사용).
  2. 바깥쪽 경계가 영숫자에 붙어 있으면(`x**2`) 강조 시도로 보지 않는다 — 전혀 보고하지 않고 그대로
     둔다(변환도 에러도 아님).
  3. 순차 쌍짓기(1+2번째, 3+4번째, ...)만 사용 — 같은 marker는 CommonMark에서도 자기 자신 안에 중첩될
     수 없으므로, "**a **b** c**"는 중첩이 아니라 "a "와 " c"가 각각 bold인 두 개의 분리된 span으로
     읽는다. 서로 다른 marker(`**`/`__`)가 중첩 없이 겹치면(`**foo __bar** baz__`) 양쪽 다 ambiguous로
     표시한다.
  4. span이 표 셀 경계(이스케이프되지 않은 `|`)를 넘으면 ambiguous — 한 쪽 셀에서 닫지 않은 `**`가
     다음 셀의 `**`를 닫는 것으로 잘못 읽는 사고를 막는다.
- `convert_markdown_emphasis(content, block_id)` (541행): 안전(ambiguous=False)한 span만
  `\textbf{...}`로 치환하고, 나머지는 원본 그대로 둔 채 위치·이유가 담긴 설명 문자열 목록을 반환한다.
  안전 span이 하나도 없으면 아무 것도 바뀌지 않는다 — **idempotent**(이미 변환된 내용을 다시 넣어도
  동일).
- Dunder/식별자(`__init__`) 보호는 **의도적으로 넣지 않았다**: 처음에 넣었다가 "__three__"처럼 공백
  없는 한 단어짜리 진짜 bold("__bold__" 변형)를 깨뜨리는 회귀를 테스트로 발견하고 제거했다 — 이
  스키마에는 inline-code/literal span이 따로 없어서 둘을 구문만으로 구별할 방법이 없고, 계획에도
  이런 보호를 요구하는 문구가 없다 (`find_markdown_emphasis_matches`의 docstring에 트레이드오프 명시).

## A3. 생성·복구·조립 연결

- **생성 직후 (TextAgent)**: `_validate_one_block`(text_agent.py:1190 부근, structural 체크 바로
  다음, 다른 content-quality 체크보다 먼저)에서 `convert_markdown_emphasis`를 호출해
  `content_block.content`를 **직접 변환**한다 — 이 메서드의 다른 모든 체크와 달리 의도적으로
  side-effect가 있다. 안전하게 변환되지 않은 span이 남으면 `SemanticBlockError("markdown_emphasis",
  ...)`를 내서 기존 attempt-retry 루프(같은 예산·재시도 상한)로 넘긴다 — 새 메커니즘이 아니라 기존
  block 복구 경로를 그대로 재사용.
- **title/caption**: 둘 다 LaTeX를 전혀 통과하지 않고 `_escape_latex`로만 처리되므로(Markdown 개념이
  없는 escaper), 변환이 아니라 **항상 거부**한다 — `validate_plain_title`은 title, 신규
  `validate_plain_caption`은 caption에 같은 규칙(백슬래시 명령 + Markdown 강조 둘 다 금지)을 적용한다.
  둘 다 같은 `enabled_validators["plain_title"]` 게이트를 공유한다 — `figure_content`가 두 코드를
  공유하는 기존 패턴과 동일.
- **저장 직전**: `_run_generation_loop`이 모든 block 통과 후에만 `.blocks.json`/`.draft.tex`를 쓰므로,
  변환된 `\textbf{...}` 형태가 이미 canonical artifact에 들어간다 — 별도 배선 불필요.
- **조립 전 잔여 검사 (LatexAssembler)**: `_validate_fragment`에 추가한 체크는 **검사만** 하고 절대
  변환하지 않는다 — TextAgent 쪽 검사가 비활성화됐거나(ablation) 오래된 artifact가 그대로 남아 있을 때,
  조립기가 출력 직전에 몰래 고치면 저장된 artifact·검증 evidence·PDF가 서로 어긋나게 된다는 계획의
  명시적 경고를 그대로 반영했다. `DEFAULT_ENABLED_ASSEMBLER_VALIDATORS["markdown_emphasis"]`로 독립
  토글 가능.
- **Validator adapter**: `src/core/adapters.py::_GATE_KEY_BY_CODE`에 `markdown_emphasis` →
  `markdown_emphasis`, `plain_caption` → `plain_title` 매핑 추가. 쪼개진 구조적/게이트 체크 순서 로직
  (`_short_circuited_before`)은 일반화돼 있어 추가 수정이 필요 없었다.

## A4. 회귀 및 실제 출력 확인

`test/test_markdown_emphasis.py`에 37 tests:

- 탐지기/변환기 단위 테스트: 본문 bold, 표 셀 bold, 여러 강조 구간, 짝 불일치(미종료 marker), 표 셀
  경계를 넘는 ambiguous span, 서로 다른 marker의 올바른 중첩 vs. 겹침, 기존 `\textbf{...}` 무변화,
  `x**2`류 코드 무시, `$...$`/`\[...\]` 내부 무시, idempotency.
- TextAgent 배선: 안전 변환 시 무오류, ambiguous 시 `markdown_emphasis` 에러, 토글 끄면 원문 그대로,
  title/caption 거부.
- LatexAssembler 배선: 미변환 잔여물 거부, 이미 변환된 내용 통과, 토글 끄면 통과.
- **E2E (`@unittest.skipUnless(shutil.which("tectonic"), ...)`, 이 환경에서 실제 실행됨)**: 스크립트된
  LLM으로 `TextAgent.process_node` 실제 재시도 루프(1차 시도에서 3개 block 안전 변환 + 1개 block
  ambiguous로 실패 → 2차 시도에서 그 1개만 재생성) → `LatexAssembler.assemble` → `LatexCompiler.
  compile_pdf`(tectonic)까지 수행해, 렌더링된 `.tex`에 `**`/`__`가 전혀 남지 않고 `\textbf{...}`로
  바뀌었음과 실제 PDF가 생성됨을 확인.

변경 전/후 전체 테스트 스위트 비교: 기존에 알려진, 이 revision과 무관한 57개 실패
(`test_text_agent_structural_toggle.py` 등 — [[docgen_revision_2026_10_15]] Stage 1 메모 참고)가
**정확히 동일하게** 유지됨을 `git stash`로 전/후 비교해 확인했다. 통과 테스트는 879 → 916 (새 테스트
37개 + 기존 metadata 테스트 조정).

## Stage 0 메타데이터와의 관계

Stage 0은 `AdditionalVerifierRunMetadata.markdown_emphasis_enabled`를 "Fix A가 없으니 항상 False"로
못 박았다. Fix A가 실제로 생겼으므로 그 전제가 깨졌다 — `run_full_pipeline.py::_build_run_result`가
이제 이 필드를 실제 `text_agent_enabled_validators` 스냅샷에서 읽어(`.get("markdown_emphasis",
True)`) 채운다. Fix B/C(`math_verifier_enabled`/`dependency_verifier_enabled`)는 여전히 False로
남는다 — 이 둘은 Fix A와 달리 지금 없는 production `ConstraintProvider`가 필요하다(Stage 0 문서의
"Known gap" 참고). `test_run_full_pipeline_additional_verifier_metadata.py`의 기존 "세 플래그 모두
False" 주장도 이에 맞춰 수정했고, 명시적 ablation(끄기)이 여전히 정확히 반영되는지 보는 테스트를
추가했다.

## 다음 단계

계획 §4 순서상 다음은 **2단계 (C1–C2, C3의 참조·순서 검사)** — Fix C의 요구사항 매핑과 source 문맥
연결이다.
