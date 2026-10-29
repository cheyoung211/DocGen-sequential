# AV Stage 0: 범위 정리와 공통 계약 확정 (P1–P4)

이 문서는 Notion "SNU / NCIA / Personal Research / Multimodal Agents / **Revision (Additional
verifiers)**" (작성 기준 2026-10-28) 계획의 **0단계** (P1–P4, 전체 범위와 공통 설계) 산출물이다. 이 revision은
별도로 진행 중인 **Revision (Graph artifacts)** (브랜치 `graph_artifact`, 0–6단계, `docs/stage1_*.md` ~
`docs/stage6_*.md`)의 artifact graph·controller 기반 위에 Markdown 강조 규제(Fix A)·계산·수식 verifier(Fix
B)·cross-section dependency 관리(Fix C) 세 가지를 추가한다. 파일명은 기존 `stageN_*.md`와의 혼동을 피하기
위해 `av_` 접두사를 쓴다.

> **범위**: 이 문서 + `src/evaluation/schemas.py`의 `AdditionalVerifierRunMetadata` + 그 실제 배선
> (`scripts/run_full_pipeline.py`) + `src/agents/text_agent.py`의 복구 전 스냅샷 보존 + `src/core/
> controller.py`의 미해결 집계 보강이 전체다. Fix A/B/C의 실제 검사 로직은 전혀 구현되지 않았다 (1~3단계의
> 일). `markdown_emphasis_enabled`/`math_verifier_enabled`/`dependency_verifier_enabled`는 지금 모두
> `False`이며, 이는 placeholder가 아니라 **현재 실제 상태**다.

코드 기준: `graph_artifact`에서 분기한 브랜치 `additional_verifier`,
commit [`278e748`](https://github.com/cheyoung211/DocGen-sequential/commit/278e7488a5ecc6c91ffc3fb4e7ce091e9f5df63b).

> **2026-10-28 수정**: 첫 버전이 P1–P4를 "기존 메커니즘으로 이미 충족"이라고 설명한 부분 중 **P2·P3·P4
> 각각에 실제로는 틀린 설명이 있었다** — 사용자 리뷰가 4건 모두에서 구체적 반례를 짚었다. 아래 각 절은
> 수정된 설명이다. 틀렸던 원인은 공통적으로 같다: "이 메커니즘이 존재한다"를 "이 메커니즘이 지금 상황에
> 실제로 연결되어 있다"와 혼동한 것 — 바로 이 프로젝트의 리뷰 역사([[docgen_stage_review_rigor]])가 반복해서
> 경고하는 패턴이다.

> **2026-10-29 수정 (2차 리뷰)**: 10-28 수정 자체도 **완전하지 않았다** — P1은 TextAgent 한 agent만
> 식별했고(Planner·Assembler 누락), P2는 호출 간(cross-call) 보존만 다뤄 **같은 호출 내부의 실패한
> attempt**(특히 첫 attempt)는 여전히 안 남았고, P3의 고침 자체가 `FinishCall(reason="resolved")`에
> 비해 있는 `unresolved_artifact_ids`를 만들 수 있는 새 모순을 **만들어냈다**. 세 건 모두 실제 코드로
> 고쳤다 (아래 각 절).

## P1. 범위·버전 기록

- **기존 결과 보존**: `outputs/`, `outputs_b4_workshop/`, Notion "Gold Labeling" 페이지는 전혀 건드리지
  않았다.
- `AdditionalVerifierRunMetadata` (`src/evaluation/schemas.py`)에 다음을 기록한다:
  - `markdown_emphasis_enabled`/`math_verifier_enabled`/`dependency_verifier_enabled`: 모두 `False`
    (아직 아무 verifier도 없음).
  - `code_commit`: 실행 시점 `git rev-parse HEAD` (`scripts/run_full_pipeline.py::_current_code_commit`,
    `@lru_cache`, 실패 시 `None`으로 degrade).
  - **`git_dirty`** *(수정 후 추가)*: `git status --porcelain`의 결과. `code_commit`만으로는 "실제 사용된
    prompt·설정"을 식별할 수 없다 — working tree에 커밋되지 않은 수정(특히 prompt를 반복 수정하는 동안)이
    있으면 `code_commit`이 가리키는 커밋과 실제 실행에 쓰인 코드가 달라진다. 이 필드는 그 상황을 명시적으로
    구분 가능하게 한다 (`_git_is_dirty`, `@lru_cache`, 실패 시 `None` — "깨끗함 확인됨"과 "확인 불가"를
    구분).
  - **`text_agent_prompt_hash`** *(수정 후 추가)*: 실제 `TextAgent.SYSTEM_PROMPT` 문자열의 SHA-256 앞
    16자 (`_text_agent_prompt_hash`). git 상태와 독립적으로 "실제 사용한 prompt"를 식별한다 — working
    tree가 dirty여도 이 값은 정확하다.
  - **`planner_prompt_hash`** *(2차 리뷰 후 추가)*: 2차 리뷰가 지적한 대로, 10-28 수정은 TextAgent 한
    agent만 식별했다 — Planner는 `PLANNER_STRATEGY_PROMPT` + 조건부 3개 조각(`PLANNER_NO_IMAGES_PROMPT`/
    `PLANNER_REQUIRED_SECTIONS_PROMPT`/`PLANNER_LEGACY_OUTPUT_FORMAT_PROMPT`)을 매 호출마다 다르게
    조합하므로(`allow_figures`/`required_sections` 플래그에 따라), 단일 `SYSTEM_PROMPT`가 없다. 4개 고정
    템플릿 문자열을 이어붙여 해싱(`_planner_prompt_hash`)해 "planner 프롬프트의 코드 버전"을 식별한다 —
    TextAgent 쪽도 호출별 user prompt는 해싱하지 않는 것과 동일한 범위. **`LatexAssembler`는 이 코드베이스에
    `SYSTEM_PROMPT` 같은 모듈 상수가 전혀 없어** (`grep`으로 확인) `assembler_prompt_hash` 필드를 의도적으로
    만들지 않았다 — 없는 것을 해싱해서 있는 것처럼 보이게 하지 않는다.
  - **`text_agent_enabled_validators_snapshot` / `planner_enabled_validators_snapshot` /
    `assembler_enabled_validators_snapshot`** *(2차 리뷰 후 세 필드로 분리)*: 원래 `enabled_validators_
    snapshot` 하나로 TextAgent만 담았던 것을, 세 agent 각각의 실제 인스턴스 `.enabled_validators` 딕셔너리로
    분리했다 (static `DEFAULT_ENABLED_*`가 아니라, ablation 실행이 `TextAgent(enabled_validators={...})`
    등으로 override한 실제 값). `DocGenPipeline.run()`의 `_write_log` 클로저가
    `self.text_agent`/`self.planner`/`self.assembler` 각각의 `.enabled_validators`를 전달한다
    (`scripts/run_full_pipeline.py`).
  - `plan_doc_version`: `"2026-10-28"` (이 코드가 기준으로 한 계획 작성일).
  - **`rubric_version`** *(수정)*: 기본값을 `None`에서 `"gold_labeling_draft_2026-10-26"`로 바꿈 — Gold
    Labeling 페이지의 마지막 수정일을 "현재 적용 중인 초안"의 식별자로 삼는다. `None`은 "rubric이 전혀
    없음"과 구분되지 않아 부정확했다.
  - **`rubric_finalized`** *(수정 후 추가)*: `False` (고정). G1(5단계)에서 실제 버전 체계가 정해지기 전까지
    "확정 전"이라는 상태 자체를 bool로 명시한다 — 문자열 네이밍 관례("draft_...")에만 의존하지 않는다.

## P2. 세 검사군의 Evidence 연결, 그리고 복구 전 보존

`Evidence`/`Constraint`/`CheckOutcome`/`UnknownReason` (`src/core/schemas.py:144-259`)는 범용적으로 설계되어
있어, A/B/C 세 검사군 모두 새 스키마 없이 재사용한다 — 이 부분은 원래 설명대로 맞다.

**원래 설명이 틀렸던 부분**: "`*.draft.tex` + `ArtifactGraphStore.save_checkpoint()`가 복구 전 원본 보존을
이미 보장한다"고 썼으나, 사용자 리뷰가 지적한 대로 **둘 다 실제로는 보장하지 않는다**.

- `grep -rn "ArtifactGraphStore\|save_checkpoint\|from src.core" scripts/run_full_pipeline.py` 결과가
  **0건**이다 — 실제 파이프라인은 `src/core/`(artifact graph·controller)를 전혀 참조하지 않는다. 즉
  `save_checkpoint()`는 real pipeline 경로에서 호출될 방법이 없다 (존재하지만 연결되지 않은 기능).
- `{node_id}.draft.tex`/`{node_id}.blocks.json`은 **고정된 단일 경로**다 (`text_agent.py`의
  `_run_generation_loop`). `regenerate_blocks()`가 성공하면 `_run_generation_loop`가 같은 경로에
  덮어쓴다 — 복구 전 내용은 메모리상의 `persisted`/`accepted`에만 잠깐 존재했다가 디스크에서는 영원히
  사라진다. `process_node()`가 이미 DRAFTED된 node에 다시 호출되는 섹션 단위 복구(`_repair_and_recompile`의
  section 분기)도 동일하게 덮어쓴다.

**실제 수정**: `TextAgent._snapshot_before_overwrite(sections_dir, node_id)` (신규)를
`process_node()`와 `regenerate_blocks()` 양쪽의 복구 진입점 맨 앞에서 호출한다. 첫 draft(아직
`{node_id}.blocks.json`이 없음)에는 no-op이고, 이미 드래프트된 node에 다시 호출될 때만
`{node_id}.pre_repair_{N}.blocks.json`/`.draft.tex`로 현재 내용을 번호 붙여 보존한다 (`N`은 같은
디렉터리의 기존 `pre_repair_*` 개수+1로 결정, 여러 복구 라운드가 서로 덮어쓰지 않음). `ArtifactGraphStore`
전체를 real pipeline에 연결하는 일(= P1의 공백 2번과 동일한 작업량)은 여전히 하지 않았고, 이건 그보다 작고
로컬한 대안이다 — 진짜 artifact graph 연결은 4단계(I1) 작업으로 남는다.

테스트: `test/test_text_agent.py::TextAgentPreRepairSnapshotTest` (첫 draft는 스냅샷 없음, `process_node`
재호출/`regenerate_blocks` 모두 이전 내용을 정확히 보존, 2회 연속 복구가 `pre_repair_1`/`pre_repair_2`로
분리됨을 확인).

**2차 리뷰가 지적한 부분 (2026-10-29)**: 위 수정은 **호출 간**(cross-call) 보존만 다룬다 — `process_node`/
`regenerate_blocks`가 다시 호출되기 *전에* 이미 디스크에 있던 내용만 보존한다. 하지만 **같은 호출 내부**,
즉 `_run_generation_loop`의 attempt 루프 안에서 검증에 실패한 attempt(특히 그 node의 **첫 attempt**)는
어디에도 기록되지 않았다 — 성공한 마지막 attempt만 `_write_semantic_artifacts`로 디스크에 쓰인다. 실패
원인 분석에는 바로 이 실패한 attempt의 실제 내용(어떤 블록이 왜 거부됐는지)이 필요하다.

**실제 수정**: 새 `TextAgent._persist_failed_attempt(sections_dir, node_id, attempt=..., error_message=...,
failing_blocks=..., raw_response=...)`를 `_run_generation_loop`의 `SemanticBlockErrors`/일반 `Exception`
두 분기 모두에서 호출한다. `SemanticBlockErrors`의 경우 실제로 거부된 블록 내용(`exc.blocks`를
`failing_ids`로 필터링)까지 보존하고, 일반 예외의 경우 raw LLM 응답(획득했다면)과 에러 메시지를 보존한다.
파일명은 `{node_id}.failed_attempt_{k}.json`이며 `k`는 (루프의 `attempt` 번호가 호출마다 1로 리셋되는 것과
달리) 디스크에 이미 있는 `failed_attempt_*` 개수로 매번 새로 스캔해 결정 — 같은 node에 대한 서로 다른 호출의
실패 기록이 서로 덮어쓰지 않는다. `raw_response`는 루프 반복마다 맨 앞에서 `None`으로 리셋한다 —
`self.llm.generate()` 자체가 실패하면 이전 attempt의 값이 새어 들어가는 걸 막는다.

테스트: `test/test_text_agent.py::TextAgentFailedAttemptPreservationTest` (첫 attempt 일반 실패 보존,
`SemanticBlockErrors`의 실제 거부 내용 보존, 3회 모두 실패해 ERROR로 끝나는 경우도 3개 전부 보존·순서대로
번호 매겨짐을 확인).

## P3. 미해결 상태의 종료 시 보고

**원래 설명이 틀렸던 부분**: "`_aggregate_unresolved`가 이미 미해결을 전부 집계한다"고 썼으나, 사용자
리뷰가 지적한 대로 이 함수(`src/core/controller.py`)는 `_compute_unresolved`(stale/blocked/
needs_recheck_keys) + 현재 FAIL/UNKNOWN evidence + 마지막 compile 실패만 본다 — **이 넷 모두 "무언가가
이미 일어났다"는 흔적이 있어야 잡힌다**. 한 번도 CheckCall로 이름이 불린 적 없는 필수 criterion은 stale도,
blocked도, FAIL/UNKNOWN evidence도, 무효화된 적도 없으므로 네 경로 어디에도 걸리지 않고 조용히
`unresolved_artifact_ids`에서 빠진다.

**실제 수정**: 새 `_never_checked_required_ids(store, tracked_ids, constraint_provider)`가
`constraint_provider(ref)`로 얻은 **요구되는** constraint_id 집합과 `store.live_evidence(artifact_id)`로
얻은 **실제로 한 번이라도 유효하게 검사된** constraint_id 집합을 비교해, 차집합이 있는 artifact를 미해결로
추가한다. "required"는 스펙(`constraint_provider`)에서만 가져온다 — 로그(`live_evidence`)에서 "required"를
역산하지 않는다 (Stage 6 자체가 이미 겪은 동일 함정, [[docgen_stage_review_rigor]] 패턴 #72).

`_aggregate_unresolved(state, store, constraint_provider=None)`에 **선택적** 세 번째 인자로 추가했다 —
기존 7개 호출부 중 `ControllerRuntime._controller_node`(line ~1264) 한 곳만 실제로 넘긴다. 나머지 6곳
(`RuleBasedController`/`LLMController` 내부의 자기 사전 점검)은 그대로 둬도 안전하다: 이 모듈 자체의
docstring(point 14)이 이미 확립한 설계대로, `_controller_node`는 **어떤 정책이 만든 FinishCall이든** 여기서
무조건 한 번 더 `_aggregate_unresolved`를 통과시키는 "choke point"이기 때문이다 — 6곳을 전부 고치는 대신
이 한 곳만 고치면 외부에 보이는 최종 보장(`ControllerRuntime.run()`의 결과)은 동일하게 맞다
([[docgen_stage_review_rigor]] 패턴 #53의 "한 곳의 선택지점" 원칙).

테스트: `test/core/test_controller.py`에 5개 추가 — `_never_checked_required_ids`의 단위 테스트 3개
(빈 required 집합 무시, 미검사 required 포착, live evidence로 해소됨), 기존 stale 경로와 중복 안 됨,
`constraint_provider=None` 하위호환, 그리고 실제 `_controller_node`를 통한 end-to-end repro
(`test_finish_call_includes_artifact_with_a_required_but_never_checked_constraint`). 기존
36개 테스트 전부 변경 없이 통과 (42개 전체 통과).

**2차 리뷰가 지적한 부분 (2026-10-29)**: 위 수정이 **새 모순을 만들었다**. `RuleBasedController.decide()`는
모든 게 해결됐다고 판단하면 `FinishCall(reason="resolved", unresolved_artifact_ids=[])`를 **하드코딩**으로
반환한다 (이 판정은 `_compute_unresolved`만 보고 내려지며, `_never_checked_required_ids`가 잡는 "요구되지만
한 번도 안 검사됨" 케이스는 이 판정에 전혀 반영되지 않는다). choke point가 이 호출 뒤에 enriched
`unresolved_artifact_ids`를 union하면서, `reason="resolved"`인데 `unresolved_artifact_ids`가 비어있지 않은
**직접적인 모순 상태**가 새로 도달 가능해졌다 — 이 모듈의 reason 어휘 중 "아무 문제 없음"을 주장하는 건
`"resolved"` 하나뿐이고, 나머지(`"unresolved_remaining"`, `"evidence_inconclusive"`, `"compile_failed"`,
`"cannot_repair"`, 두 guard reason)는 이미 문제를 인정하는 문구라 이 모순이 발생하지 않는다.

**실제 수정**: `_controller_node`에서 enriched `unresolved_artifact_ids`를 계산한 뒤, `call.reason ==
"resolved"`이면서 그 목록이 비어있지 않으면 reason을 새 문자열을 만들지 않고 이미 존재하는
`"unresolved_remaining"`으로 바꾼다. `_FINISH_REASON_MAP`에서 `"resolved"`와 `"unresolved_remaining"`
둘 다 이미 동일한 `ControllerStopReason.FINISHED`로 매핑되므로, `_tool_node`가 기록하는 stop_reason에는
영향이 없다 — `FinishCall.reason` 문자열만 바뀐다.

테스트: `test_resolved_reason_is_corrected_when_the_enriched_union_is_not_actually_empty`(모순 상황에서
`"unresolved_remaining"`으로 교정됨 확인), `test_resolved_reason_is_left_alone_when_the_union_is_genuinely_
empty`(진짜 해결된 경우 `"resolved"`가 그대로 유지됨 확인 — 과교정 방지).

## P4. 비용·조건 분리

**원래 설명의 빠진 부분**: "`CostTrackingProducer`가 비용을 집계한다"는 맞지만, 사용자가 물은 "결정적
검사·변환 비용은 어느 경로로 기록되는가"에는 답하지 않았다. 다시 확인한 결과:

- **LLM 호출 비용** (Producer.produce()): `ToolExecutor.run_generation()` → `cost_tracking.
  CostTrackingProducer`가 감싸 실제 토큰/모델 비용의 `CostRecord`를 만들고 `_charge()`로 기록한다.
- **결정적 검사 비용** (Validator.check(), 즉 CheckCall 포함 모든 ToolCall): `ToolExecutor._execute_inner`가
  **모든 ToolCall 종류(CHECK 포함)를 wall-clock latency 기반 `CostRecord(stage=f"tool_executor:
  {action}")`로 자동 기록**한다 (`src/core/tool_executor.py:265-317`) — 성공/실패/timeout 모두. 이미
  완전히 별도 경로로 분리되어 있었다 — 처음 문서가 이 경로를 확인하지 않고 "비어 있다"고 암묵적으로
  가정한 것이 틀렸다.

**단, 이 둘 다 `src/core/controller.py` 기반 실행(ToolExecutor를 통한 dispatch)에만 적용된다.** P2에서
확인한 것과 같은 단절 — `run_full_pipeline.py`는 `src.core`를 전혀 참조하지 않으므로, 오늘
`TextAgent._validate_one_block`처럼 실제 파이프라인에서 직접 호출되는 결정적 검사에는 **비용 기록 경로가
전혀 없다** (LLM 생성 토큰은 `token_usage_records`로 잡히지만, 순수 Python 정규식·구조 검사는 토큰을 쓰지
않으므로 그 경로에도 안 잡힌다). Fix A/B/C가 `src/core/controller.py` 실행 경로로 구현되면 비용 기록이
자동으로 따라오지만, 지금처럼 `TextAgentValidatorAdapter`/`_validate_one_block` 스타일로 실제 파이프라인에
직접 끼워 넣으면 **새 비용 기록 경로가 필요하다** — 이것도 4단계(I1, 실행 경로 연결)의 일이지 0단계에서
풀 수 있는 범위가 아니다.

**비교 조건 표** (사용자가 요청한 "표로 고정"): 지금 결정 가능한 것은 베이스라인 한 줄뿐이다 — 나머지는
A/B/C가 실제로 구현되는 1~4단계에서 값이 채워지는 **구조 고정**이다.

| 조건 | prompt 버전 | 정규화 | 활성 검사 | feedback | 복구 범위 | 비용 기록 경로 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (현재, additional verifiers 없음) | `text_agent_prompt_hash`/`planner_prompt_hash` (현재 커밋 기준 고정) | 없음 (markdown 강조 규제 미적용) | 각 agent의 `DEFAULT_ENABLED_*`만 (`{text_agent,planner,assembler}_enabled_validators_snapshot`으로 기록됨) | `TextAgent.process_node`의 내부 재시도 피드백만 (always-on, 끌 수 없음) | 기존 `_repair_and_recompile` (block/section) | LLM: `token_usage_records`; 결정적 검사: 없음 |
| +Fix A (markdown_emphasis) | TBD (1단계에서 SYSTEM_PROMPT 수정 시 새 hash) | TBD: 평문 강조만 제한적 변환, 모호한 경우는 block 복구로 | `markdown_emphasis_enabled=True` 추가 | TBD (A3에서 결정) | TBD (A3) | TBD — 4단계 전까지는 real pipeline에 새 경로 직접 추가 필요 |
| +Fix A+C (reference/order) | TBD | TBD | +`dependency_verifier_enabled=True` | TBD | TBD | TBD |
| +Fix A+B+C (전체) | TBD | TBD | 전부 `True` | TBD | TBD | TBD |

이 표의 "TBD" 행은 각 단계 설계 문서(`docs/av_stage1_*.md` 등)에서 구체적 값으로 채워져야 완료된다 — 0단계는
표의 **열 구조와 베이스라인 행**을 고정하는 것까지다.

**2차 리뷰 확인 (2026-10-29)**: 2차 리뷰는 이 "TBD" 행들과 "결정적 검사 비용 기록 경로 없음"이 여전히
그대로라고 짚었다 — 맞다, 둘 다 이미 위에서 **의도적으로** 1~4단계로 이연한 것이다 (A/B/C 자체가 없는데
그 조건의 정규화·feedback·복구 범위를 지금 확정할 수는 없다). 이번 라운드에서 추가 조치는 하지 않았다.

## 발견된 공백 — 0단계에서 풀지 않고 이후 단계로 이연

1. **production `ConstraintProvider`가 코드베이스 어디에도 없다.** (원래 설명 그대로 유효) — 4단계/I1 작업.
2. **Markdown 강조 구문 탐지가 전혀 없다.** (원래 설명 그대로 유효) — 1단계 작업.
3. **real pipeline과 `src/core/`(artifact graph·controller·ToolExecutor) 사이의 단절**이 P2(체크포인트)와
   P4(검사 비용 기록) 둘 다에서 같은 원인으로 드러났다 — 공백 1번과 사실상 동일한 근본 원인의 세 번째
   증상이다. P2는 로컬 대안(`_snapshot_before_overwrite`)으로 부분 완화했지만, P4의 "결정적 검사 비용
   기록"은 그런 로컬 대안을 아직 만들지 않았다 — 1~3단계에서 `_validate_one_block` 스타일로 Fix A/B/C를
   붙일 경우 비용 기록이 필요해지는 시점에 다시 다뤄야 한다.
4. Fix C의 계약 스키마는 이미 존재한다 (원래 설명 그대로 유효) — 2~3단계 작업.

## Gold Labeling → 수정 매핑 (회귀 테스트용 실제 사례)

| 샘플 | 관찰된 결함 | 해당 수정 |
|---|---|---|
| sample_001, 002, 003, 004, 006, 007, 009 | `**bold**`/`__bold__`가 PDF에 그대로 노출 (총 7개 샘플, 가장 빈번) | Fix A |
| sample_001 | `0.998^999` 계산 오류 (재계산으로 확정) | Fix B (수치 재계산) |
| sample_005 | 굽힘응력 10배, 처짐 25배 과다 계산 (재계산으로 확정) | Fix B (수치 재계산) |
| sample_000 | proof_sketch에서 `T_n(x)`를 `f(x)`로 암묵적 대체 (논리 비약) | Fix B (후속 범위, B6 — 논리 검토) |
| sample_003, 007, 008 | 동일 대상에 다른 표기 혼용 ($p(D\|\theta)$ vs $\mathcal{L}(\theta)$; `u` vs `U`; `Δ%` vs `RG`) | Fix B (공유 값·표기 일관성) |
| sample_002, 006 | `worked_examples`/`case_or_application`이 요구된 정의를 실제로 재사용하지 않음 (`must_reuse` 미충족) | Fix C |
| sample_005 | summary가 analysis 섹션을 `\Cref`로 인용하지 않음 (`must_reference` 미충족) | Fix C |
| sample_008 | summary가 인용한 섹션에 없는 내용을 서술 (내용 불일치) | Fix C |
| 다수 샘플의 `contract_result` 과소/과대 채점, `blocks.json` kind 불일치 | 자동 채점기 자체의 버그 | **범위 밖** — Notion 계획 §5가 명시 |

## 완료 기준 확인

- [x] 어떤 산출물의 어떤 버전을 무엇으로 검사했고, 왜 수정했으며, 무엇이 미해결인지를 한 실행 로그에서
  추적할 수 있다 — P1의 버전 메타데이터, P2의 복구 전 스냅샷, P3의 (한 번도 검사 안 된 required 포함)
  미해결 집계가 서로 다른 세 가지를 각각 책임지고, 셋 다 이제 실제로 연결되어 있다. 추적할 "검사" 자체는
  아직 없으므로 (A/B/C 미구현) 실제 추적은 1단계에서 처음 end-to-end로 검증된다.
- [x] 기존 결과 보존 확인 (outputs/, Gold Labeling 불변).
- [x] 검사 범위와 로그 계약 확정: P1-P4 각각이 가리키는 기존/신규 메커니즘, 그리고 P4의 비교 조건 표 구조.
- [x] 1차 리뷰(2026-10-28) 4건, 2차 리뷰(2026-10-29) 3건 모두 반영 완료. 2차 리뷰 반영: P1에 planner
  prompt hash + 3-agent validator snapshot 분리, P2에 호출-내부 실패 attempt 보존
  (`_persist_failed_attempt`), P3의 `reason="resolved"`/비어있지 않은 unresolved 모순 교정. 회귀 테스트
  총 24개 추가 (controller.py 7개, text_agent.py 7개, schemas/wiring 테스트 갱신), 전체 테스트 스위트
  변경 없이 통과 (사전 존재하던 57건의 무관한 실패는 이 세션 변경 전/후 동일함을 `git stash`로 확인).
