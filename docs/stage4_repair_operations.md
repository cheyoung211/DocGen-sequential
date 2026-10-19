# Stage 4 — 고정 복구 정책과 분기 재실행

Notion "Revision 2026.10.15" 4단계 구현 기록. Stage 1(`docs/stage1_experiment_design.md`),
Stage 2(`docs/stage2_artifact_graph.md`), Stage 3(`docs/stage3_tool_executor.md`)와 동일한
형식을 따른다.

## 0. 코드 위치

- `src/core/repair_operations.py` — `RepairOperationExecutor`(4개 복구 연산 + graph-aware
  재검사/선택적 재생성 cascade), `FixedGraphAwareRepairPolicy`("fixed graph-aware" 비교군),
  `AlwaysRegenerateDocumentPolicy`("whole-document regeneration" 비교군).
- `src/core/branching_runner.py` — `BranchingRunner`(같은 checkpoint에서 여러 분기 실행,
  후속 정책 고정, `check_timing`으로 "final-only verification" 비교군의 trigger 타이밍 표현).
- `src/core/adapters.py` — `BlockPatchProducer` 추가(`TextAgent.regenerate_blocks`를
  `Producer`로 래핑, PATCH_BLOCK에 실제 producer를 제공).
- `src/core/schemas.py` — `RepairOutcome`, `RegeneratedArtifactRecord`,
  `RepairExecutionResult`, `BranchExecutionResult`, `BranchComparisonResult` 추가.
- `test/core/test_repair_operations.py`(16), `test/core/test_branching_runner.py`(11),
  `test/core/test_adapters.py`에 `BlockPatchProducer` 테스트 6개 추가.

Zero changes to `scripts/run_full_pipeline.py` 또는 agent 코드 — Stage 1-3과 동일한 원칙으로
순수 추가(additive)만 했다.

---

## 1. Notion 4단계 체크리스트 대응

### 1.1 "block patch, section regeneration, subgraph replan, whole-document regeneration 연산을 구현한다."

`RepairOperationExecutor.execute()`가 네 가지 `RepairActionType` 전부를 동일한 메커니즘으로
수행한다 — 차이는 `action.target_artifacts`가 무엇을 가리키는지뿐이다(block 1개 / section 1개 /
명시적 다중 artifact seed / document). PATCH_BLOCK은 새로 추가한 `adapters.BlockPatchProducer`
(`TextAgent.regenerate_blocks` 래핑)로, REGENERATE_SECTION/REGENERATE_DOCUMENT는 기존
`TextAgentProducer`/`PlannerAgentProducer`로 실제 생성을 수행한다. REPLAN_SUBGRAPH는 agent
코드에 "부분 재계획" 메서드가 없으므로(§3 결정 1) 대상 artifact 각각을 해당 kind의 기존
Producer로 재생성한 뒤 아래 1.2의 cascade로 영향을 전파하는 방식으로 구현했다.

### 1.2 "수정 이후 영향받는 artifact를 재검사하고, 필요한 항목만 재생성하는 고정 graph-aware 정책을 구현한다."

이 항목은 "정책"과 "연산"이 아니라 하나의 메커니즘으로 구현했다: *범위*(어떤 artifact를 직접
대상으로 할지)는 `RepairPolicy.decide()`가 결정하고, *그 범위에서 재검사/선택적 재생성이 얼마나
전파되는지*는 `RepairOperationExecutor`가 결정한다. 후자는 네 연산 모두 동일하다.

cascade 알고리즘(`RepairOperationExecutor.execute()`):
1. 직접 지정된 target마다 실제 Producer를 호출해 재생성한다.
2. 재생성(혹은 재확인, 아래 참고)된 artifact마다 `ArtifactGraphStore.impact_candidates`로
   보수적 영향 후보(`I(R)=R∪Desc_G(R)`)를 구한다.
3. 각 후보를 **항상 새로** `Validator.check()`한다(캐시된 evidence 재사용 없음). FAIL이면
   재생성 대상으로 큐에 넣어 같은 과정을 반복하고, PASS이면 **재확인(reaffirm)**한다.
4. 재확인은 내용을 바꾸지 않고 `add_version`을 다시 호출하되, 원래 선언된 입력을 전부
   **현재 최신 버전으로 재고정(re-pin)**한다. Stage 2의 `add_version`은 이미 입력이
   outdated/stale인지 판정하는 로직을 갖고 있으므로, 재고정된 입력이 실제로 다 fresh하면
   `born_stale=False`로 기록되어 staleness가 깨끗하게 해소된다 — Stage 2 코드에 새 메서드를
   추가하지 않고 기존에 이미 여러 차례 검증된 메커니즘만 재사용했다(상세 이유는
   `repair_operations.py`의 `RepairOperationExecutor` docstring, "Why reaffirm" 단락).

제약이 등록되지 않은 artifact(`constraint_provider`가 빈 리스트 반환)는 건너뛴다 — 아무것도
단정하지 않는다. 이것이 하위 reachability를 잃지 않는 이유: `impact_candidates`는 건너뛴
노드를 거쳐야 도달하는 artifact까지 이미 한 번에(transitively) 반환하기 때문이다
(`test_a_candidate_with_no_constraints_is_skipped_without_losing_further_reachability`).

**방향 1의 finer-grained kind(DEFINITION/SYMBOL/EQUATION/TABLE/CLAIM/FIGURE/SOURCE_DATA/
CALCULATION)에는 실제 Producer가 없다** — 오늘 파이프라인은 plan 전체(DOCUMENT) 또는 section의
block 전체(SECTION/BLOCK)만 생성하며, "이 수식만" 재생성하는 agent 진입점이 없다. 이런 kind를
재생성해야 하는 상황은 `RepairOutcome.NO_PRODUCER`로 정직하게 보고하고 끝낸다 — 가짜 producer를
만들거나 조용히 건너뛰지 않는다.

### 1.3 "같은 checkpoint에서 여러 repair action을 실행하고 후속 정책을 고정해 비교하는 분기 runner를 만든다."

`BranchingRunner.run_branches()`가 `docs/stage1_experiment_design.md` §5의 checkpoint 분기
설계를 그대로 실행한다:
- 각 `BranchSpec`(label + 초기 정책)마다 `ArtifactGraphStore.from_checkpoint`로 독립된 store를
  복원한다(branch_id = label).
- 모든 분기의 2회차 이후 실행은 `downstream_policy_factory()`로 매 분기마다 새로 만든 **같은
  규칙의 독립 인스턴스**를 쓴다 — 공유 인스턴스 하나가 아니다. `budget_factory()`도 분기마다
  새로 호출한다.

**왜 factory인가(코드 작성 중 직접 발견한 문제):** 처음에는 `downstream_policy`/`budget`을
분기 간에 공유하는 설계였다. 테스트를 작성하던 중, `BlindRetryRepairPolicy`처럼 artifact별
재시도 횟수를 내부에 들고 있는 정책을 공유하면 분기 A의 2회차가 올린 카운터를 분기 B의 2회차가
그대로 물려받는다는 것을 발견해 factory 방식으로 바꿨다. 같은 종류의 문제가 **validator**에도
있었다: `BasicConstraintChecker`는 생성 시점에 특정 `ArtifactGraphStore`에 바인딩되므로
(`tool_executor.ToolExecutor`와 동일한 "one instance = one store" 설계), 분기 간에 validator
하나를 공유하면 각 분기가 복원한 자신의 store가 아니라 처음 바인딩된 store만 계속 읽게 된다.
`BranchingRunner`는 `validator_factory(store)`로 분기마다 그 분기의 store에 바인딩된 validator와
`RepairOperationExecutor`를 새로 만들어 이 문제를 막는다(`producer_registry`/
`constraint_provider`는 agent나 순수 함수에 묶여 있어 공유해도 안전하므로 그대로 공유한다).

`check_timing`(`"continuous"`/`"final_only"`)은 이 harness가 표현할 수 있는 한 가지 "when" 축이다
— `docs/stage1_experiment_design.md` 비교표의 "whole-document regeneration"(trigger=on)과
"final-only verification"(trigger=off 중간, on 최종)은 scope 규칙이 완전히 같고(둘 다
`AlwaysRegenerateDocumentPolicy`) trigger 타이밍만 다르다는 점에 착안해, 분기의 **초기** 결정만
이 옵션으로 제어한다. 후속 round는 실행기 자신의 cascade가 만들어내는 fresh evidence로 계속
움직이므로 "检查를 미룬다"는 개념이 적용될 지점이 없다 — 진짜 controller(Stage 5)가 있어야
의미 있게 구현된다.

### 1.4 "작은 자연 실패·주입 실패 pilot으로 복구 가능성, 정상 내용 손상, 비용 로그를 점검한다."

사용자와 사전에 범위를 확정했다(§3). 이번 단계는 **메커니즘 구현 + Fake 기반 테스트**까지만
완료한다 — 실제 로컬 Qwen3.6 vLLM으로 문서를 생성하는 작은 규모의 자연/주입 실패 pilot은 코드
리뷰 이후 별도로 진행한다. 메커니즘 수준에서 "자연 실패"(예: `test_a_descendant_that_still_
fails_after_reverify_gets_regenerated_and_reported`의 cascade 중 발생하는 2차 실패)와
"주입 실패"(production이 그대로 실패하도록 구성한
`test_production_failure_does_not_crash_and_does_not_cascade`,
`test_a_production_failure_is_recorded_and_the_branch_stops_honestly`)에 해당하는 시나리오는
이미 테스트로 커버했다 — 실제 파이프라인 pilot 전 단계로서 "복구 가능성/정상 내용 손상/비용 로그"가
구조적으로 올바르게 기록되는지는 확인된 상태다.

---

## 2. 완료 기준 대응

> "controller 학습 없이 비교 가능한 복구 연산과 replay 데이터. 같은 초기 상태에서 행동과 후속
> 결과를 연결할 수 있고, 실패한 복구도 남는다."

- **controller 학습 없음**: `FixedGraphAwareRepairPolicy`/`AlwaysRegenerateDocumentPolicy` 둘 다
  고정 규칙이며 학습 파라미터가 없다. `RepairOperationExecutor`도 고정된 cascade 규칙이다.
- **비교 가능한 복구 연산 + replay 데이터**: `BranchExecutionResult`가 `source_checkpoint_id`,
  `initial_action`, `initial_round`, `downstream_rounds`, `final_checkpoint`를 모두 보존하므로
  한 checkpoint에서 나온 여러 분기를 나란히 재생/비교할 수 있다.
- **행동과 후속 결과의 연결**: `RepairExecutionResult`가 `action` 자체와 `regenerated`(어떤
  artifact가 실제로 재생성됐는지, 각각의 production/outcome/재검사 evidence)를 함께 들고
  있어, "이 행동 → 이 결과"를 직접 추적할 수 있다.
- **실패한 복구도 남는다**: `RepairOutcome.{STILL_FAILING, PRODUCTION_FAILED, NO_PRODUCER}`가
  예외를 던지지 않고 정상적인 결과값으로 기록된다. `BranchingRunner`는 production이 완전히
  실패해 더 이상 평가할 evidence가 없는 경우를 `stopped_reason="repair_could_not_be_attempted"`로
  구분해서 보고한다 — "no_more_failures"(정상 해소)와 혼동되지 않도록 코드 작성 중 직접 발견하고
  고친 지점이다(§3의 review 없이도 테스트 작성 과정에서 드러난 문제, 아래 §4 참고).

---

## 3. 사용자 결정 사항 (구현 전 확정)

구현 전 2가지 설계 질문을 사용자에게 확인했다:

1. **REPLAN_SUBGRAPH 구현 방식**: "타겟별 Producer + 전파" 채택 — 대상 artifact 각각을 해당
   kind의 기존 Producer로 재생성한 뒤 그 subgraph의 `impact_candidates`에 대해서만 재검사/필요시
   재생성을 전파한다. PlannerAgent의 전체 재계획 호출을 끼워 넣지 않는다. 이에 따라
   `FixedGraphAwareRepairPolicy`는 fan-out>1인 실패 artifact에 대해 "artifact 본인 + 알고 있는
   직접 의존 artifact들"만 seed로 지정하고, 더 깊은 전파는 실행기의 cascade에 맡긴다.
2. **Pilot 범위**: "메커니즘 구현 + Fake 테스트만" 채택 — 실제 로컬 Qwen3.6 LLM으로 문서를
   생성하는 pilot 실행은 이번 세션에 하지 않고, 코드 리뷰 이후 별도로 진행한다.

---

## 4. 구현 중 직접 발견하고 고친 문제 (사용자 리뷰 전)

Stage 1-3과 달리 이번 라운드는 아직 사용자 리뷰를 받지 않았다. 다만 테스트를 작성하는 과정에서
직접 3가지 문제를 발견해 리뷰 전에 고쳤다 — `docgen_stage_review_rigor` 메모의 패턴들과 같은
종류이므로 기록해 둔다.

1. **분기 간 정책 상태 누수**: `downstream_policy`를 분기 간에 공유하는 초안이었으나, 상태를
   가진 정책(예: `BlindRetryRepairPolicy`류)의 재시도 카운터가 분기 A→B로 새는 것을
   `test_a_stateful_downstream_policy_does_not_leak_retry_state_across_branches` 설계 중 발견.
   `downstream_policy_factory`(분기마다 새 인스턴스)로 수정.
2. **분기 간 validator store 바인딩 불일치**: 위와 같은 테스트를 작성하며, `BasicConstraintChecker`가
   생성 시점에 특정 store에 바인딩된다는 사실을 다시 확인 — `BranchingRunner`가 validator를
   한 번만 만들어 공유하면 모든 분기가 "원래" store만 계속 보게 된다. `validator_factory(store)`로
   분기마다 그 분기의 store에 바인딩된 validator+executor를 새로 만들도록 수정.
3. **"no_more_failures"가 실패를 성공처럼 보이게 함**: production이 완전히 실패하면
   `reverify_evidence`가 비어 있으므로(재검사를 시도조차 못 했으므로), 다음 round 진입 조건
   `if not any(FAIL in current_evidence)`가 "실패 없음 = 해결됨"으로 잘못 읽혔다. `overall_outcome`
   자체는 `PRODUCTION_FAILED`로 정확히 기록되지만 `stopped_reason`이 이를 가릴 뻔했다.
   `_stalled_without_evidence()`를 추가해 "evidence가 없는 이유가 전부 PASS인지, 아니면 아무것도
   시도조차 못 했는지"를 구분하고, 후자는 `stopped_reason="repair_could_not_be_attempted"`로
   별도 보고하도록 수정(`test_a_production_failure_is_recorded_and_the_branch_stops_honestly`).

---

## 5. 하지 않은 것 (Stage 4 범위 밖, 다음 단계로 명시적으로 미룸)

- **실제 파이프라인 연결**: `DocumentGraph`/`PlannerAgent`/`TextAgent`가 실제로
  `ArtifactGraphStore.add_version`을 호출하도록 바꾸는 작업은 없다 — Stage 1-3의 adapter들과
  같은 원칙으로, 이번 단계는 메커니즘을 테스트로 증명하는 단계다. 실제 연결은 Stage 5(controller)
  또는 그 이후의 일이다.
- **실제 로컬 LLM pilot 실행**: §3에서 사용자와 합의한 대로, 실제 Qwen3.6 vLLM으로 문서를 생성해
  자연/주입 실패를 수집하는 pilot은 이번 세션에 하지 않았다. 코드 리뷰 이후 별도로 진행한다.
- **CSP_SMT/SYMBOLIC 제약 검사**: Stage 3에서 이미 후속 작업으로 남긴 범위이며 이번 단계에서도
  바꾸지 않았다.
- **무한 loop/반복 수정 방지 일반화**: `BranchingRunner`는 `max_downstream_rounds`로 자신의
  round 수만 제한한다 — "동일 수정 반복 방지" 같은 더 일반적인 안전장치는 Notion 5단계 체크리스트
  항목이며, 이번 단계는 명시적으로 요구하지 않았다.
- **방향 1의 finer-grained artifact kind에 대한 실제 Producer**: DEFINITION/SYMBOL/EQUATION 등을
  독립적으로 재생성하는 agent 기능은 오늘 파이프라인에 없고, 이번 단계에서 새로 만들지 않았다
  (`RepairOutcome.NO_PRODUCER`로 정직하게 보고하는 것까지가 이번 단계의 범위).
