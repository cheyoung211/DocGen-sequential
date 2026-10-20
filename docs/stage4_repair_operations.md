# Stage 4 — 고정 복구 정책과 분기 재실행

Notion "Revision 2026.10.15" 4단계 구현 기록. Stage 1(`docs/stage1_experiment_design.md`),
Stage 2(`docs/stage2_artifact_graph.md`), Stage 3(`docs/stage3_tool_executor.md`)와 동일한
형식을 따른다. **1차 구현(§1-4)은 사용자 리뷰에서 8건의 실제 결함이 발견되어 전부 수정했다 —
§5가 그 수정 기록이며, 아래 §0-4의 코드 설명은 수정이 반영된 최종 상태를 기준으로 한다.**

## 0. 코드 위치

- `src/core/repair_operations.py` — `RepairOperationExecutor`(선택적 cascade로 PATCH_BLOCK/
  REGENERATE_SECTION/REPLAN_SUBGRAPH를 수행 + REGENERATE_DOCUMENT를 위한 무조건 재생성 경로),
  `FixedGraphAwareRepairPolicy`("fixed graph-aware" 비교군), `AlwaysRegenerateDocumentPolicy`
  ("whole-document regeneration" 비교군).
- `src/core/branching_runner.py` — `BranchingRunner`(같은 checkpoint에서 여러 분기 실행,
  후속 정책 고정, `check_timing`으로 "final-only verification" 비교군의 trigger 타이밍 표현).
- `src/core/adapters.py` — `BlockPatchProducer` 추가(`TextAgent.regenerate_blocks`를
  `Producer`로 래핑), 기존 세 Producer(`PlannerAgentProducer`/`TextAgentProducer`/
  `BlockPatchProducer`) 모두 실제 산출물·입력을 `ProductionResult`에 싣도록 수정(§5.1).
- `src/core/schemas.py` — `ProductionResult`에 `content`/`read_input_versions` 추가;
  `RepairOutcome`(RESOLVED/STILL_FAILING/**INCONCLUSIVE**/PRODUCTION_FAILED/NO_PRODUCER),
  `RegeneratedArtifactRecord`, `RepairExecutionResult`(+ **left_stale**/**budget_exhausted**),
  `RepairRound`, `BranchExecutionResult`(**rounds: List[RepairRound]**), `BranchComparisonResult`.
- `test/core/test_repair_operations.py`(29), `test/core/test_branching_runner.py`(16),
  `test/core/test_adapters.py`에 `BlockPatchProducer`/content 테스트 추가.

Zero changes to `scripts/run_full_pipeline.py` 또는 agent 코드 — Stage 1-3과 동일한 원칙으로
순수 추가(additive)만 했다.

---

## 1. Notion 4단계 체크리스트 대응

### 1.1 "block patch, section regeneration, subgraph replan, whole-document regeneration 연산을 구현한다."

네 연산은 더 이상 하나의 메커니즘이 아니다 — PATCH_BLOCK/REGENERATE_SECTION/REPLAN_SUBGRAPH는
`RepairOperationExecutor._execute_selective`(선택적 cascade, §1.2)를 공유하고,
REGENERATE_DOCUMENT는 전용 `_execute_unconditional`을 쓴다(§5.8이 이 분리가 필요했던 이유).
PATCH_BLOCK은 `adapters.BlockPatchProducer`(`TextAgent.regenerate_blocks` 래핑)로, REGENERATE_
SECTION/REGENERATE_DOCUMENT는 `TextAgentProducer`/`PlannerAgentProducer`로 실제 생성을
수행한다. REPLAN_SUBGRAPH는 agent 코드에 "부분 재계획" 메서드가 없으므로(§3 결정 1) 대상
artifact 각각을 해당 kind의 기존 Producer로 재생성한 뒤 선택적 cascade로 영향을 전파한다.

### 1.2 "수정 이후 영향받는 artifact를 재검사하고, 필요한 항목만 재생성하는 고정 graph-aware 정책을 구현한다."

`_execute_selective`의 cascade:
1. 직접 지정된 target마다 실제 Producer를 호출해 재생성한다(비용/허용은 Stage 3의
   `ToolExecutor.run_generation()`을 통해서만 — §5.6).
2. 재생성(또는 재확인)된 artifact마다 `impact_candidates`로 보수적 영향 후보를 구해 **고정점
   sweep**에 추가한다 — 한 번의 방문으로 끝나지 않고, `store.is_stale()`을 매 sweep마다 새로
   다시 확인하며 더 이상 진행이 없을 때까지(또는 안전 상한까지) 반복한다. 이는 §5.2에서 수정한
   처리 순서 문제(아래) 때문에 필요하다.
3. 각 후보를 **항상 새로** `ToolExecutor.execute(CheckCall(...))`로 재검사한다(캐시 재사용
   없음, 비용도 기록됨 — §5.6). 결과는 FAIL/PASS/판정불가 세 갈래로 분류한다(`_classify_evidence`,
   §5.3): FAIL이면 재생성 대상으로 큐에 넣고, PASS면 **재확인(reaffirm)**하고, 둘 다 아니면
   (UNKNOWN만 있거나 evidence가 없으면) 그대로 둔다 — 아무것도 단정하지 않는다.
4. 재확인은 내용을 바꾸지 않고 `add_version`을 다시 호출하되, 원래 선언된 입력을 전부
   **현재 최신 버전으로 재고정(re-pin)**한다. 또한 (a) 그 artifact를 향한 INFERRED edge를
   새 버전으로 이어주고(`_carry_forward_inferred_edges`), (b) 재확인에 쓰인 PASS evidence를
   새 버전에 다시 묶어 기록한다(§5.4) — 둘 다 1차 구현에 없던 단계다.

제약이 등록되지 않은 artifact, 또는 재검사 결과가 UNKNOWN뿐인 artifact는 `RepairExecutionResult.
left_stale`에 담겨 "해결되지 않았지만 시도도 하지 않았다"를 명시적으로 보여준다 — 1차 구현은
이런 경우를 완전히 누락시켰다(§5.3). `left_stale`은 하위 reachability를 잃지 않는다:
`impact_candidates`는 건너뛴 노드를 거쳐야 도달하는 artifact까지 이미 한 번에(transitively)
반환하기 때문이다.

**방향 1의 finer-grained kind**(DEFINITION/SYMBOL/EQUATION/TABLE/CLAIM/FIGURE/SOURCE_DATA/
CALCULATION)에는 실제 Producer가 없다 — `RepairOutcome.NO_PRODUCER`로 정직하게 보고한다.

### 1.3 "같은 checkpoint에서 여러 repair action을 실행하고 후속 정책을 고정해 비교하는 분기 runner를 만든다."

`BranchingRunner.run_branches()`가 `docs/stage1_experiment_design.md` §5의 checkpoint 분기
설계를 실행한다. 각 `BranchSpec`마다 독립된 store를 복원하고, **독립된 `Validator`/
`ToolExecutor`/`RepairOperationExecutor` 세트도 분기마다 새로 만든다**(`validator_factory
(store)`) — `BasicConstraintChecker`처럼 생성 시점에 특정 store에 바인딩되는 validator를
공유하면 분기가 복원한 자신의 store가 아니라 처음 바인딩된 store만 읽게 되기 때문이다(§5.6의
리뷰에서 발견). `downstream_policy_factory()`/`budget_factory()`/`context_factory(spec)`
모두 분기마다 새로 호출해 상태 누수를 막는다(후자 둘은 §5.7에서 추가/수정).

`BranchExecutionResult.rounds: List[RepairRound]`가 모든 round를 순서대로 담는다 —
`rounds[0]`이 분기 고유의 초기 round, 그 이후는 공유 후속 정책이 결정한 round다. 각 `RepairRound.
executions`는 그 round의 `decide()`가 반환한 **모든** 행동을 실행한 결과를 담는다 — 1차 구현은
첫 번째 행동만 실행하고 나머지를 버렸다(§5.5).

`check_timing`(`"continuous"`/`"final_only"`)은 분기의 **초기** round에만 영향을 준다 —
"whole-document regeneration"(trigger=on, `triggering_evidence`를 그대로 씀)과 "final-only
verification"(trigger=off 중간, on 최종 — 이 runner가 한 번만 직접 검사해 그 결과로 결정)의
scope 규칙은 동일(둘 다 `AlwaysRegenerateDocumentPolicy`)하다는 점에 착안했다. 이 한 번의
검사도 이제 `ToolExecutor.execute(CheckCall(...))`를 통해 비용·허용이 기록된다(§5.6).

### 1.4 "작은 자연 실패·주입 실패 pilot으로 복구 가능성, 정상 내용 손상, 비용 로그를 점검한다."

사용자와 사전에 합의한 대로 **메커니즘 구현 + Fake 기반 테스트**까지만 완료했다 — 실제 로컬
Qwen3.6 vLLM pilot은 이번 리뷰 반영 이후 별도로 진행한다.

---

## 2. 완료 기준 대응

> "controller 학습 없이 비교 가능한 복구 연산과 replay 데이터. 같은 초기 상태에서 행동과 후속
> 결과를 연결할 수 있고, 실패한 복구도 남는다."

- **controller 학습 없음**: 두 정책, cascade 규칙 모두 고정이며 학습 파라미터가 없다.
- **비교 가능한 복구 연산 + replay 데이터**: `BranchExecutionResult`가 `source_checkpoint_id`,
  `rounds`(모든 round의 모든 실행), `final_checkpoint`를 보존한다.
- **행동과 후속 결과의 연결**: `RepairExecutionResult`가 `action` 자체와 `regenerated`/
  `reverified_untouched`/`left_stale`을 함께 들고 있어 "이 행동 → 이 결과"를 직접 추적할 수
  있다.
- **실패한 복구도 남는다**: `RepairOutcome.{STILL_FAILING, INCONCLUSIVE, PRODUCTION_FAILED,
  NO_PRODUCER}`가 예외 없이 결과값으로 기록된다. `RepairExecutionResult.budget_exhausted`와
  `left_stale`은 "끝까지 가지 못했다"를 `overall_outcome=INCONCLUSIVE`로 정직하게 반영한다
  (1차 구현의 `stopped_reason="no_more_failures"` 오판 버그는 §5.5/5.6에서 수정됐다 — 이제는
  매 action마다, round 안에서도 budget을 확인하고, 처리되지 않은 독립 실패를 다음 round로
  정확히 넘긴다).

---

## 3. 사용자 결정 사항 (구현 전 확정, 1차 구현)

1. **REPLAN_SUBGRAPH 구현 방식**: "타겟별 Producer + 전파" — 대상 artifact 각각을 기존
   Producer로 재생성한 뒤 그 subgraph의 `impact_candidates`에 대해서만 재검사/필요시재생성을
   전파한다. PlannerAgent의 전체 재계획 호출을 끼워 넣지 않는다.
2. **Pilot 범위**: "메커니즘 구현 + Fake 테스트만" — 실제 로컬 LLM pilot은 리뷰 반영 이후.

---

## 4. 구현 중 직접 발견하고 고친 문제 (1차 구현, 사용자 리뷰 전)

1. 분기 간 정책 상태 누수 → `downstream_policy_factory`.
2. 분기 간 validator store 바인딩 불일치 → `validator_factory(store)`.
3. "no_more_failures"가 production 실패를 성공처럼 보이게 함 → (1차 수정은 `stopped_reason`
   전용 분기였으나, §5.5/5.6의 재설계로 지금은 `pending` 맵 기반 추적 + `budget_exhausted`로
   대체됐다.)

---

## 5. 사용자 재검토 후 수정 (1차, 2026.10.19)

사용자가 8건의 실제 결함을 코드 위치·재현 시나리오와 함께 지적했다. 전부 수정했고, 각 수정마다
회귀 테스트를 추가했다. `docgen_stage_review_rigor` 메모에도 패턴으로 기록했다.

**1. 복구한 실제 내용이 저장되지 않음.** `_regenerate()`가 Producer의 반환 artifact를 반영하지
않고 `content=None`으로 새 버전을 만들었고, 입력 관계도 실제 사용한 입력이 아니라 이전 입력을
재고정한 값이었다. **수정**: `ProductionResult`에 `content: Optional[str]`/
`read_input_versions: Optional[List[ArtifactRef]]`을 추가(`None`=미보고, `[]`=실제로 입력 0개
사용 — 둘을 구분). `_regenerate()`는 이제 `production.content`를 그대로 저장하고,
`production.read_input_versions`가 있으면 그것을, 없으면(Producer가 모를 때) 실행기가
의도한 재고정 값(`context["read_input_versions"]`로 전달한 것)을 쓴다 — 어느 쪽도 조작된
값이 아니다. `adapters.py`의 세 Producer(`PlannerAgentProducer`/`TextAgentProducer`/
`BlockPatchProducer`) 모두 `self.last_output.model_dump_json()`을 `content`로, 전달받은
`read_input_versions`를 그대로 echo하도록 수정. (`test_regeneration_persists_the_producers_
real_content_not_none`, `test_regeneration_records_what_the_producer_actually_reports_
using_not_a_blind_repin`, `test_producer_that_reports_nothing_falls_back_to_the_intended_
repinned_inputs`)

**2. 의존성 순서 때문에 stale artifact가 남아도 RESOLVED가 됨.** a→z→b에서 a를 복구하면 z,b가
모두 stale이 되는데, 정렬 순서상 b가 z보다 먼저 처리되어 z의 **구버전**에 재고정된 채 resolve되고,
이후 z가 갱신되며 b가 다시 stale이 되어도 재방문하지 않았다. **수정**: 한 번 방문하면 끝나는
단일 pass를 **고정점(fixed-point) sweep**으로 바꿨다 — 매 sweep마다 `is_stale()`을 다시 확인하고,
더 이상 진행(resolve/regenerate)이 없을 때까지(상한: `len(affected_ids)+1`) 반복한다. b는 z가
갱신된 뒤의 sweep에서 다시 방문되어 z의 **최종** 버전에 재고정된다.
(`test_a_candidate_resolved_before_its_own_upstream_is_revisited_once_the_upstream_changes`)

**3. UNKNOWN을 검증 성공처럼 처리함.** FAIL이 없으면 무조건 재확인하거나 RESOLVED로 처리해,
`BasicConstraintChecker`가 UNKNOWN만 반환해도 stale이 해제됐다. **수정**: `_classify_evidence()`
가 FAIL/PASS/판정불가(UNKNOWN뿐이거나 evidence 없음) 세 갈래로 나눈다. 새 `RepairOutcome.
INCONCLUSIVE`(Notion 0단계의 `recovery_outcome` 어휘와 동일)가 "재생성/재확인은 했지만 검증은
확정적이지 않음"을 나타내며 RESOLVED와 분명히 구분된다. 판정불가인 cascade candidate는 아예
재생성/재확인을 시도하지 않고 `left_stale`에 담는다. (`test_unknown_only_reverify_after_
regeneration_is_inconclusive_not_resolved`, `test_unknown_only_cascade_candidate_stays_
stale_and_is_reported_left_stale`, `test_a_candidate_with_no_constraints_is_left_stale_
not_fabricated_as_resolved`)

**4. 본문을 보존하는 재확인에서 INFERRED 관계가 사라짐 + evidence가 구버전에만 묶임.**
`_reaffirm()`이 새 버전을 만들 때 그 artifact를 향한 INFERRED edge를 이어주지 않아, 다음
`impact_candidates`에서 빠지고 이후 변경에도 무효화되지 않았다. 또한 재확인에 쓴 PASS evidence가
구버전에만 기록되어 새 버전의 `live_evidence()`가 비었다. **수정**: `_carry_forward_inferred_
edges(old_ref, new_ref)`가 `old_ref`를 향한 모든 INFERRED edge를 `new_ref`로 다시 선언한다
(`from_artifact`는 자신의 현재 버전으로 재고정 — `add_inferred_edge`의 소급 무효화 규칙과
자연스럽게 맞물린다). `_reaffirm()`은 evidence를 `model_copy(update={"checked_artifact":
new_ref})`로 재바인딩해 다시 기록한다. (`test_reaffirm_carries_forward_an_inferred_edge_
onto_the_new_version`, `test_reaffirmed_evidence_is_rebound_to_the_new_version_so_live_
evidence_is_not_empty`)

**5. 분기 runner가 남은 실패를 놓치고 정상 종료함.** `decide()`가 반환한 첫 행동만 실행하고
나머지를 버려, 독립적인 A/B FAIL 중 A만 처리되고 B는 그대로인데도 "no_more_failures"로 종료됐다.
한 행동 안에서 일부는 생성 실패하고 일부는 PASS여도 같은 오판이 나왔다. **수정**:
`BranchExecutionResult.rounds[i].executions`가 그 round의 `decide()`가 반환한 **모든** 행동의
실행 결과를 담는다. 다음 round에 넘길 evidence는 `current_evidence`를 통째로 교체하지 않고,
`_apply_round_to_pending()`이 실행된 각 결과를 `pending`(artifact_id → 최신 FAIL evidence) 맵에
반영한다 — RESOLVED면 제거, STILL_FAILING/INCONCLUSIVE(FAIL 포함)면 갱신, PRODUCTION_FAILED/
NO_PRODUCER면 **그대로 둠**(원래 있던 trigger가 계속 유효). 어떤 행동도 건드리지 않은 artifact는
자동으로 계속 남는다. (`test_two_independent_failures_both_get_a_decided_action_executed`,
`test_an_unresolved_independent_failure_is_not_silently_dropped_after_the_other_resolves`)

**6. 복구 cascade가 Stage 3의 예산 차단과 검사 비용 기록을 우회함.** 개별 생성/검사 호출 사이에
예산 확인이 없어, 예산 50에 호출당 100토큰인 대상 두 개가 둘 다 실행되어 200토큰을 썼다. 직접
호출한 validator 검사의 비용도 기록되지 않았다. **수정**: `RepairOperationExecutor`가
생성자에서 `ToolExecutor`를 받아, 모든 생성 호출은 `tool_executor.run_generation()`으로, 모든
검사 호출은 `tool_executor.execute(CheckCall(...))`로 나간다 — 허용 확인과 비용 기록이 Stage 3가
이미 구현·검증한 지점에서 정확히 한 번 일어난다. `BranchingRunner`도 분기마다 독립된
`ToolExecutor`를 만들어 round 안의 각 행동 실행 전에 `tool_executor.is_suspended`/
`budget.remaining`을 확인한다(round 사이가 아니라 행동 사이에도). `check_timing="final_only"`의
1회 검사도 같은 경로를 쓴다. (`test_budget_is_checked_between_individual_generation_calls_
within_one_execute`, `test_budget_is_enforced_between_actions_within_one_round_not_only_
between_rounds`)

**7. 분기별 생성 입력이 완전히 독립적이지 않음.** 모든 분기가 같은 `context` dict(딥카피 아님)를
공유해, Producer가 그 안의 mutable 객체(예: 실제 `DocumentGraph`)를 바꾸면 분기끼리 서로 영향을
줬다. **수정**: `BranchingRunner.run_branches()`가 `context_factory(spec)`을 분기마다 호출해
독립된 context를 얻는다 — `downstream_policy_factory`/`budget_factory`와 같은 패턴.
(`test_each_branch_gets_an_independent_context_not_a_shared_mutable_one`)

**8. whole-document baseline이 전체 문서 재생성을 보장하지 않음.** `AlwaysRegenerateDocumentPolicy`
가 REGENERATE_DOCUMENT를 고르면 executor가 다른 연산과 같은 **선택적** cascade를 수행해, PASS인
section은 재생성되지 않고 기존 본문을 유지했다 — "fixed graph-aware"가 REGENERATE_DOCUMENT로
후퇴한 경우와 실질적으로 구분되지 않았다. **수정**: `RepairOperationExecutor.execute()`가
`action.action_type == REGENERATE_DOCUMENT`일 때 전용 `_execute_unconditional()`로 분기한다 —
document와 그 `impact_candidates` 전체를 무조건, 선택 없이 재생성한다(실패 여부 보고는 여전히
한다). 두 baseline의 차이는 이제 "이 scope를 얼마나 자주 고르는가"에만 있다 — 한번 고르고 나면
항상 동일하게 "전부" 다시 만든다. (`test_regenerate_document_forces_every_section_even_ones_
that_would_pass`)

**추가 (8건과 별도).** `use_diagnostic_feedback=False`여도 `base_context`에 남아있던
`error_message`가 그대로 전달됐다. **수정**: feedback이 꺼지면 `context.pop("error_message",
None)`으로 명시적으로 제거하고, `BlockPatchProducer`처럼 메시지가 필수인 경우에만 중립 값
`"retry"`로 대체한다. (`test_blind_action_clears_a_stale_error_message_from_base_context`,
`test_blind_patch_block_gets_a_neutral_placeholder_not_a_stale_message`)

**검증**: `test/core/`에 18개 테스트 추가(`test_repair_operations.py` 16→29,
`test_branching_runner.py` 11→16, `test_adapters.py` +content 검증 — 192→210). 전체 저장소
테스트 실행 결과도 기존과 동일한 57건(무관, Stage 1부터 기록된 결함) 외 회귀 없음 — 756 passed.
이번 리뷰는 코드 리뷰 전이 아니라 **첫 공개 이후** 받은 리뷰이며, Stage 1-3과 마찬가지로 다음
리뷰 라운드가 또 있을 수 있다는 전제로 다룬다.

(이 수치는 2026.10.20 리뷰에서 사용자가 직접 정정했다 — 192→210 증가량은 18개이며, 처음
보고했던 "45개 테스트 추가"는 오기였다. 자세한 내용은 §5-2 참고.)

---

## 5-2. 사용자 재검토 후 수정 (2차, 2026.10.20)

1차 수정 이후 사용자가 다시 코드를 검토해 4건의 실제 결함(과 1건의 문서 오기)을 재현 시나리오와
함께 지적했다. 전부 수정했고, 각 수정마다 회귀 테스트를 추가했다 — 수정 전 코드에 그 테스트를
그대로 돌려 실제로 실패하는지까지 확인했다(`docgen_stage_review_rigor` 메모에 패턴으로 기록).

**1. 복수 대상 복구와 전체 재생성에서는 여전히 stale이 남아도 RESOLVED.** 두 재현:
(a) a→b에서 복구 대상을 `[b, a]` 순서로 지정하면, b가 a보다 먼저 처리되어 "resolved"로 기록된
뒤 a가 나중에 바뀌며 b가 다시 stale이 되는데, `_execute_selective()`가 명시적으로 지정한 대상을
재검사 후보(`affected_ids`)에서 빼버려서 b가 다시는 점검되지 않았다. (b) `doc→z→b`(doc이
상류, z는 doc을 읽고, b는 z를 읽음) 전체 문서 재생성에서 `_execute_unconditional()`이 여전히
ID 정렬 순서(`doc, b, z`)로 처리해, doc이 z의 **구버전**을 기준으로 재생성된 뒤 z가 나중에
바뀌어도 아무도 doc을 다시 보지 않았다. **수정**: (i) `_execute_selective()`에서 명시적 대상을
재검사 후보에서 빼는 코드를 제거해 sweep이 모든 대상을 계속 추적하게 했다. (ii)
`_execute_unconditional()`에 `_execute_selective()`와 동일한 고정점 sweep을 추가하고, 1차
처리 순서도 `_dependency_order()`(Kahn 알고리즘, 순환이 있으면 ID 정렬로 대체 — `artifact_
graph.py`가 순환을 명시적으로 허용하므로 toposort를 정확성의 전제로 삼지 않음)로 바꿔 sweep이
해야 할 일을 줄였다. (iii) 두 메서드 모두 최종 `left_stale` 판정을 "한 번이라도 손댔으면
제외"가 아니라 `last_outcome`(artifact_id별 가장 최근 결과, 실행 순서대로 갱신)이 RESOLVED이거나
아예 없는 경우에만 `is_stale()`을 다시 확인하도록 바꿨다 — STILL_FAILING/INCONCLUSIVE/
PRODUCTION_FAILED/NO_PRODUCER는 이미 `regenerated`로 정직하게 보고되므로 제외. (`test_multiple_
explicit_targets_out_of_dependency_order_leave_no_target_stale`, `test_regenerate_document_
settles_a_multi_hop_chain_regardless_of_processing_order`)

**2. 복구 도중 새로 발견한 하위 실패는 branching runner에서 사라짐.** a만 실패했다고 알려주고
a를 복구하면 cascade가 하위 b의 실패를 새로 발견하는 상황에서, b는 원래 `pending`에 없었기
때문에 b가 PRODUCTION_FAILED이거나 UNKNOWN/제약없음(`left_stale`)이어도 `_apply_round_to_
pending()`이 되돌릴 FAIL evidence가 없어 `pending`이 비게 되고 `stopped_reason="no_more_
failures"`로 오판했다. **수정**: `_run_one_branch()`에 branch 전체에서 유지되는 `unresolved_ids`
집합을 추가 — `_apply_round_to_pending()`이 PRODUCTION_FAILED/NO_PRODUCER/left_stale이 된
artifact_id를 여기 추가하고 RESOLVED/reaffirm PASS가 되면 제거한다. 이 집합은 **`decide()`에
절대 전달하지 않는다**(UNKNOWN을 FAIL로 조작하지 않는다는 사용자의 명시적 요구) — 오직
`_finalize_stopped_reason()`이 "no_more_failures"/"*_policy_returned_nothing"을 그대로 쓸지
`"unresolved_remaining"`으로 바꿔 쓸지 고르는 데만 쓰인다. (`test_a_newly_discovered_production_
failure_is_not_read_as_no_more_failures`, `test_a_newly_discovered_left_stale_candidate_is_not_
read_as_no_more_failures`)

**3. `BlockPatchProducer`가 patch한 block 하나의 내용으로 section 전체를 반환.**
`regenerate_blocks()`는 patch한 block과 그대로 보존된 다른 block을 모두 포함한
`TextAgentOutput`을 반환하는데, `produce()`가 `self.last_output.model_dump_json()` 전체를
이 **BLOCK 단위** artifact의 content로 저장해, block 하나만 바뀌어야 하는 경우에도 저장된
내용에 다른 block까지 섞여 들어갔다. **수정**: 반환된 `blocks`에서 `block_id == artifact_id`인
항목만 찾아 그 블록의 `model_dump_json()`을 저장한다. 일치하는 블록이 없으면(이론상 있을 수
없지만) FAILED로 정직하게 보고한다. (`test_block_patch_producer_extracts_only_the_patched_
block_not_the_whole_section`, `test_block_patch_producer_reports_failure_when_no_returned_
block_matches`)

**4. 검사기 실행 오류를 예산 소진으로 잘못 보고.** `RepairOperationExecutor._check()`와
`BranchingRunner`의 `check_timing="final_only"` 최초 검사가 `ToolExecutor.execute(CheckCall(...))`
의 `not observation.ok`를 전부 예산/행동수/suspend로 간주했다. 실제로는 `Validator.check()`
자체가 예외를 던지는 경우(`ToolExecutor`가 `ToolErrorCode.EXECUTION_FAILED`로 정확히 보고)도
같은 경로를 타는데, 예산이 10,000 남고 executor도 suspend되지 않은 상태에서 전체 cascade/branch
가 "예산 소진"으로 중단됐다. **수정**: `schemas.py`에 공개 상수 `BUDGET_ERROR_CODES =
{BUDGET_EXCEEDED, MAX_ACTIONS_EXCEEDED, SUSPENDED}`를 추가하고, `_check()`와 `final_only`의
초기 검사 모두 이 집합에 속하는 코드만 "예산 소진"(또는 `final_only`의 경우
`"initial_check_failed"`)으로 분류하게 했다. 그 외 오류(`EXECUTION_FAILED`/`TIMEOUT`)는 빈
evidence로 돌아와 기존 UNKNOWN/`left_stale` 경로가 그대로 정직하게 처리한다 — 새 반환 상태를
추가하지 않았다. (`test_a_validator_crash_is_not_misreported_as_budget_exhaustion`,
`test_final_only_initial_check_crash_is_not_misreported_as_budget_exhaustion`)

**검증**: `test/core/`에 8개 테스트 추가(`test_repair_operations.py` 29→32,
`test_branching_runner.py` 16→19, `test_adapters.py` +2 — BlockPatchProducer 관련 기존 2개
테스트는 실제 `TextAgentOutput`/`ContentBlock`을 쓰도록 수정만 하고 새로 세지 않음). 210→218.
각 새 테스트는 수정 전 코드에 그대로 돌려 실제로 깨지는 것까지 확인했다. 전체 저장소 테스트도
기존과 동일한 57건 외 회귀 없음 — 756→764 passed.

---

## 6. 하지 않은 것 (Stage 4 범위 밖, 다음 단계로 명시적으로 미룸)

- **실제 파이프라인 연결**: `DocumentGraph`/`PlannerAgent`/`TextAgent`가 실제로
  `ArtifactGraphStore.add_version`을 호출하도록 바꾸는 작업은 없다.
- **실제 로컬 LLM pilot 실행**: 이번 리뷰 반영 이후 별도로 진행한다.
- **CSP_SMT/SYMBOLIC 제약 검사**: Stage 3에서 이미 후속 작업으로 남긴 범위다.
- **무한 loop/반복 수정 방지 일반화**: `BranchingRunner`는 `max_downstream_rounds`로 자신의
  round 수만 제한한다 — Notion 5단계 체크리스트의 더 일반적인 안전장치는 범위 밖이다.
- **방향 1의 finer-grained artifact kind에 대한 실제 Producer**: `RepairOutcome.NO_PRODUCER`로
  정직하게 보고하는 것까지가 이번 단계의 범위다.
