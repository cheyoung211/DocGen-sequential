# Stage 5 — LangGraph 기반 자율 controller MVP

Notion "Revision 2026.10.15" 5단계 구현 기록. Stage 1-4 문서와 동일한 형식을 따른다.

## 0. 코드 위치

- `src/core/controller.py` — `ControllerPolicy` Protocol, `RuleBasedController`(Stage 4의
  `FixedGraphAwareRepairPolicy`를 그대로 재사용해 scope를 결정), `LLMController`(로컬/API
  LLM에게 구조화된 JSON으로 다음 행동을 직접 받음), `ControllerRuntime`(LangGraph
  `StateGraph` — `controller`/`tool` 두 노드 + 조건부 edge), `compute_ready_artifacts`
  (readiness 정의, 병렬 실행에는 연결하지 않음).
- `src/core/schemas.py` — `PatchToolResult`/`ReplanToolResult`(PATCH/REPLAN의 진짜 실행
  결과 schema), `ControllerStopReason`, `ControllerStepRecord`, `ControllerState`,
  `LLMControllerDecision` 추가.
- `src/core/tool_executor.py` — **Stage 3 코드 자체를 수정** (사용자 결정, 아래 §2 참고):
  `producer_registry`/`constraint_provider`/`repair_context`/`model_name`을 constructor에
  추가로 받으면 PATCH/REPLAN을 `repair_operations.RepairOperationExecutor`로 실제 디스패치한다.
  둘 다 주지 않으면(모든 기존 호출부·테스트) Stage 3의 NOT_IMPLEMENTED 동작이 그대로 유지된다.
- `test/core/test_controller.py`(12개 신규), `test/core/test_tool_executor.py`(+4개, 기존
  NOT_IMPLEMENTED 테스트는 네이밍만 명확히 함).
- `requirements.txt` — `langgraph==1.2.16` 추가(정확히 고정 — `qwen_vllm`의 `vllm==0.19.1`과
  같은 이유: 메이저 버전 간 동작을 매번 재검증하지 않는 의존성은 고정한다).

Zero changes to `scripts/run_full_pipeline.py` 또는 agent 코드 — Stage 1-4와 동일. 유일한
예외는 `tool_executor.py`이며, 이것도 **기존 동작을 전혀 바꾸지 않는 순수 추가**다(§2).

---

## 1. Notion 5단계 체크리스트 대응

### 1.1 "typed state, conditional routing, checkpoint/resume을 사용해 controller→tool/worker→observation→controller 루프를 구성한다."

`ControllerRuntime`이 `langgraph.graph.StateGraph(ControllerState)`를 빌드한다.
`ControllerState`(순수 pydantic model, `langgraph` import 없음 — schema와 behavior를 분리하는
기존 원칙 유지)는 `document_ref`/`checkpoint`/`budget`/`current_evidence`/`compiled`/
`step_count`/`history`/`pending_tool_call`/`done`/`stop_reason`을 담는다. 그래프는 정확히 두
노드다:

- `controller` 노드(`_controller_node`): anti-loop guard를 먼저 확인하고, 안 걸리면
  `ControllerPolicy.decide(state, store)`를 호출해 `pending_tool_call`을 정한다.
- `tool` 노드(`_tool_node`): `pending_tool_call`을 이 runtime의 `ToolExecutor`로 실제
  실행하고, 관찰 결과(evidence/budget/compiled/done/stop_reason)를 state에 반영한다.

`controller → tool`은 무조건 edge, `tool → (controller | END)`는 `state.done`에 따른 조건부
edge다 — "관찰 결과에 따라 다음 행동이 달라지는" 루프를 그대로 구현한다.

**Checkpoint/resume은 두 층으로 구현했다** (아래 §3에서 상세):
1. LangGraph 자체 checkpointer(`InMemorySaver`, 기본값) — 같은 프로세스 안에서 `thread_id`로
   재생/재개 가능.
2. `ControllerRuntime.resume_from_checkpoint` — Stage 2의 `Checkpoint`를 새 `branch_id`로
   복원해, 프로세스가 죽은 뒤에도(LangGraph의 in-memory checkpointer가 사라진 뒤에도) 이어서
   실행할 수 있다.

### 1.2 "남은 예산과 검증 결과를 읽는 고정 규칙 controller와 LLM controller를 각각 연결한다."

둘 다 `ControllerPolicy.decide(state, store) -> ToolCall`를 구현한다 (Stage 1의
`RepairPolicy.decide()`와 동일한 **side-effect-free** 원칙 — anti-loop guard는 이 메서드
안이 아니라 `_controller_node`가 호출하기 *전에* 확인한다).

- **`RuleBasedController`**: evidence가 없으면 CHECK, FAIL이 있으면 Stage 4의
  `FixedGraphAwareRepairPolicy`(수정 없이 그대로)에게 scope를 물어 그 중 하나를 PATCH/REPLAN
  으로 변환해 실행, 실패가 없고 아직 컴파일 안 했으면 COMPILE, 컴파일했으면 FINISH.
- **`LLMController`**: `LLMGenerator` Protocol(`generate`/`extract_json_block` — 실제
  `scripts.llm_client.LLMClient`가 그대로 만족)에게 현재 budget/evidence/history를 요약한
  prompt를 주고 `LLMControllerDecision`(JSON) 하나를 받아 파싱한다. 파싱이
  `max_attempts`번 다 실패하면 안전하게 `FinishCall`로 종료한다 — "LLM이 보고하는 confidence는
  보정된 확률로 취급하지 않는다"는 방향 2의 주의와 같은 이유로, 파싱 실패도 그대로 재시도만 하다
  멈추지 않고 명시적으로 끝낸다.

로컬 Qwen3.6(vLLM)은 `LLMClient.supports_structured_output()`이 `False`를 반환하므로
(provider="local"), `LLMController`는 `generate_structured()`가 아니라 평범한 `generate()` +
`extract_json_block()`(기존 `TextAgent`/`PlannerAgent`가 쓰는 것과 동일한 메서드)으로 직접
JSON을 파싱한다.

### 1.3 "finish·재계획·반복 검사 조건을 구현하고 무한 loop, 동일 수정 반복, 예산 초과를 막는다."

세 guard 모두 `_controller_node`/`_tool_node`에만 있다 — 어떤 `ControllerPolicy`도 이걸 알
필요가 없다(Stage 3의 `ToolExecutor`가 budget/action-count를 강제하는 것과 같은 층 분리).

- **무한 loop / 최대 step**: `max_steps`. `step_count >= max_steps`이면 `_controller_node`가
  `decide()`를 호출하지 않고 직접 `FinishCall(reason="max_steps_reached", ...)`을 만든다.
- **동일 수정 반복**: `_repeated_action(history, threshold=repeat_threshold)`. 최근
  `threshold`개 step의 `call_summary`(행동+target+version을 식별하는 한 줄 문자열)가 전부
  동일하고 전부 `ok=True`일 때만 발동한다 — 같은 대상을 계속 고쳐도 실패가 계속되는 경우(값은
  매번 다른 버전이라 `call_summary`가 달라짐)는 이 guard가 아니라 `max_steps`로만 멈추게
  설계했다(실제 동작을 `test_a_run_of_still_failing_repairs_does_not_trip_the_repeat_guard`로
  확인).
- **예산 초과**: `_tool_node`가 거부된 호출의 `error.code`가 `schemas.BUDGET_ERROR_CODES`에
  속하면 즉시 `done=True`/`stop_reason=BUDGET_EXHAUSTED`로 끝낸다 — 컨트롤러가 스스로 FINISH를
  고르지 않아도 멈춘다.

### 1.4 "의존성이 충족된 독립 artifact만 병렬 처리하도록 readiness를 정의한다. 병렬 실행은 버전 충돌 처리까지 준비된 뒤 적용한다."

`compute_ready_artifacts(store, candidate_ids)`: candidate가 (a) 현재 stale하고 (b) 그
artifact의 모든 입력(= `to_artifact`가 이 artifact의 현재 버전인 edge들의 `from_artifact`)이
stale하지 않을 때만 "ready"로 판정한다. **`ControllerRuntime`의 루프 어디에서도 호출하지
않는다** — 체크리스트가 명시한 대로 병렬 실행 자체는 이번 단계에 포함하지 않았고, 이 함수는
개념을 지금 정의·테스트해 두어 나중에 버전 충돌 처리가 준비됐을 때 바로 쓸 수 있게 하는 것만이
목적이다.

---

## 2. 사용자 결정: PATCH/REPLAN을 어디서 실제로 연결할 것인가

구현 전 사용자에게 두 가지 방식을 제안해 결정받았다:

- (a) Stage 3 `ToolExecutor`는 그대로 두고, LangGraph의 `tool` 노드가 PATCH/REPLAN일 때만
  `RepairOperationExecutor.execute()`를 직접 호출.
- (b) `ToolExecutor` 자체를 수정해 PATCH/REPLAN 분기 안에서 내부적으로
  `RepairOperationExecutor`를 호출하도록 바꾼다.

**사용자가 (b)를 선택했다.** `tool_executor.py`를 다음과 같이 수정했다:

- `ToolExecutor.__init__`이 `producer_registry`/`constraint_provider`/`repair_context`/
  `model_name` 네 개의 **선택적** keyword-only 인자를 새로 받는다. `producer_registry`와
  `constraint_provider`를 둘 다 주면, `__init__`이 그 자리에서 `self`를 넘겨
  `RepairOperationExecutor(store, self, producer_registry, constraint_provider, ...)`를
  만들어 `self._repair_executor`에 보관한다 — `RepairOperationExecutor`가 이미 Stage 4부터
  "자기 budget 체크를 위해 ToolExecutor를 하나 받는다"는 설계였으므로, 같은 인스턴스(`self`)를
  넘기면 PATCH/REPLAN의 중첩 generate/check 호출도 이 executor의 budget 원장(ledger)
  하나로만 집계된다 — 별도 ledger가 생기지 않는다.
- 순환 import 문제: `repair_operations.py`가 모듈 최상단에서 `tool_executor.py`를 import
  하므로, 반대 방향 import는 `__init__` **안에서** 지연 import(`from src.core.repair_operations
  import RepairOperationExecutor`)로 처리했다 — 클래스 정의 시점에는 문제가 없고, 인스턴스
  생성 시점에는 두 모듈 다 이미 로드되어 있으므로 안전하다.
- 둘 다 주지 않으면(`None`, 기본값) `self._repair_executor`는 `None`으로 남고, PATCH/REPLAN은
  Stage 3 원문 그대로 `NOT_IMPLEMENTED`를 반환한다 — budget도 action count도 건드리지 않는다.
  **기존 48개 `test_tool_executor.py` 테스트(이름만 명확히 바꿨을 뿐 동작은 그대로) +
  `repair_operations.py`/`branching_runner.py`의 모든 기존 테스트가 전혀 수정 없이 그대로
  통과한다** — 이것이 "순수 추가"라는 주장의 근거다.
- `PatchCall`/`ReplanCall` → `RepairActionType` 매핑(`_dispatch_patch`/`_dispatch_replan`):
  - `PatchCall.target.kind`: BLOCK→PATCH_BLOCK, SECTION→REGENERATE_SECTION, 그 외→
    REPLAN_SUBGRAPH(단일 target).
  - `ReplanCall.targets`: 단일 DOCUMENT target→REGENERATE_DOCUMENT(executor의 **무조건**
    경로), 단일 SECTION target→REGENERATE_SECTION, 그 외(다중 target 포함)→REPLAN_SUBGRAPH.

  이 매핑은 라벨만 고르는 선택이 아니다 — REGENERATE_DOCUMENT는
  `RepairOperationExecutor._execute_unconditional`이라는 **별개의 실행 경로**(선택적 skip
  없음)를 타므로, 단일 DOCUMENT target을 `PatchCall`로 잘못 보내면(그 매핑 규칙은 "BLOCK/
  SECTION 아니면 REPLAN_SUBGRAPH") 조용히 선택적 cascade로 격하되어 "항상 전체 재생성"이라는
  의미가 깨진다. `test_replan_with_a_document_target_runs_the_unconditional_path`로 이
  매핑이 올바른지 직접 확인했다.

---

## 3. Checkpoint/resume: 왜 두 층인가

`ControllerRuntime`은 한 episode 동안 **살아있는** `ArtifactGraphStore`/`ToolExecutor`를
자기 자신(`self`)에 들고 있다 — LangGraph의 state(`ControllerState`)에서 매번
`from_checkpoint`로 재구성하지 않는다. 이유: `ArtifactGraphStore.from_checkpoint`는 항상
**새 `branch_id`**를 요구하도록 Stage 2에서 설계되어 있다(복원 후 LLM을 다시 부르는 실행은
별도 run으로 기록한다는 규칙). 매 node 호출마다 재구성했다면, 정상적인 한 episode가 step마다
새 branch로 쪼개져 "하나의 연속된 실행"이라는 전제 자체가 깨진다.

대신 `state.checkpoint`는 `_tool_node`가 매 dispatch 뒤에 `store.save_checkpoint()`로 갱신하는
**스냅샷**일 뿐이다 — LangGraph 자신의 checkpointer(`InMemorySaver`)가 이 전체 typed state를
프로세스 안에서 재생/복제할 수 있게 해 주는 역할과, 실제로 프로세스가 죽었을 때 복구할 재료
역할 두 가지를 겸한다.

`ControllerRuntime.resume_from_checkpoint(checkpoint, ..., new_branch_id=...)`가 실제 복구
경로다: `ArtifactGraphStore.from_checkpoint`로 **새 branch**에 store를 재구성하고,
`checker_factory(store)`로 그 새 store에 바인딩된 새 `Validator`를 만든다(Stage 4의
`BranchingRunner`가 `validator_factory(store)`를 쓰는 것과 똑같은 이유 — `Validator`가 생성
시점에 특정 store에 바인딩되므로 분기/재개마다 새로 만들어야 한다). `budget`은 checkpoint가
들고 있지 않으므로(Stage 2의 `Checkpoint`는 artifact graph 범위만 다룬다) 호출자가 마지막으로
알던 남은 예산을 직접 넘겨야 한다.

`test_resume_from_checkpoint_reconstructs_the_store_under_a_new_branch_id`가 이 전체를
확인한다: 새 `Validator`가 재구성된 store에 바인딩됐는지, 재개된 store가 원본과 완전히 다른
인스턴스인지, 재개된 episode를 실행해도 원본 branch(`store`)의 `doc`은 전혀 건드려지지 않는지.

---

## 4. 자체 발견한 버그 (사용자 리뷰 전)

테스트를 작성하는 중 발견한 한 가지, Stage 1-4에서도 반복됐던 패턴과 같은 종류의 문제:

**`test_tool_executor.py`에 새 fake를 추가할 때 이름이 충돌했다.** 처음에는 새 PATCH/REPLAN
real-dispatch 테스트용 fake를 `_FakeProducer`/`_AlwaysPassValidator`라는 이름으로 썼는데, 이
파일 **뒤쪽**(`run_generation` 섹션)에 이미 `_FakeProducer`라는 다른 클래스(생성자 시그니처도
다르고 `content=None`/`kind=SECTION`을 고정 반환)가 있었다. Python은 같은 모듈 안에서 나중에
나온 `class` 문이 그 이름을 덮어쓰므로, 내 새 테스트들은 전부 "그 다른 클래스"의 fake를 쓰고
있었다 — `Producer` Protocol을 똑같이 만족하니 아무 예외도 나지 않고, 그냥 조용히 틀린 결과
(`content=None`, 엉뚱한 kind)를 돌려줬다. 처음 작성한 assertion이 그 틀린 값(`content is
None`, `budget.spent == 0`)을 "정상"이라고 착각하게 만들 뻔한 걸, 더 구체적인 assertion
(`store.latest("a").content == "fixed:a"`, `budget.spent > before`)을 넣어서 직접 잡았다.
새 fake들을 `_FakeRepairProducer`/`_AlwaysPassRepairValidator`로 고유하게 이름 지어 해결했고,
그 클래스 docstring에 왜 이 이름을 골랐는지(파일 순서에 의존하지 않는 유일한 해법) 남겼다.

**교훈 (`docgen_stage_review_rigor`에 추가할 후보)**: 같은 테스트 파일에 여러 섹션이 쌓이면서
"관례적인" fake 이름(`_FakeProducer`, `_FakeValidator`, ...)이 재사용되기 쉽다 — Python은 이걸
import 에러도 아니고 타입 에러도 아닌, 조용한 이름 재바인딩으로만 처리하므로 테스트가 그냥
"통과"해 버린다(틀린 객체를 대상으로). 새 fake를 추가하기 전에 같은 파일에서
`grep -n "class _Fake"` 한 번 해보는 것이 저렴한 예방책이다.

---

## 5. 다음 단계로 미룬 것

- 실제 로컬 Qwen3.6 파이프라인으로 돌리는 pilot 실행(rule-based/LLM controller 둘 다) — 사용자가
  명시적으로 "메커니즘 + fake 테스트만" 이번 단계 범위로 확정했다. 코드 리뷰 이후 별도 진행.
- 병렬 실행(버전 충돌 처리가 준비된 뒤 — Notion 5단계 체크리스트 자신의 명시적 유보).
- `LLMController`가 실제 토큰 비용을 측정해 budget에 반영하는 것 — 이번 단계의 `LLMGenerator`
  Protocol은 `generate()`가 `usage_sink`를 받지 않는 얇은 인터페이스라, 실제 `LLMClient`를
  연결할 때 `cost_tracking.CostTrackingProducer`와 같은 방식의 wrapper가 필요할 것이다. 지금은
  의도적으로 범위 밖에 두었다(별도 "controller 자신의 LLM 호출 비용" 트래킹은 아직 설계되지
  않음 — 방향 2의 "컨트롤러"가 쓰는 토큰과 "생성기"가 쓰는 토큰을 어떻게 같은 budget에 합산할지
  결정이 필요).
- `ArtifactGraphStore.log.tool_calls`(branch별 영구 로그)와 `ControllerState.history`(episode
  trace, resume을 넘어 생존)가 지금은 서로 독립적으로 쌓인다 — 둘을 사후에 하나의 timeline으로
  재구성하는 유틸리티는 만들지 않았다.

---

## 6. 사용자 1차 재검토 후 수정 (2026-10-20)

1차 공개 후 사용자가 코드 위치·재현 시나리오와 함께 7건의 실제 결함을 지적했다. 전부 수정했고
각 수정마다 회귀 테스트를 추가했다. Stage 1-4가 모두 거친 다회차 리뷰 패턴이 Stage 5에도
그대로 적용됐다 — 상세는 `src/core/controller.py`의 모듈 docstring(이번 라운드 전체를
7개 항목으로 기록)과 메모리 `docgen_stage_review_rigor`(패턴 #42 이후)에도 동일하게 남겼다.

### 6.1 같은 artifact의 여러 제약 결과가 하나로 덮임 + 무효화된 evidence가 남음

`_merge_evidence()`가 `artifact_id`만 키로 써서, 같은 문서에 제약 A=FAIL·제약 B=PASS가 있으면
리스트에서 마지막으로 온 것만 남고 나머지는 사라졌다. **수정**: `(artifact_id, constraint_id)`
쌍으로 키를 바꿔 서로 다른 제약은 항상 독립적으로 보존한다.

별도로, 컨트롤러가 들고 있는 evidence 캐시는 저장소의 `live_evidence()`가 적용하는 무효화
규칙(입력이 바뀌면 PASS도 무효화)과 동기화되지 않았다. **수정**: `ArtifactGraphStore`에
새 public 메서드 `is_evidence_still_valid()`를 추가했다 — 기존 private `_evidence_is_live()`를
그대로 감싼 wrapper로, `live_evidence()`가 추가로 거는 "artifact 자신이 stale이면 안 됨"
조건은 빼고 노출한다(이유는 §6.2). 컨트롤러는 매 merge마다 `_prune_invalidated_evidence()`로
이 메서드를 호출해 무효화된 항목을 걸러낸다.

### 6.2 실패한 복구와 하위 cascade가 발견한 실패가 다음 판단에서 사라짐

`live_evidence()`는 **stale한 artifact에 대해 항상 빈 리스트를 반환**하도록 설계되어 있다(그
artifact의 "현재 settled 상태"에 대한 신뢰할 수 있는 판정이 있을 수 없다는 뜻). 그런데
PRODUCTION_FAILED/NO_PRODUCER로 끝난 복구는 새 버전을 쓰지 않으므로 staleness가 절대
풀리지 않고, 동시에 그 복구 시도 자체는 check까지 가지도 못해 fresh evidence도 생기지 않는다
— evidence 기반 추적만으로는 이 상태를 영원히 표현할 방법이 없었다.

재현해보니 실제로 **두 가지 다른 원인**이 있었다:
- (a) 원래 stale했던 artifact(예: a→b에서 a가 바뀌어 b가 stale)가 NO_PRODUCER로 막히면,
  `is_stale()`은 계속 True로 남는다 — 이 경우는 `is_stale()`만 봐도 잡힌다.
- (b) **애초에 한 번도 stale이었던 적이 없는** artifact(의존성이 아니라 자기 콘텐츠 자체가
  실패하는 경우, 예: 문서 초안 자체에 결함)가 PRODUCTION_FAILED면, `is_stale()`은 처음부터
  끝까지 False다 — 이 경우는 `is_stale()`로 전혀 잡히지 않는다. 사용자가 지적한 "a의 생성이
  PRODUCTION_FAILED여도 기존 FAIL이 삭제되어 COMPILE로 진행" 시나리오가 정확히 이 경우였다
  (다른 artifact에 이미 PASS가 있으면 `_classify()`가 그 PASS만 보고 깨끗하다고 읽음).

**수정**: `ControllerState`에 두 개의 집합을 추가했다.
- `tracked_artifact_ids`: 이 episode가 호출하거나 cascade가 건드리거나 발견한 모든
  artifact_id. (a)를 잡기 위해 `store.is_stale()`과 함께 쓴다.
- `blocked_artifact_ids`: 가장 최근 복구 시도가 PRODUCTION_FAILED/NO_PRODUCER로 끝난
  artifact. (b)를 잡기 위한 것 — 나중에 실제로 생성이 성공하면(그 결과가 PASS든 FAIL이든
  UNKNOWN이든) 즉시 해제된다.

새 `_compute_unresolved(store, tracked_ids, blocked_ids)`가 두 집합을 합쳐 반환하고,
`RuleBasedController.decide()`는 FAIL evidence를 처리한 뒤 이 결과가 비어있지 않으면 — 아직
확인 안 한 것이 있으면 CHECK를, 확인했는데도 안 풀린 것이 있으면
`FinishCall(reason="unresolved_remaining")`을 — COMPILE보다 먼저 고려한다.

### 6.3 "resolved" 종료 조건 보강

UNKNOWN evidence만 있어도 `compiled=True`면 그대로 resolved로 끝났다. **수정**:
`RuleBasedController`가 이제 "FAIL 없음"이 아니라 "엄밀히 PASS"를 요구하고, UNKNOWN이면
`FinishCall(reason="evidence_inconclusive")`로 명시적으로 끝낸다.

컴파일 성공 후 PATCH로 내용이 바뀌어도 `compiled`가 무효화되지 않아 재컴파일 없이 종료할 수
있었다. **수정**: `_tool_node`가 PATCH/REPLAN이 성공적으로 디스패치될 때마다(결과와 무관하게)
`compiled`/`last_compile_failed`를 무조건 `False`로 리셋한다.

추가로, 규칙 controller가 컴파일 실패 후 같은 COMPILE을 반복guard로만 멈추던 문제도 이번에
같이 고쳤다: `last_compile_failed` 플래그를 추가해, 이미 한 번 실패한 컴파일을 재시도하지 않고
`FinishCall(reason="compile_failed")`로 바로 정직하게 멈춘다 — 규칙 기반 정책은 컴파일러
진단을 해석할 방법이 없으므로, 반복 guard에 기대기보다 스스로 멈추는 쪽을 선택했다.

### 6.4 LLM controller가 도구의 실제 observation을 받지 못함

도구 결과는 저장소 로그에 남지만, 다음 LLM 입력에는 `inspect:doc@latest → ok`처럼 실행
요약만 들어갔다 — INSPECT/QUERY가 실제로 무엇을 찾았는지, COMPILE이 왜 실패했는지(엔진·
return code·fatal error 수·첫 오류 유형)가 전혀 전달되지 않았다. **수정**: 새
`ControllerStepRecord.detail` 필드 + `_describe_observation()`이 매 dispatch의 실제 payload를
사람이 읽을 수 있는 한 줄로 기록하고, `LLMController._user_prompt()`가 최근 history의
`detail`을 그대로 포함한다.

### 6.5 Durable resume이 controller 실행 상태를 이어받지 않음

artifact 저장소는 새 branch로 복원됐지만, `run()`은 항상 새 `ControllerState()`를 만들어
`history`/`step_count`/`current_evidence`/`compiled`를 초기화했고, 새 `ToolExecutor`도
action count·suspension을 0/False로 다시 열었다 — "재개"가 아니라 artifact graph만 이어받은
새 episode였다.

**수정**: `ControllerState`에 `actions_taken`/`suspended`/`suspend_reason`을 추가해
`ToolExecutor`의 admission 상태를 그대로 복제하고, `ToolExecutor.__init__`에 같은 이름의
선택적 인자 3개를 추가했다(기존 호출부는 전혀 영향 없음). `resume_from_checkpoint`는 이제
bare `Checkpoint`+`Budget`이 아니라 **직전 전체 `ControllerState`**를 받아, 그 안의
`checkpoint`로 store를 새 branch에 복원하면서 `actions_taken`/`suspended`/`suspend_reason`도
새 `ToolExecutor`에 그대로 넘기고, `run()`도 `resume_state`가 있으면 `history`/`step_count`/
`current_evidence`/`tracked_artifact_ids`/`blocked_artifact_ids`/`compiled`를 그 상태에서
이어간다(`checkpoint` 필드만 재개 시점의 새 스냅샷으로 갱신).

### 6.6 Controller 자신의 LLM 호출이 예산을 우회함 + 호출 자체의 예외가 전파됨

`LLMController.decide()`가 `generator.generate()`를 직접 호출해 `budget.cap=0`이어도 호출이
나갔고, 비용도 원장에 전혀 반영되지 않았다. 또한 `generate()`가 예외(timeout 등)를 던지면
그대로 `decide()` 밖으로 전파됐다 — JSON 파싱 실패만 처리하고 있었다.

**수정**: 새 `_LLMDecisionProducer`(`Producer` Protocol 구현)가 "LLM에게 다음 행동을
물어보는 것"을 하나의 generate 호출로 감싸고, `LLMController.decide()`는 이를
`tool_executor.run_generation()`으로 실행한다 — 호출 전 admission 확인(cap=0이면 모델을
전혀 안 부름), 호출 후 Stage 3의 같은 원장으로 비용 반영까지 기존 generate 호출과 동일한
경로를 탄다. `_LLMDecisionProducer.produce()` 자신의 재시도 루프 안에서 JSON 파싱 실패와
`generate()`의 예외를 **같은 방식으로** 잡아 제한된 횟수만 재시도하고, 끝까지 실패해도 예외를
밖으로 내보내지 않고 `ProductionResult(FAILED, ...)`를 반환한다 — `ControllerPolicy.decide()`
자체는 이 한 가지 side effect(예산 소비)만 예외적으로 허용한다는 점을 Protocol docstring에
명시했다(이 runtime이 `decide()`를 매 step 정확히 한 번만, 항상 결과를 실행하는 방식으로
부르기 때문에 안전하다는 근거와 함께).

### 검증

`test/core/test_controller.py`가 12개→27개로 늘었다(신규 15개, 7개 버그 각각에 최소 1개
이상의 회귀 테스트). `test/core/` 236→251개 전부 통과. 전체 저장소 기준 797 passed, 기존과
동일한 57건(무관, Stage 1부터 기록, Stage 8 전 정리 예정) + 끊어진 테스트 파일 1개(동일) 외
회귀 없음. 변경된 파일: `src/core/artifact_graph.py`(`is_evidence_still_valid` 추가, 순수
추가), `src/core/tool_executor.py`(`actions_taken`/`suspended`/`suspend_reason` 선택적
인자 추가, 기존 동작 불변), `src/core/schemas.py`(`ControllerState`에 `tracked_artifact_ids`/
`blocked_artifact_ids`/`last_compile_failed`/`actions_taken`/`suspended`/`suspend_reason`
추가, `ControllerStepRecord`에 `detail` 추가), `src/core/controller.py`(위 6.1-6.6 전체).
