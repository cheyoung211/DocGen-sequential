# Stage 3 설계: 도구 실행기, 기본 제약 검사, 비용 집계

이 문서는 Notion "SNU / NCIA / Personal Research / Multimodal Agents / Revision 2026.10.15"
계획의 **Stage 3** 산출물이다. Stage 1(검증·복구·평가 분리, `docs/stage1_experiment_design.md`)과
Stage 2(versioned artifact graph, `docs/stage2_artifact_graph.md`)는 이미 완료되었다. 근거는
`file:line` 형태로 표기한다.

> **범위**: 이 문서와 `src/core/{tool_executor,constraint_checks,cost_tracking}.py`, `src/core/schemas.py`의
> 신규 schema는 **추가(additive)** 구현이다. `scripts/run_full_pipeline.py`나 기존 agent 코드는
> 전혀 바뀌지 않았다 (동작 변화 0) — Stage 1·2와 동일한 원칙. 오늘의 파이프라인을 이 executor로
> 실제 연결하는 것은 Stage 4/5의 일이다.

---

## 0. 코드 위치

| 구분 | 위치 |
|---|---|
| Stage 3 schema (`ToolActionKind`/`ToolErrorCode`/`ToolExecutionError`/7개 Call·Result/`ToolObservation`) | `src/core/schemas.py` (신규, 파일 끝에 추가) |
| 도구 실행기 (dispatch, budget/max-actions/suspend-resume 강제) | `src/core/tool_executor.py::ToolExecutor` |
| 기본 제약 검사 (`schema_contract`/`recompute`, `Validator` 구현) | `src/core/constraint_checks.py::BasicConstraintChecker` |
| 비용·지연·GPU 측정, 로컬/API 공통 모델 adapter | `src/core/cost_tracking.py::GpuUtilizationSampler`, `CostTrackingProducer` |
| 테스트 | `test/core/test_tool_executor.py` (24개), `test/core/test_constraint_checks.py` (14개), `test/core/test_cost_tracking.py` (13개) |

---

## 1. Notion 3단계 체크리스트 대응

### 1.1 "inspect/query/check/patch/replan/compile/finish의 입력·출력 schema와 실행 오류 형식을 정의한다."

- 방향 2의 7개 행동 후보("inspect_artifact, query_dependencies, check_constraints /
  check_calculation, patch_block, replan_subgraph, compile_document, finish")를 Stage 3
  체크리스트 자신의 7개 짧은 이름으로 접는다 — 체크리스트 문구 자체가
  "check_constraints / check_calculation"을 **"check" 하나**로 접었으므로, `ToolActionKind`에는
  `CHECK` 하나만 있다. 어떤 `ConstraintKind`를 실제로 검사하는지는 action 이름이 아니라
  `CheckCall.constraints`에 담긴 `Constraint` 목록의 속성이다 (`schemas.py`, `ToolActionKind` 바로
  위 주석).
- 7개 action 각각 `{Name}Call`/`{Name}Result` pydantic 모델로 입출력 schema를 고정했다
  (`InspectArtifactCall/Result`, `QueryDependenciesCall/Result`, `CheckCall/Result`, `PatchCall`,
  `ReplanCall`, `CompileCall/CompileToolResult`, `FinishCall/FinishToolResult`). 모든 Call은
  `action: Literal[...]`로 discriminated union(`ToolCall`)을 이룬다.
- 실행 오류 형식은 7개 action이 공유하는 하나의 `ToolExecutionError(code, action, message)` —
  `code`는 `ToolErrorCode`(`BUDGET_EXCEEDED, MAX_ACTIONS_EXCEEDED, SUSPENDED, NOT_IMPLEMENTED,
  UNKNOWN_ARTIFACT, TIMEOUT, EXECUTION_FAILED`)로 고정되어, 각 action이 제각기 오류 모양을
  발명하지 않는다. 모든 `ToolExecutor.execute()` 호출은 `ok`/`output`/`error`/`cost`를 가진
  하나의 `ToolObservation`을 반환하며, `output`/`error`는 상호 배타적임을 `model_post_init`이
  강제한다(`ProductionResult`의 success-xor-failure 패턴을 그대로 적용, `schemas.py`).

### 1.2 "참조·필수 구성요소·숫자 재계산부터 구현하고, 필요한 좁은 제약군에만 solver를 적용한다."

- `BasicConstraintChecker`(`constraint_checks.py`)가 Stage 1의 `Validator` Protocol을 구현한다 —
  새 병렬 인터페이스가 아니다. `adapters.TextAgentValidatorAdapter`(legacy `ContentBlock`/
  `LayoutBlock` 도메인)와는 완전히 분리된, `ArtifactGraphStore`/`ArtifactRef`/`Constraint` 위에서만
  동작하는 별도의 `Validator` 구현이다.
- **SCHEMA_CONTRACT**: "참조 무결성"과 "필수 구성요소"는 이 체커 수준에서 같은 메커니즘으로
  수렴한다 — `Constraint.applies_to`에 나열된 모든 `ArtifactRef`가 실제로 저장되어 있는지만
  확인한다(`_check_schema_contract`). artifact_id 존재뿐 아니라 정확한 `(artifact_id, version)`
  실재 여부까지 확인한다(`store.get_version`) — Stage 2의 `_validate_artifact_ref`가 이미 구분한
  "존재 여부 vs 버전 실재 여부"와 동일한 구분을 재사용. **이 체커는 artifact의 content가 실제로
  그 참조를 인용하는지는 검사하지 않는다** — 그건 레퍼런스 스캐너가 할 일이고, Stage 2가 이미
  명시적으로 미룬 범위다(`docs/stage2_artifact_graph.md` §3, "실제 레퍼런스 스캔 기반 INFERRED
  edge 자동 탐지").
- **RECOMPUTE**: 기대값/관측값 비교만 한다(`_check_recompute`). 숫자를 문서 content나 source
  data artifact에서 직접 추출하는 로직은 이 체커의 책임이 아니다 — `Validator.check(...,
  context=...)`의 기존 확장점(Stage 1의 `TextAgentValidatorAdapter.check()`가 이미 같은 방식으로
  `context`에 도메인별 정보를 담는 패턴)을 그대로 써서, 호출자가
  `context["recompute"][constraint_id] = {"expected": ..., "observed": ..., "tolerance": ...}`로
  값을 넘긴다. 값이 없으면 FAIL이 아니라 UNKNOWN/NOT_CHECKED.
- **CSP_SMT·SYMBOLIC**: 이번 단계는 미구현으로 남겼다 — 체크할 구체적인 narrow 제약 사례가 아직
  없고, z3/sympy 모두 `requirements.txt`에 없다. 항상 UNKNOWN/NOT_REPRESENTABLE을 반환하며
  (`_IMPLEMENTED_KINDS`에 없는 kind는 `check()`의 첫 분기에서 걸러짐), `set_enabled()`로도 켤 수
  없다 — "비활성화"가 아니라 "구현 없음"이므로 토글 자체가 `ValueError`. **Notion
  Progress.ver2에 이후 구현해야 할 후속 작업으로 기록**(사용자 결정, 아래 §3 참고).

### 1.3 "도구 결과에 evidence, 관련 version, unknown·timeout 정보를 포함한다."

- CHECK의 `CheckResult.evidence`는 `Evidence`(Stage 1) 그대로 — `unknown_reason`까지 포함.
- INSPECT/QUERY는 `ArtifactRef`/`StoredArtifact`를 통해 버전 정보를 그대로 노출한다
  (`InspectArtifactResult.stored.ref.version`, `QueryDependenciesResult.edges[*].{from,to}_artifact.version`).
- `ToolErrorCode.TIMEOUT`은 schema에 정의되어 있으나 **이번 단계에서 실제로 발생시키는 호출
  경로는 없다** — 모든 Stage 3 action(inspect/query/check/compile)이 로컬 연산이거나 로컬
  subprocess(`tectonic`)라 실제 타임아웃 시나리오가 아직 없다. `GpuUtilizationSampler`의
  `nvidia-smi` 호출에는 2초 타임아웃이 있지만, 이는 GPU 측정 실패일 뿐 도구 호출 자체의 실패가
  아니므로 `None` 측정값으로만 드러나고 `ToolErrorCode.TIMEOUT`을 발생시키지 않는다. 실제 LLM
  호출(Stage 4/5에서 patch/replan이 실제로 구현되면)이 연결되면 이 코드가 쓰일 자리다.

### 1.4 "초기 생성과 모든 후속 호출의 비용·지연을 기록하고 controller 외부의 executor에서 budget cap, 최대 행동 수, 중단·재개 조건을 강제한다. 필요 시 로컬/API 공통 모델 adapter를 둔다."

- **"로컬/API 공통 모델 adapter"는 `CostTrackingProducer`다** (`cost_tracking.py`) — 오늘 실제로
  존재하는 유일한 LLM 호출 표면인 Stage 1의 `Producer`(`PlannerAgentProducer`/
  `TextAgentProducer`)를 감싼다. "초기 생성"의 비용·지연이 바로 이 `produce()` 호출이다. 매
  `produce()` 호출마다 정확히 하나의 `CostRecord`를 만든다 — 내부 재시도 루프가 여러 번 실제
  LLM을 호출했어도(`TextAgent._run_generation_loop`), `usage_sink`에 쌓인 모든 항목을 합산하므로
  **실패한 내부 attempt의 토큰도 자동으로 포함**된다(완료 기준의 "실패한 시도 ... 도 비용에서
  누락되지 않는다"), 각 attempt를 따로 가로챌 필요 없이.
- **GPU 사용량 실측**: `GpuUtilizationSampler`가 로컬 모델(`qwen`으로 시작하는 model_name) 호출
  구간에만 백그라운드 스레드로 `nvidia-smi --query-gpu=utilization.gpu`를 주기적으로 샘플링해
  `평균 utilization × 경과 시간`을 `gpu_seconds`로 추정한다. **이것은 근사치다, 정확한
  per-process GPU 계정이 아니다** — `nvidia-smi`는 디바이스 전체 사용률만 보여주므로, 같은
  GPU를 동시에 쓰는 다른 프로세스가 있으면 구분할 수 없다. 새 의존성(`pynvml`/`nvidia-ml-py`)을
  추가하지 않기로 결정(사용자 결정, 아래 §3) — 이미 호스트에 있는 `nvidia-smi` CLI만 사용.
  API 모델은 `gpu_seconds=None`(측정할 것이 이쪽에는 없음).
- **budget cap/최대 행동 수/중단·재개는 `ToolExecutor`(controller 외부)에서 강제** —
  `RepairPolicy.decide()`가 side-effect-free여야 한다는 Stage 1의 제약과 같은 이유로, 이
  enforcement는 "결정"이 아니라 "집행"이므로 decide()류 메서드에 넣지 않고 전용 executor
  클래스에 둔다. `execute()`는 디스패치 **전**에 `_check_admission()`으로 (suspended →
  budget_exceeded → max_actions_exceeded 순서로) 거부 여부를 확인하고, 거부된 호출은 **전혀
  charge되지 않는다**(`test_a_rejected_call_is_never_charged_a_cost_record`). 디스패치된 모든
  호출(토큰 0개인 inspect/query/check/compile 포함)은 `tool_executor:{action}` stage 이름으로
  `CostRecord`를 만들어 `ArtifactGraphStore.record_cost()`에 반드시 기록한다 — 토큰 비용이
  0이어도 "비용 기록에서 빠지지 않는다"는 완료 기준은 이 기록 자체를 가리킨다(토큰 기반
  `Budget.spent`가 늘지 않는 것과는 별개).
- **자동 suspend**: budget이 소진되거나 max_actions에 도달하면 그 **직후** 자동으로 suspend
  상태로 전환된다(`_charge()`) — 다음 호출이 그 사실을 "재발견"하는 것이 아니라, 이미 suspended
  상태이므로 그 다음부터는 전부 `ToolErrorCode.SUSPENDED`로 거부된다(구체적 원인은
  `suspend_reason` 문자열로 보존). `resume(extra_budget=..., extra_actions=...)`로 cap을 올리고
  재개할 수 있다.
- **FINISH는 예외**: suspended/exhausted 상태와 무관하게 항상 실행된다 — 바로 그 상태일 때
  깨끗하게 종료할 수 있어야 하기 때문이다(`FinishCall`의 docstring).

---

## 2. 완료 기준 대응

> "controller가 읽을 수 있는 구조화된 observation. 오래된 검증 결과를 재사용하지 않으며, 실패한
> 도구 호출과 downstream 재계산도 비용에서 누락되지 않는다."

- **구조화된 observation**: `ToolObservation` 하나로 7개 action이 전부 수렴 (§1.1).
- **오래된 검증 결과 재사용 금지**: `_dispatch_check`는 `store.live_evidence()`를 조회해
  건너뛰는 shortcut이 전혀 없다 — 매번 `self._checker.check(...)`를 새로 호출하고 그 결과를
  무조건 `store.record_evidence()`에 기록한다(`test_check_runs_fresh_every_time_never_reuses_a_prior_result`).
- **실패한 도구 호출도 비용에서 누락되지 않음**: `execute()`의 dispatch 단계에서 예외가 나도
  (`EXECUTION_FAILED`) `succeeded=False`인 `CostRecord`를 만들어 charge한다
  (`test_dispatch_exception_is_converted_to_a_structured_error_not_raised`). `CostTrackingProducer`도
  대칭적으로, 감싼 `Producer`가 `ProductionResult(FAILED)`를 반환하든 (계약 위반으로) 실제
  예외를 던지든 두 경우 모두 `last_cost`에 `succeeded=False` 레코드를 남긴다
  (`test_failed_production_result_is_still_recorded_as_a_cost_with_succeeded_false`,
  `test_an_unexpected_raise_from_the_wrapped_producer_still_propagates_and_still_records_cost`).
  **단, 거부된(admission에서 막힌) 호출은 "실패한 도구 호출"이 아니다** — 아무것도 실행되지
  않았으므로 charge하지 않는다(§1.4). 이 둘을 구분하는 것이 설계의 핵심이다: "시도했으나 실패"는
  반드시 비용에 남고, "시도 자체가 거부됨"은 비용이 없다.
- **downstream 재계산**: Stage 3에는 아직 실제 downstream 재계산을 일으키는 복구 연산(Stage
  4+)이 없으므로, 이 부분은 "비용 집계 메커니즘 자체가 이를 빠뜨리지 않음"을 지금 보장하는
  것으로 대응한다 — PATCH/REPLAN이 실제로 Producer를 호출하게 되면(Stage 4), 그 비용은
  `CostTrackingProducer`를 거쳐 똑같이 하나도 빠짐없이 `record_cost()`로 들어간다; 지금 구조를
  바꿀 필요가 없다(Stage 2 문서가 §2에서 이미 예견한 바와 동일한 패턴: "Stage 3의
  inspect/query/check/patch/replan/compile 도구 schema가 만들어지면 그 출력도 같은 로그에
  쌓이도록 확장하면 된다 (지금 구조를 바꿀 필요는 없다)").

---

## 3. 사용자 결정 사항 (구현 전 확정)

구현 전, 범위가 모호한 4가지를 질문해 확정했다:

1. **patch_block/replan_subgraph**: schema + plumbing만 (실제 복구 연산은 Stage 4). →
   `ToolExecutor.execute()`에서 항상 `NOT_IMPLEMENTED`, budget/action-count 모두 건드리지 않음.
2. **compile_document**: 실제로 `LatexCompiler.compile_pdf`를 호출 (schema-only가 아님) — 컴파일러는
   파이프라인 전체 연결과 무관하게 독립 호출 가능한 결정적 도구이기 때문.
3. **CSP_SMT/SYMBOLIC**: 이번 단계는 미구현으로 남기고, **Notion Progress.ver2에 이후 구현해야 할
   후속 작업으로 기록** (아래 "Notion 기록" 참고) — z3/sympy 의존성 추가 없음.
4. **비용·지연의 "로컬/API 공통 모델 adapter"**: GPU 사용량까지 실측 — `nvidia-smi` subprocess
   기반 (새 Python 의존성 없음), 근사치임을 명시.

---

## 4. 하지 않은 것 (Stage 3 범위 밖, 다음 단계로 명시적으로 미룸)

- **CSP_SMT/SYMBOLIC 제약 검사**: 사용자 결정에 따라 미구현 — Notion Progress.ver2에 후속 작업으로
  기록(§3). z3/sympy는 `requirements.txt`에 추가하지 않았다.
- **patch_block/replan_subgraph의 실제 실행**: schema·오류 형식만 정의, 실제 "어느 범위를 고칠지
  결정하고 수행하는" 로직은 Stage 4의 "고정 복구 정책"이다(`docs/stage2_artifact_graph.md`가 이미
  명시한 경계와 동일).
- **오늘 파이프라인과의 실제 연결**: `scripts/run_full_pipeline.py`가 `ToolExecutor`/
  `CostTrackingProducer`를 실제로 호출하도록 바꾸는 작업은 없다 — Stage 1·2와 동일하게, 표현
  가능함과 독립 동작을 테스트로 증명하는 단계다.
- **정확한 per-process GPU 계정**: `GpuUtilizationSampler`는 디바이스 전체 utilization 기반
  근사치만 제공한다. `pynvml`/`nvidia-ml-py` 같은 정밀 계정 메커니즘은 추가하지 않았다(§3).
- **CheckCall의 RECOMPUTE가 실제 문서 content/source data에서 숫자를 추출**: 이 체커는 비교만
  한다 — 어디서 숫자를 읽어와 `context["recompute"]`에 채울지는 호출자(Stage 4/5가 실제 artifact
  content를 연결할 때)의 몫이다.
- **ToolErrorCode.TIMEOUT을 실제로 발생시키는 경로**: schema에는 있으나, 이번 단계의 모든
  dispatch 경로(로컬 연산·로컬 subprocess)에는 의미 있는 타임아웃 시나리오가 없다 — 실제 LLM
  호출이 patch/replan을 통해 연결되는 Stage 4/5에서 쓰일 자리다.
- **budget 소진이 토큰 비용으로 "실행 중간에" 크로스되는 경로의 실제 가동**: 이번 단계가
  디스패치하는 모든 action(inspect/query/check/compile)은 토큰 비용이 항상 0이므로,
  `Budget.remaining`이 실행 도중 LLM 토큰 소비로 음수를 넘어가는 경로는 지금 "이미 cap=0으로
  시작한 경우"로만 테스트된다(§1.4). `_charge()`의 로직 자체는 0이 아닌 토큰을 가진
  `CostRecord`에도 동일하게 적용되므로 코드를 바꿀 필요는 없다 — patch가 실제 Producer를 호출하게
  되는 Stage 4/5에서 이 경로가 처음으로 실제 가동된다.
