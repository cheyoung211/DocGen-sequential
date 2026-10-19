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

> **2026.10.19 사용자 재검토 반영 (3회)**: 1차 구현 공개 후 사용자가 세 차례에 걸쳐 재현 중심으로
> 결함을 발견했다 — 1차는 4가지 핵심 결함 + 추가 2건(§5), 2차는 그중 "evidence 유효성"·"budget
> hard cap"·"checkpoint JSON 직렬화" 세 영역에서 추가 3건(§6), 3차는 2차가 도입한
> `Evidence.missing_inputs` 메커니즘 자체의 결함 2건(§7). 이 문서는 **현재(세 차례 수정 후)
> 구현**을 기준으로 적는다 — 무엇이 어떻게 바뀌었는지는 §5/§6/§7에 리뷰 이력으로 남긴다.

| 구분 | 위치 |
|---|---|
| Stage 3 schema (`ToolActionKind`/`ToolErrorCode`/`ToolExecutionError`/7개 Call·Result·`GenerateResult`/`ToolObservation`/`ToolCallLogEntry`) | `src/core/schemas.py` (신규, 파일 끝에 추가) |
| 도구 실행기 (dispatch, `run_generation`, budget/max-actions/suspend-resume 강제, 모든 호출 로깅) | `src/core/tool_executor.py::ToolExecutor` |
| 기본 제약 검사 (`schema_contract`: kind까지 확인 / `recompute`: 비교+실제 재계산 두 모드, `Validator` 구현) | `src/core/constraint_checks.py::BasicConstraintChecker` |
| 비용·지연·GPU 측정, 로컬/API 공통 모델 adapter | `src/core/cost_tracking.py::GpuUtilizationSampler`, `CostTrackingProducer` |
| `live_evidence`(입력 버전+정확한 버전 단위의 결측 여부까지 확인, constraint당 최신 1건을 먼저 고른 뒤 유효성 검사)/`evidence_history`(전체 이력)/`record_tool_call` | `src/core/artifact_graph.py::ArtifactGraphStore` (Stage 2 소유 파일에 Stage 3가 추가) |
| `Evidence.missing_inputs`는 `List[str]`(artifact_id)가 아니라 `List[ArtifactRef]`(artifact_id+version) | `src/core/schemas.py::Evidence` |
| `RunLog.tool_calls`/`log_tool_call`, `Checkpoint.tool_calls` | `src/core/run_log.py`, `src/core/schemas.py::Checkpoint` |
| 실제 compile timeout (`subprocess.run(timeout=...)`, 전파만 함) | `src/agents/latex_compiler.py::LatexCompiler.compile_pdf` (신규 `timeout` 파라미터, 기본값 `None`=기존 동작 그대로 — Stage 3에서 유일하게 agent 코드를 수정한 지점) |
| `CompileToolResult.compile_result`는 `Optional[Any]`가 아니라 `Optional[CompileResult]`(`src/evaluation/schemas.py`에서 import) | `src/core/schemas.py::CompileToolResult` |
| 테스트 | `test/core/test_tool_executor.py` (44개), `test/core/test_constraint_checks.py` (30개), `test/core/test_cost_tracking.py` (13개), `test/core/test_artifact_graph.py` (48개) |

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
  수렴한다 — `Constraint.applies_to`에 나열된 모든 `ArtifactRef`가 실제로 저장되어 있는지, **그리고
  저장된 kind가 claim한 kind와 정확히 일치하는지**(`stored.ref.kind == ref.kind`)까지 확인한다
  (`_check_schema_contract`). **사용자 재검토에서 발견된 결함**: 1차 구현은 kind 비교를 빠뜨려,
  TABLE을 요구하는 참조가 실제로는 같은 id/version의 DEFINITION을 가리켜도 PASS했다 — 지금은
  kind가 다르면 "not found"와 구분되는 사유("expected kind=..., actually stored as kind=...")로
  FAIL한다. artifact_id 존재뿐 아니라 정확한 `(artifact_id, version)` 실재 여부까지 확인하는 것은
  Stage 2의 `_validate_artifact_ref`가 이미 구분한 "존재 여부 vs 버전 실재 여부"와 동일한 구분을
  재사용. **2차 사용자 재검토에서 발견된 결함**: `applies_to` 전체를 무조건
  `Evidence.input_versions`에 담던 1차 수정은, 아예 존재하지 않는 참조(필수 table 누락 등)를 마치
  "지금 존재하는 버전"인 것처럼 기록해 `live_evidence()`의 무효화 방향을 거꾸로 뒤집었다 — 누락을
  찾아낸 FAIL이 즉시 무효로 취급되고, 누락이 해소되는 순간 오히려 "유효"로 되살아났다(§6/§2).
  **수정**: 실제로 존재하는(단, kind가 틀린 경우 포함) 참조만 `input_versions`에, 아예 존재하지
  않는 참조는 `Evidence.missing_inputs`에 분리해서 담는다 — PASS여도 그 참조들이 나중에 바뀌면 이
  판정이 더 이상 유효하지 않아야 하기 때문. **3차 사용자 재검토에서 발견된 결함(§7)**:
  `missing_inputs`를 artifact_id만으로(버전 없이) 기록해, table@1이 이미 존재하는 상태에서
  table@2가 요구되어도 "table이라는 id는 존재한다"는 이유로 FAIL이 즉시 무효화됐다. **수정**:
  `missing_inputs`를 `List[str]`에서 `List[ArtifactRef]`로 바꿔 artifact_id와 version을 모두
  기록하고, `_still_missing()`이 정확히 그 version을 `get_version()`으로 확인하도록 고쳤다.
  **이 체커는 artifact의
  content가 실제로 그 참조를 인용하는지는 검사하지 않는다** — 그건 레퍼런스 스캐너가 할 일이고,
  Stage 2가 이미 명시적으로 미룬 범위다(`docs/stage2_artifact_graph.md` §3, "실제 레퍼런스 스캔
  기반 INFERRED edge 자동 탐지").
- **RECOMPUTE — 두 모드, 사용자 재검토 후 이름을 정확히 분리**: 1차 구현은 "기대값/관측값 비교"
  단 하나만 있었는데, 이것은 "숫자 재계산"이 아니라 "숫자 비교"라는 지적을 받았다(체크리스트
  문구는 "숫자 재계산"). 지금은 두 모드로 나뉜다:
  1. **직접 비교**(기존 모드): 호출자가 `expected`/`observed`를 이미 계산해 넘긴다 —
     `context["recompute"][constraint_id] = {"expected": ..., "observed": ..., "tolerance": ...,
     "source_refs": [...]}`. 이 체커는 비교만 한다 — "비교"로 정확히 부른다.
  2. **구조화된 source artifact에서 실제 재계산**(신규): `{"source_ref": ArtifactRef, "operation":
     "sum", "observed": ..., "tolerance": ...}`를 넘기면, 체커가 `ArtifactGraphStore`에서
     `source_ref`의 저장된 content를 직접 읽어 JSON 숫자 배열로 파싱하고 `operation`으로 지정한
     연산(현재는 `"sum"` 하나만 — 일반 수식 평가기가 아니라 좁고 구체적인 사례 하나)을 실행해
     `expected`를 **실제로 재계산**한다 — 이것이 "숫자 재계산"이라는 이름에 부합하는 모드다.
  두 모드 모두 **`constraint.applies_to`와 각 모드 고유의 source 참조를 합친 뒤 지금 실제로
  존재하는지 하나씩 확인해서 분리**(`_resolve_input_refs`, `(artifact_id, version)` 기준 중복
  제거 후 존재하면 `input_versions`, 존재하지 않으면 `missing_inputs`)한다(직접 모드:
  `applies_to` + 호출자가 선언한 `source_refs`; 재계산 모드: `applies_to` + `source_ref` 자신).
  **2차 사용자 재검토에서 발견된 결함**: 원래는 `source_refs`로만 기록해, source를 `applies_to`
  에만 선언한 호출자의 경우 `input_versions`가 비어 source가 바뀌어도 PASS가 영원히 live로
  남았다 — 이를 "무조건 합쳐서 `input_versions`에" 담는 것으로 고쳤다(당시 이름: `_merge_input_refs`).
  **3차 사용자 재검토에서 발견된 결함(§7)**: 그 수정이 존재 여부를 확인하지 않고 무조건
  `input_versions`에 담았기 때문에, 아직 생성되지 않은 같은 source를 `Constraint.applies_to`와
  재계산의 `source_ref`에 동시에 지정하면 그 참조가 `input_versions`(존재해야 유효)와
  `missing_inputs`(존재하지 않아야 유효)에 동시에 들어가 evidence가 영원히 유효할 수 없는
  자기 모순이 발생했다. **수정**: `_resolve_input_refs`로 이름을 바꾸고, 합친 뒤 각 참조의 존재
  여부를 실제로 확인해 정확히 한쪽에만 들어가도록 했다. 값이 없으면 FAIL이 아니라
  UNKNOWN/NOT_CHECKED, source artifact가 아예 없으면 UNKNOWN/MISSING_DEPENDENCY, content가
  숫자 배열이 아니면 UNKNOWN/NOT_REPRESENTABLE.
- **CSP_SMT·SYMBOLIC**: 이번 단계는 미구현으로 남겼다 — 체크할 구체적인 narrow 제약 사례가 아직
  없고, z3/sympy 모두 `requirements.txt`에 없다. 항상 UNKNOWN/NOT_REPRESENTABLE을 반환하며
  (`_IMPLEMENTED_KINDS`에 없는 kind는 `check()`의 첫 분기에서 걸러짐), `set_enabled()`로도 켤 수
  없다 — "비활성화"가 아니라 "구현 없음"이므로 토글 자체가 `ValueError`. **Notion
  Progress.ver2에 이후 구현해야 할 후속 작업으로 기록**(사용자 결정, 아래 §3 참고).

### 1.3 "도구 결과에 evidence, 관련 version, unknown·timeout 정보를 포함한다."

- CHECK의 `CheckResult.evidence`는 `Evidence`(Stage 1) 그대로 — `unknown_reason`까지 포함.
- INSPECT/QUERY는 `ArtifactRef`/`StoredArtifact`를 통해 버전 정보를 그대로 노출한다
  (`InspectArtifactResult.stored.ref.version`, `QueryDependenciesResult.edges[*].{from,to}_artifact.version`).
- `ToolErrorCode.TIMEOUT`은 이제 실제로 발생한다. **사용자 재검토에서 발견된 결함**: 1차 구현은
  `LatexCompiler.compile_pdf`의 `subprocess.run()`에 timeout이 전혀 없어 무한 루프를 유발하는
  문서가 영원히 멈추지 않을 수 있었고, 그 상태를 모의로 재현(`TimeoutExpired` 주입)했을 때도
  `ToolErrorCode.TIMEOUT`이 아니라 일반 `EXECUTION_FAILED`가 반환됐다(당시 어떤 코드 경로도
  `subprocess.TimeoutExpired`를 실제로 던지지 않았기 때문). **수정**: `compile_pdf`에 `timeout`
  파라미터를 추가해(기본값 `None` = 기존과 동일하게 무제한) `subprocess.run(timeout=...)`에
  그대로 전달한다 — 타임아웃 시 `subprocess.run`이 자식 프로세스를 실제로 kill하고
  `TimeoutExpired`를 던진다. `CompileCall.timeout_s`(기본 120초)가 이 값을 채우고,
  `ToolExecutor._execute_inner`가 `subprocess.TimeoutExpired`를 (일반 `Exception`보다 먼저)
  별도로 잡아 `ToolErrorCode.TIMEOUT`으로 변환하며, 소요 비용(`latency_s`,
  `succeeded=False`)도 다른 실패 경로와 똑같이 charge된다 — 실제 tectonic 호출로(mock 없이)
  0.001초 timeout을 줘도 동일하게 동작함을 테스트로 확인했다. `GpuUtilizationSampler`의
  `nvidia-smi` 호출 자체의 2초 타임아웃은 여전히 별개(GPU 측정 실패일 뿐 도구 호출의 실패가
  아니므로 `None` 측정값으로만 드러남).

### 1.4 "초기 생성과 모든 후속 호출의 비용·지연을 기록하고 controller 외부의 executor에서 budget cap, 최대 행동 수, 중단·재개 조건을 강제한다. 필요 시 로컬/API 공통 모델 adapter를 둔다."

- **"로컬/API 공통 모델 adapter"는 `CostTrackingProducer`다** (`cost_tracking.py`) — 오늘 실제로
  존재하는 유일한 LLM 호출 표면인 Stage 1의 `Producer`(`PlannerAgentProducer`/
  `TextAgentProducer`)를 감싼다. "초기 생성"의 비용·지연이 바로 이 `produce()` 호출이다. 매
  `produce()` 호출마다 정확히 하나의 `CostRecord`를 만든다 — 내부 재시도 루프가 여러 번 실제
  LLM을 호출했어도(`TextAgent._run_generation_loop`), `usage_sink`에 쌓인 모든 항목을 합산하므로
  **실패한 내부 attempt의 토큰도 자동으로 포함**된다(완료 기준의 "실패한 시도 ... 도 비용에서
  누락되지 않는다"), 각 attempt를 따로 가로챌 필요 없이.
- **사용자 재검토에서 발견된 결함**: `CostTrackingProducer`가 `last_cost`를 만들어도, 이를
  **`ToolExecutor`의 `Budget`에 반영하는 경로가 전혀 없었다** — `ArtifactGraphStore.record_cost()`
  를 불러도 로그만 쌓이고 `self._budget`은 바뀌지 않는다. 재현: cap=10 tokens에서 모의 생성이
  12 tokens를 쓰면, 로그에는 기록되지만 `executor.budget.spent`는 0으로 남고 다음 도구 호출이
  그대로 허용됐다. **수정**: `ToolExecutor.run_generation(producer, artifact_id, context, *,
  model_name, stage)`를 추가했다 — ① 호출 **전** `_check_admission(GENERATE)`로 허용량을
  확인하고(budget이 이미 소진된 상태라면 producer를 전혀 호출하지 않음), ② `CostTrackingProducer`로
  실제 호출하며, ③ 성공·실패·예외 모두 동일하게 `_charge()`(기존 `ToolCall` 디스패치와 같은
  경로)로 budget을 차감하고 `record_cost()`에 기록한다. **2차 사용자 재검토에서 발견된 결함**:
  ①을 "엄격한 상한(strict cap)" 보장이라고 표현한 것이 부정확했다 — admission 확인은 호출 *전*
  남은 예산이 0보다 큰지만 보고, 그 호출 자체의 실제 비용이 남은 예산 안에 들어올지는 보지 않는다.
  cap=10에서 12 tokens를 쓰는 호출은 (10 > 0이므로) 그대로 허용되고 12 전액이 charge되어
  `spent`가 `cap`을 넘어선다 — 다음 호출부터만 확실히 막힌다(§6). **문서 수정**: "엄격한 상한"이
  아니라 "소진 후 신규 호출 차단"으로 정확히 표현한다 — 총 초과분은 "허용된 호출 하나의 비용"으로
  bound되지만 0으로 bound되지는 않는다. 진짜 pre-emptive hard cap(호출 전 비용 예측·예약, 또는
  wrapped Producer의 출력 토큰 수 제한)은 LLM 호출 자체(agent/llm_client.py)를 건드려야 하므로
  Stage 3의 "agent 코드는 건드리지 않는다" 범위를 벗어나 구현하지 않았다. `ToolActionKind.GENERATE`를 schema에
  추가했지만 controller가 선택하는 7개 action에는 포함되지 않는다(direction 2는 "생성"을 action
  후보로 나열하지 않음) — `execute()`/`ToolCall`과는 별도 진입점이며, `Producer` 참조가
  pydantic으로 직렬화 가능한 값이 아니므로 애초에 같은 진입점을 쓸 수 없다. 모의 Producer만으로
  (`FakeProducer`) 검증했다 — 실제 파이프라인 연결은 여전히 Stage 4/5의 일이다.
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
  **1차 사용자 재검토에서 발견된 결함**: `live_evidence()` 자신은 `checked_artifact`의 버전/stale
  여부만 봤지, 검사에 실제로 쓰인 *다른* 입력(예: RECOMPUTE의 source artifact)이 바뀌었는지는
  전혀 보지 않았다 — source artifact가 v1→v2로 바뀌어도 기존 PASS가 계속 live로 남았다. 또한
  같은 constraint_id를 재검사했을 때(필수 table 누락→FAIL, table 추가 후 재검사→PASS) 과거
  FAIL과 새 PASS가 **동시에** live로 반환됐다 — "매번 새로 검사한다"는 것만으로는 이전 결과가
  무효화되지 않았기 때문이다. **1차 수정**: `Evidence`에 `input_versions` 필드를 추가해 체커가
  실제로 참조한 다른 artifact 버전을 담게 하고(§1.2), `live_evidence()`가 (a)
  `checked_artifact` 버전/stale 조건에 더해 (b) `input_versions`의 모든 항목이 여전히 현재·
  non-stale 버전인지까지 확인하며, (c) 같은 constraint_id에 대해서는 **가장 최근에 기록된 것
  하나만** 반환하도록(append-only 순서를 그대로 "최신"으로 사용) 다시 작성했다. 전체 이력은
  별도의 새 메서드 `evidence_history(artifact_id)`로 분리해, "현재 유효한 판정"과 "전체 이력"이
  서로 다른 접근자로 명확히 구분된다.
  **2차 사용자 재검토에서 발견된 결함(§6)**: 이 1차 수정은 두 가지를 더 틀렸다. ① "참조가
  존재해야 유효"라는 규칙을 "참조가 아예 없음을 확인한 FAIL"에도 그대로 적용해, 누락을 찾아낸
  FAIL이 즉시 무효 처리되고 누락이 해소되면 오히려 되살아나는 **반대 방향**의 오류를 만들었다.
  ② (b) 필터링을 (c) "constraint_id당 최신 선택"보다 **먼저** 적용해, 최신 판정이 무효화된
  경우 그보다 오래된(이미 대체된) 판정으로 조용히 되돌아갔다. **2차 수정**: `missing_inputs`
  필드를 추가해 "참조가 없어야 유효"라는 반대 방향 조건을 별도로 표현하고(`_still_missing`),
  `live_evidence()`를 "constraint_id당 최신 선택 → 그 다음에만 유효성 검사"로 순서를 바꿨다
  (`_evidence_is_live`). 세부는 §6.
  **3차 사용자 재검토에서 발견된 결함(§7)**: `missing_inputs`를 artifact_id만으로 기록해 ①
  요청한 버전이 아닌 다른 버전이 이미 존재해도 FAIL이 무효화됐고, ② 같은 참조를 `applies_to`와
  recompute의 `source_ref`에 동시에 지정하면 `input_versions`·`missing_inputs` 양쪽에 모두
  들어가 "존재해야 유효"와 "없어야 유효"가 동시에 요구되는 자기 모순이 생겼다. **3차 수정**:
  `missing_inputs`를 `List[ArtifactRef]`로 바꿔 정확한 버전까지 기록하고(`_still_missing`이
  `get_version(artifact_id, version)`으로 그 버전 자체를 확인), 모든 참조를 합친 뒤 실제
  존재 여부를 확인해서 정확히 한쪽에만 넣도록(`_resolve_input_refs`) 했다. 세부는 §7.
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
  `CostTrackingProducer`/`run_generation`을 실제로 호출하도록 바꾸는 작업은 없다 — Stage 1·2와
  동일하게, 표현 가능함과 독립 동작을 테스트로 증명하는 단계다.
- **정확한 per-process GPU 계정**: `GpuUtilizationSampler`는 디바이스 전체 utilization 기반
  근사치만 제공한다. `pynvml`/`nvidia-ml-py` 같은 정밀 계정 메커니즘은 추가하지 않았다(§3).
- **RECOMPUTE의 recompute-from-source 모드가 지원하는 연산은 `"sum"` 하나뿐**: 일반 수식
  평가기가 아니다 — 다른 연산(비율·평균 등)이 필요해지면 `_RECOMPUTE_OPERATIONS`에 추가하면 되는
  구조이지만, 지금은 "좁은 제약군 하나"만 구현했다(§1.2).
- **`run_generation()`의 호출 자체는 `record_tool_call()`로 로깅되지 않는다**: `Producer` 참조가
  로그에 직렬화 가능한 값이 아니기 때문 — 비용/budget은 §1.4대로 정확히 연결되지만, "이 생성
  호출이 정확히 언제·무엇으로 일어났는지"의 call/observation 로그 자체는 Stage 4/5가 실제
  Producer 연결을 설계할 때 필요하면 추가한다.
- **`SCHEMA_CONTRACT`가 artifact content에서 실제 참조를 스캔**: 여전히 `applies_to`에 명시된
  것만 확인한다 — Stage 2가 이미 미룬 범위와 동일(§1.2).
- **`run_generation()`의 budget admission은 pre-emptive hard cap이 아니다**: 호출 전 "남은
  예산 > 0"만 확인하며, 그 호출 자체의 비용이 남은 예산을 넘는지는 사전에 알 수 없다(LLM 호출의
  실제 토큰 수는 호출 전에 알 수 없음) — 그래서 총 `spent`가 `cap`을 넘을 수 있고, bound는 "허용된
  호출 하나의 비용"까지다(§1.4, §6). 이를 진짜 hard cap으로 만들려면 wrapped `Producer`의 출력
  토큰 수 자체를 제한해야 하는데, 이는 agent/`llm_client.py` 레벨의 변경이 필요해 "agent 코드는
  건드리지 않는다"는 Stage 3 범위를 벗어난다 — 2차 사용자 재검토에서 이 한계를 명시적으로
  인정하기로 확정(§6).

---

## 5. 사용자 재검토 후 수정 (1차, 2026.10.19)

1차 구현을 공개한 뒤 재현 중심 리뷰에서 4가지 핵심 결함과 추가 2건을 발견해 전부 수정했다 —
`src/core/{schemas,tool_executor,constraint_checks,artifact_graph,run_log}.py`,
`src/agents/latex_compiler.py` 수정, `test/core/`에 재현 테스트 추가 (`test_tool_executor.py`
24→39개, `test_constraint_checks.py` 14→22개, `test_artifact_graph.py`에 신규 5개).

1. **생성 비용이 executor의 budget을 줄이지 않음** — §1.4에 기록. 재현: cap=10 tokens, 모의
   생성 12 tokens 사용 → `budget.spent`는 0으로 남고 다음 도구 호출이 허용됨. 수정:
   `ToolExecutor.run_generation()` 추가, 호출 전 admission 확인 + `_charge()`로 budget 연결.
2. **검사 입력이 바뀌어도 과거 evidence가 유효하게 남음** — §2에 기록. 재현: (a) RECOMPUTE의
   source artifact를 v1→v2로 바꿔도 기존 PASS가 live로 유지됨, (b) FAIL 기록 후 원인이
   해소되어 PASS를 다시 기록해도 둘 다 live로 반환됨. 수정: `Evidence.input_versions` 추가 +
   `live_evidence()`가 이를 확인하고 constraint_id당 최신 1건만 반환; `evidence_history()`를
   별도로 추가해 전체 이력과 분리.
3. **기본 제약 검사의 범위가 좁음** — §1.2에 기록. 재현: TABLE을 요구하는 참조가 실제로는
   DEFINITION이어도 PASS. 수정: kind 비교 추가. 또한 RECOMPUTE가 "비교"만 하고 "재계산"이라는
   이름에 부합하지 않는다는 지적을 받아, 구조화된 source artifact에서 실제로 합계를 계산하는
   두 번째 모드를 추가하고 문서/이름을 정확히 분리했다.
4. **TIMEOUT이 정의돼 있지만 실행 경로에서 처리되지 않음** — §1.3에 기록. 재현: 실제
   `subprocess.run()`에 timeout이 없었고, `TimeoutExpired`를 주입해도 `EXECUTION_FAILED`가
   반환됨. 수정: `compile_pdf`에 `timeout` 파라미터 추가(기본 `None`=기존 동작 유지) +
   `ToolExecutor`가 `subprocess.TimeoutExpired`를 `ToolErrorCode.TIMEOUT`으로 매핑, 비용도 다른
   실패와 동일하게 charge.
5. **`ToolObservation` 검증 불충분** — 1차 구현은 "틀린 필드가 없는지"만 확인하고 "맞는 필드가
   있는지"는 확인하지 않아, `ok=True`+`output=None`이나 `action=CHECK`를 달고
   `FinishToolResult`를 담은 observation도 전부 유효한 객체로 통과했다. 수정:
   `model_post_init`이 이제 ok=True→output 필수+output.action==action, ok=False→error
   필수+error.action==action을 모두 강제한다(`schemas.py::ToolObservation`).
6. **실행 관측이 저장되지 않음** — `FinishCall`의 종료 이유·미해결 항목이 checkpoint에 전혀
   남지 않았고, 일반 도구 호출의 입출력도 저장 경로가 없었다. 수정: 새 `ToolCallLogEntry`
   schema + `RunLog.log_tool_call`/`tool_calls` + `ArtifactGraphStore.record_tool_call` +
   `Checkpoint.tool_calls`를 추가해, `ToolExecutor.execute()`의 **모든** 호출(허용·거부 불문)을
   call+observation 쌍으로 로깅하고 checkpoint에 포함시킨다 — Stage 4의 분기 재실행이 필요로
   할 연결을 미리 마련했다.

모든 수정은 재현 테스트로 남겼다 — 각 테스트 이름이 구체적 재현 시나리오를 그대로 설명한다
(예: `test_schema_contract_fails_on_kind_mismatch_even_when_id_and_version_resolve`,
`test_live_evidence_excludes_evidence_whose_declared_input_moved_to_a_newer_version`,
`test_run_generation_charges_the_executors_own_budget`,
`test_compile_timeout_reports_timeout_not_execution_failed`,
`test_tool_observation_rejects_an_action_output_mismatch`,
`test_execute_logs_a_finish_calls_reason_and_unresolved_ids`). 기존 §0–4의 테스트는 수정 없이
그대로 통과한다; `test/evaluation/`의 38개 테스트도 회귀 없음.

---

## 6. 사용자 재검토 후 수정 (2차, 2026.10.19)

1차 수정을 공개한 뒤 같은 날 두 번째 재현 중심 리뷰에서 추가 3건을 발견해 전부 수정했다 —
`src/core/{schemas,artifact_graph,constraint_checks}.py` 수정(이번 차수는 agent 코드 변경 없음),
`test/core/`에 재현 테스트 추가(`test_tool_executor.py` 39→42개, `test_constraint_checks.py`
22→27개, `test_artifact_graph.py` 44→47개).

1. **`live_evidence()`가 필요한 FAIL을 숨기거나 과거 판정을 다시 반환함** — §2/§1.2에 기록. 세
   가지를 재현: (a) 필수 table 누락 FAIL이 `live_evidence()`에서 빈 목록으로 숨겨짐, (b) table을
   추가하고 재검사하지 않으면 그 누락-FAIL이 다시 반환됨, (c) 과거 PASS 이후 새 FAIL을 기록하고
   그 FAIL의 입력이 바뀌면 과거 PASS가 다시 반환됨. 원인: 1차 수정은 "참조가 존재해야 유효"라는
   규칙 하나만 있어서 "존재하지 않음을 확인한 FAIL"을 표현할 수 없었고(→ (a)(b)), 유효성 필터링을
   constraint_id당 최신 선택보다 먼저 적용해 무효화된 최신 판정이 그보다 오래된 판정으로
   되돌아갔다(→ (c)). 수정: `Evidence.missing_inputs: List[str]` 추가(버전이 아니라 artifact_id로,
   "아직 존재하지 않음"을 직접 표현) + `ArtifactGraphStore._still_missing`/`_evidence_is_live` 추가
   + `live_evidence()`를 "constraint_id당 최신 선택 → 그 다음에만 유효성 검사" 순서로 재작성.
   `BasicConstraintChecker._check_schema_contract`/`_check_recompute_from_source`도 존재하지 않는
   참조는 `missing_inputs`에, 존재하지만 틀린/변경 가능한 참조는 `input_versions`에 분리해서
   담도록 수정. 추가로, 직접 비교 모드가 `source_refs` context 키로만 input을 추적해 `applies_to`
   에만 선언된 source는 추적되지 않던 것도 `_merge_input_refs`로 수정(§1.2) — recompute-from-source
   모드도 동일하게 `applies_to`를 병합하도록 통일.
2. **비용 연결은 됐지만 "엄격한 상한"은 성립하지 않음** — §1.4에 기록. 재현: cap=10에서 12
   tokens를 쓰는 호출이 그대로 실행되고 12 전액이 charge되어 `spent`(12) > `cap`(10)이 됨;
   다음 호출부터만 막힌다. 이것은 코드 버그가 아니라 **문서 표현의 오류**였다 — admission 확인은
   호출 전 "남은 예산 > 0"만 보고, 호출 자체의 실제 비용은 사전에 알 수 없으므로 그 호출이
   예산을 넘을 수 있다는 것은 애초에 피할 수 없는 설계였다(진짜 pre-emptive hard cap은
   Producer의 출력 토큰 수 제한이 필요 → agent 코드 변경 → Stage 3 범위 밖, §4). **수정**: 코드
   동작은 바꾸지 않고, "엄격한 상한" 표현을 "소진 후 신규 호출 차단, 초과분은 허용된 호출 하나의
   비용까지 bound"로 정확히 고쳤다(`tool_executor.py` 모듈 docstring, `run_generation` docstring,
   이 문서, 테스트 docstring). 이 정확한 bound 자체를 `test_budget_admission_bounds_overspend_to_
   at_most_one_calls_cost`로 못박아, "표현을 다시 과장되게 되돌리면" 테스트가 그 사실을 상기시키게
   했다.
3. **저장된 관측을 checkpoint JSON에서 복원하면 `compile_result`가 dict로 바뀜** — §0에 기록.
   재현: 실제 compile 결과를 포함한 checkpoint를 `model_dump_json()` → `model_validate_json()`으로
   왕복하면, 저장 전 `CompileResult` 객체였던 `observation.output.compile_result`가 복원 후 평범한
   `dict`가 되어 `.pdf_exists` 같은 속성 접근이 깨짐. 원인: `CompileToolResult.compile_result`가
   `Optional[Any]`로 선언되어 있어 pydantic이 복원 시 참조할 schema가 없었다 — 값은 보존되지만
   타입 정보가 사라짐. 수정: `compile_result`를 `Optional[CompileResult]`(`src/evaluation/schemas.py`
   에서 import)로 변경 — `src/core/adapters.py`가 이미 `src.evaluation.metrics...`를 같은 방향으로
   import하고 있고, `src/evaluation/schemas.py`는 `src.core`를 전혀 import하지 않으므로 순환
   import 위험이 없음을 확인했다.

모든 수정은 재현 테스트로 남겼다(예: `test_live_evidence_returns_a_fail_that_correctly_found_its_
declared_input_missing`, `test_live_evidence_does_not_fall_back_to_an_earlier_verdict_when_the_
latest_one_for_the_same_constraint_is_invalidated`, `test_check_through_the_real_executor_a_
missing_required_table_stays_live_until_added_then_rechecked`(실제 checker+executor를 통한
end-to-end 재현), `test_budget_admission_bounds_overspend_to_at_most_one_calls_cost`,
`test_compile_result_survives_a_checkpoint_json_round_trip_as_a_typed_object`). 검증: `test/core/`
+ `test/evaluation/` 192개 전부 통과(1차 수정 후 181개에서 +11); 전체 저장소 기준으로도
`test/test_pipeline_dependency_ordering.py`(수집 자체가 깨진, Stage 3 이전부터의 무관한 기존
결함) 제외 700 passed / 57 failed로, 57건은 1차 수정 때와 정확히 동일한 기존 무관 결함 목록이라
회귀 없음을 확인했다. `scripts/run_full_pipeline.py`와 agent 코드는 이번 차수에서 전혀 수정하지
않았다.

---

## 7. 사용자 재검토 후 수정 (3차, 2026.10.19)

2차 수정으로 도입한 `Evidence.missing_inputs` 메커니즘 자체에 재현 중심 리뷰로 2건을 더 발견해
수정했다 — `src/core/{schemas,artifact_graph,constraint_checks}.py` 수정(이번 차수도 agent 코드
변경 없음), `test/core/`에 재현 테스트 추가(`test_tool_executor.py` 42→44개,
`test_constraint_checks.py` 27→30개, `test_artifact_graph.py` 47→48개).

두 문제 모두 "누락 상태를 evidence에 기록하는 방식" 하나에서 비롯됐다 — 2차 수정 자체의 근본
원인이었다.

1. **artifact는 존재하지만 요청한 버전이 없으면 유효한 FAIL이 숨겨짐** — §1.2/§2에 기록. 재현:
   `table@1`만 저장된 상태에서 `table@2`를 요구하면 checker는 정상적으로 FAIL을 반환하지만, 2차
   수정은 이를 `missing_inputs=["table"]`(artifact_id만)로 기록했다. `_still_missing()`은 "그
   artifact_id의 **어떤** 버전이든 존재하는가"만 확인했으므로, `table@1`이 이미 (체크 전부터)
   존재한다는 사실 자체가 "더 이상 누락이 아님"으로 잘못 해석되어 FAIL이 즉시 무효화됐다 —
   "ID 전체의 부재"와 "특정 버전의 부재"를 구분하지 못한 것이 원인이다. **수정**:
   `Evidence.missing_inputs`를 `List[str]`에서 `List[ArtifactRef]`로 바꿔 artifact_id와 version을
   모두 보존하고, `ArtifactGraphStore._still_missing(ref)`가 `self.get_version(ref.artifact_id,
   ref.version)`으로 정확히 그 버전의 존재 여부만 확인하도록 고쳤다 — `table@1`이 이미 존재해도
   `table@2`가 실제로 생기기 전까지는 FAIL이 계속 유효하다.
2. **없는 source를 `Constraint.applies_to`에도 선언하면 UNKNOWN evidence가 숨겨짐** — §1.2에
   기록. 재현: 같은 미생성 source를 `Constraint.applies_to`와 재계산의 `source_ref`에 동시에
   지정하면, 2차 수정은 `input_versions=list(constraint.applies_to)`(무조건, 존재 여부 확인
   없음)와 `missing_inputs=[source_ref.artifact_id]`를 **동시에** 기록했다 — 같은 참조가
   "존재해야 유효"(`input_versions`)와 "없어야 유효"(`missing_inputs`) 양쪽 조건에 걸려, 그
   evidence는 outcome과 무관하게 영원히 유효 목록에 들어갈 수 없는 자기 모순 상태가 됐다. 원인은
   `applies_to`를 병합할 때 각 참조가 실제로 존재하는지 확인하지 않고 무조건 `input_versions`에
   담은 것이다. **수정**: 기존 `_merge_input_refs`(병합만 하고 분류하지 않음)를
   `_resolve_input_refs`(병합 후 각 참조를 `get_version()`으로 존재 여부 확인, 존재하면
   `input_versions`, 존재하지 않으면 `missing_inputs`로 분류해 반환)로 교체했다. 이제 같은 참조가
   두 목록에 동시에 들어가는 경우는 구조적으로 불가능하다 — RECOMPUTE 두 모드의 모든 분기
   (직접 비교, source 없음/content 없음/숫자 아님/성공)가 이 하나의 분류 로직을 거친다.

두 문제의 공통 교훈: "이 입력이 없다"는 사실을 기록할 때 **정확히 무엇의 부재인지**(어떤
artifact_id의 어떤 버전)와 **그 부재가 다른 어떤 기록과 충돌하지 않는지**(같은 참조가
input_versions에도 들어가 있지 않은지)를 둘 다 보장해야 한다 — artifact_id만으로는 버전을
구분할 수 없고, 병합만으로는 존재 여부를 알 수 없다.

모든 수정은 재현 테스트로 남겼다(예:
`test_schema_contract_missing_ref_tracks_the_exact_version_not_just_the_artifact_id`,
`test_live_evidence_treats_a_missing_input_as_version_specific_not_id_specific`,
`test_check_through_the_real_executor_a_fail_for_a_missing_specific_version_stays_live_even_when_
an_older_version_already_exists`(실제 checker+executor를 통한 end-to-end 재현),
`test_direct_comparison_mode_declaring_the_same_missing_source_via_applies_to_and_source_refs_
does_not_self_contradict`,
`test_declaring_the_same_missing_source_via_applies_to_and_source_ref_does_not_self_contradict`,
`test_check_through_the_real_executor_declaring_the_same_missing_source_via_applies_to_and_
source_ref_is_still_live`). 검증: `test/core/` + `test/evaluation/` 198개 전부 통과(2차 수정 후
192개에서 +6); 전체 저장소 기준으로도 706 passed / 57 failed로, 57건은 매 차수와 정확히 동일한
기존 무관 결함 목록이라 회귀 없음을 확인했다. `scripts/run_full_pipeline.py`와 agent 코드는
이번 차수에서도 전혀 수정하지 않았다.
