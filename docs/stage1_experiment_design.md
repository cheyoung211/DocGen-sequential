# Stage 1 실험 설계: 검증·복구·평가 분리

이 문서는 Notion "SNU / NCIA / Personal Research / Multimodal Agents / Revision 2026.10.15"
계획의 **Stage 1**(검증·복구·평가 분리와 실험 설계) 산출물이다. Stage 0(기존 결과 재현·용어 정의·범위
확정)은 이미 완료되었다 (`src/evaluation/paper_tables.py`, 브랜치 `graph_artifact`). 근거는
`docs/method_architecture.md`와 동일하게 `file:line` 형태로 표기한다.

> **범위**: 이 문서와 `src/core/{schemas,interfaces,adapters}.py`는 **설계와 실험 사전 등록**이다.
> `scripts/run_full_pipeline.py`나 기존 agent 코드는 전혀 바뀌지 않았다 (동작 변화 0). 아래 7개 비교군 중
> "오늘 구현됨"으로 표시되지 않은 항목은 이름과 정의만 여기 있고, 실제 구현은 명시된 이후 단계
> (Stage 4/5/8)에서 이루어진다. Human annotation도 실제 수집은 Stage 6이며, 여기서는 프로토콜만 정한다.

---

## 0. 공통 인터페이스 (코드 위치)

| 구분 | 위치 |
|---|---|
| Artifact·Constraint·Evidence·RepairAction·Cost·Budget·Quality 스키마 | `src/core/schemas.py` |
| Producer / Validator / RepairPolicy / FinalEvaluator Protocol | `src/core/interfaces.py` |
| 오늘 설정을 새 인터페이스로 표현한 adapter | `src/core/adapters.py` |
| 테스트 | `test/core/test_interfaces.py`, `test/core/test_adapters.py` |

세 가지 독립 축(리뷰 문제 #1에 대한 직접 대응):

| 축 | 정의 | 코드 위치 |
|---|---|---|
| trigger | 어떤 검사가 켜져 있는가 | `Validator.config` (`ValidatorConfig`, 오늘의 `DEFAULT_ENABLED_VALIDATORS` `text_agent.py:89-99`와 동일 shape) |
| diagnostic feedback | 실패 메시지를 producer에게 넘기는가 | `RepairAction.use_diagnostic_feedback` |
| repair scope | 얼마나 넓게 재생성/재검사하는가 | `RepairAction.action_type` |

오늘의 `DocGenPipeline._repair_and_recompile` (`scripts/run_full_pipeline.py:425-537`)는 이 세 축이
하나로 고정되어 있다 (trigger=항상 on, scope=에러 위치에 따라 block/section 중 선택, feedback=**block
분기에서만** on — section 분기(`process_node` 호출, `run_full_pipeline.py:508-511`)는 컴파일러 에러
메시지를 아예 넘기지 않는다, 한 round당 정확히 하나의 위치만 처리). `src/core/adapters.py::
TodayFixedRepairPolicyAdapter`가 이 결정 규칙(scope·feedback의 분기별 차이와 cardinality 모두)을 그대로
`RepairPolicy.decide()`로 표현한다.

`BlindRetryRepairPolicy`는 **독립적인 scope 규칙을 새로 발명하지 않는다** — 비교 대상 정책(`wrapped`)의
`decide()`를 **공통 제한(retry cap)을 적용한 동일 evidence 목록으로 정확히 한 번만 호출**하고, 반환된
행동에서 feedback만 제거한다. 리뷰 문제 #1이 요구하는 "동일한 허용 복구 정책과 비용 상한에서 blind
retry와 비교한다"를 글자 그대로 따른 설계다 — `BlindRetryRepairPolicy(wrapped=TodayFixedRepairPolicyAdapter())`는
동일 Evidence에 대해 `TodayFixedRepairPolicyAdapter()` 단독과 **정확히 같은 scope와 cardinality**를
내고, feedback 여부만 다르다 (`test_blind_retry_shares_the_wrapped_policys_scope_rule`,
`test_blind_retry_matches_the_wrapped_policys_cardinality_exactly`). 초기 버전은 (a) scope를 직접
하드코딩했고, (b) `wrapped.decide()`를 artifact마다 따로 호출했다 — (b)는 오늘 정책이 "한 번에 하나만"
처리하는 cardinality를 우회해, 실패한 artifact가 2개면 오늘=1개 행동인데 blind=2개 행동이 나오는 식으로
scope·feedback 외의 것까지 달라지는 비교가 됐다 — 지금 설계로 교체했다.

retry 횟수 집계도 분리했다: `decide()` 자체는 side effect가 없고(몇 번을 호출해도 예산/시도 횟수를
소모하지 않음), 실행기가 실제로 복구를 시작할 때만 `record_attempt_started()`를 호출해 집계한다
(`test_blind_retry_only_counts_an_attempt_once_execution_is_recorded`). retry cap은 `wrapped.decide()`를
호출하기 *전에* evidence 목록 자체에서 이미 소진된 artifact의 항목을 제외하는 방식으로 적용된다 — 정책을
여러 번 호출하는 방식이 아니다.

**"blind"가 제거하는 feedback의 범위**: `use_diagnostic_feedback=False`는 이 RepairAction을 실행할 때
*producer 호출에 명시적으로 넘기는* 진단 텍스트만 제거한다. `TextAgent.process_node`의 내부 재시도
루프(`_run_generation_loop`)는 매 attempt마다 자기 자신이 발견한 검증 오류를 다음 attempt의 prompt에
무조건 넣는다(`text_agent.py:765`, `previous_error=last_error`) — 이를 끌 수 있는 파라미터가 없고, 이
adapter 계층에서도 끌 수 없다(agent 코드 수정은 Stage 1 범위 밖). 따라서 "blind" 조건은 **외부
RepairPolicy가 이 행동에 붙이는 진단 텍스트** 단계에서만 성립하며, producer 내부의 자기 피드백 루프는
그대로 남는다 — `RepairAction.use_diagnostic_feedback`와 `adapters.TextAgentProducer`의 docstring에
명시했다.

### 0.1. Producer/Validator 수정 (2026-10-16)

- **Producer의 생성 실패가 반환값에 드러남**: `TextAgent.process_node`는 재시도 소진 시 노드를 `ERROR`로
  표시하고 `None`을 반환하지만, `PlannerAgent.generate_plan`은 대신 예외를 던진다
  (`planner.py:714-716`). 두 메커니즘이 서로 다르다는 사실 자체가 `Producer` 인터페이스가 하나로
  통일해야 하는 이유다 — `Producer.produce()`는 이제 `ArtifactRef`가 아니라 `ProductionResult`(`status`
  /`artifact`/`error`)를 반환하고, `status=SUCCESS`이면 `artifact`가 반드시 있고 `status=FAILED`이면
  `artifact`가 반드시 없음을 스키마 자체가 강제한다. `adapters.PlannerAgentProducer`/`TextAgentProducer`가
  두 실패 경로를 모두 잡아 이 한 가지 모양으로 정규화한다.
- **Validator 설정 변경이 실제 검사에 반영됨**: `TextAgentValidatorAdapter.config`는 읽기 전용 snapshot
  (복사본)이라고 명시했고, 실제로 반영되는 변경은 `set_enabled(code, enabled)`가 `text_agent.
  enabled_validators`를 직접(같은 dict 객체) 수정한다.
- **check()가 요청받은 constraints를 실제로 사용함**: 요청된 constraint마다 평가한다 — 지원하지 않는
  constraint_id, 이 block kind에 적용 불가능한 constraint, 앞선 구조적 실패(`type_mismatch`/
  `empty_content`)로 실행되지 않은 gated 검사, trigger가 꺼진 검사를 각각 `UNKNOWN`
  (`NOT_REPRESENTABLE` 또는 `NOT_CHECKED`)으로 구분해 돌려준다. 자세한 매핑은 `adapters.py`의
  `_GATE_KEY_BY_CODE`/`_APPLICABLE_KINDS_BY_CODE`/`_EXCLUDED_KINDS_BY_CODE`.

### 0.2. 추가 수정 (2026-10-17)

- **TodayFixedRepairPolicyAdapter의 feedback이 분기에 따라 달라짐**: block 분기(`PATCH_BLOCK`)만
  `use_diagnostic_feedback=True`이고, section 분기(`REGENERATE_SECTION`)는 `False`다 — 실제
  `_repair_and_recompile`의 section 복구 호출(`process_node`)이 컴파일러 에러 메시지를 전달받지 않기
  때문이다. 이전 버전은 두 분기 모두 `True`로 반환해 실제 동작과 달랐다.
- **BlindRetryRepairPolicy가 wrapped 정책을 artifact마다 따로 호출하던 cardinality 버그**: 이제
  `wrapped.decide()`를 (retry cap으로 미리 걸러진) 전체 evidence 목록에 대해 정확히 한 번만 호출하고,
  반환된 행동에서 feedback만 제거한다 — `TodayFixedRepairPolicyAdapter`를 감쌀 때 cardinality까지
  동일해진다(`test_blind_retry_matches_the_wrapped_policys_cardinality_exactly`).
- **empty_content의 조기 종료 미반영**: `_validate_one_block`은 `type_mismatch` → `empty_content` →
  나머지 gated 검사 순으로 조기 반환한다(`text_agent.py:1020-1037`). 이전 버전은 `type_mismatch`/
  `empty_content`를 "항상 실행됨"으로 취급해, 타입이 틀리고 내용도 비어 있는 블록에서 `empty_content`가
  `PASS`로 잘못 보고됐다(실제로는 한 번도 평가되지 않음). `_short_circuited_before()`가 두 구조적 검사의
  순서를 명시적으로 모델링해, `empty_content` 자신도 `type_mismatch`에 의해 `UNKNOWN`/`NOT_CHECKED`가
  될 수 있게 했다. Kind 불일치(`NOT_REPRESENTABLE`)는 이 short-circuit 여부와 무관하게 먼저 확인한다 —
  더 근본적인 사유이기 때문이다(`test_validator_adapter_prioritizes_kind_mismatch_over_short_circuit`).

---

## 1. 독립 평가 rubric

완료 기준("평가기가 온라인 keyword rule의 결과를 그대로 최종 점수로 사용하지 않는다")을 지키기 위해,
`FinalEvaluator.evaluate()` (`src/core/interfaces.py`)는 online `Evidence`/`VerificationEvent` 스트림을
입력으로 받지 않는다 — 타입 시그니처 자체가 그런 인자를 허용하지 않는다.

| 평가 축 | 정의 | 오늘 구현 상태 | 구현 시점 |
|---|---|---|---|
| Quality | 최종 문서가 benchmark contract를 얼마나 만족하는가 | `evaluate_contract` (`src/evaluation/metrics/contract_satisfaction.py`)를 `ContractOnlyFinalEvaluator`로 감싸 재사용 — online validator와는 별도 함수 | 구현됨 (Stage 1) |
| Content preservation | 복구가 무관한 정상 내용을 얼마나 손상시켰는가 | `ContentPreservationScore.not_implemented_reason`로 명시된 미구현 — 버전 비교(수정 전/후)가 필요하며 Stage 2의 versioned artifact graph에 의존 | Stage 2 이후 |
| Verifier precision/recall | online validator의 판정이 독립 gold와 얼마나 일치하는가 | `VerifierPrecisionRecall.sample_size=0` — human gold 없음 | Stage 6 |

각 축은 "오늘 못 하면 조용히 0/1로 채우지 않고 `None` + 이유를 남긴다"는 원칙을 따른다
(`src/core/schemas.py::QualityScore.not_implemented`가 기존 `ContractResult.not_implemented`
(`src/evaluation/schemas.py:201-203`) 패턴을 그대로 따름).

---

## 2. Human annotation 방식 (프로토콜만 — 실제 수집은 Stage 6)

- **표본 추출**: trigger 표본(온라인 validator가 최소 1회 fail을 기록한 artifact)과 non-trigger 표본(한
  번도 fail하지 않은 artifact)을 각각 추출한다. `VerificationEvent.result` (`src/evaluation/schemas.py:101`)
  로 trigger 여부를 판정하되, 이 라벨은 annotation 대상 선정에만 쓰고 annotation 결과 자체에는 영향을 주지
  않는다 (라벨링 화면에 verifier 이름/결과를 노출하지 않는다 — 그렇지 않으면 verifier precision/recall이
  순환 참조가 된다).
- **층화**: `src/dataset/schemas.py::Domain`(4종)·`Difficulty`(4종)·stage(`Stage` enum,
  `src/evaluation/schemas.py:18-30`)별로 최소 표본 수를 확보한다.
- **라벨러 수와 rubric**: 항목당 최소 2인, pass/fail/unknown 3지선다 + 자유서술 사유. 두 번째 라벨러는
  첫 번째 라벨을 보지 않는다(blind).
- **불일치 처리**: 2인이 불일치하면 제3의 라벨러가 재검토(다수결), 재검토 후에도 불일치하면 `unknown`으로
  기록하고 해당 항목은 precision/recall 분모에서 제외하되 불일치율 자체를 별도로 보고한다.
- **독립성 보장**: 같은 사람이 같은 run의 online validator 디버깅에 참여했다면 그 run의 annotation에서
  제외한다.

---

## 3. 비교군 (7개)

| 이름 | 정의 | trigger | feedback | scope | 상태 |
|---|---|---|---|---|---|
| single-pass | 검증·재시도 없이 1회 생성 | off | off | 없음 | Stage 8에서 구현 (오늘 파이프라인에서 모든 validator를 off + `max_compile_repair_attempts=0`으로 재현 가능 — `--disable-*-validator` 플래그들, `run_full_pipeline.py:593-637`) |
| blind retry | 비교 대상 정책과 **동일한 scope 규칙을 공유**(wrapping)하되 feedback 없이 재시도 | on | off | 비교 대상 정책과 동일 | **구현됨** — `adapters.BlindRetryRepairPolicy(wrapped=...)` |
| final-only verification | 생성 중에는 검사하지 않고, 완성된 문서에서만 1회 검사 | off(중간), on(최종) | n/a | whole-document | Stage 4에서 구현 |
| whole-document regeneration | 어떤 실패든 문서 전체를 재생성 | on | on/off 무관 | 항상 REGENERATE_DOCUMENT | Stage 4에서 구현 |
| planner-only | TextAgent/ImageAgent 검증·복구 없이 planner 출력만으로 진행 | planner만 on | on | 없음 | 오늘 리뷰에서 이미 강력한 baseline으로 관찰됨(§리뷰 문제, Table 4 NO-VERIFIER) — Stage 8에서 공정 비교로 재확인 |
| fixed graph-aware | dependency graph 기반으로 영향 받은 artifact만 재검사/재생성, 정책은 고정 규칙 | on | on | artifact 종류·영향범위에 따라 결정 | Stage 4에서 구현 (방향 1) |
| proposed controller | 검증 결과·남은 예산을 보고 inspect/verify/patch/replan/compile/stop 중 선택 | on (동적 선택 가능) | 정책이 선택 | 정책이 선택 | Stage 5+7에서 구현 (방향 2) |

오늘 구현된 `TodayFixedRepairPolicyAdapter`는 이 7개 중 어떤 것도 아니다 — 오늘의 실제 컴파일 복구 규칙을
그대로 옮긴 것으로, "fixed graph-aware"의 선행 형태(dependency graph 없이 컴파일 에러 위치만 사용)다.

---

## 4. 비용 상한

`src/core/schemas.py::CostRecord`/`Budget`을 쓴다. `CostRecord`의 필드명은 기존
`TokenUsage` (`src/evaluation/schemas.py:207-212`)와 겹치게 설계되어 있어 합산에 변환이 필요 없다.

- 단위는 비교군 간에는 토큰 수로 통일한다(로컬 Qwen3.6 vLLM과 API 모델이 섞인 비교에서는 `latency_s`/
  `gpu_seconds`를 별도 축으로 같이 보고 — 토큰 수만으로 비용을 동치 취급하지 않는다, 방향 2의 명시적 요구).
- 비용에는 초기 생성, 모든 검사, **실패한 시도**, 복구 이후 downstream 재계산을 전부 포함한다
  (`CostRecord.succeeded=False`인 레코드도 `Budget.spent`에 합산).
- 여러 예산 구간(quality–cost curve)을 Stage 8에서 보고하며, 구체적 구간 값은 Stage 3에서 실측한 1회
  평균 비용을 보고 정한다 (지금 숫자를 미리 못박지 않는다).

---

## 5. Checkpoint 분기 실험 설계 (방향 4 포인터)

실행은 Stage 4 이후이지만 설계는 지금 고정한다:

1. 복구 전 상태를 versioned checkpoint로 저장한다 (Stage 2의 artifact graph에 의존).
2. 동일 checkpoint에서 서로 다른 `RepairPolicy`로 분기한다 — 이때 "동일 input"이 정확히 보장되려면
   checkpoint가 `ArtifactRef.version`까지 고정해야 한다 (`src/core/schemas.py::ArtifactRef`).
3. 각 분기의 후속 실행은 동일한 후속 정책(예: 둘 다 그 뒤로는 `TodayFixedRepairPolicyAdapter`)으로 고정해
   비교한다 — 그렇지 않으면 1단계 선택 효과와 이후 정책 효과가 혼재된다.
4. 효과는 해당 checkpoint·행동·후속 정책 조건에만 한정해서 해석한다 (Notion 6번 항목의 명시적 제약).

---

## 6. 반복 단위

- pilot: item당 n=3 반복부터 시작 (Notion 8단계 문구: "예산 계획은 3회 pilot부터 시작할 수 있다").
- 본실험 반복 수는 pilot에서 관측한 분산과 원하는 추정 정밀도로 정한다 — 지금 고정 숫자를 못박지 않는다.
- 신뢰구간 계산은 item을 cluster로 취급한다 (반복 run을 독립된 "문제 수"로 세지 않음) — 한 item의
  n회 반복은 그 item 1개에 대한 분산 추정치이지, n개의 서로 다른 문제가 아니다.

---

## 7. Train/dev/test 분할

`src/dataset/schemas.py`의 기존 필드를 재사용해 층화한다 — 새 분할 스키마를 만들지 않는다:

- **층화 축**: `Domain`(4종: mathematics/statistics_ml/scientific_engineering/financial_economic,
  `src/dataset/schemas.py:48-52`), `Difficulty`(4종: basic/intermediate/advanced/extreme, `:55-59`),
  그리고 Stage 2 이후에는 artifact-graph 구조(의존 깊이·fan-out)도 축에 추가한다 (지금은 섹션 그래프만
  있어 이 축은 Stage 6에서 추가).
- **중복 방지**: 같은 `sample_id`의 template/topic/graph 구조가 train/dev와 test에 걸쳐 중복되지 않게
  분할한다 (Notion 6단계: "template·topic·graph 구조를 고려한 train/dev/test 분할을 고정하고, held-out
  test를 정책 개선에 사용하지 않는다").
- **고정 시점**: 실제 분할 산출물은 Stage 6에서 만들어지고, 그 이후 Stage 7(정책 개선)은 test를 전혀
  보지 않는다 — dev에서 고른 정책을 Stage 8에서 한 번만 test에 적용한다.

---

## 8. 하지 않는 것 (Stage 1 범위 밖, 재확인)

- `scripts/run_full_pipeline.py`/agent 코드 수정 없음.
- 실제 human annotation 수집 없음 (§2는 프로토콜만).
- final-only/whole-doc-regen/planner-only/graph-aware/controller `RepairPolicy`의 실제 구현 없음
  (§3 표의 "상태" 열 참고).
- Artifact graph의 실제 저장/무효화 로직 없음 (Stage 2).
