# Method Architecture: Verifier 기반 Multi-Agent LaTeX 문서 생성 파이프라인

이 문서는 `document_generation_api`의 method architecture를 정리한다: verifier 구조,
multi-agent orchestration 흐름, 오류 탐지 → local recovery 로직, failure taxonomy,
로깅/메트릭 수집 방식. 모든 근거는 `file:line` 형태로 표기한다.

> **주의**: 코드베이스에는 `verifiers/` 같은 별도 모듈이 존재하지 않는다. "verifier"는
> 각 에이전트(`planner.py`, `text_agent.py`, `latex_assembler.py`, `latex_compiler.py`) 내부에
> 임베드된 validator 함수/메서드이며, 모두 공통 `EventLogger.log_verification(...)`
> (`src/evaluation/event_logger.py:61-95`)을 통해 `VerificationEvent`로 기록된다.
> "deterministic / execution-grounded / semantic" 3분류는 코드에 명시된 용어가 아니라,
> 실제 동작을 기준으로 분류한 것이다.

---

## 0. 전체 파이프라인 개요

```
PlannerAgent          →  TextAgent (+ImageAgent)   →  LatexAssembler   →  LatexCompiler
(blueprint 생성)          (섹션별 콘텐츠 생성)          (LaTeX 조립)         (실제 컴파일)
      │                          │                        │                    │
      ▼                          ▼                        ▼                    ▼
 deterministic              deterministic            deterministic       execution-grounded
 verifier                   verifier                 verifier            verifier
 (blueprint validator)      (semantic block          (layout/reference/  (real LaTeX engine
                             validator)               frame validator)    compile)
```

오케스트레이터는 별도 파일이 아니라 `DocGenPipeline` 클래스
(`scripts/run_full_pipeline.py:203-537`)이며, `run()`이 Planning → Content generation →
Integration & compile 3단계를 순차 실행한다.

---

## 1. Verifier 구조

### 1.1 Deterministic verifier (규칙 기반, 정적 검사)

LLM 판단이나 컴파일 없이 텍스트/JSON 산출물을 고정 규칙(정규식, allowlist, 참조 무결성)으로 검사.

| 위치 | 함수 | 파이프라인 호출 지점 | Input | Output/Verdict |
|---|---|---|---|---|
| Planner | `_validate_blueprint` (`src/agents/planner.py:719-898`) | `generate_plan` 내부 (`planner.py:702-706`), plan 파싱 직후 | 파싱된 `PlannerOutput`(graph nodes + `DocumentBlueprint`), in-memory | 실패 시 `ValueError` → `AttemptTracker`가 `VerificationEvent(stage="planner", verifier="planner_blueprint_validator", result="fail")`로 기록. `signal_type` ∈ `{json_parse_error, schema_validation_error, blueprint_validation_error, generation_error}` |
| TextAgent | `_validate_one_block` (`src/agents/text_agent.py:1003-1144`) | 블록 생성 직후, 각 `ContentBlock`을 대응 `LayoutBlock`과 대조 | 1개 `ContentBlock`(LLM JSON) + 대응 `LayoutBlock`(blueprint spec) | `SemanticBlockError(code, block_id, message)` 리스트 → `SemanticBlockErrors`. 통과한 블록은 별도로 보존되어 **부분 수용(partial acceptance)** 가능 |
| LatexAssembler | `validate_layout_policy`, `figure_asset_completeness`, `_validate_semantic_blocks`, `validate_figure_single_use`, `validate_references`, `validate_document_frame` (`src/agents/latex_assembler.py:694-877`) | `assemble()` 내부(`586-692`), 렌더링 직후/도중, 컴파일 이전 | 렌더링된 LaTeX 텍스트 + in-memory blueprint/그래프 (PDF 없음) | 각 `validate_*`는 `List[str]`(빈 리스트=pass) 반환 → `_log_list_check`가 pass/fail `VerificationEvent`로 변환. 비어있지 않으면 `ValueError`로 `assemble()` 중단(컴포저 자체는 재시도하지 않음) |

Planner/TextAgent 단계 validator는 각각 `DEFAULT_ENABLED_PLANNER_VALIDATORS`
(`planner.py:368-373`), `DEFAULT_ENABLED_VALIDATORS` (`text_agent.py:89-99`),
`DEFAULT_ENABLED_ASSEMBLER_VALIDATORS` (`latex_assembler.py:543-561`)로 개별
on/off 가능하며, CLI에서 `--disable-planner-validator` 등으로 verifier ablation 실험을
지원하도록 설계되어 있다.

### 1.2 Execution-grounded verifier (실제 컴파일)

| 위치 | 함수 | 파이프라인 호출 지점 | Input | Output/Verdict |
|---|---|---|---|---|
| LatexCompiler | `compile_pdf` (`src/agents/latex_compiler.py:100-151`) | Integration & compile 단계, `LatexAssembler.assemble()` 성공 직후 (`run_full_pipeline.py:409-411`) | 이미 조립된 LaTeX 프로젝트 디렉토리(`main.tex` + `sections/*.tex` + assets) | 실제 LaTeX 엔진(`tectonic` 우선, `pdflatex` fallback)을 `subprocess`로 실행 → `compile_success`(bool, PDF 존재 여부). 로그는 `parse_compile_log`/`classify_error_type` (`src/evaluation/compile_log_parser.py:60-179`)로 `(fatal_error_count, warning_count, first_error_type)` 산출. `compile_result.json`(`CompileResult` pydantic 모델)로 영속화, `VerificationEvent(stage="compile", verifier="latex_compiler")`로 로깅 |

컴파일 오류는 `extract_error_locations`(tectonic 전용, `compile_log_parser.py:137-158`)와
`LatexAssembler.locate_block_at_line`(렌더 시 삽입된 `% <layout-anchor:{block_id}>` 주석을
역추적, `latex_assembler.py:940-963`)로 **컴파일러 라인 번호 → 구체적 semantic block**까지
attribution되며, 이것이 §3의 recovery loop를 구동한다.

### 1.3 "Semantic" verifier — 사후(post-hoc) 어휘 기반 검사, LLM-judge 아님

**중요**: 파이프라인 내에 LLM-as-judge 형태의 진짜 의미론적 verifier는 존재하지 않는다
(README `126-130`, `contract_satisfaction.py` docstring `1-8`이 명시적으로 향후 과제로 유보).
대신 존재하는 것은 결정론적 키워드/어휘 매칭 기반의 "semantic 근사" 검사다.

| 위치 | 함수 | 파이프라인 호출 지점 | Input | Output/Verdict |
|---|---|---|---|---|
| Evaluation | `evaluate_contract` (`src/evaluation/metrics/contract_satisfaction.py:395-435`) | 매 run 종료 후 `run_full_pipeline.py:77`(`_build_run_result`)에서 호출, batch 재채점 시 `aggregate.py:147`에서도 호출 — **생성 루프 밖, retry를 게이팅하지 않음** | `project_dir/plan.json` + `project_dir/sections/*`(컴파일 도달 시 `.tex`, 아니면 `.blocks.json`) + benchmark `BenchmarkItem` contract | `ContractResult`: 카테고리별(required_sections, required_components, terminology, notation) float 점수(0-1) + `ContractCategoryDetail`(status ∈ `satisfied\|partially_satisfied\|missing\|not_representable\|not_applicable`). `content_coverage`/`cross_section_dependency`는 미구현으로 항상 `None` |

이 검사는 recovery/retry 루프에 연결되어 있지 않고, `RunResult`/집계 메트릭(§5의 Metric 7)에만
반영된다.

---

## 2. Multi-Agent Orchestration 흐름

오케스트레이터: `DocGenPipeline` (`scripts/run_full_pipeline.py:203-537`, 별도 orchestrator
파일 없음). `PlannerAgent`, `TextAgent`, `LatexAssembler`, `LatexCompiler`는 항상 생성되고,
`ImageAgent`는 `image_model != 'none'`일 때만 생성(`204-278`) — planner는 기본적으로
`FIGURE` 노드를 생성하지 않도록 지시되어 ImageAgent는 사실상 호출되지 않는 경우가 많음
(`planner.py:572-581`, README `22-24`).

### 실행 순서 (`DocGenPipeline.run()`, `run_full_pipeline.py:279-423`)

1. **Phase 1 — Planning** (`324-337`): `PlannerAgent.generate_plan(...)` → `PlannerOutput`
   (graph nodes + `DocumentBlueprint`), `plan.json`으로 영속화. `DocumentGraph`에 노드/계층
   엣지 추가, `graph.set_blueprint(...)`로 blueprint 부착.
2. **Phase 2 — Content generation loop** (`343-391`): 외부 루프
   `for i in range(max_iterations)` (기본 `--iterations 3`). 매 iteration마다
   `PENDING`/`ERROR` 상태 노드를 수집 → `SECTION`/`SUBSECTION`은
   `TextAgent.process_node(...)`, `FIGURE`는 `ImageAgent.process_figure_node(...)`로 위임
   (표는 별도 노드 타입 없이 TextAgent 내부의 `pandas_csv_to_latex_table` 도구로 생성).
   전 노드가 `DRAFTED`에 도달하지 못하면 `RuntimeError`(`content_generation_error`)로 전체
   요청 중단.
3. **Phase 3 — Integration & compile** (`396-423`): `LatexAssembler.assemble(...)` 실패 시
   `integration_error`(컴파일 로그가 아직 없는 시점). 성공 시 `LatexCompiler.compile_pdf(...)`
   실행, 실패 시 `_repair_and_recompile(...)` 호출(§3). 최종 상태는
   `success \| content_generation_error \| integration_error \| compile_error` 중 하나로
   `run_result.json`에 기록.

### 에이전트 간 전달되는 데이터 구조

- `PlannerInput` → `PlannerOutput`(`src/common/schemas.py:152-198`): `DocumentBlueprint`
  (`sections: List[SectionLayout]`, 각 `SectionLayout.blocks: List[LayoutBlock]`) —
  모든 하위 에이전트가 참조하는 단일 계약(contract).
- `TextAgent`는 `TextAgentOutput`(`blocks: List[ContentBlock]`)을
  `sections/{node_id}.blocks.json`으로 **파일에 기록**하고, `LatexAssembler`는 이를
  in-memory 전달이 아니라 **디스크에서 다시 읽어들인다**(`latex_assembler.py:965-995`) —
  프로세스보다 오래 남을 수 있는 JSON 아티팩트로서의 의도적인 파일 경계
  (`latex_assembler.py:488-498` 주석).
- 단일 mutable `DocumentGraph`(`src/common/state.py`)가 전체 단계를 관통하며,
  노드 상태(`NodeStatus`: `PENDING, RUNNING, DRAFTED, REVIEWED, VALIDATED, BLOCKED, ERROR`),
  figure registry, cross-reference label registry를 보유.

### 루프 구조와 종료 조건 — 2단(two-level) retry 아키텍처

- **Inner loop (에이전트/노드 단위)**: `PlannerAgent.generate_plan`, `TextAgent.process_node`
  /`regenerate_blocks` 각각 자체 `for attempt in range(1, max_attempts+1)` 루프 수행
  (기본 Planner 3회, TextAgent 신규 초안 3회 / `regenerate_blocks` 2회). validation 오류가
  다음 프롬프트에 피드백되는 **generate → verify → recover(retry-with-feedback) → re-verify**
  구조. 성공 시 즉시 종료, attempt 소진 시 노드는 `ERROR`로 마킹(또는 planner는 예외 발생).
- **Outer loop (`DocGenPipeline.run`, Phase 2)**: 여전히 `PENDING`/`ERROR` 상태인 노드를
  최대 `iterations`(기본 3)회까지 재디스패치 — 각 에이전트의 inner loop 위에 얹힌 더 굵은
  단위의 재시도.
- **Compile-repair loop**: 별도의 세 번째 루프(§3 참고).

---

## 3. 오류 탐지 → Local Recovery 로직

**복구 단위는 가능한 한 block 단위, 불가능하면 section 단위로 fallback, planner는
plan 전체 단위** — 코드는 attribution 가능한 만큼 스코프를 의도적으로 좁힌다.

### 3.1 TextAgent — block 단위 점진적 축소 (가장 세밀한 복구 단위)

`_run_generation_loop` (`text_agent.py:728-821`):
- `SemanticBlockErrors` 실패 시, **실패한 `block_id`들만** 다음 attempt의 `pending_ids`에
  재투입. 이미 통과한 블록은 `accepted`에 보존되어 다시 생성되지 않음(`776-792`).
- 구조적 실패(잘못된 JSON, id/순서/개수 오류 등 하나의 블록에 귀속시킬 수 없는 경우)는
  대기 중인 블록 집합 전체를 그대로 재시도(`793-800`).
- → 이 시스템에서 가장 fine-grained한 복구 단위는 **section이 아니라 단일 semantic block**.

### 3.2 Compile-repair loop — block 단위, section 단위로 fallback

`DocGenPipeline._repair_and_recompile` (`run_full_pipeline.py:425-537`), `LatexCompiler.compile_pdf`
실패 후에만 호출(`413-416`). `--max-compile-repair-attempts`(기본 2)로 제한.

절차(매 attempt):
1. `compile.log` 파싱 → `extract_error_locations`(tectonic 전용)
2. `RecoveryAction.REGENERATE_CONTENT`로 분류된 오류만 필터링 — 진짜로 콘텐츠 수정이 필요한
   LaTeX 문법 오류만 해당. `DETERMINISTIC_REPAIR`(패키지/파일 누락)와
   `INFRASTRUCTURE_FAILURE`는 분류만 되고 **자동 복구되지 않음**
   (`RecoveryAction` docstring, `src/evaluation/schemas.py:39-45`).
3. `section_id`(tectonic 파일 경로에서 무료로 확보) + `block_id`
   (`LatexAssembler.locate_block_at_line`)를 resolve
4. block이 resolve되면 → `TextAgent.regenerate_blocks(section_id, block_id, error_message, ...)`가
   **그 블록 하나만** 재생성, 컴파일러 오류 메시지를 attempt 1부터 프롬프트에 미리 시딩
   (`text_agent.py:654-726`)
5. block이 resolve되지 않으면 → 해당 **section 전체**를 `TextAgent.process_node`로 재초안
   (`run_full_pipeline.py:506-511`)
6. **문서 전체 재생성은 절대 일어나지 않음**
7. `LatexAssembler.assemble` + `LatexCompiler.compile_pdf` 재실행, 성공 시 즉시 반환,
   실패 시 다음 repair attempt 또는 포기

### 3.3 Planner — sub-plan 단위 복구 없음

Blueprint validation 실패 시 **plan 전체**를 재생성(`planner.py:671-717`). blueprint의
특정 섹션 하나만 patch하는 메커니즘은 존재하지 않음.

### 3.4 Retry 횟수 / 종료 조건 (모두 bounded, unbounded loop 없음)

| 단계 | 최대 attempt |
|---|---|
| Planner | 3 (`max_attempts=3`, `planner.py:624`) |
| TextAgent 신규 초안 | 3 (`text_agent.py:551`) |
| TextAgent `regenerate_blocks`(compile-repair 경로) | 2 (`text_agent.py:662`) |
| Outer pipeline iteration | 3 (`--iterations`, `run_full_pipeline.py:577`) |
| Compile-repair round | 2 (`--max-compile-repair-attempts`, `run_full_pipeline.py:579-582`) |

ImageAgent와 컴포저의 구조적 검사는 **retry/recovery 경로가 전혀 없음** — 실패 시
`failure_id=None`으로 로깅되며 "복구 경로 없이 탐지된 실패" 버킷(Metric 2)에 집계됨
(`image_agent.py:150-154`, `event_logger.py:41-59`).

### 3.5 Failure ↔ Recovery 페어링

`AttemptTracker` (`src/evaluation/event_logger.py:135-252`)가 inner-loop의 실패 →
(다음 attempt 결과)를 `RecoveryEvent(failure_id, recovery_action, success, ...)`로 페어링.
- `record_failure`: 새 `failure_id` 발급, 이전 대기 중인 실패가 있으면 즉시
  `success=False`로 종료 기록(`184-212`)
- `record_success`: 대기 중인 실패를 `success=True`로 종료(`214-241`)
- `record_exhausted`: 상태만 초기화 — 루프가 포기하기 직전의 마지막 실패는
  `RecoveryEvent` 없이 남음(의도된 설계: "이에 대응한 재시도가 없었으므로",
  `event_logger.py:150-157`)

---

## 4. Failure Taxonomy

단일 통합 enum은 없고, 목적별로 여러 taxonomy가 분리되어 있다.

| Taxonomy | 위치 | 값 |
|---|---|---|
| **Stage** (실패 발생 단계) | `schemas.py:18-30` | `PLANNER, TEXT_AGENT, IMAGE_AGENT, COMPOSER, COMPILE`. `GENERATION_STAGES = {PLANNER, TEXT_AGENT, IMAGE_AGENT}` |
| **RecoveryAction** (컴파일 실패 시 필요한 조치 종류) | `schemas.py:39-50` | `REGENERATE_CONTENT`(유일하게 자동 복구됨), `DETERMINISTIC_REPAIR`, `INFRASTRUCTURE_FAILURE`, `UNKNOWN`(모두 분류만 되고 자동 복구 안 됨) |
| **signal_type** (생성 오류, 에이전트별 plain string) | Planner `planner.py:339-350` | `json_parse_error, schema_validation_error, blueprint_validation_error, generation_error` |
| | TextAgent `text_agent.py:31-46` | `json_parse_error, schema_validation_error, content_validation_error, generation_error` |
| | ImageAgent `image_agent.py:17-26` | `json_parse_error, content_validation_error, image_generation_error` |
| | Composer semantic check `latex_assembler.py:488-498` | `missing_semantic_artifact, invalid_semantic_artifact, composer_semantic_check_error` |
| **SemanticBlockError.code** (블록별 콘텐츠 규칙 위반) | `text_agent.py:261-330` | `type_mismatch, empty_content, forbidden_environment, manual_label, table_columns, table_shape, equation_shape, plain_title, figure_asset_mismatch, figure_forbidden_content, unicode_math_symbol, bare_math_notation, compile_error`(compile-repair 경로에서 주입) |
| **first_error_type** (컴파일 로그 분류) | `compile_log_parser.py:47-65` | `missing_math_delimiter, undefined_control_sequence, undefined_reference_or_command, missing_begin_document, unbalanced_environment, missing_file, latex_error, package_error, missing_delimiter_or_argument, other` + `latex_compiler.py`에서 합성되는 `main_tex_missing, no_latex_engine_found` |
| **ContractItemStatus** (contract 충족 판정) | `schemas.py:157-159` | `satisfied, partially_satisfied, missing, not_representable, not_applicable` |
| **Composer 구조적 검사별 signal_type** | `latex_assembler.py` (`_log_list_check` 호출부) | `layout_policy_violation, missing_figure_asset, figure_single_use_violation, undefined_reference, document_frame_incomplete` |
| **VerificationResultValue** (모든 verifier 공통 최상위 판정) | `schemas.py:52` | `Literal["pass", "fail"]` |

---

## 5. 로깅 / 메트릭 수집 방식

### 5.1 이벤트 스키마 (`src/evaluation/schemas.py`)

- **`VerificationEvent`** (`61-107`): `run_id, stage, verifier, artifact_id, section_id,
  block_id, attempt, outer_iteration, sequence_id, result(pass/fail), signal_type, message,
  timestamp, producer_model, failure_id`. `sequence_id`는 run 내 전역 단조 증가
  (`attempt`는 outer 재디스패치마다 리셋되므로 별도 필요).
- **`RecoveryEvent`** (`109-130`): `run_id, failure_id, stage, artifact_id, recovery_action,
  attempt, outer_iteration, sequence_id, success, resulting_artifact_id`.
- **`CompileResult`** (`133-154`): `compile_success, engine, return_code, fatal_error_count,
  warning_count, first_error_type, recovery_action, pdf_exists`.
- **`RunResult`** (`236-281`): run 단위 최상위 레코드 — `run_id, benchmark_id,
  producer_model(s), seed, config_name, verification_config, completed, status,
  error_type/message, compile_success, compile_result, verification_events[],
  recovery_events[], contract_result, total_generation_attempts, total_recovery_attempts,
  token_usage, elapsed_time`.

### 5.2 기록/영속화 경로

- `event_logger`를 인자로 받는 모든 호출부가 `EventLogger.log_verification`/
  `log_recovery`(`event_logger.py:23-132`)로 직접 기록(`None`이면 no-op). `EventLogger`는
  run 1회당 1개 생성(`run_full_pipeline.py:293`)되어 메모리에 누적.
- `_write_run_log`/`_build_run_result`(`run_full_pipeline.py:55-176`)가 매 요청 종료 시
  (성공/실패 무관) 전체 `RunResult`를 `<project_dir>/run_result.json`에 기록(계측 버그가
  성공한 run을 깨뜨리지 않도록 try/except로 격리, 실패 시 legacy flat dict로 fallback).
- `LatexCompiler`는 컴파일 시도마다 별도 `compile_result.json`과 raw 엔진 로그
  `compile.log`도 기록.
- **DB 없음** — 모든 영속화는 `outputs/`, `evaluation/` 아래 flat JSON/JSONL 파일.

### 5.3 계산되는 메트릭 (`src/evaluation/metrics/*.py`, `aggregate.py`가 오케스트레이션)

1. **Verification Trigger Rate** (`trigger_rate.py:76-81`): `triggered(fail) / checked`,
   `event_level`(전체 이벤트) / `initial_attempt_only`(attempt==1만) 두 축, 각각
   `overall`/`by_stage`/`by_verifier`/`by_producer_model`로 분해 + 실패에 대한
   `by_signal_type` 분포.
2. **Recovery Success Rate** (`recovery_success.py:13-66`):
   `successfully_recovered / failures_with_recovery_attempt`. "복구 경로 자체가 없음"
   (`detected_failures_without_recovery_path`)과 "경로는 있으나 재시도 없이 소진됨"
   (`detected_failures_without_recovery_attempt`)을 별도 버킷으로 구분, `by_stage` 분해.
3. **Compile Success Rate** (`compile_success.py:12-47`):
   `successfully_compiled / generation_runs` + `runs_with_fatal_errors`,
   `runs_with_warnings_only`(의도적으로 `compile_success`와 독립 유지), `by_first_error_type`.
4. **Contract Satisfaction Rate** (Metric 7; `contract_satisfaction.py:438-455`):
   카테고리별(`overall, required_sections, required_components, terminology, notation`)
   배치 평균, `content_coverage`/`cross_section_dependency`는 항상 `not_implemented`.

### 5.4 집계 엔트리 포인트

`aggregate.py:110-176`(`aggregate_batch`)가 `discover_sample_dirs`/`split_factor_sample`
(`src/evaluation/discovery.py:14-38`, `plan.json`/`meta.json` 마커로 탐색)로 배치 디렉토리를
순회, 샘플별 `run_result.json`을 로드(계측 이전 run은 `compile_result.json`/`compile.log`로부터
`CompileResult`를 재구성하는 fallback 존재), 모든 이벤트를 pool하여
`evaluation/foundation_metrics/<batch_name>_foundation_metrics_eval.json`에 통합 JSON으로
기록.

---

## 6. 부가 설계 노트

- **Verifier ablation이 1급 설계 요소**: Planner/TextAgent/Assembler의 각 validator dict는
  이름 단위로 독립 on/off 가능하며, CLI 플래그(`--disable-planner-validator`,
  `--disable-text-validator`, `--disable-assembler-validator`)로 각 verifier의
  recovery/compile-success 기여도를 개별 측정하도록 설계됨. TextAgent와 LatexAssembler에
  이름이 겹치는 중복 검사(예: `forbidden_environment`)가 있는데, 이 둘은 의도적으로 미러링되지
  않으므로 `_warn_verifier_pairing_mismatches`(`run_full_pipeline.py:179-200`)가 활성화 상태
  불일치를 경고.
- **LLM-judge형 semantic verifier는 파이프라인에 존재하지 않음** — §1.3의 contract
  satisfaction은 전부 키워드/어휘 매칭이며, 이는 README와 코드 docstring에서 명시적으로
  향후 과제로 유보된 부분.
- **`src/baseline_single_llm/single_llm.py`**: verifier 없는 단일 LLM 호출 baseline —
  본 multi-agent+verifier 파이프라인과의 비교 대조군(control condition)으로 존재.
