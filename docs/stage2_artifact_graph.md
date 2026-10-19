# Stage 2 설계: Versioned artifact graph와 로그·checkpoint

이 문서는 Notion "SNU / NCIA / Personal Research / Multimodal Agents / Revision 2026.10.15"
계획의 **Stage 2** 산출물이다. Stage 1(검증·복구·평가 분리, `docs/stage1_experiment_design.md`)은
이미 완료되었다. 근거는 `file:line` 형태로 표기한다.

> **범위**: 이 문서와 `src/core/artifact_graph.py`/`src/core/run_log.py`/`src/core/schemas.py`의
> 신규 스키마는 **추가(additive)** 구현이다. `scripts/run_full_pipeline.py`나 기존 agent 코드는
> 전혀 바뀌지 않았다 (동작 변화 0). `src/common/state.py::DocumentGraph`(오늘의 섹션 전용 실행
> 상태)와도 독립이다 — 방향 1이 명시한 대로 **실행 그래프와 artifact 그래프를 분리**했다. 오늘의
> 파이프라인을 이 store로 실제 연결하는 것은 Stage 4/5의 일이다.

---

## 0. 코드 위치

| 구분 | 위치 |
|---|---|
| Stage 1 스키마 (`ArtifactRef`/`DependencyEdge`/`Evidence`/`CostRecord`/`RepairAction`/...) | `src/core/schemas.py` (기존) |
| Stage 2 스키마 (`ArtifactStatus`/`StoredArtifact`/`InvalidationRecord`/`ActionLogEntry`/`AddVersionResult`/`Checkpoint`) | `src/core/schemas.py` (신규, 파일 끝에 추가) |
| Versioned artifact graph store (버전·typed edge·change-impact·staleness·evidence 바인딩·checkpoint) | `src/core/artifact_graph.py::ArtifactGraphStore` |
| action/evidence/cost 로그 | `src/core/run_log.py::RunLog` (store가 소유, `src/evaluation/event_logger.py::EventLogger`와는 독립된 별개 개념) |
| 테스트 | `test/core/test_artifact_graph.py` (44개 중 19개가 Stage 2 신규) |

---

## 1. Notion 2단계 체크리스트 대응

### 1.1 "artifact ID/version, 읽은 입력 버전, typed edge, 출처를 저장한다. 명시된 관계와 추론된 관계를 구분한다."

- `ArtifactGraphStore.add_version(artifact_id, kind, content, read_input_versions=..., produced_by=...)`
  가 `artifact_id`별로 1부터 순차 증가하는 `version`을 발급한다 — `ArtifactRef`의 기존 docstring이
  "Stage 1 only defines the shape; Stage 2 owns actually assigning/incrementing `version`"이라고
  명시했던 바로 그 책임이다 (`schemas.py::ArtifactRef`).
- `read_input_versions`에 넘긴 각 입력마다 `DependencyEdge(relation=EXPLICIT)`을 자동 생성한다 —
  producer가 "이것을 읽었다"고 **선언**한 관계만 EXPLICIT이다.
- `ArtifactGraphStore.add_inferred_edge(from_artifact, to_artifact, reason)`는 선언되지 않은 관계를
  별도로 기록한다 (`relation=INFERRED`) — 예: 예제가 생성 입력으로는 정의를 선언하지 않았지만 본문에서
  그 label을 인용하는 경우. **Stage 2는 이 구분을 저장·질의하는 메커니즘만 제공한다 — 레퍼런스를
  자동으로 스캔해 추론 엣지를 찾아내는 NLP/정규식 탐지기는 구현하지 않았다**; 그런 추론 엣지는 지금은
  호출자가 명시적으로 `add_inferred_edge`를 불러줘야 생긴다.
- "출처"(provenance)는 두 곳에 나뉘어 저장된다: `ArtifactRef.read_input_versions`(어느 버전을
  읽었는지, Stage 1부터 존재)와 새 필드 `StoredArtifact.produced_by`(무엇이 만들었는지, 자유
  문자열 — Producer 클래스명이나 실행된 action id 등).

### 1.2 "변경 영향 후보 계산, stale 상태, 검증 evidence의 버전 바인딩·무효화를 구현한다."

- `ArtifactGraphStore.impact_candidates(artifact_id)`가 방향 1의 수식 `I(R) = R ∪ Desc_G(R)`을
  그대로 구현한다 — `_dependents` 인접 목록 위의 BFS. **순환을 명시적으로 허용한다**: Notion 문서가
  "문서 내 상호 참조에는 순환이 있을 수 있으므로 모든 관계를 DAG로 강제하지 않는다"고 적었으므로
  toposort가 아니라 visited-set BFS로 구현했다 (`test_impact_candidates_terminates_on_a_cycle`로
  순환 입력에서도 종료함을 직접 확인).
- 이 메서드는 **순수 질의**다 — 몇 번을 불러도 `self`를 바꾸지 않는다 (stale 집합·edge 수가 불변임을
  `test_impact_candidates_is_a_pure_query`로 확인). 실제로 stale 상태를 세우는 것은
  `add_version`뿐이다 — Stage 1의 `RepairPolicy.decide()`가 side effect 없이 유지되어야 한다는
  설계 원칙([[docgen_stage_review_rigor]]의 bug #1)을 이 "계산" 계층에도 그대로 적용했다.
- `add_version`은 새 버전을 쓸 때마다 `impact_candidates(artifact_id) - {artifact_id}`를
  STALE로 표시하고 `InvalidationRecord`를 남긴다 (누가·왜 stale이 됐는지 감사 가능). **artifact
  자신은 절대 stale로 표시하지 않는다** — 방금 새로 생성된 그 버전이 바로 "fresh"의 근거이기
  때문이다.
- Evidence의 버전 바인딩·무효화: `ArtifactGraphStore.live_evidence(artifact_id)`는 "현재 최신
  버전에 대해 기록되었고(**버전 일치**) 그 이후 stale이 되지 않은(**non-stale**) Evidence"만
  돌려준다. **두 조건 모두 필요한 이유**: 공통 정의가 바뀌어도 그 하위 artifact 자신의 버전 번호는
  그대로다(아직 재생성되지 않았으므로) — 그래서 버전 일치만 보면 "여전히 유효"로 잘못 판정된다.
  `test_live_evidence_excludes_evidence_whose_artifact_became_stale`이 이 정확한 시나리오를
  검증한다.

### 1.3 "action/evidence/cost를 추가 기록하는 로그와, 복구 전 checkpoint 저장·복원을 구현한다."

- `RunLog`(`src/core/run_log.py`)가 `ActionLogEntry`/`Evidence`/`CostRecord`를 각각 append-only
  리스트로 누적한다 — `EventLogger`(`src/evaluation/event_logger.py`)와 같은 "누적만 하고 아무것도
  결정하지 않는다" 패턴을 공유하지만, 둘은 서로 다른 개념(오늘의 agent-내부 재시도 vs 새 controller
  레벨 action)이라 서로 읽거나 쓰지 않는다.
- `ArtifactGraphStore.record_action()`은 **실행기가 실제로 action을 시작했을 때만** 호출해야
  한다 — `RepairPolicy.decide()` 내부에서는 절대 호출하면 안 된다(`interfaces.py`의 기존 제약과
  동일한 이유).
- `ArtifactGraphStore.save_checkpoint(label=...)`가 artifact 전 버전·edge·stale 집합·invalidation
  이력·action/evidence/cost 로그 전체를 **독립적인(live 참조 없는) 딥카피**로 `Checkpoint`에 담는다.
  `ArtifactGraphStore.from_checkpoint(checkpoint, branch_id)`는 그 `Checkpoint`의 데이터만으로
  (원본 store 인스턴스 없이도) 새 store를 재구성한다 —
  `test_from_checkpoint_rebuilds_from_serialized_json_alone`이 `model_dump_json()` →
  `model_validate_json()` 라운드트립 후에도 재구성이 되는지 직접 확인한다 (완료 기준의 "replay
  가능한 상태"에 대한 가장 직접적인 증거).
- **분기(branch) 독립성**: `restore_checkpoint(checkpoint_id, branch_id)`는 새 `branch_id`를
  반드시 명시적으로 받는다. 복원된 store에서 새로 호출하는 `add_version`/`record_action`은 전부 그
  새 `branch_id`로 기록되고, 원본 store와 서로 영향을 주지 않는다 — 완료 기준의 "LLM을 다시 호출하는
  확률적 후속 실행은 별도 run으로 기록한다"를 만족한다
  (`test_restoring_a_checkpoint_does_not_let_either_branch_affect_the_other`). 복원된
  artifact 자체의 `branch_id` 필드는 **원래 찍힌 값 그대로** 남는다 — 복원이 과거 기록을 다시 쓰지
  않는다(`test_restored_artifacts_keep_their_original_branch_provenance`).
- `latest()`/`get_version()`/`edges`는 전부 **딥카피를 반환**한다 — 호출자가 반환값을 in-place로
  수정해도 store의 저장된 히스토리는 바뀌지 않는다(버전은 불변 기록이어야 한다는 전제를 실제로
  강제). `test_latest_and_edges_return_copies_mutation_does_not_corrupt_the_store`로 확인.

### 1.4 "공통 정의 변경이 여러 section에 영향을 주는 작은 사례로 선택적 무효화와 무관한 artifact 보존을 확인한다."

`test_shared_definition_change_invalidates_dependents_but_preserves_unrelated_artifact`가
방향 1의 예시 문장("정의의 가정이 잘못되어 정리와 예제까지 영향을 받았다면, 정의를 수정하고 관련
정리·예제를 재검사한다. 무관한 비교 표는 그대로 유지한다")을 그대로 구현한다: 정의 `def_1`을
정리 `thm_1`(EXPLICIT 관계)과 예제 `ex_1`(INFERRED 관계) 둘 다 참조하게 하고, 무관한 비교 표
`tbl_1`은 어떤 관계도 갖지 않게 한 뒤 `def_1`을 변경한다. `impact_candidates`가
`{def_1, thm_1, ex_1}`만 반환하고, `thm_1`/`ex_1`만 STALE이 되며, `tbl_1`은 stale 집합에도
invalidation 이력에도 전혀 등장하지 않고 내용도 그대로임을 확인한다.

---

## 2. 완료 기준 대응

> "artifact store와 replay 가능한 상태. 저장된 입력·산출물·도구 관측은 동일하게 복원되며, LLM을
> 다시 호출하는 확률적 후속 실행은 별도 run으로 기록한다."

- **저장된 입력·산출물**: `Checkpoint.artifacts`/`edges`가 `ArtifactRef`/`StoredArtifact`를
  그대로 보관하므로 복원 시 byte-for-byte 동일하다 (pydantic 모델 동등성으로 확인).
- **도구 관측**: 지금 단계에서 "도구 관측"에 대응하는 것은 `Evidence`(검사 결과)와
  `CostRecord`(실행 비용)다 — 둘 다 checkpoint에 포함되어 동일하게 복원된다. Stage 3의
  inspect/query/check/patch/replan/compile 도구 schema가 만들어지면 그 출력도 같은 로그에 쌓이도록
  확장하면 된다 (지금 구조를 바꿀 필요는 없다).
- **확률적 후속 실행의 분리 기록**: 위 §1.3의 branch 독립성으로 구현됨.

---

## 3. 하지 않은 것 (Stage 2 범위 밖, 다음 단계로 명시적으로 미룸)

- **실제 레퍼런스 스캔 기반 INFERRED edge 자동 탐지**: `add_inferred_edge`는 메커니즘만 제공한다 —
  "이 섹션이 저 label을 인용한다"를 실제로 찾아내는 로직(정규식/NLP)은 구현하지 않았다.
- **"수정/재검사/재생성 대상" 3분류 및 실제 복구 연산**: `impact_candidates`는 방향 1이 말하는
  "보수적 영향 **후보**"만 계산한다 — 그중 무엇을 patch/regenerate/replan할지 **결정**하는 것은
  Stage 4의 고정 정책, Stage 5+의 controller 몫이다.
- **constraint hypergraph**: Stage 1의 `Constraint.applies_to: List[ArtifactRef]`가 다중
  artifact에 걸친 제약을 이미 표현할 수 있어, 별도 hypergraph 구조를 새로 만들지 않았다.
- **Budget 소비/강제**: `RunLog`는 `CostRecord`를 누적만 하고 `Budget.spent`를 갱신하거나 cap을
  강제하지 않는다 — 비용 집계·budget cap enforcement는 Stage 3의 명시적 범위다.
- **오늘 파이프라인과의 실제 연결**: `DocumentGraph`/`PlannerAgent`/`TextAgent`에서 실제로
  `ArtifactGraphStore.add_version`을 호출하도록 바꾸는 작업은 없다 — Stage 1의 adapter들과 마찬가지로,
  이 store가 "표현 가능함"을 테스트로 증명하는 단계이며 실제 연결은 Stage 4/5다.
- **파일 기반 checkpoint 영속화**: `Checkpoint`는 pydantic 모델로 완전히 직렬화 가능함을
  (`model_dump_json`/`model_validate_json`) 테스트로 확인했지만, 디스크 경로·디렉터리 레이아웃을
  정해 실제로 파일에 쓰는 코드는 아직 만들지 않았다 — 지금 필요하지 않은 디렉터리 규약을 미리
  설계하지 않기 위함이며, 필요해지면 `Checkpoint.model_dump_json()`을 그대로 쓰면 된다.
