# Stage 6 — 실패·행동 데이터셋과 human gold 구축

Notion "Revision 2026.10.15" 6단계 구현 기록. Stage 1-5 문서와 동일한 형식을 따른다.

> **범위 (구현 전 사용자 결정 3건)**: (1) 이번 단계는 분류·집계 메커니즘과 Protocol-conforming
> fake 기반 테스트까지만 완료한다 — 실제 로컬 Qwen3.6 pipeline을 돌려 자연 실패를 실제로
> 수집하는 pilot은 Stage 3-5와 동일한 원칙으로 별도 후속 작업으로 미룬다. (2) 4가지 주입
> 유형(가정 누락/분모 변경/기호 불일치/불충족 blueprint)의 실제 injector는 이번 단계에서
> 구현한다 — schema만 두고 미루지 않는다. (3) human annotation은 **도구만** 구축한다 — AI는
> 독립적인 human annotator 역할을 할 수 없으므로, 실제 라벨링은 사용자 또는 고용된
> annotator가 이 도구를 통해 수행할 후속 작업으로 남긴다. 이 문서의 테스트에 쓰인 라벨은
> 모두 합성 fixture이며 실제 human judgment로 취급하지 않는다.
>
> 이 결정에 따라 **Notion 체크리스트의 체크박스는 이번 커밋에서 올리지 않았다** — "수집한다"/
> "사람이 평가한다"는 문자 그대로 아직 실행되지 않았고, 이전 단계들의 "메커니즘 완료, pilot은
> 후속"이라는 표현을 체크된 박스에 괄호로 덧붙이는 방식보다, 실제 데이터/실제 human label이
> 존재하기 전까지는 체크하지 않는 쪽이 이 프로젝트의 overclaiming 민감도(`docgen_stage_review_rigor`
> 메모리)에 더 부합한다고 판단했다. §5에 아직 남은 일을 명시한다.

## 0. 코드 위치

- `src/core/fault_injection.py` — `FaultInjector` (4개 주입 유형 각각 메서드 하나씩,
  `ArtifactGraphStore.add_version`만으로 구현 — 실제 Producer/LLM 호출 없음).
- `src/core/failure_dataset.py` — `classify_failure_difficulty`(의존 깊이·fan-out·공유 정의
  사용·관계 누락 4축), `build_failure_record`, `compute_repair_gain_score`/
  `build_candidate_action_record`(Stage 4의 `BranchExecutionResult`에서 repair gain·손상·
  총비용을 계산).
- `src/core/annotation.py` — `select_balanced_sample`, `resolve_annotations`(동의/조정/미해결),
  `compute_agreement_report`, `compute_stratum_weights`, `compute_verifier_precision_recall`.
- `src/core/dataset_split.py` — `build_fixed_split`(template/topic/graph-structure 단위로
  묶어서 분할), `train_sample_ids`/`dev_sample_ids`/`held_out_test_sample_ids`(명시적
  `acknowledge_test_use=True` 없이는 거부).
- `src/core/schemas.py` — `FaultInjectionKind`/`InjectedFault`, `FailureOrigin`/
  `FailureDifficultyProfile`/`FailureRecord`, `RepairGainScore`/`CandidateActionRecord`,
  `AnnotationOutcome`/`AnnotationStratum`/`AnnotationTask`/`AnnotationLabel`/
  `ResolvedLabelStatus`/`ResolvedAnnotation`/`AnnotationAgreementReport`/`StratumWeight`,
  `SplitName`/`SplitFamilyKey`/`SplitAssignment`/`DatasetSplit` 추가.
- `src/core/artifact_graph.py` — 작은 추가 2건: `direct_dependents`(Stage 5의
  `is_evidence_still_valid`와 같은 전례 — 기존 private `_current_dependents()`를 공개
  accessor로 노출), `direct_dependencies`(그 반대 방향, `dependency_depth` 계산용으로 신규
  추가). 둘 다 순수 조회이며 `self`를 변경하지 않는다.
- `test/core/test_fault_injection.py`(10개), `test/core/test_failure_dataset.py`(17개),
  `test/core/test_annotation.py`(16개), `test/core/test_dataset_split.py`(9개) — 신규 52개
  전부 통과. `test/core/` 260→312개(260+52=312, §4의 collect-only 결과로 확인), `test/evaluation/`
  38개는 변경 없음.

`scripts/run_full_pipeline.py`/agent 코드는 전혀 수정하지 않았다 — Stage 1-5와 동일한 순수
추가(additive) 원칙. `artifact_graph.py`에 메서드 2개를 추가한 것은 Stage 2 자체의 동작을
바꾸지 않는, 이미 존재하는 private 로직을 공개하거나(`direct_dependents`) 그와 대칭인 조회를
신규로 추가한 것(`direct_dependencies`)뿐이다.

---

## 1. Notion 6단계 체크리스트 대응

### 1.1 "자연 실패와 주입 실패를 구분해 수집하고 의존 깊이·fan-out·공유 정의·관계 누락에 따라
분류한다."

`FailureOrigin`(NATURAL/INJECTED)을 모든 `FailureRecord`에 명시적으로 둔다 — `injected_fault`가
있는지 여부로 추론하지 않는다 (`FailureRecord.model_post_init`이 둘의 일치를 강제). 4개 분류축은
`classify_failure_difficulty(store, artifact_id) -> FailureDifficultyProfile`:

- **의존 깊이**: `_dependency_depth`가 `ArtifactGraphStore.direct_dependencies`를 따라 올라가는
  가장 긴 현재 경로의 길이를 재귀적으로 계산한다. `impact_candidates`와 동일하게 그래프가
  acyclic이라고 가정하지 않는다 — 이미 방문 중인(현재 recursion stack에 있는) 노드로
  되돌아가는 edge는 깊이에 0만큼만 기여한다고 처리해(진짜 longest-*simple*-path는 cycle이
  있는 그래프에서 NP-hard이므로) 종료를 보장한다 (`test_dependency_depth_terminates_on_a_genuine_cycle`).
- **fan-out**: `len(store.direct_dependents(artifact_id))` — Stage 4의 `FixedGraphAwareRepairPolicy`
  가 REPLAN_SUBGRAPH로 올릴지 결정할 때 쓰는 것과 같은 "직접 의존자 수" 개념.
- **공유 정의 사용**: `kind not in {BLOCK, SECTION, DOCUMENT} and fan_out > 1` — 역시
  `FixedGraphAwareRepairPolicy._scope_for`의 3번 분기 조건을 그대로 재사용했다(재정의하지
  않음), "난이도"와 "복구 범위 선택"이 서로 다른 기준으로 갈라지지 않도록.
- **관계 누락**: `store.edges` 전체에서 이 artifact가 endpoint인 `relation=INFERRED` edge
  개수. INFERRED edge 자체가 "생성 시점에 선언되지 않았다가 나중에 발견된 관계"라는 사실의
  직접적인 증거이므로, 별도 발견 로직을 새로 만들지 않고 Stage 2가 이미 구분해 둔 값을 그대로
  센다.

주입 실패는 `src/core/fault_injection.py::FaultInjector`가 담당한다 (아래 §2).

### 1.2 "checkpoint별 후보 행동과 반복 후속 결과를 저장해 repair gain, 손상, 총비용을 계산한다."

Stage 4가 이미 만든 `BranchComparisonResult`/`BranchExecutionResult`가 "같은 checkpoint에서
여러 후보 행동을 실행하고 반복된 후속 결과를 저장"하는 메커니즘 자체를 전부 갖고 있다 — Stage
6은 그 결과를 **다시 실행하지 않고** 집계만 한다.

- **repair gain**: `compute_repair_gain_score(before_checkpoint, branch)`가
  `RepairExecutionResult.final_outcome_by_artifact`(그 schema 자신의 docstring이 이미 "이것만
  읽고 `regenerated`/`reverified_untouched`를 직접 스캔하지 말라"고 경고하는 필드)를 라운드
  순서대로 병합한 뒤(`_merge_final_outcomes` — 나중 라운드가 항상 이긴다), `RESOLVED` 비율을
  gain으로 낸다. 분모가 0이면(예산이 처음부터 0이라 아무 artifact도 최종 상태에 도달하지
  못한 경우) `gain=None` + `not_implemented_reason`을 남긴다 — 0.0으로 조작하지 않는다.
- **손상**: `_compute_damage`가 `before_checkpoint`와 `branch.final_checkpoint`를 비교한다 —
  비교 대상은 **이 branch가 한 번도 건드리지 않은(=`final_outcome_by_artifact`에 없는)**
  artifact뿐이다. 거기서 content가 달라졌다면 그건 이 복구가 고치려던 것과 무관한 손상이다.
  이것으로 `interfaces.FinalEvaluator`/`schemas.ContentPreservationScore`가 Stage 1부터
  "Stage 2의 versioned artifact graph가 있어야 계산 가능"이라고 미뤄뒀던 계산을 **처음으로
  실제 구현**한다.
- **총비용**: `branch.total_cost_tokens`를 그대로 쓴다 (Stage 4가 이미 Stage 3의 `ToolExecutor`
  를 통해 집계한 값 — 재합산하지 않음).

`build_candidate_action_record`가 `(checkpoint, 후보 행동, 결과)` 하나를 `CandidateActionRecord`
로 묶는다 — direction 4의 benchmark 저장 단위("초기 상태, 오류 근거, 의존관계, 후보 복구 범위,
독립 평가 기준, 행동별 결과와 비용")를 코드로 표현한 것.

### 1.3 "trigger/non-trigger와 성공/실패를 포함한 표본을 사람이 평가하고, 평가자 간 불일치
처리 절차를 기록한다."

`docs/stage1_experiment_design.md` §2가 적어 둔 프로토콜을 코드로 옮겼다 (`src/core/annotation.py`):

- **표본 추출/층화**: `select_balanced_sample(candidates, min_per_stratum, seed)` — trigger/
  non-trigger × domain × difficulty 조합(`AnnotationStratum.key`)마다 최소 개수를 뽑는다.
  seed 기반으로 결정적이다.
- **blind 2인 + 조정자**: `AnnotationTask`는 verifier 이름/결과를 어디에도 담지 않는다(§2의
  "라벨링 화면에 verifier 이름/결과를 노출하지 않는다" 요구를 타입 자체로 강제). `AnnotationLabel.
  is_adjudicator`로 3번째(조정) 라벨을 1·2번째와 구분한다.
- **불일치 처리**: `resolve_annotations`가 2인 일치 → `AGREED`, 불일치 → 조정자 필요 →
  다수결 `ADJUDICATED`, 조정 후에도 다수결이 없으면 `UNRESOLVED`(`outcome=None`)로 떨어진다.
  Notion 원문의 "불일치하면 unknown으로 기록"을 `outcome=AnnotationOutcome.UNKNOWN`이 아니라
  `status=UNRESOLVED`로 구현했다 — 전자였다면 "annotator들이 독립적으로 정말 unknown이라고
  판단한 경우"와 "합의에 실패해서 그냥 unknown 처리한 경우"가 구분 불가능해진다. `UNRESOLVED`는
  `compute_verifier_precision_recall`이 분모에서 그냥 건너뛰므로 Notion이 요구하는 실제 효과
  (분모 제외)는 동일하게 달성한다.
- **불일치율 별도 보고**: `compute_agreement_report`가 전체 및 stratum별 불일치율을 낸다.
- **계층별 가중치**: `compute_stratum_weights(population_counts, sample_counts)` —
  `weight = population_count / sample_count`. 샘플링되지 않은 population stratum이 있으면
  에러를 낸다(조용히 무시하면 가중 추정치가 편향되는데 그 사실조차 드러나지 않는다).
- **verifier precision/recall**: `compute_verifier_precision_recall(predictions, gold)` —
  FAIL을 positive class로 두고, gold가 `UNRESOLVED`이거나 verifier 예측이 UNKNOWN인 항목은
  분모에서 제외한다. `VerifierPrecisionRecall.sample_size`가 실제 라벨이 들어오기 전까지는
  여전히 0이다 — 이 계산 함수 자체는 구현됐지만, 입력으로 줄 진짜 `gold`가 아직 없다(§5).

### 1.4 "template·topic·graph 구조를 고려한 train/dev/test 분할을 고정하고, held-out test를
정책 개선에 사용하지 않는다."

`src/core/dataset_split.py::build_fixed_split`가 `SplitFamilyKey(document_type, topic,
graph_structure_bucket)`로 샘플을 먼저 family 단위로 묶고, family 전체를 통째로 한 split에
배정한다 — 개별 sample 단위로 배정하면 같은 template/topic/graph 구조가 train과 test에 걸쳐
중복될 수 있다는 Notion의 지적을 구조적으로 막는다. family를 target 비율(`train_frac`/
`dev_frac`/`test_frac`, 기본 0.7/0.15/0.15)에서 가장 모자란 split에 greedy하게 배정한다 —
family 크기가 고르지 않으면 정확한 비율은 못 맞출 수 있지만(그건 "family를 절대 쪼개지 않는다"
는 하드 제약의 당연한 대가), 결과는 `seed` 고정으로 재현 가능하다.

"held-out test를 정책 개선에 사용하지 않는다"는 **코드로 강제**했다: `train_sample_ids`/
`dev_sample_ids`는 바로 쓸 수 있지만, `held_out_test_sample_ids`는 `acknowledge_test_use=True`
를 명시적으로 넘기지 않으면 `HeldOutSplitLockedError`를 던진다 — `DatasetSplit.locked` 필드는
감사용 메타데이터일 뿐이고, 실제 게이트는 이 접근자 자체가 무조건 요구하는 kwarg다. Stage 7
(정책 개선)은 이 함수를 호출할 이유가 전혀 없게, Stage 8(최종 1회 평가)만 이 플래그를 쓰도록
설계했다.

(함수 이름이 `test_sample_ids`가 아니라 `held_out_test_sample_ids`인 이유: pytest가 `test_`로
시작하는 top-level 함수를 전부 테스트로 수집하려고 시도한다 — 첫 draft에서 바로 이 문제로
`ERROR at setup of test_sample_ids: fixture 'split' not found`가 났다. 클래스 이름도
`TestSplitLockedError`가 아니라 `HeldOutSplitLockedError`로 지은 것은 같은 이유다.)

---

## 2. FaultInjector의 4가지 주입 유형

Notion의 "주입 예시" 4개를 그대로 구현했다 (`src/core/fault_injection.py`):

| 메서드 | 건드리는 artifact 수 | 방식 |
|---|---|---|
| `inject_missing_assumption` | 1 | 지정한 가정 문장을 content에서 제거 |
| `inject_altered_denominator` | 1 (TABLE kind만 허용) | 지정한 분모 문자열을 다른 값으로 교체 |
| `inject_inconsistent_symbol_meaning` | 2 (서로 다른 두 section) | 같은 기호에 서로 다른 의미를 각 section에 추가 |
| `inject_unsatisfiable_blueprint` | 1 | 서로 충족 불가능한 요구사항 ≥2개를 문서 content에 추가 |

모두 `ArtifactGraphStore.add_version`만으로 구현했다 — 실제 Producer/LLM 호출 없음(§범위 결정
1). `InjectedFault.pre_injection_refs`/`injected_refs`를 같은 길이의 병렬 리스트로 설계한 것은
`inject_inconsistent_symbol_meaning`만 2개 artifact를 건드리기 때문 — 단일 `injected_ref`
필드였다면 이 유형에 맞지 않았다. `predicted_affected_artifact_ids`는 주입 직후
`store.impact_candidates()`로 계산한 "예측값"이며, 나중에 분류/복구 단계가 실제로 관찰하는
영향과 의도적으로 분리되어 있다(`InjectedFault`의 docstring 참고) — "예측"과 "관찰"을 같은
필드에 섞으면 나중에 "이 injector가 예측한 전파"와 "실제로 일어난 전파"를 비교할 수 없어진다.

각 메서드는 대상 artifact가 한 번도 저장되지 않았거나, 지정한 텍스트/kind가 실제로 없으면
`UnsupportedInjectionTargetError`를 던진다 — 아무것도 바뀌지 않는 "주입"은 주입이 아니라는
원칙이다.

---

## 3. 왜 `direct_dependents`/`direct_dependencies`를 Stage 2 코드에 추가했는가

`classify_failure_difficulty`는 fan-out(하류)과 dependency_depth(상류) 둘 다 필요하다.
`ArtifactGraphStore`는 이미 `_current_dependents()`(private, 하류 방향)를 갖고 있었지만
공개 접근자는 `impact_candidates`(transitive closure, 1-hop이 아님)뿐이었다. 두 가지 선택이
있었다: (a) `failure_dataset.py`가 `store.edges`(public, 전체 audit trail)를 직접 읽어
"currently applicable edge" 필터링을 재구현하거나, (b) Stage 2에 이미 있는 필터링 로직을
공개 메서드로 노출한다. (a)는 `_current_dependents`의 필터 규칙(교재/dependents 쪽만 현재
버전인지 확인하고 upstream 쪽은 확인하지 않음)과 미묘하게 달라질 위험이 있다 — 두 모듈이 같은
"현재 적용 가능한 edge"를 서로 다르게 계산하게 될 수 있다. (b)를 선택했다 — Stage 5가
`is_evidence_still_valid`를 똑같은 이유로 추가한 전례를 그대로 따른 것이다. `direct_dependents`
는 기존 로직을 그대로 노출하고, `direct_dependencies`는 그 반대 방향으로 새로 작성했지만 같은
필터 규칙("이 artifact_id 쪽 endpoint가 현재 최신 버전인 edge만")을 대칭적으로 적용한다. 둘 다
순수 조회이며 `ArtifactGraphStore`의 기존 55개(Stage 2)+... 테스트에 영향이 없음을 전체
테스트 재실행으로 확인했다(§4).

---

## 4. 검증

`test/core/` 신규 파일 4개, 52개 테스트 전부 통과(`test_fault_injection.py` 10,
`test_failure_dataset.py` 17, `test_annotation.py` 16, `test_dataset_split.py` 9 —
`--collect-only`로 직접 확인한 값). 기존 `test/core/` 260개(Stage 5 3차 재검토까지) + 신규
52개 = `test/core/` 312개(`--collect-only`가 보고하는 값과 일치). `test/evaluation/` 38개
변경 없음. `test/core/`+`test/evaluation/` 합계 350개 전부 통과 — `python3 -m pytest
test/core/ test/evaluation/ -q` 결과.

전체 저장소 기준(`test/test_pipeline_dependency_ordering.py` 수집 오류 제외, Stage 1부터
이미 알려진 기존 결함): **858 passed, 57 failed** — Stage 0/1에서 이미 기록된 것과 정확히
같은 57건, Stage 6이 새로 깨뜨린 것은 없다.

---

## 5. 하지 않은 것 (Stage 6 범위 밖, 다음에 할 일)

- **실제 자연 실패 수집**: 로컬 Qwen3.6 pipeline을 실제로 돌려 자연 발생 실패를 수집하는
  일은 전혀 하지 않았다. `FailureRecord`/`classify_failure_difficulty`는 메커니즘만 갖췄다.
- **실제 human annotation**: `select_balanced_sample`/`resolve_annotations`/
  `compute_agreement_report`/`compute_stratum_weights`는 전부 합성 테스트 데이터로만
  검증했다. 실제 라벨은 사용자(또는 고용된 annotator)가 이 도구를 통해 직접 만들어야 한다 —
  AI는 독립적인 human judgment를 대신할 수 없다(§범위 결정 3, 사용자와 합의).
- **실제 train/dev/test 분할 산출물**: `build_fixed_split`는 함수만 완성됐고, 실제
  benchmark 샘플 전체에 대해 호출해 파일로 저장하는 일은 하지 않았다 — 그러려면 실제
  `graph_structure_bucket`을 샘플마다 어떻게 정의할지(현재는 `SplitFamilyKey`가 호출자가
  직접 넘기는 문자열) 먼저 benchmark 쪽(`src/dataset/`)과 연결해야 한다.
- **`VerifierPrecisionRecall`/`ContentPreservationScore`를 실제 FinalEvaluator에 연결**:
  `compute_repair_gain_score`/`compute_verifier_precision_recall`은 계산 함수로서는
  완성됐지만, `adapters.ContractOnlyFinalEvaluator`(Stage 1)가 이 값들을 실제로 채워 넣도록
  연결하는 일은 하지 않았다 — 그 연결은 실제 human gold가 존재해야 의미가 있다.

이 중 어느 것도 Notion 체크리스트 박스를 올리지 못한 이유다(§범위 결정 참고) — 이번 라운드는
"무엇을 측정/수집/분할할지"와 "그 계산이 올바른지"를 메커니즘 수준에서 완성한 것이고, 실제
데이터가 들어오면 그 위에서 바로 동작하도록 설계했다.
