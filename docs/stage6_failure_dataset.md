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

---

## 6. 사용자 재검토 후 수정 (2026-10-22)

1차 공개 후 사용자가 코드 위치·재현 시나리오와 함께 8건의 실제 결함을 지적했다 — Stage 1-5가
모두 거친 "첫 리뷰에서 다수 결함" 패턴이 Stage 6에도 똑같이 적용됐다. 전부 수정했고 각
수정마다 회귀 테스트를 추가했다.

1. **Fault injection이 기존 입력 의존관계를 끊음** (`fault_injection.py`): 4개 injector
   모두 `read_input_versions=[current.ref]`(자기 자신의 이전 버전)을 새 버전의 입력으로
   선언했다 — 이는 (a) 버전 이력(store가 이미 관리함)을 가짜 semantic dependency로
   오염시키고, (b) 자기 자신을 의존자로 만드는 self-loop을 만들며, (c) 진짜 상위 입력으로의
   edge를 조용히 끊었다(`_current_dependents`의 "to_artifact가 현재 최신 버전인 edge만
   유효" 필터 때문에, 새 버전이 실제 입력을 재선언하지 않으면 과거 edge가 즉시 무효화된다).
   재현: A→B→T에서 T에 주입하면 `direct_dependencies("T")`가 `{"T"}`로, `dependency_depth`
   가 2→1로, `fan_out`이 0→1로 왜곡됐고, 이후 A를 바꿔도 T가 stale이 되지 않았다. **수정**:
   `read_input_versions=current.ref.read_input_versions`(실제 상위 입력을 그대로 재선언)로
   4곳 전부 교체했다. 회귀 테스트:
   `test_injection_preserves_real_upstream_dependency_not_a_self_loop`.
2. **분할이 template/topic/graph의 개별 중복을 차단하지 못함** (`dataset_split.py`):
   `SplitFamilyKey.key`(세 필드의 튜플 동치)로만 그룹을 묶어서, 같은 document_type·같은
   graph_structure_bucket이지만 topic만 다른 두 샘플이 서로 다른 그룹이 되어 train/dev/test
   경계를 넘을 수 있었다. "세 축 중 어느 것도 경계를 넘지 않는다"는 요구는 세 속성 값 중
   **하나라도** 공유하면 같은 그룹으로 묶는 union-find(연결 요소)로만 만족 가능하다 — 어느
   한 속성 값을 공유하는 두 샘플이 다른 컴포넌트에 있으면 그 속성 값 자체가 두 split에
   걸치게 된다. **수정**: `_group_by_shared_attribute`(document_type/topic/
   graph_structure_bucket 각각을 join key로 쓰는 union-find)를 추가해 튜플 동치 그룹화를
   대체했다. 기존 테스트 fixture `_families()`도 모든 family가 세 속성을 전부 겹치지 않게
   고쳤다(이전엔 graph_structure_bucket을 전부 공유해서 새 구현에서는 모두 한 그룹으로
   뭉쳐버리는, 더 이상 유효하지 않은 fixture였다). 회귀 테스트:
   `test_build_fixed_split_merges_samples_sharing_any_single_attribute`,
   `test_build_fixed_split_transitively_merges_through_a_shared_attribute_chain`,
   `test_no_individual_attribute_value_ever_crosses_a_split_boundary`.
3. **미해결 artifact가 있어도 repair gain이 1.0으로 나옴** (`failure_dataset.py`):
   `compute_repair_gain_score`의 분모가 `final_outcome_by_artifact`(그 branch 자신의
   cascade가 실제로 방문한 것만)였다 — `RepairExecutionResult.left_stale`은 그 필드
   자체에 **존재하지 않는다**(schema 자신의 docstring: "Absent from this map entirely means
   never actually resolved"). 재현: A만 trigger로 주어지고 A의 복구가 B를 stale로 만들지만
   B에 등록된 constraint가 없어 B가 left_stale로 남는 경우, gain이 1/1=1.0으로 나왔다(B가
   분모에 전혀 들어가지 않음). **수정**: 분모를 branch 자신이 아니라 **모든 비교 대상
   branch에 걸쳐 공통으로 정한** `target_artifact_ids`로 바꿨다 — 새 함수
   `compute_common_target_artifact_ids(comparison, triggering_evidence)`가 (a) 원래
   trigger로 주어진 FAIL 전체와 (b) 비교 대상 **모든** branch의 실행 이력이 한 번이라도
   final_outcome을 보고했거나 left_stale로 남긴 모든 artifact_id의 합집합을 계산한다 —
   어떤 branch가 B를 전혀 발견하지 못했어도, 다른 branch가 B를 문제로 식별했다면 B는 모든
   branch의 공통 평가 대상에 포함된다. `compute_repair_gain_score`는 이제
   `target_artifact_ids`를 필수 키워드 인자로 받고, 그 집합의 모든 id를 정확히 하나의
   outcome 버킷(진짜 `RepairOutcome` 값, 또는 판정불가를 FAIL로 바꾸지 않는 새 sentinel
   `LEFT_STALE_OR_UNRESOLVED`)에 배정한다 — `sum(outcome_counts.values()) ==
   len(target_artifact_ids)`가 항상 성립. `gain`이 `None`이 되는 유일한 경우는
   `target_artifact_ids` 자체가 빈 집합일 때뿐이다(이전엔 "이 branch가 아무것도
   final_outcome에 못 남겼을 때"도 None이었는데, 이는 실제로는 유효한 0.0이어야 했다 —
   `test_repair_gain_score_unresolved_target_is_a_real_zero_not_a_fabricated_none`). 회귀
   테스트: `test_repair_gain_score_orphaned_dependent_counts_against_a_branch_that_ignored_it`
   (좁은 branch A와 넓은 branch B를 실제로 실행해 A의 gain이 0.5로, B는 1.0으로 나옴을
   확인).
4. **복구가 정상 내용을 손상시켜도 보존율이 1.0으로 나옴** (`failure_dataset.py`):
   `_compute_damage`가 "이 branch의 cascade가 건드린 적 있는 artifact"를 평가에서 제외했다
   — 넓은 복구 범위에 휩쓸려 들어간 정상 artifact가 실제로 FAIL이 돼도 "건드렸다"는 이유로
   제외됐다. 또한 "건드리지 않은" 쪽 판정 기준이 단순 `before_content != after_content`
   문자열 비교였다 — 성공적인 재생성/reaffirm으로 내용이 바뀌는 것(실제 오류 아님)과 진짜
   새 오류를 구분하지 못했다. **수정**: 평가 대상을 "이 branch의 cascade가 방문했는지"가
   아니라 "복구 **전에** 정한 `target_artifact_ids`에 속하는지"로 바꿔 — 즉 `before_checkpoint`
   의 모든 artifact_id에서 `target_artifact_ids`를 뺀 집합(복구가 건드릴 의도가 전혀 없던
   "정상으로 남아야 할" 집합)을 평가 대상으로 쓴다. "손상 여부"도 문자열 비교 대신
   `branch.final_checkpoint`를 복원한 store에서 `is_stale()` 또는 live FAIL evidence가
   있는지로 판정한다(`ArtifactGraphStore.from_checkpoint`로 읽기 전용 재구성). 회귀 테스트:
   `test_damage_ignores_a_harmless_content_change_with_no_real_error`(내용은 바뀌지만
   PASS → 손상 아님), `test_damage_flags_a_confirmed_new_failure_even_with_unchanged_content`
   (내용은 그대로지만 FAIL → 손상), `test_damage_considers_every_non_target_artifact_even_ones_the_branch_touched`
   (넓은 복구에 휩쓸려 FAIL이 된 비-target artifact가 실제로 damage로 잡힘).
5. **복수의 초기 행동이 첫 번째 행동 하나로 기록됨** (`schemas.py`/`failure_dataset.py`):
   `CandidateActionRecord.action`(단수)이 `branch.rounds[0].executions[0].action`만
   저장했다 — `FixedGraphAwareRepairPolicy.decide()`처럼 한 번의 호출이 독립적인 실패마다
   하나씩, 여러 개의 `RepairAction`을 반환하는 게 정상인데, `outcome`(gain/손상/비용)은
   그 라운드가 실제로 실행한 **모든** action의 결합 효과를 반영하면서 `action`은 그중
   하나만 가리켰다. `branching_runner.py`가 이미 겪은 "`List[X]`는 복수 계약이다, 0번째만
   쓰는 건 암묵적 단일성 가정이다" 교훈이 기록 계층에서 그대로 재발한 것. **수정**:
   `CandidateActionRecord.action: RepairAction` → `actions: List[RepairAction] =
   Field(min_length=1)`로 바꾸고, `build_candidate_action_record`가
   `branch.rounds[0].executions`의 모든 `.action`을 모은다. 회귀 테스트:
   `test_build_candidate_action_record_keeps_every_initiating_action_not_just_the_first`
   (a·b가 동시에 실패해 한 라운드에 2개 action이 실제로 실행되는 것을 직접 확인한 뒤,
   `record.actions`에 둘 다 남는지 검증).
6. **Verifier precision/recall 집계의 두 문제** (`annotation.py`): (a) 사람이 합의한
   `AnnotationOutcome.UNKNOWN`(annotator들이 독립적으로 "판정 불가"에 합의한 경우)이
   `gold.outcome is None`(= UNRESOLVED) 필터만으로는 걸러지지 않아 "actual_fail이
   아님"으로 암묵 처리됐다 — gold=UNKNOWN + prediction=FAIL 한 건이 precision을 0.0으로
   끌어내렸다. **수정**: 공유 필터 `_included_gold_items`를 추가해 gold가 PASS/FAIL이 아닌
   모든 경우(UNRESOLVED 포함 UNKNOWN 포함)를 명시적으로 제외한다. (b) `compute_stratum_weights`
   가 가중치 테이블을 만들어도 어떤 지표 계산도 그 값을 실제로 쓰지 않았다 — 층화 표본을
   오버샘플한 희귀 stratum이 모집단 비율로는 다른 결과를 내야 하는데도 단순 비가중 recall과
   똑같은 값이 나왔다. **수정**: 새 함수 `compute_weighted_verifier_precision_recall`
   (Horvitz-Thompson 방식 가중 추정)을 추가했다 — 기존 `compute_verifier_precision_recall`
   (표본 지표, 비가중)은 그대로 두고 완전히 분리된 별도 함수로 둬서 "표본 지표"와 "가중
   모집단 추정치"가 하나의 반환값에 섞이지 않는다. 회귀 테스트:
   `test_compute_verifier_precision_recall_excludes_a_genuine_agreed_unknown_gold`,
   `test_compute_weighted_verifier_precision_recall_differs_from_the_unweighted_sample_metric`
   (비가중 recall=0.5, 가중 recall=0.1로 정확히 재현).
7. **독립 평가자 조건을 레코드 개수로만 검사함** (`annotation.py`): `_resolve_one`이
   `len(primary) >= 2`만 확인하고 "서로 다른 사람인지"는 전혀 보지 않아, 같은
   `annotator_id`의 동일 라벨 2개도 AGREED로 통과했다 — 독립성은 "라벨이 몇 개인가"가
   아니라 "누가 라벨링했는가"의 조건이다. 조정자가 원래 두 primary 중 한 명과 같은
   `annotator_id`인 경우(자기 불일치를 자기가 조정)도 막히지 않았다. **수정**: 한 task의
   모든 라벨(primary+adjudicator)에서 `annotator_id`가 중복되면(둘 중 어느 역할이든) 거부
   — 이 하나의 규칙이 "같은 primary가 두 번" 경우와 "adjudicator가 자기 자신" 경우를 동시에
   잡는다. 추가로 task_id 불일치(그룹화 키와 라벨 자신의 task_id가 다름)도 거부하도록
   했다. **추가로 지적된 범위 공백**: `AnnotationStratum`에 성공/실패·stage 축이 없었다 —
   Notion 6단계 체크리스트 원문이 "trigger/non-trigger와 **성공/실패**를 포함한 표본"을
   명시적으로 요구하는데, 기존 schema는 `trigger`만 있었다. **수정**: `AnnotationStratum`에
   `stage: Optional[str]`/`success: Optional[bool]`를 추가하고 `.key`에도 반영했다 — 같은
   non-trigger라도 실패한 trajectory(미스)와 성공한 trajectory는 서로 다른 stratum이
   된다. 회귀 테스트: `test_resolve_annotations_raises_on_duplicate_annotator_for_the_same_task`,
   `test_resolve_annotations_raises_when_adjudicator_is_also_a_primary_labeler`,
   `test_resolve_annotations_raises_on_task_id_mismatch`,
   `test_select_balanced_sample_treats_trigger_and_success_as_independent_axes`.
8. **'관계 누락' 지표가 현재 누락이 아니라 과거 INFERRED edge 수였음** (`schemas.py`/
   `failure_dataset.py`): `missing_relations_count`가 `store.edges`(전체 이력)에서
   INFERRED edge를 세어, 그 관계가 이후 재생성으로 현재 버전에 더 이상 적용되지 않아도
   값이 줄지 않았다(과거에 한 번이라도 발견됐으면 영원히 누적). "지금 몇 개가 누락됐는가"로
   읽히는 필드명이면서 실제로는 "지금까지 몇 번 발견됐는가"를 세고 있었다. **수정**: 필드를
   `historical_inferred_relations_count`로 이름 바꿔 누적/이력 값임을 명시하고, 완전히
   별도의 새 필드 `current_missing_relations_vs_reference`(기본값 `None`)를 추가했다 — 이
   값은 호출자가 **알고 있는 기준**(`reference_dependency_ids`)과 현재
   `direct_dependencies()`를 비교해서만 채워진다(일반적인 자동 탐지가 아니라, Stage 2
   자신의 "자동 reference scanner는 만들지 않는다"는 비목표를 그대로 따름). 새 함수
   `count_missing_relations_against_reference`가 이 비교를 구현한다.
   `build_failure_record`는 `injected_fault`가 주어지면 그 fault의 주입 전
   `read_input_versions`(= 그 artifact가 실제로 무엇에 의존했는지에 대한 알려진 "주입
   기록")을 기준으로 자동으로 채운다 — 사용자가 제안한 두 방법("알려진 기준 그래프·주입
   기록") 중 실제로 가진 데이터인 "주입 기록" 쪽을 구현했다. 회귀 테스트:
   `test_historical_inferred_relations_count_stays_set_after_the_relation_stops_applying`,
   `test_current_missing_relations_vs_reference_detects_a_dropped_dependency`,
   `test_build_failure_record_injected_derives_reference_from_the_injected_fault`.

**검증**: `test/core/`에 22개 테스트 추가(`test_fault_injection.py` 10→11,
`test_failure_dataset.py` 17→26, `test_annotation.py` 16→25, `test_dataset_split.py`
9→12). `test/core/` 312→334개, `test/evaluation/` 38개 변경 없음. 전체 저장소 기준 880
passed / 57 failed(Stage 1부터 이미 알려진 것과 정확히 같은 수, 이번 라운드가 새로 깨뜨린
것 없음).

**아직 하지 않은 것(§5와 동일, 이번 라운드로 바뀌지 않음)**: 실제 자연/주입 실패 수집,
실제 human annotation 수집, 실제 benchmark에 대한 분할 산출물, `FinalEvaluator` 연결.
이번 라운드는 메커니즘 자체의 정확성만 다뤘다 — Notion 체크리스트 박스는 여전히 올리지
않았다.

## 7. 사용자 2차 재검토 후 수정 (2026.10.22)

6절의 8개 수정을 반영한 뒤, 사용자가 같은 코드를 다시 검토해 4개의 추가 버그를 지적했다.
이번 라운드는 모두 6절에서 고친 메커니즘들 사이의 **상호작용**에서 생긴 문제였다 — 개별
수정은 각자 맞았지만, 두 수정을 합치면 한쪽이 다른 쪽이 막았던 구멍을 다시 열었다.

1. **공통 target 생성과 손상 계산을 연결하면 정상 artifact의 손상이 다시 빠짐**
   (`failure_dataset.py`): 6절의 버그 3 수정(`compute_common_target_artifact_ids`)과
   버그 4 수정(`_compute_damage`가 `target_artifact_ids`의 여집합만 평가)을 그대로
   합치면, "넓은 복구가 어떤 artifact를 건드렸다"는 사실 자체가 그 artifact를
   `target_artifact_ids`(공통 집합의 구성요소 (b))에 넣어버리고, 그 순간 `_compute_damage`가
   그 artifact를 평가 대상에서 빼버린다. A는 실패, B는 정상인 상태에서 좁은 복구는 B를
   보존하고 넓은 복구는 B를 망가뜨렸는데, B가 공통 target에 포함되면서 **두 branch 모두**
   B에 대한 손상 평가가 왜곡됐다(좁은 쪽은 "B를 안 건드렸다"는 이유로 gain만 깎이고, 넓은
   쪽은 preserved_fraction=1.0으로 손상이 안 보였다). **수정**: 손상의 평가 대상(="복구
   전 정상이었던 집합")을 `target_artifact_ids`가 아니라 완전히 독립된 새 함수
   `_originally_failing_artifact_ids(triggering_evidence)`의 여집합으로 고정한다 — 이
   함수는 `compute_common_target_artifact_ids`의 구성요소 (a)만 떼어낸 것으로, 어떤
   branch가 나중에 무엇을 건드렸는지와 무관하게 "애초에 실제로 실패했던 것"만 가리킨다.
   `compute_repair_gain_score`/`build_candidate_action_record`에 새 필수 키워드
   `triggering_evidence`가 추가됐다(gain 쪽 `target_artifact_ids`는 그대로 둔다 — 이건
   의도된 더 넓은 집합이다). 회귀 테스트:
   `test_damage_of_a_branch_is_not_hidden_by_a_siblings_collateral_sweep`(A/B 완전
   무관, 넓은 branch의 전용 fake policy `_AlsoRegenerate`가 B를 직접 건드려 깨뜸),
   기존 `test_repair_gain_score_orphaned_dependent_counts_against_a_branch_that_ignored_it`
   에도 동일 원리의 damage 단언을 추가.
2. **한 번 RESOLVED였다가 다시 stale이 된 artifact를 여전히 해결된 것으로 셈**
   (`failure_dataset.py`): `_merge_final_outcomes`는 라운드 순서대로 last-write-wins만
   적용한다 — B가 라운드 1에서 RESOLVED로 기록된 뒤, 라운드 2가 (B와 무관한 이유로) A를
   다시 바꿔 B를 stale로 되돌려도, 라운드 2는 B에 대해 아무것도 기록하지 않으므로 병합
   결과는 여전히 "B: RESOLVED"로 남는다. 실제로는 `final_checkpoint`에서 B가 stale인데도
   gain이 1.0으로 나왔다. **수정**: `compute_repair_gain_score`가 각 target의 RESOLVED
   판정을 그대로 믿지 않고, `branch.final_checkpoint`를 복원한 store에 대해
   `_is_broken_in_final_state`(= `is_stale` 또는 live FAIL evidence, `_compute_damage`와
   공유하는 동일한 판정)로 다시 확인한다 — 모순되면 FAIL로 바꾸는 게 아니라
   `LEFT_STALE_OR_UNRESOLVED` sentinel로 내려간다(사용자 지적대로 "판정불가를 FAIL로
   바꾸라는 뜻은 아님"과 동일한 원칙). 회귀 테스트:
   `test_repair_gain_score_does_not_trust_a_resolved_verdict_the_final_state_has_since_invalidated`
   (라운드 1이 B를 RESOLVED, 라운드 2가 A를 다시 바꿔 B를 stale로 되돌리는 것을
   `_merge_final_outcomes(branch)["b"] == RESOLVED`로 전제를 먼저 확인한 뒤 gain=0.0,
   `outcome_counts == {LEFT_STALE_OR_UNRESOLVED: 1}`을 단언).
3. **Injector가 INFERRED 의존관계를 보존하지 않음** (`fault_injection.py`/
   `schemas.py`/`failure_dataset.py`): 6절 버그 1의 수정은
   `read_input_versions=current.ref.read_input_versions`만 재선언한다 —
   `read_input_versions`는 EXPLICIT 입력만 담을 수 있고, `add_inferred_edge`로 별도
   등록된 INFERRED edge는 애초에 거기 들어가지 않는다. A→T를 INFERRED edge로 등록한 뒤
   T에 주입하면, 주입이 쓰는 새 버전은 EXPLICIT 입력만 재선언하므로 INFERRED edge가
   가리키던 "T의 이전 버전"이 더 이상 현재 버전이 아니게 되는 순간 그 관계가 조용히
   사라진다(`direct_dependencies`가 빈 집합이 됨) — 6절 버그 1과 정확히 같은 매커니즘의,
   한 단계 더 깊은 재발이다. 기준선도 마찬가지로 왜곡됐다: `build_failure_record`가
   `pre_injection_refs[i].read_input_versions`에서만 기준을 가져오므로
   `current_missing_relations_vs_reference`가 항상 0으로 나왔다(기준 집합 자체가
   INFERRED 관계를 담을 수 없는 필드에서 왔으므로). **수정**: `InjectedFault`에 새 필드
   `pre_injection_dependency_ids_by_artifact: Dict[str, List[str]]`를 추가해
   `store.direct_dependencies()`(EXPLICIT+INFERRED 전체)로 주입 직전 상태를 캡처한다.
   `fault_injection.py`의 새 헬퍼 `_carry_forward_inferred_edges`가 이 중
   `read_input_versions`로 커버되지 않는 id만 걸러 원래 INFERRED edge를 찾아 새 버전에
   대해 `add_inferred_edge`로 재등록한다. `build_failure_record`는 이제
   `pre_injection_refs[...].read_input_versions` 대신
   `InjectedFault.pre_injection_dependency_ids_by_artifact`를 기준으로 쓴다. 회귀
   테스트: `test_injection_preserves_a_pre_existing_inferred_dependency`(주입 후에도
   `direct_dependencies`가 그대로임을 확인), `test_build_failure_record_reference_baseline_includes_inferred_dependencies`
   (기준 집합에 INFERRED 의존성이 실제로 포함됐음을, 이후 그 의존성을 완전히 떨어뜨려
   `current_missing_relations_vs_reference`가 0→1로 바뀌는 것으로 증명).
4. **같은 데이터·같은 seed라도 입력 순서가 바뀌면 split이 달라짐** (`dataset_split.py`):
   `_group_by_shared_attribute`는 union-find의 내부 root(어느 샘플이 "대표"가 되는지)를
   그대로 반환 dict의 key로 썼는데, 이 root는 처리 순서(= 입력 dict의 순회 순서)에 따라
   달라진다. `build_fixed_split`이 이 key들을 `sorted()`한 뒤 `seed`로 shuffle하므로,
   실제 그룹 구성(어떤 샘플들이 같은 그룹인지)은 입력 순서와 무관하게 동일해도, 그
   root-label이 바뀌면 `sorted()` 결과 순서가 바뀌고 따라서 동일한 `seed`로 shuffle한
   결과도 완전히 달라진다. **수정**: 그룹의 key를 union-find root가 아니라
   `min(members)`로 다시 매긴다 — 이 값은 그룹의 최종 구성원 집합만으로 결정되는 순수
   함수이므로 처리 순서를 전혀 반영하지 않는다. 회귀 테스트:
   `test_build_fixed_split_is_independent_of_input_iteration_order`(동일한
   families dict를 순서만 뒤집어 넣고 결과가 같은지 확인 — 처음에는 숫자 순서대로 증가하는
   sample_id로 작성했는데 우연히 재정렬 후에도 상대 순서가 보존돼 버그를 못 잡았고, 뒤섞인
   sample_id로 바꾸자 수정 전 코드에서 실제로 8~12개 샘플이 이동하는 것을 확인한 뒤에야
   이 테스트를 확정했다).

**검증**: `test/core/`에 5개 테스트 추가(`test_fault_injection.py` 11→12,
`test_failure_dataset.py` 26→29, `test_dataset_split.py` 12→13, `test_annotation.py`
25개 변경 없음). `test/core/` 334→339개. 전체 저장소 기준 885 passed / 57 failed(이번
라운드도 기존에 알려진 것과 정확히 같은 수 — `test/test_pipeline_dependency_ordering.py`의
collection error는 `scripts/run_full_pipeline.py`에 없는 `_dependency_ordered`를
import하려다 생기는 것으로, 이번 Stage 6 변경과 무관한 훨씬 오래된 사전 존재 이슈임을
`git log`로 확인했다).

**아직 하지 않은 것**: §5/§6과 동일, 이번 라운드로 바뀌지 않음.

## 8. 사용자 3차 재검토 후 수정 (2026.10.22)

7절의 4개 수정을 반영한 뒤, 사용자가 다시 검토해 3개의 추가 버그를 지적함.

1. **[P1] 정상 내용을 그대로 보존한 좁은 복구의 gain이 여전히 낮게 계산됨**
   (`failure_dataset.py`): A는 실패, B는 이미 유효한 PASS인 상태에서, 좁은 분기는 A만
   복구하고 B는 건드리지 않았는데, 넓은 분기가 A·B를 모두 재생성(둘 다 정상으로 끝남)하면
   그 사실만으로 B가 공통 `target_artifact_ids`(`compute_common_target_artifact_ids`의
   구성요소 (b))에 들어간다. 좁은 분기는 B에 대한 실행 이력이 전혀 없으므로 기존 로직은
   무조건 `LEFT_STALE_OR_UNRESOLVED`로 집계해 gain=0.5가 됐다 — 더 선택적이고 효율적인
   복구가 오히려 더 나쁜 점수를 받는 구조. **수정**: 새 sentinel
   `ALREADY_HEALTHY_UNTOUCHED`를 추가 — target이 (a) 원래 트리거가 아니고, (b) 이
   branch 실행 이력에 전혀 없고, (c) 이 branch의 최종 상태에서 실제로 건강함이 확인되는
   경우에만 적용되며, gain의 분자에 RESOLVED와 함께 합산된다. 진짜 미해결(예: 실제로
   cascade되어 stale이 됐지만 재방문되지 않은 dependent)은 여전히
   `LEFT_STALE_OR_UNRESOLVED`로 남아 이 sentinel로 새지 않음을 별도 테스트로 확인했다.
   회귀 테스트: `test_repair_gain_score_already_healthy_untouched_counts_toward_gain`,
   `test_repair_gain_score_already_healthy_untouched_does_not_apply_to_a_genuinely_broken_target`.
2. **[P2] 복구 전부터 stale이었던 artifact를 '새로 발생한 손상'으로 집계함**
   (`failure_dataset.py`): 기존 `healthy_before_artifact_ids` 계산은 "원래 트리거가
   아닌 모든 artifact = 복구 전 정상"이라고 가정했다 — 이는 그 artifact가 이번 복구와
   무관한 이유로 이미 stale/FAIL이었을 가능성을 놓친다. B가 이미 stale인 checkpoint에서
   A만 복구했더니(B는 전혀 건드리지 않았는데도) B의 상태가 그대로인데도
   `newly_broken_artifacts=["b"]`가 나왔다. **수정**: 새 함수
   `_pre_repair_healthy_artifact_ids(before, triggering_evidence)`가
   `before_checkpoint`를 복원한 store의 실제 ground truth(`_is_broken_in_final_state`)로
   한 번 더 걸러낸다 — 원래 트리거가 아니면서 **동시에** 복구 전 실제로 건강했던
   artifact만 손상 평가 대상에 들어간다. 정상→손상 전환만 손상으로 집계되고, 복구 전부터
   깨져 있던 상태가 그대로 유지되는 경우는 평가 대상에서 완전히 제외된다. 회귀 테스트:
   `test_damage_excludes_an_artifact_that_was_already_broken_before_the_repair`.
3. **[P2] 동일 artifact의 다른 버전을 참조하는 INFERRED edge가 주입 과정에서 사라짐**
   (`fault_injection.py`): 7절에서 추가한 `_carry_forward_inferred_edges`는
   `direct_dependencies()`(artifact ID만 반환)의 결과를 새 EXPLICIT
   `read_input_versions`의 ID 집합과 **ID만으로** 차집합해 "INFERRED로 추정되는 것"을
   걸러냈다. T가 A@2를 EXPLICIT으로, A@1을 INFERRED로 참조하는 경우 두 관계가 모두 ID
   "a"로 collapse되어 차집합이 빈 집합이 되고, 실제 INFERRED A@1→T edge가 조용히
   누락됐다 — 그 결과 그 edge가 유일한 원인이었던 T의 retroactive staleness도 함께
   사라졌다(의존관계를 복구한 적이 없는데 `add_version`의 "입력이 모두 정상이면
   staleness 해제" 로직에 의해 정상화됨). **수정**: `_carry_forward_inferred_edges`를
   재작성해 `store.edges`에서 `to_artifact`가 정확히 `(artifact_id, version)` 쌍으로
   일치하는 INFERRED edge만 직접 조회하고, EXPLICIT과의 ID 기반 차집합 로직을 완전히
   제거했다 — 같은 ID의 다른 버전은 절대 같은 사실이 아니므로 "이미 커버됐는지" 비교 자체가
   불필요하다. 회귀 테스트:
   `test_injection_preserves_an_inferred_edge_to_a_different_version_than_the_explicit_one`.

**검증**: `test/core/`에 4개 테스트 추가(`test_fault_injection.py` 12→13,
`test_failure_dataset.py` 29→32). `test/core/` 339→343개. 전체 저장소 기준 889 passed /
57 failed(기존과 동일). 수정 전 코드에 대해 각 회귀 테스트를 직접 실행(스크립트로 구 로직을
재현)해 실제로 실패함을 확인한 뒤에야 "수정됨"으로 기록했다.

**아직 하지 않은 것**: §5/§6/§7과 동일, 이번 라운드로 바뀌지 않음.
