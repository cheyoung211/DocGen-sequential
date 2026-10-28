# AV Stage 0: 범위 정리와 공통 계약 확정 (P1–P4)

이 문서는 Notion "SNU / NCIA / Personal Research / Multimodal Agents / **Revision (Additional
verifiers)**" (작성 기준 2026-10-28) 계획의 **0단계** (P1–P4, 전체 범위와 공통 설계) 산출물이다. 이 revision은
별도로 진행 중인 **Revision (Graph artifacts)** (브랜치 `graph_artifact`, 0–6단계, `docs/stage1_*.md` ~
`docs/stage6_*.md`)의 artifact graph·controller 기반 위에 Markdown 강조 규제(Fix A)·계산·수식 verifier(Fix
B)·cross-section dependency 관리(Fix C) 세 가지를 추가한다. 파일명은 기존 `stageN_*.md`와의 혼동을 피하기
위해 `av_` 접두사를 쓴다.

> **범위**: 이 문서 + `src/evaluation/schemas.py`의 `AdditionalVerifierRunMetadata` 추가 +
> `scripts/run_full_pipeline.py::_build_run_result`의 배선이 전체다. Fix A/B/C의 실제 검사 로직은 전혀
> 구현되지 않았다 (1~3단계의 일). `markdown_emphasis_enabled`/`math_verifier_enabled`/
> `dependency_verifier_enabled`는 지금 모두 `False`이며, 이는 placeholder가 아니라 **현재 실제 상태**다.
> 기존 파이프라인 동작(생성·검증·복구·컴파일)은 전혀 바뀌지 않았다 — 추가된 것은 `RunResult`에 실려 나가는
> 메타데이터 한 필드뿐이다.

코드 기준: `graph_artifact`에서 분기한 브랜치 `additional_verifier`,
commit [`278e748`](https://github.com/cheyoung211/DocGen-sequential/commit/278e7488a5ecc6c91ffc3fb4e7ce091e9f5df63b).
`git rev-parse HEAD`로 이 브랜치의 현재 HEAD가 Notion 계획 문서가 명시한 commit과 **정확히 일치**함을
확인했다 — 계획 작성 이후 이 브랜치에 다른 커밋이 올라간 적이 없다.

---

## P1. 범위·버전 기록

- **기존 결과 보존**: `outputs/`, `outputs_b4_workshop/`, Notion "Gold Labeling" 페이지는 이 단계에서 전혀
  건드리지 않았다. 변경은 `src/evaluation/schemas.py`, `scripts/run_full_pipeline.py`, 테스트, 그리고 이
  문서뿐이다.
- **버전 기록 장치**: `VerificationConfigSnapshot` (`src/evaluation/schemas.py:215-233`)을 재사용하지
  않기로 결정했다 — 이 클래스는 자신의 docstring에서 이미 "toggle 시스템이 없어 항상 고정된 기본값만
  기록한다"고 밝히고 있고, `scripts/run_full_pipeline.py::_build_run_result`도 실제로 이 필드를 한 번도
  채우지 않는다 (항상 default). 이미 "깨진 채 방치"로 알려진 클래스를 확장하면 같은 문제(측정값이 실제
  상태에 연결되지 않음, [[docgen_stage_review_rigor]]의 패턴 #15)를 반복하게 된다.
- 대신 새 `AdditionalVerifierRunMetadata` (`src/evaluation/schemas.py`, `VerificationConfigSnapshot` 바로
  다음)를 추가하고 `RunResult.additional_verifier_metadata`로 연결했다. 필드:
  - `markdown_emphasis_enabled` / `math_verifier_enabled` / `dependency_verifier_enabled`: 세 수정이
    실제로 켜졌는지. 지금은 셋 다 정직하게 `False` — 어떤 verifier도 아직 존재하지 않기 때문이다.
  - `plan_doc_version`: 이 코드가 어느 시점의 Notion 계획을 기준으로 작성됐는지 (`"2026-10-28"`).
  - `code_commit`: 실행 시점의 실제 git HEAD. `scripts/run_full_pipeline.py::_current_code_commit()`이
    `git rev-parse HEAD`를 실행해 채운다 (`@lru_cache`로 프로세스당 1회만 — batch 실행에서 샘플마다 git을
    shell-out하지 않도록). git이 없거나 실패해도 파이프라인 실행 자체는 멈추지 않도록 `try/except`로 감싸
    `None`으로 degrade한다 (`cost_tracking.py`의 `GpuUtilizationSampler`가 `nvidia-smi` 실패를 다루는
    방식과 동일한 원칙).
  - `rubric_version`: Gold Labeling rubric의 버전. 지금은 placeholder (`None`) — G1(5단계)에서 실제
    버전 체계가 정해지기 전까지는 의미 있는 값을 넣을 수 없다.
- **구현 착수 시 재대조 결과**: 계획 문서가 명시한 6개 "주요 수정 위치" 파일 (`text_agent.py`,
  `constraint_checks.py`, `artifact_graph.py`, `controller.py`, `schemas.py`, `run_full_pipeline.py`) 중
  이번 0단계에서 실제로 건드린 것은 `src/evaluation/schemas.py` (계획에는 없던 파일이지만 P1의 버전
  기록을 위해 필요)와 `scripts/run_full_pipeline.py`(배선) 뿐이다. 나머지는 1~4단계에서 다룬다.

## P2. 세 검사군의 Evidence 연결

`Evidence`/`Constraint`/`CheckOutcome`/`UnknownReason` (`src/core/schemas.py:144-259`)는 이미 범용적으로
설계되어 있어, A/B/C 세 검사군 모두 **새 스키마 없이 그대로 재사용**한다:

- `constraint_id`: 검사 criterion ID (예: 향후 `"markdown_emphasis"`, `"cross_section_must_reuse"`).
- `checked_artifact` + `input_versions`/`missing_inputs`: 대상과 그 검사가 실제로 의존한 모든 source
  버전 — `ArtifactGraphStore.live_evidence()`가 이미 이 필드로 무효화를 처리한다.
- `outcome` + `unknown_reason`: PASS/FAIL/UNKNOWN과 판정 불가 사유.
- `detail`: 진단 정보·복구 대상 위치.

**새 `ConstraintKind`는 0단계에서 추가하지 않는다.** `TextAgentValidatorAdapter`
(`src/core/adapters.py:327-420`)가 이미 보여주는 패턴 — `ConstraintKind` enum이 아니라 `constraint_id`
문자열로 라우팅하는 방식 — 이 A(markdown_emphasis)와 C(dependency relation)의 MVP에 그대로 들어맞는다. B는
기존 `ConstraintKind.RECOMPUTE`를 연산 종류만 확장하는 것으로 시작한다(계획 B2 문구와 일치: "기존 합계
재계산을 필요한 연산으로 확장"). 각 수정이 실제로 새 `ConstraintKind`가 필요한지는 해당 단계(1/3) 설계
문서에서 결정한다 — 지금 미리 정하지 않는다.

**"원본 초안과 복구 전 checkpoint"는 이미 실재한다**, 새로 만들 필요 없음:
- 원본 초안: `TextAgent._render_draft_with_anchors`가 `{node_id}.draft.tex`를 쓴다
  (`src/agents/text_agent.py:1147-1164`). Gold Labeling 페이지의 모든 "복구로 생긴 손상 확인" 판정이
  이미 이 draft.tex와 최종본을 비교하는 방식으로 이루어져 있다 (예: sample_002, sample_004, sample_005,
  sample_007).
- 복구 전 checkpoint: `ArtifactGraphStore.save_checkpoint()` (Stage 2, `graph_artifact` 브랜치)가 독립된
  branch 복사본을 제공한다.

## P3. 미해결 상태의 종료 시 보고

새 메커니즘 불필요 — 이미 구현되어 있다: `FinishCall.unresolved_artifact_ids`, 여러 호출부에 흩어져
있던 "무엇이 미해결인가"를 한 곳으로 모은 `_aggregate_unresolved` (Stage 5 round 3,
[[docgen_stage_review_rigor]] 패턴 #53), `RepairOutcome.INCONCLUSIVE` (no-FAIL이 PASS가 아님을 명시하는
3값 분류, 패턴 #31/#60/#69). **향후 Fix A/B/C는 이 경로를 통해서만 미해결을 보고해야 한다** — 각자 독립된
"미해결 목록"을 새로 발명하지 않는다. 예산 소진으로 종료된 경우를 검증 완료로 표시하지 않는다는 규칙도
`RepairOutcome.INCONCLUSIVE`가 이미 표현한다.

## P4. 비용·조건 분리

새 메커니즘 불필요: `CostTrackingProducer` (Stage 3)가 모든 `Producer.produce()` 호출의 비용을
`CostRecord`로 집계하고, `RepairAction.use_diagnostic_feedback` + `action_type`이 trigger/feedback/scope
세 축을 이미 분리해 기록한다 (Stage 1, `docs/stage1_experiment_design.md` §0). **향후 Fix A/B/C의 검사·
변환·실패한 재시도·재계산 비용은 이 기존 경로로만 집계한다** — 측정만 하고 budget enforcement에 연결되지
않는 값을 새로 만들지 않는다 (Stage 3 review가 이미 발견한 패턴, [[docgen_stage_review_rigor]] #15).

## 발견된 공백 — 0단계에서 풀지 않고 이후 단계로 이연

1. **production `ConstraintProvider`가 코드베이스 어디에도 없다.**
   `ConstraintProvider = Callable[[ArtifactRef], List[Constraint]]`
   (`src/core/repair_operations.py:157`)는 `controller.py`/`branching_runner.py`/`tool_executor.py`/
   `failure_dataset.py`에 널리 배선되어 있지만, grep 결과 `src/`에서 실제 dataset 계약
   (`HardConstraint`/`CrossSectionDependency`/`TerminologyConstraint`/`NotationConstraint`,
   `src/dataset/schemas.py`)으로부터 런타임 `Constraint`를 만드는 코드는 **어디에도 없다** — Stage 1-6
   전체가 테스트 fake provider로만 검증되었다는 뜻이다. 계획 §4의 **I1**("등록만 된 검사가 실행되지 않는
   경우를 확인한다")이 바로 이 공백을 가리킨다. 4단계 작업이며, 지금 풀지 않는다.
2. **Markdown 강조 구문 탐지가 전혀 없다.** `latex_assembler.py`에는 Markdown **표** 처리
   (`is_markdown_table`, `_markdown_table_to_tabularx`, `latex_assembler.py:356-442`)만 있고, `**bold**`
   류 강조 구문을 다루는 코드는 없다 — Fix A(1단계)는 기존 코드의 확장이 아니라 완전히 새 작업이다.
3. Fix C의 **계약(요구사항) 쪽 스키마는 이미 존재한다**: `MUST_REUSE`/`MUST_REFERENCE`/
   `MUST_BE_CONSISTENT_WITH`/`MUST_PRECEDE` (`src/dataset/schemas.py:124-127`), `reference_required`/
   `depends_on` (283-284), `cross_section_dependencies` (396). 2~3단계가 만들어야 하는 것은 **검증기**와
   requirement-ID → 생성된 artifact-ID 매핑뿐이다.

## Gold Labeling → 수정 매핑 (회귀 테스트용 실제 사례)

| 샘플 | 관찰된 결함 | 해당 수정 |
|---|---|---|
| sample_001, 002, 003, 004, 006, 007, 009 | `**bold**`/`__bold__`가 PDF에 그대로 노출 (총 7개 샘플, 가장 빈번) | Fix A |
| sample_001 | `0.998^999` 계산 오류 (재계산으로 확정) | Fix B (수치 재계산) |
| sample_005 | 굽힘응력 10배, 처짐 25배 과다 계산 (재계산으로 확정) | Fix B (수치 재계산) |
| sample_000 | proof_sketch에서 `T_n(x)`를 `f(x)`로 암묵적 대체 (논리 비약) | Fix B (후속 범위, B6 — 논리 검토) |
| sample_003, 007, 008 | 동일 대상에 다른 표기 혼용 ($p(D\|\theta)$ vs $\mathcal{L}(\theta)$; `u` vs `U`; `Δ%` vs `RG`) | Fix B (공유 값·표기 일관성) |
| sample_002, 006 | `worked_examples`/`case_or_application`이 요구된 정의를 실제로 재사용하지 않음 (`must_reuse` 미충족) | Fix C |
| sample_005 | summary가 analysis 섹션을 `\Cref`로 인용하지 않음 (`must_reference` 미충족) | Fix C |
| sample_008 | summary가 인용한 섹션에 없는 내용을 서술 (내용 불일치) | Fix C |
| 다수 샘플의 `contract_result` 과소/과대 채점, `blocks.json` kind 불일치 | 자동 채점기 자체의 버그 | **범위 밖** — Notion 계획 §5가 명시: 이번 세 수정이 평가기 전체를 고쳤다고 간주하지 않음 |

## 완료 기준 확인

- [x] 어떤 산출물의 어떤 버전을 무엇으로 검사했고, 왜 수정했으며, 무엇이 미해결인지를 한 실행 로그에서
  추적할 수 있다 — 단, 추적할 "검사"가 아직 없으므로 (A/B/C 미구현) 지금은 이 경로가 **연결되어 있다는
  것**만 확인했다. 실제 추적 가능성은 Fix A가 생기는 1단계에서 처음 검증된다.
- [x] 기존 결과 보존 확인 (outputs/, Gold Labeling 불변).
- [x] 검사 범위와 로그 계약 확정 (P2-P4가 가리키는 기존 메커니즘 목록 + 신규 `AdditionalVerifierRunMetadata`).
