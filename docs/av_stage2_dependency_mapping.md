# Stage 2 (Revision "Additional verifiers") -- C1: 요구사항-artifact 매핑

Notion "Revision (Additional verifiers)" (작성 2026-10-28, 코드 기준
[`graph_artifact`@`278e748`](https://github.com/cheyoung211/DocGen-sequential/commit/278e7488a5ecc6c91ffc3fb4e7ce091e9f5df63b))
수정 C(Cross-section dependency) 체크리스트 C1의 구현 범위와 설계를 기록한다.

**진행 순서에 대한 기록**: Notion §4 "권장 구현 순서" 표는 순서 2 = C1-C2(+C3 참조/순서),
순서 3 = B1-B3(+C3 재사용/일관성)로 되어 있다. 사용자는 2026-11-03 두 차례 확인을 거쳐
"계획대로 C1-C2를 먼저" 진행하고 "C를 먼저 모두 구현하고 B를 진행"하기로 결정했다 -- Fix B의
B1-B3는 이 결정에 따라 Fix C가 끝난 뒤로 미뤄졌다. 단, 계획 문서 자체가 C3의 두 번째 항목
(must_reuse/must_be_consistent_with)에 "B의 registry/계산 검사를 활용한다"고 명시하고 있어
C→B 단방향 선행만으로는 C3 전체를 끝낼 수 없는 상호 의존이 실재한다 (B3도 "C의 dependency
매핑으로 각 target의 사용 위치와 연결한다"고 요구하므로 역방향 의존도 있다). 이번 라운드는
C1만 구현한다 -- C3의 B-의존 부분에 도달하면 다시 확인한다.

## 0. 현재 코드 상태 (구현 전 감사)

기존에 "요구사항→생성 artifact" 매핑으로 쓸 수 있는 것은 **섹션 레벨 하나뿐**이고, 그조차
한 곳에서만 쓰인다:

- `src/evaluation/metrics/contract_satisfaction.py::_match_required_section`
  (L180-217)과 `_keywords`(L121-123)가 `RequiredSection.section_id` ->
  생성된 `SectionLayout` 매칭을 구현한다 (section_id 동등 -> 정규화 title 동등 ->
  title/purpose 키워드 overlap, 이 순서). `src/agents/planner.py`가 이 두 함수를
  **재사용**해 (`from src.evaluation.metrics.contract_satisfaction import _keywords,
  _match_required_section`, L23) 블루프린트 검증 시 같은 매칭을 수행한다
  (`_validate_blueprint`, L876-898). 이것이 이 코드베이스에 이미 있는 "매칭기
  재사용" 전례이며, 아래 설계는 이걸 중복 구현하지 않고 그대로 가져다 쓴다.
- `RequiredComponent`(`src/dataset/schemas.py` L277-289)와
  `CrossSectionDependency`(L305-315)에 대해서는 **생성 시점 매핑이 전혀 없다**:
  `PlannerInput`(`src/common/schemas.py`)에 `required_components` 필드 자체가
  없고, `src/agents/planner.py` 전체에 `RequiredComponent`에 대한 참조가 하나도
  없다 -- Planner는 `required_sections`만 보고 `required_components`/
  `cross_section_dependencies`는 전혀 보지 못한 채 블루프린트를 만든다.
- 평가 시점도 마찬가지로 약하다: `_score_required_components`(contract_satisfaction.py
  L304-337)는 component_id별로 어떤 블록이 그걸 만족하는지 식별하지 않고, "이 문서
  어딘가에 이 kind의 블록이 하나라도 있는가"만 본다(L308-313의 `kinds_present`는
  문서 전체 블록 kind의 집합일 뿐 section/component별 위치 정보가 없다). `cross_section_
  dependency`는 `evaluate_contract`가 `not_implemented`로 명시한다(L452-453) --
  구현이 아니라 의도적 미구현이다.
- `src/dataset/validators.py`에는 `component_id`/`section_id`/`dependency_id`의
  **중복 검사가 없다** -- `BenchmarkItem`의 pydantic 스키마도 `min_length=1`만
  강제하고 유일성은 강제하지 않는다. 오래된/수동 작성 benchmark 파일이 중복 id를
  가질 수 있고, 지금은 아무것도 그걸 잡지 않는다.

**핵심 확인 사항 (C1 설계의 근거)**: `src/dataset/instruction_builder.py`를 직접
읽어, 벤치마크를 생성하는 쪽에서 이미 다음 불변식을 지키고 있음을 코드로 확인했다 --
새로 발명한 규칙이 아니라 기존 데이터 생성기의 실제 동작이다:

- `RequiredComponent.placement`는 항상 **같은 `BenchmarkItem`의 어떤
  `RequiredSection.section_id`와 동일한 문자열**이다 (`placement_section =
  slot_id_by_index.get(comp["body_slot_index"], intro_id)`, L477, 이어서
  `RequiredComponent(..., placement=placement_section, ...)`, L481).
- `CrossSectionDependency.source_section`/`target_sections`도 항상 같은
  `required_sections`의 `section_id` 값이다 (`core_definition_section_id =
  slot_id_by_index.get(1, intro_id)`, L571; `target_sections=[worked_examples_id,
  summary_id]`, L578; 이후 두 dependency도 동일 패턴, L590-610).

즉 component/cross-section 매핑은 **섹션 매핑 위에 얹는 2차 매핑**으로 설계할 수 있다 --
`placement`/`source_section`/`target_sections`를 먼저 (기존) 섹션 매칭기로 생성
section_id로 바꾸고, 그 다음에야 "이 섹션 안의 어느 블록"을 찾는 문제가 된다. 벤치마크
자체가 이 계층 구조를 이미 전제하고 있으므로, 매핑기를 그 반대로(섹션 매칭과 무관하게
전체 문서에서 컴포넌트를 찾는 방식으로) 설계하면 벤치마크가 실제로 인코딩한 정보를
버리는 것이 된다.

## 1. 허용 label·관계별 판정 기준 (Gold Labeling에서 확정한 실제 판정 원칙)

Notion "Gold Labeling" 페이지(사람이 10개 샘플을 직접 판독한 기록, 최종 수정
2026-10-26)에서 실제로 적용된 원칙을 다음과 같이 코드화한다. 전부 실제 사람 판정
사례에서 역산한 것이고, 이 문서가 새로 지어낸 규칙이 아니다.

1. **직접 라벨 인용** (`\Cref{<해당 컴포넌트의 label>}`)이 항상 충족.
2. **섹션 레벨 인용 허용** (`\Cref{sec:<source_section>}`)도 참조 충족으로 인정한다
   -- sample_000: "직접 라벨 인용은 없으나 섹션 레벨 인용... 참조 충족으로 인정하기로
   함". 이는 사용자가 명시적으로 결정한 원칙이며, 사후에 "아무 인접 라벨이나 전부
   인정"으로 확대하지 않는다 (Notion 본문: "임의의 인접 label이면 모두 인정하거나...
   바꾸지 않는다").
3. **인접 라벨 인용**은 오직 **해당 컴포넌트 자체가 라벨을 갖지 않을 때만** 인정한다
   (예: `boxed_result`는 라벨이 없는 tcolorbox라서 `default_block_label`이 `None`을
   반환함 -- `src/common/state.py` L207-213의 `_LABEL_PREFIX_BY_KIND`에 KEY_TAKEAWAY류가
   없음과 정확히 대응). sample_002/003/007의 `boxed_result`가 이 경우다 -- "박스
   자체에는 라벨이 없지만, summary가 바로 그 결과를 뒷받침하는 수식 라벨을 직접
   인용함 -> 충족으로 처리". **이 판정은 본문 내용을 보고 "바로 그 결과를 뒷받침하는"
   것이 맞는지 사람이 확인한 것이라 구조적으로 미리 열거할 수 없다** -- C1은 라벨이
   없는 컴포넌트에 대해 "인접 라벨 허용 가능" 플래그만 남기고, 실제로 어느 라벨이
   인접한지는 C3(본문 스캔)의 몫으로 둔다.
4. **`reference_required=False`인 컴포넌트는 인용이 전혀 없어도 충족** -- sample_004
   definition_core: "요구사항이 직접 인용을 요구하는 게 아니므로 충족으로 처리". 이건
   "인용 안 해도 패스"가 아니라 "애초에 인용 의무가 없다"는 뜻이고, `depends_on`의
   **내용 재사용**은 별도로 확인된다(다음 항목).
5. **`depends_on`(컴포넌트 레벨)은 내용 재사용 여부로 판정**하며 인용 존재와는 별개다
   -- sample_002 worked_example_1: "T\_n(x)/f hat(x) 회귀 설정을 전혀 쓰지 않고...
   정의도 인용하지도 않음 -> 확정 미충족"; sample_000 worked_example_1: "T\_n(x) 공식
   재사용 확인 -> 충족" (인용 여부는 언급조차 없음). C1은 `depends_on`을 "라벨
   의무"가 아니라 "내용 재사용 의무"로 기록하고, 실제 내용 비교는 B/C3에 위임한다.
6. **`must_reference`는 실제 교차참조 명령이 있어야 하며, 본문에서의 단순 언급으로는
   불충분**하다 -- sample_005: "summary.tex 전체에 Cref가 단 한 번도 없음
   (\"the analysis section\"이라는 서술만 있고 실제 교차참조 명령 없음) -> 확정
   미충족", 10개 중 유일한 사례. `src/agents/latex_assembler.py`의
   `REFERENCE_COMMAND`(L1127: `\ref|Cref|cref|eqref|nameref|autoref|pageref`)가
   정확히 이 구분(명령 vs. 평문)에 쓸 수 있는 기존 정규식이다.
7. **`must_reuse`/`must_be_consistent_with`는 다중 target을 개별 판정**한다 -- 하나의
   `CrossSectionDependency.target_sections`에 여러 target이 있을 때 일부만 충족해도
   "부분 미충족"으로 기록하지 전체를 PASS나 FAIL로 뭉뚱그리지 않는다 (sample_002:
   "worked_examples는 미충족... summary는 충족"; sample_006: "summary는 충족...
   case\_or\_application은... 전혀 인용하지 않고... 전혀 언급하지 않음"). C1의
   obligation 스키마는 애초에 target마다 별도 레코드를 만들어 이 "부분 미충족"을
   자연스럽게 표현한다 (아래 2절).
8. **인용이 존재해도 그 근거가 실제로 없으면 별도 결함** -- sample_008: "summary.tex는
   revenue\_growth\_analysis 섹션이 customer acquisition과 CAC를 보여줬다고 서술하는데,
   실제... 전혀 나오지 않음... 날조". 이건 "참조 유효성"(라벨이 존재하는가)과는
   완전히 다른 검사(인용된 섹션이 실제로 그 내용을 담고 있는가)이고, Notion 본문도
   "참조 유효성과 별도 평가한다"고 명시한다. C1은 이 검사를 전혀 시도하지 않는다 --
   obligation을 "인용 여부"까지만 구조화하고, "인용된 내용이 사실인가"는 C3 이후의
   몫으로 완전히 분리해 둔다.

**rubric 버전**: 위 8개 원칙은 Gold Labeling 페이지의 `page_last_edited_at =
2026-10-26T07:16:53Z` 기준 내용에서 추출했다 -- `AdditionalVerifierRunMetadata.
rubric_version`(`src/evaluation/schemas.py` L298)의 기존 기본값
`"gold_labeling_draft_2026-10-26"`과 **같은 날짜**이므로 새 rubric_version 문자열을
만들지 않고 그 필드를 그대로 재사용한다. 이 문서가 사람 판정 사례를 바꿔 쓸 때마다
(향후 라운드에서) 이 날짜/필드를 다시 확인해야 한다.

## 2. 스키마 설계 -- `src/core/dependency_mapping.py` (신규)

`src/core/schemas.py`(Constraint/Evidence/ArtifactRef)나
`src/dataset/schemas.py`(BenchmarkItem 자체)를 확장하지 않고 새 모듈을 둔다 --
이 매핑은 "벤치마크 계약 vs. 한 번의 생성 결과"를 잇는 변환 결과물이라
`BenchmarkItem`(계약 자체)도 `Constraint`(런타임 체크 대상)도 아닌, 그 사이에 있는
별개의 개념이기 때문이다(`constraint_checks.py`가 `ArtifactGraphStore` 전용 별도
모듈인 것과 같은 이유).

```python
class MappingStatus(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"       # 복수의 동등하게 그럴듯한 후보, 추측하지 않음
    NOT_REPRESENTABLE = "not_representable"  # 이 파이프라인에 대응 LayoutBlockKind가 없음 (컴포넌트만)
    DUPLICATE_ID = "duplicate_id" # 입력 계약 자체가 이 id를 두 번 이상 선언함

class SectionMapping(BaseModel):
    section_id: str                       # RequiredSection.section_id (벤치마크)
    status: MappingStatus
    matched_section_id: Optional[str] = None   # 실제 생성 SectionLayout.section_id
    matched_via: Optional[str] = None          # _match_required_section의 matched_via 그대로
    reason: str = ""

class ComponentMapping(BaseModel):
    component_id: str
    status: MappingStatus
    placement_section_id: str             # 벤치마크 RequiredComponent.placement (원본)
    matched_section_id: Optional[str] = None   # placement가 매칭된 생성 section_id
    matched_block_id: Optional[str] = None
    matched_label: Optional[str] = None        # label_registry에 실제 등록된 label (없으면 None)
    matched_via: Optional[str] = None          # "kind_only" | "kind_and_keyword"
    reason: str = ""

class CrossSectionObligation(BaseModel):
    dependency_id: str
    relation: DependencyRelation          # src.dataset.schemas.DependencyRelation 재사용
    description: str
    source_section_id: str                # 벤치마크 원본
    target_section_id: str                # target_sections를 1개씩 분해한 것 (원칙 7)
    status: MappingStatus                 # source/target 섹션 매칭 상태를 합성
    matched_source_section_id: Optional[str] = None
    matched_target_section_id: Optional[str] = None
    allowed_labels: List[str] = Field(default_factory=list)
    content_check_required: bool = False  # must_reuse/must_be_consistent_with는 True (원칙 5/7)
    adjacent_label_possible: bool = False  # source 섹션 안에 라벨 없는 컴포넌트가 있을 때 True (원칙 3)
    order_hint: Optional[Tuple[int, int]] = None  # (source, target)의 blueprint.sections 내 인덱스 -- must_precede용
    reason: str = ""

class ComponentDependencyObligation(BaseModel):
    """RequiredComponent.depends_on (컴포넌트->컴포넌트, 원칙 5)."""
    component_id: str                     # 의존하는 쪽
    depends_on_component_id: str          # 의존 대상
    status: MappingStatus
    matched_component_block_id: Optional[str] = None
    matched_depends_on_block_id: Optional[str] = None
    content_check_required: bool = True   # 항상 내용 재사용 검사 대상 (원칙 5)
    reason: str = ""

class DependencyMappingReport(BaseModel):
    rubric_version: str = "gold_labeling_draft_2026-10-26"
    section_mappings: List[SectionMapping]
    component_mappings: List[ComponentMapping]
    cross_section_obligations: List[CrossSectionObligation]
    component_dependency_obligations: List[ComponentDependencyObligation]
    data_quality_issues: List[str] = Field(default_factory=list)  # 중복 id 등, 매칭 전 단계에서 발견
```

설계 결정 근거:

- **target마다 별도 `CrossSectionObligation`** (원칙 7) -- 하나로 뭉쳐서 "이 dependency는
  pass/fail"로 기록하면 sample_002/006/008의 부분 미충족을 표현할 수 없다. 대신
  `(dependency_id, target_section_id)` 쌍마다 하나씩 만든다.
- **`allowed_labels`는 직접+섹션 레벨만, 인접 라벨은 넣지 않는다** (원칙 2/3) --
  인접 라벨 인정은 본문 내용을 봐야 판단 가능한 사실이라 매핑 단계(생성물의 본문을
  읽지 않는 구조적 매칭)에서 미리 확정할 수 없다. 대신 `adjacent_label_possible`
  플래그로 "이 source 섹션에는 라벨 없는 컴포넌트가 있다 -> C3가 본문에서 인접
  라벨을 찾아봐야 한다"는 사실만 남긴다. 리스트를 억지로 채우면 "구조적으로 확정된
  허용 라벨"과 "내용을 봐야 아는 허용 라벨"이 같은 필드에 섞여 어느 쪽인지 알 수
  없게 된다 (이 프로젝트가 반복해서 경계하는 "같은 사실을 나타내는 두 필드가 결국
  어긋난다" 패턴, `docs/stage2_artifact_graph.md` 라운드 2 참고).
- **`content_check_required`를 명시적 플래그로 분리** (원칙 5/7) -- "매칭됐다"와
  "충족됐다"를 절대 같은 뜻으로 쓰지 않는다. `MATCHED`는 "source와 target이 실제
  생성물의 무언가에 대응된다"는 뜻일 뿐, must_reuse/depends_on의 진짜 판정(내용
  재사용 여부)은 아직 전혀 하지 않았다는 걸 이 플래그로 못박는다.
- **`DUPLICATE_ID`는 매칭 루프 전에 끝낸다** -- 중복 id를 가진 요구사항은 "어느 것과
  매칭해야 할지 모호"가 아니라 "입력 계약 자체가 깨졌다"는 다른 종류의 문제라서
  별도 상태값과 `data_quality_issues`(최상위, 개별 매핑 레코드가 아니라 리포트
  전체에 대한 경고)로 분리한다.

## 3. 매칭 알고리즘

1. **사전 검사**: `required_sections`/`required_components`/`cross_section_dependencies`
   각각에서 id 중복을 검사한다. 중복된 id는 전부 `DUPLICATE_ID`로 기록하고
   (중복 그룹의 각 항목이 아니라 "이 id가 N번 나타남" 한 줄을) `data_quality_issues`에
   추가하며, 매칭 루프에서 완전히 건너뛴다(추측해서 하나를 고르지 않는다).
2. **섹션 매핑**: 중복 아닌 `required_sections` 전체에 대해 기존
   `_match_required_section`(+ `claimed_ids` 공유)를 **그대로 호출**해 `SectionMapping`
   리스트를 만든다 -- 0절에서 확인한 재사용 전례를 따른다.
3. **컴포넌트 매핑**: `required_components`를 순서대로 처리하며, 섹션별로 이미 이
   컴포넌트 매핑 루프가 소비한 block id를 추적하는 `claimed_block_ids: Set[str]`를
   쓴다(섹션 매칭의 `claimed_ids`와 같은 원리, 섹션 레벨이 아니라 블록 레벨).
   - `placement`가 가리키는 벤치마크 section_id를 2단계의 `SectionMapping`에서
     찾는다. 못 찾거나(DUPLICATE_ID로 건너뛴 경우 포함) 그 섹션 자체가 UNMATCHED면
     `ComponentMapping.status = UNMATCHED`(reason에 "placement section 자체가
     매칭되지 않음"을 남긴다).
   - `COMPONENT_TYPE_TO_LAYOUT_KIND[comp.type]`(contract_satisfaction.py에서 **재사용**,
     재정의하지 않음)가 `None`이면 `NOT_REPRESENTABLE`.
   - 매칭된 생성 section의 blocks 중 `claimed_block_ids`에 없고 kind가 일치하는
     후보를 모은다. 0개면 `UNMATCHED`. 1개면 `MATCHED`(`matched_via="kind_only"`)이고
     그 block id를 claim한다. 2개 이상이면 `_keywords`(재사용)로
     `comp.description`과 각 후보의 `title`/`instructions`의 overlap을 비교해
     유일한 최고 점수가 있으면 `MATCHED`(`matched_via="kind_and_keyword"`), 동점이거나
     전부 0점이면 `AMBIGUOUS`(reason에 경합 후보 id들을 전부 나열).
   - 매칭되면 `src/common/state.py::default_block_label`로 라벨을 구해 `matched_label`에
     채운다(없으면 `None`으로 두고 억지 라벨을 만들지 않는다 -- 원칙 3이 바로 이
     `None`인 경우를 전제한다).
4. **cross-section obligation**: 각 `CrossSectionDependency`의 `source_section`과
   `target_sections`의 각 원소를 2단계 `SectionMapping`에서 조회해 하나씩
   `CrossSectionObligation`을 만든다(target마다 별도 레코드, 원칙 7). `status`는
   source/target 둘 다 MATCHED면 `MATCHED`, 하나라도 UNMATCHED/AMBIGUOUS/DUPLICATE_ID면
   그 상태를 그대로 전파한다(새로 추측하지 않는다). 둘 다 매칭됐을 때만:
   - `allowed_labels`: matched source section 자체의 `sec:{id}` 라벨 + `placement ==
     source_section_id`인 모든 매칭된 `ComponentMapping`의 `matched_label`(None 제외).
   - `adjacent_label_possible`: 위 컴포넌트들 중 `matched_label is None`인 것이
     하나라도 있으면 True.
   - `content_check_required`: `relation in {MUST_REUSE, MUST_BE_CONSISTENT_WITH}`.
   - `order_hint`: `(blueprint.sections.index(matched_source), blueprint.sections.index(matched_target))`.
5. **component dependency obligation**: 각 `RequiredComponent`의 `depends_on`
   원소마다 하나씩 만든다. 둘 다 `ComponentMapping`에서 MATCHED일 때만
   `matched_*_block_id`를 채우고, 아니면 해당 상태를 전파한다. `content_check_required`는
   항상 True(원칙 5 -- depends_on은 애초에 내용 재사용 전용 개념).

## 3.1. 구현 중 자체 발견한 버그 (사용자 리뷰 전 수정)

구현 직후 자체 재검토에서: 중복 `component_id`를 가진 컴포넌트의 `depends_on` 쪽
루프가 `if comp.component_id in component_dup_ids: continue`로 **아무 레코드도 남기지
않고 건너뛰었다** -- `ComponentMapping`은 DUPLICATE_ID로 보고되지만
`ComponentDependencyObligation`은 완전히 사라지는, 0절 "중복을 명시적으로 보고한다"
원칙을 스스로 어긴 사례였다. `cross_section_dependencies`의 중복 처리(모든 target에
대해 DUPLICATE_ID 레코드를 만듦)와 대칭이 맞지 않는 것으로 발견했다. 고쳐서
`depends_on` 항목마다 `DUPLICATE_ID` 상태의 `ComponentDependencyObligation`을 만들도록
수정했고, 회귀 테스트(`test_duplicate_component_id_still_reports_its_depends_on_edges`)
를 추가했다. (`depends_on_component_id` 쪽이 중복인 경우는 기존 `UNMATCHED` 처리로
이미 레코드가 남으므로 손대지 않았다 -- 이유만 "did not resolve"로 덜 구체적일 뿐
누락은 아니다.)

## 4. C2 구현 (2026-11-03)

**C2 체크리스트 1 (source 문맥 제공 + 입력 버전 기록)**: 기존에 완전히 미구현이었던
(`required_components`/`cross_section_dependencies`가 Planner에 전달되지도 않았고,
`test_planner_cross_section_dependencies.py`/`test_text_agent_dependency_context.py`
두 테스트 파일이 이미 이 정확한 기능의 스펙으로 작성되어 있었으나 실제 구현은 없어
기존 "57개 무관 실패" 목록에 섞여 있었음을 발견) 기능을 구현해 두 파일(7개 테스트)을
통과시켰다.

- `src/common/schemas.py::PlannerInput.cross_section_dependencies` 필드 추가.
- `src/agents/planner.py::PlannerAgent._wire_cross_section_dependencies` (신규,
  `_validate_blueprint`에서 호출, `required_sections` QA 토글과 무관하게 항상 실행):
  `_match_required_section`을 재사용해 `source_section`/`target_sections`를 실제 생성
  node id로 해석하고, **관계별로 다르게** 기록한다 -- `must_reuse`/
  `must_be_consistent_with`만 `node["dependencies"]`(readiness 게이트)와
  `spec["dependency_notes"]` 둘 다, `must_reference`는 `dependency_notes`만(라벨만
  있으면 되고 forward reference가 정당하므로 readiness를 막지 않음), `must_precede`는
  둘 다 안 씀(순수 조립 순서라 C1의 `order_hint`가 담당). Notion 본문의 "forward
  reference와 문서 순서 요구를 생성 입력 관계와 구분한다"를 이 3분기로 구현.
- `src/agents/text_agent.py::TextAgent._build_dependency_context` (신규, 프롬프트에
  주입): `must_reuse`/`must_be_consistent_with`는 source가 DRAFTED/REVIEWED/
  VALIDATED일 때만(아니면 완전히 생략, placeholder 없음) 실제 content excerpt(500자
  cap, "[…]" 표시)+key_points+관계/설명을 보여준다. `must_reference`는 draft 여부와
  무관하게 항상 라벨 의무+설명만 보여주고(excerpt 없음 -- 라벨은 이미 "Available
  cross-reference labels" 목록에 있음) forward reference를 막지 않는다. 호출마다
  `node.spec["dependency_versions_used"]`에 실제로 읽은 source의 `attempt_count`/
  `last_updated`를 기록(legacy `DocumentNode`에 진짜 버전 번호가 없어 이게 지금
  가능한 유일한 "입력 버전" 근사치 -- Stage 0의 기존 known gap과 동일한 한계).

**C2 체크리스트 2 (`compute_ready_artifacts` 개념 연결 + 누락·stale·순환 처리)**:
`src/common/state.py::DocumentGraph`에 3개 메서드 추가 -- `compute_ready_sections`
(Stage 5 `compute_ready_artifacts`와 동일한 "정의하고 테스트하되 실제 루프에는
연결하지 않는다" 원칙, `run_full_pipeline.py`는 건드리지 않음), `find_dependency_cycles`
(DFS 기반, candidate 범위 제한 가능), `find_stale_dependents`(`dependency_versions_used`
스냅샷이 source의 현재 attempt_count/last_updated와 달라지면 stale로 표시 -- "입력
버전을 기록"과 "stale 입력을 처리"가 같은 메커니즘으로 연결됨). `test/
test_document_graph_readiness.py` 16개 테스트.

**회귀 확인**: 전체 테스트 스위트, 기존 57개 무관 실패 중 정확히 7개(지금 구현한 두
스펙 파일)만 PASS로 바뀌었고 나머지 50개는 `git stash` 비교로 완전히 동일함을 확인.

## 5. C3 구현 (2026-11-03, 첫 번째 bullet만)

신규 `src/core/dependency_verifier.py`. C1의 `DependencyMappingReport` + 실제 렌더링된
section 텍스트(`Dict[str, str]`)를 입력으로 받아 `CrossSectionObligation`마다
(target 단위로, dependency_id로 뭉치지 않음) `Evidence`와 같은 구조의
`DependencyVerificationRecord`(PASS/FAIL/UNKNOWN + `unknown_reason` + `detail` +
`cited_labels_found` + `matched_at_line`)를 만든다.

- **`must_reference`**: `latex_assembler.REFERENCE_COMMAND`를 재사용해 실제 인용
  명령(`\Cref`/`\ref`/...)만 인정하고 평문 언급은 FAIL(Gold Labeling sample_005).
  콤마로 묶인 다중 라벨 인용(`\Cref{a,b}`)도 분리해서 처리(기존
  `validate_references`는 이걸 분리하지 않는 것으로 확인했으나, 그 메서드의 더 단순한
  목적(라벨이 존재하는가)에는 문제가 안 되므로 그쪽은 손대지 않음).
- **`must_precede`**: C1의 `order_hint`(blueprint.sections 인덱스)로 선후 관계를
  확인. `order_hint`가 없으면(source/target 둘 다 안 잡힘) UNKNOWN.
- **`must_reuse`/`must_be_consistent_with`**: Fix B의 registry/계산 검사가 없는
  현재로서는 전부 UNKNOWN/NOT_REPRESENTABLE -- "검사기가 없다"는 이유로 PASS가
  되는 일은 없다.
- `status != MATCHED`인 obligation(UNMATCHED/AMBIGUOUS/DUPLICATE_ID)도 레코드
  자체는 만들되 UNKNOWN으로 남긴다 -- 매핑 안 된 요구사항이 결과에서 조용히
  사라지지 않는다.
- `DependencyVerificationResult`는 `total_targets`/`checked_targets`/`pass_count`/
  `fail_count`/`unknown_count`를 **저장된 필드가 아니라 `records`에서 매번 계산하는
  property**로 제공 -- 두 출처가 서로 어긋날 수 있는 상태를 원천적으로 차단
  (`docs/stage_review_rigor` 패턴 #10과 동일 원칙).

`test/core/test_dependency_verifier.py` 11개 테스트, 전체 회귀는 여전히 기존 50개
무관 실패만 남고 동일함을 재확인.

**C3의 두 번째 bullet은 아직 구현하지 않았다.** Notion 본문이 "must_reuse와
must_be_consistent_with를... B의 registry/계산 검사를 활용"한다고 명시하고 있고, B(B1-B3)는
사용자의 2026-11-03 결정에 따라 아직 시작하지 않았다 -- 이 문서 맨 위 "진행 순서에 대한
기록" 참고. 이번 라운드는 여기서 멈춘다.

## 5.1. C4 구현 (2026-11-03) -- controller 연결

사용자 결정(2026-11-03): C3의 두 번째 bullet(must_reuse/must_be_consistent_with)은
B가 끝난 뒤로 미루고, C4-C5를 먼저 진행.

핵심 발견: **생산용 `ConstraintProvider`가 코드베이스 어디에도 없다는 Stage 0의 기존
known gap은, dependency 검증에 대해서는 생각보다 훨씬 작은 작업으로 닫힌다.**
`repair_operations.ConstraintProvider = Callable[[ArtifactRef], List[Constraint]]`
하나뿐이고, `ToolExecutor`/`FixedGraphAwareRepairPolicy`/`ArtifactGraphStore.
live_evidence`/`Checkpoint`는 이미 Stage 2-4에서 "어떤 constraint kind든" 제너릭하게
동작하도록 완성되어 있었다 -- 새 kind를 추가해도 그 기존 기계장치를 다시 만들 필요가
없었다.

- `ConstraintKind.DEPENDENCY` 추가 (`src/core/schemas.py`).
- `BasicConstraintChecker._check_dependency` (`src/core/constraint_checks.py`):
  `context["dependency"][constraint_id]`로 relation/allowed_labels/order_hint를
  받는, 기존 `_check_recompute`와 동일한 확장점 패턴. **재구현이 아니라 adapter** --
  `dependency_verifier._verify_must_reference`/`_verify_must_precede`를 그대로
  호출해 Evidence로 변환만 한다(두 경로가 같은 사실에 대해 서로 다른 답을 낼 수
  없게). must_reuse/must_be_consistent_with는 여기서도 UNKNOWN/NOT_REPRESENTABLE.
- `src/core/dependency_constraint_provider.py` (신규) -- `DependencyConstraintProvider`:
  `DependencyMappingReport` + `ArtifactGraphStore`로부터 실제 `ConstraintProvider`를
  만든다. source ref는 **항상 `store.latest()`로 현재 버전을 조회** -- 이미 존재하는
  source에 추측 버전 번호를 쓰면 invalidation이 영원히 그 버전에 고정되는 버그가
  되므로, "한 번도 저장 안 됨"일 때만 `version=1` placeholder를 쓰고
  `_resolve_input_refs`가 스스로 "missing"으로 찾아내게 둔다. `context_for()`가
  `__call__()`과 같은 obligation 목록에서 동시에 만들어져 둘이 어긋날 수 없다.
- **두 번째 발견(기존 RECOMPUTE에도 있던, 이름 붙여지지 않은 gap)**: `repair_
  operations.py`(`self._base_context`, 생성 시점 고정)와 `controller.py`
  (`context={}`)의 실제 재검증 호출 경로 어디도 `ConstraintProvider`가 반환한
  제약에 맞는 `context`를 실제로 조립해서 넘기지 않는다 -- RECOMPUTE도 지금까지
  오직 손으로 만든 테스트 호출로만 검증되었을 뿐, 실제 controller 루프를 통과한
  적이 없다는 뜻. 이번 session은 이 gap을 메우지 않고(그건 I1 범위), 대신
  `DependencyConstraintProvider`가 `__call__`/`context_for` 둘 다 노출해 나중에
  결합할 재료를 준비해 두는 데서 멈췄다 -- 문서에 솔직하게 기록.
- **복구 범위는 새 코드 없이 이미 맞다**: `FixedGraphAwareRepairPolicy.decide()`는
  constraint kind를 전혀 보지 않고 `Evidence.checked_artifact`(=target)의 kind만
  보고 스코프를 고른다 -- DEPENDENCY FAIL도 자동으로 `REGENERATE_SECTION`을 target에
  대해 고른다. "잘못된 공통 정의는 source와 영향받는 target을 복구"는 Fix B
  시나리오라 지금은 발생하지 않음(해당 없음).
- **budget cutoff/resume은 이미 공짜**: `ToolExecutor._dispatch_check`가 매
  CHECK마다 `store.record_evidence`를 자동 호출하고, `live_evidence`/
  `save_checkpoint`/`from_checkpoint`가 이미 Stage 2-4에서 테스트된 기계장치라서,
  DEPENDENCY Evidence도 별도 코드 없이 그대로 재사용·재검증 없는 PASS 금지·resume
  후 미해결 유지가 성립함을 테스트로 직접 확인했다(`test_dependency_controller_
  integration.py`).

`test/core/test_constraint_checks.py`(TestDependencyConstraint, 8개),
`test/core/test_dependency_constraint_provider.py`(7개),
`test/core/test_dependency_controller_integration.py`(4개, 실제
ToolExecutor+FixedGraphAwareRepairPolicy+checkpoint/restore 통과).

## 5.2. C5 구현 (2026-11-03) -- End-to-end 확인

`test/core/test_dependency_c5_end_to_end.py` (10개): `BenchmarkItem`+
`DocumentGraph` 하나(섹션 4개: intro/formal_definition/worked_examples/summary,
dependency 6개, target 7개)로 체크리스트가 요구한 시나리오를 전부 실제로 구성해
확인했다 -- 필수 인용 누락(FAIL), 잘못된 label(인용은 있으나 FAIL), 허용된 section
참조(PASS), forward reference(물리적 순서 무관 PASS), 문서 순서 요구(order_hint
PASS), "인용만 있고 재사용은 없는 경우"(summary가 reference_key_result는 PASS하면서
동시에 reuse_core는 UNKNOWN으로 남아 citation≠reuse 확인), 여러 target 중 일부만
다른 결과(reuse_core의 두 target 모두 독립 기록), 최종 미해결 집계(PASS 3/FAIL 2/
UNKNOWN 2, `DependencyVerificationResult`의 property와 실제 `store.live_evidence`
전체 스캔 둘 다 일치). C3의 직접 함수 경로와 C4의 실제 `ToolExecutor` 경로가 7개
target 전부에서 동일한 결과를 내는 것도 한 테스트로 확인(`test_the_real_executor_
path_agrees_with_the_direct_report_for_every_target`).

**명시적으로 검증하지 않은 것** (설계 문서 §6과 동일한 경계): "source 내용과 다른
요약"(인용은 유효하나 내용이 날조됨, sample_008류)은 의미 비교가 필요해 이 브랜치의
범위 밖; "정상 block 보존"은 constraint kind와 무관한 기존 Stage 4
`RepairOperationExecutor`의 책임이라 재검증하지 않음(이미 그 쪽 테스트가 커버).

**회귀 확인**: 전체 스위트, 기존 57개 무관 실패 중 여전히 정확히 7개(C2가 고친 것)만
PASS로 바뀌었고 나머지 50개는 동일. 신규 테스트 전부(C1 16 + C3 11 + C2 TextAgent/
Planner/DocumentGraph 7+6+16 + C4 19 + C5 10 = 85개) 통과.

**남은 것**: C3의 두 번째 bullet과 Fix B(B1-B6) 전체. `DependencyConstraintProvider`가
이미 must_reuse/must_be_consistent_with에 대해서도 Constraint를 만들어 두므로(항상
UNKNOWN), B가 실제 registry/recompute 검사를 들고 오면 `_check_dependency`의 해당
분기와 `dependency_verifier`의 해당 분기만 채우면 되고, 이 라운드의 provider/
executor/checkpoint 연결은 그대로 재사용된다.

## 6. 명시적으로 미룬 것 (C1 범위 밖)

- **인접 라벨의 실제 판별**(원칙 3) -- `adjacent_label_possible` 플래그까지만, 본문을
  스캔해 "이 라벨이 정말 그 결과를 뒷받침하는가"를 판단하는 건 C3.
- **내용 재사용/일관성의 실제 검사**(`content_check_required=True`인 모든 레코드) --
  B3(공유 값 registry)와 C3가 할 일. C1은 "이게 검사되어야 한다"는 사실과 두 쪽의
  정확한 artifact 참조만 만들어 둔다.
- **인용된 내용의 사실 여부**(원칙 8, sample_008의 날조 사례) -- 참조 유효성과
  별도로, C3 이후 어딘가에서 다뤄야 할 전혀 다른 검사. 이 문서/구현 어디에도
  "인용이 있으면 내용도 맞다"는 가정을 넣지 않는다.
- **`must_precede`의 실제 순서 위반 판정** -- `order_hint`라는 원재료만 제공.
  실제 "선후가 지켜졌는가"는 블루프린트 순서와 실제 조립 순서가 다를 수 있다는 점
  (Notion 본문: "생성 순서와 별도 관리") 때문에 C3가 조립 단계 정보까지 같이 보고
  판단해야 한다.
- **Planner/TextAgent 프롬프트에 이 매핑을 주입하는 것** -- C2의 일.
- **생성 파이프라인(`run_full_pipeline.py`)과의 실제 연결** -- Stage 0 문서가 이미
  적어 둔 기존 gap(I1, production `ConstraintProvider` 부재)과 동일한 종류의 문제이고
  여기서 새로 풀지 않는다. `DependencyMappingReport`는 지금은 `BenchmarkItem` +
  `DocumentGraph`(블루프린트+label_registry)만 입력으로 받는 순수 함수로 만든다.
  **(2026-11-04 갱신: `run_full_pipeline.py`의 readiness/cycle 연결은 §7에서 실제로
  닫혔다 -- 단, `DependencyConstraintProvider`를 그 파이프라인의 실제 생성 루프에
  연결하는 것 자체는 여전히 I1 범위다. 이 두 가지는 서로 다른 "연결"이다: §7이 고친
  건 레거시 `DocumentGraph`/`planner.py`/`text_agent.py` 경로가 자신이 이미 짜 둔
  `dependencies`를 실제로 지키게 만든 것이고, I1이 아직 닫지 않은 건 그 레거시
  경로와 전혀 별개인 `ArtifactGraphStore`/`ToolExecutor`/`RuleBasedController` 스택을
  `run_full_pipeline.py`에 아예 연결하는 것이다.)**

## 7. 리뷰 발견 수정 (2026-11-04) -- C1/C2/C4 완료 후 외부 리뷰 5건

C4-C5를 끝낸 뒤, 외부 리뷰(P1 3건, P2 2건)가 이 구현을 실제로 돌려 다섯 가지 결함을
찾았다. 전부 코드를 다시 읽어 재현한 뒤 고쳤다 -- 아래 각 항목은 리뷰가 지목한 파일·
줄을 직접 확인하고 수정한 것이다.

**[P1 · C1] `reference_required`가 매핑 결과에서 사라짐.** `ComponentMapping`이
`RequiredComponent.depends_on`은 반영하면서 `reference_required`는 전혀 읽지
않았다 -- 리뷰가 지적한 그대로, 이 값을 True↔False로 바꿔도 보고서가 완전히
동일했다. `ComponentMapping`에 `reference_required: bool` 필드를 추가하고 6개
생성 지점(DUPLICATE_ID/UNMATCHED×2/NOT_REPRESENTABLE/AMBIGUOUS/MATCHED) 전부에서
`comp.reference_required`를 그대로 보존하도록 고쳤다 -- 매칭 상태와 무관하게 "이
요구사항이 원래 인용을 요구했는가"라는 사실 자체는 항상 살아남아야 하기 때문이다
(원칙 4). `test_reference_required_is_preserved_on_component_mapping`이 리뷰의
재현 그대로(True/False 두 리포트의 `model_dump()`가 달라야 함)를 검증한다.

**[P2 · C1] 섹션 매핑의 동점을 MATCHED로 처리.** `build_section_mappings`가 재사용하는
`_match_required_section`의 title/keyword 두 tier는 모두 "동점이면 리스트 순서상
첫 번째"로 해소한다 -- 재사용 대상인 그 함수 자체를 바꾸지 않고(다른 호출자인
planner.py 검증/contract_satisfaction.py 채점에 영향 주지 않기 위해), `_tied_section_
candidates`를 새로 추가해 `_match_required_section`이 실제로 이긴 tier에서만 동점
후보를 재계산한다. 동점이 있으면 `AMBIGUOUS`로 보고하고 어느 쪽도 claim하지 않는다.
`matched_via == "section_id"`(정확한 id 일치)는 애초에 동점이 날 수 없으므로 재검사
대상에서 제외. 섹션 순서를 뒤집어도 항상 AMBIGUOUS가 나오는지 확인하는 테스트
(`test_section_mapping_reports_ambiguous_on_title_tie_regardless_of_list_order`)로
리뷰의 재현을 그대로 검증했다.

**[P2 · C2] 500자 절단이 요구사항과 무관한 접두사만 남길 수 있음.**
`TextAgent._build_dependency_context`가 `source_node.content`의 앞 500자만 잘라
붙였다 -- 필요한 값/수식/가정이 그 뒤에 있으면 아예 프롬프트에 들어가지 않았다.
`_select_relevant_excerpt`(신규, static method)를 추가해: 빈 줄 기준으로 content를
블록 단위로 나누고(`_render_draft_with_anchors`가 이미 그렇게 렌더링함), 각 블록을
dependency의 `description`과의 키워드 overlap(`_keywords`, 재사용)으로 채점해 가장
관련 있는 블록부터 예산(500자)을 채우고, 선택된 블록들을 원래 문서 순서로 되돌려
이어붙인다. 하나의 최고 관련 블록만으로 예산을 넘으면 그 블록만 직접 자른다(재배치된
텍스트의 앞부분을 맹목적으로 자르면 방금 골라낸 바로 그 내용이 잘려 나가는 버그를
직접 재현해서 고쳤다 -- 처음 구현은 이 버그가 있었다). description에 겹치는 키워드가
전혀 없거나 content에 빈 줄 구조가 전혀 없으면 예전과 동일한 "앞부분부터" 동작으로
자연스럽게 축소된다.
`test_excerpt_selects_the_relevant_paragraph_even_when_it_is_not_near_the_start`가
500자보다 긴 무관한 내용 뒤에 숨은 필요한 값이 실제로 프롬프트에 나타나는지 확인한다.

**[P1 · C4] 실제 controller/repair executor에서 dependency 검사가 항상 UNKNOWN/
not_checked로 끝남.** `controller.py`의 `RuleBasedController.decide()` 두 지점과
`repair_operations.py`의 `RepairOperationExecutor._check()`가 전부 `context={}`
또는 고정된 `self._base_context`만 썼다 -- `DependencyConstraintProvider.context_for()`를
전혀 호출하지 않는다. 리뷰가 재현한 그대로: 같은 인용 누락 obligation을
`provider.context_for(ref)`로 체크하면 FAIL, 이 두 실제 호출 경로가 쓰는 빈
context로 체크하면 UNKNOWN/NOT_CHECKED. `repair_operations.py`에 `_context_for_check(
provider, ref, base_context)` 헬퍼를 추가했다 -- `base_context`에 `provider.
context_for(ref)`(존재하면, `getattr` duck-typing)를 merge한다. 일반 함수로 된
기존 `ConstraintProvider`(`context_for` 속성이 없음)는 정확히 예전과 동일한 동작으로
축소되므로 다른 호출자의 동작은 바뀌지 않는다. 세 호출 지점(`RepairOperationExecutor.
_check`, `RuleBasedController.decide`의 두 `CheckCall`)을 모두 이 헬퍼로 바꿨다.
`test_rule_based_controller_decide_now_threads_context_so_the_check_actually_fails`
(`test_dependency_controller_integration.py`)가 리뷰의 재현을 실제
`RuleBasedController.decide()` + 실제 `ToolExecutor.execute()`로 end-to-end
재확인한다 -- 더 이상 UNKNOWN이 아니라 FAIL이 나온다.

이 수정은 "`repair_operations.py`/`controller.py`는 `run_full_pipeline.py`에 연결돼
있지 않다"는 그 모듈들 자신의 기존 scope 선언과 충돌하지 않는다 -- 그 선언은 실제
*파이프라인*에 대한 것이고, 이 버그는 그 클래스들이 (테스트에서든, 나중의 실제
호출자에서든) 이미 구성되어 쓰일 때 스스로 내는 결과가 틀렸다는, 모듈 내부
정확성 문제다.

**[P1 · C2] 실제 생성 루프가 dependency를 전달하지도, readiness를 적용하지도 않음.**
두 가지가 겹쳐 있었다:
1. `run_full_pipeline.py`의 `PlannerInput(...)` 생성 지점이 `required_sections`만
   넘기고 `cross_section_dependencies`는 항상 기본값 `[]`였다 -- C2가 `planner.py`/
   `text_agent.py`에 짜 넣은 와이어링 전체가 이 파이프라인에서는 그냥 죽은 코드였다.
   `benchmark_item.cross_section_dependencies`를 실제로 넘기도록 고쳤다.
2. 그 다음 실행 루프가 매 iteration마다 PENDING/ERROR 노드를 전부 바로 처리했다 --
   `_build_dependency_context`가 아직 drafted되지 않은 source를 프롬프트에서
   생략하는 것은 모델이 그 내용을 지어내는 것만 막을 뿐, target이 너무 일찍 drafted
   되는 것 자체는 전혀 막지 못한다(리뷰의 정확한 지적). 두 가지로 고쳤다:
   - `DocumentGraph.find_dependency_cycles()`를 그래프 구성 직후 한 번 호출해,
     기다려도 절대 풀리지 않는 순환을 생성 시작 전에 명확한 에러로 끝낸다("몇 번째
     iteration에서 뭐가 안 끝났는지"만 보여주는 기존의 모호한 에러 대신).
   - 배치(한 iteration의 pending_nodes) 처리 순서를 `_dependency_ordered`(신규)로
     재배열해 같은 배치 안의 dependency가 먼저 오게 하고, 각 노드를 실제로 drafting
     하기 직전에 `graph.compute_ready_sections({node.id})`를 **그 시점의 그래프
     상태로 다시** 확인한다 -- 같은 루프 안에서 방금 drafted된 형제 노드를 즉시
     반영하므로 흔한 경우(둘 다 이번 pass에서 성공)는 추가 iteration 없이 바로
     풀리고, source가 이번 pass에 실패하면(ERROR로 재시도 필요) dependent는 여전히
     이번 pass에서 제외되어 다음 iteration으로 미뤄진다(한 번의 snapshot만 보고
     판단하는 방식이었다면 놓쳤을 경우).
3. `_repair_and_recompile`(컴파일 에러 복구 루프)도 고쳤다 -- 이 경로는 이미
   drafted된 섹션의 내용을 그 이후에 다시 바꿀 수 있는 유일한 지점이라,
   `DocumentGraph.find_stale_dependents()`로 그 섹션을 must_reuse/must_be_
   consistent_with로 읽었던 다른 섹션을 찾아 재조립 전에 다시 drafting한다 --
   하지 않으면 더 이상 존재하지 않는 내용을 계속 인용/재사용하는 섹션이 조용히
   남는다.

**흥미로운 발견 (추정이 아니라 git 기록으로 확인)**: `_dependency_ordered`를 구현하며
`test_pipeline_dependency_ordering.py`를 다시 보니, Stage 0 라운드에서 이 파일의
import 실패를 "renamed/removed된 `_dependency_ordered`에 대한 stale 참조"라고
**추정**해 두고 손대지 않았던 기록을 발견했다. `git log --all -S"_dependency_ordered"`로
전체 히스토리를 확인한 결과 이 이름은 이 저장소 어디에도 커밋된 적이 없다 --
"renamed/removed"가 아니라 애초에 구현된 적이 없는, C2 기능을 위해 미리 작성된
스펙이었다(이 라운드 전의 `test_planner_cross_section_dependencies.py`/
`test_text_agent_dependency_context.py`와 정확히 같은 패턴). 이번에 실제로
구현해 4개 테스트가 모두 통과하고, `test/test_run_full_pipeline_additional_verifier_
metadata.py`의 잘못된 추정 docstring도 함께 고쳤다. 더 이상 `--ignore=`가 필요 없다.

**회귀 확인**: `pytest test/ -q`(이제 `test_pipeline_dependency_ordering.py`를
ignore할 필요 없음) -- 기존 50개 무관 실패와 정확히 동일(diff로 확인, 신규 실패
0건), 신규/확장 테스트 16개(dependency_mapping +5, text_agent dependency context +2,
repair_operations +2, controller +2, dependency_controller_integration +1,
pipeline_dependency_ordering +4) 전부 통과, 전체 1115 passed / 50 failed.

**명시적으로 검증하지 않은 것**: `DocGenPipeline.run()`의 실행 루프와
`_repair_and_recompile` 자체는 (이 수정 이전에도) 전용 단위 테스트가 없다 --
planner/text_agent/image_agent/assembler/compiler를 전부 엮어야 하는 무거운
통합 테스트가 필요해서다. 이번 수정이 실제로 쓰는 두 빌딩 블록(`_dependency_
ordered`, `DocumentGraph.compute_ready_sections`)은 각각 충분히 테스트되어
있지만, 이 둘을 `DocGenPipeline.run()`/`_repair_and_recompile` 안에서 **조합한
결과** 자체를 실제로 실행해 확인한 테스트는 없다 -- import/문법 검사와 코드
직접 검토로만 확인했다. 실제 Qwen3.6 파이프라인으로의 소규모 파일럿은 이
라운드의 범위 밖(Stage 0/5가 이미 적어 둔 것과 같은 종류의 후속 과제)이다.
