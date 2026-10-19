import pandas as pd
import json
import os

# 경로 설정
input_file = './dataset/processed/wikihow_grouped.csv'
output_file = './dataset/processed/stress_prompt_schema.jsonl'

def create_stress_dataset(path):
    print("Stress Test Dataset 생성 시작...")
    df = pd.read_csv(path)
    
    # 1. 스트레스 요인 및 지시문 정의
    stress_map = {
        "long_context_planning": (
            "Generate an extremely detailed guide with 6-8 main sections, including multiple subsections for each. "
            "The total length must be between 1,500 to 3,000 words, ensuring each section is deeply elaborated."
        ),
        "constraint_binding": (
            "Your response MUST include exactly one table, one figure placeholder, a specific 'Overview' section, "
            "and a 'Summary' section. Failure to include any of these four elements will be considered a task failure."
        ),
        "cross_section_dependency": (
            "In the Introduction, establish a clear categorization or framework. You MUST reuse and strictly adhere "
            "to this exact categorization in later sections and any tables produced, ensuring 100 percent logical consistency."
        ),
        "multimodal_grounding": (
            "Do not insert tables randomly. Only include a table when it serves a structural purpose, "
            "such as ingredient comparisons, cost breakdowns, or timelines. "
            "The table must correspond to a concept explicitly introduced in the text (e.g., cost, steps, timeline)."
        ),
        "reference_consistency": (
            "Explicitly reference 'Figure 1' and 'Table 1' at least twice in different parts of the main text. "
            "The references must be contextually accurate and explain the visual elements in those specific locations."
        ),
        "formatting": (
            "Use at least 5 LaTeX formatting elements in total. "
            "These must include at least 3 different types chosen from: enumerate, itemize, equation, table, figure, and cross-references."
        )
        # "ambiguity": (
        #     "Based on the somewhat underspecified prompt provided, use your agentic planning capabilities to infer "
        #     "missing details, resolve ambiguities, and proactively fill in the gaps to create a complete guide."
        # )
    }
    
    stress_factors = list(stress_map.keys())
    
    # 2. Title별 그룹화
    grouped = df.groupby('title')
    dataset = []
    
    for i, (title, group) in enumerate(grouped):
        # 라운드 로빈 방식으로 스트레스 요인 선택
        factor_name = stress_factors[i % len(stress_factors)]
        instruction = stress_map[factor_name]
        
        overview = group['overview'].iloc[0]
        
        # 섹션 데이터 구성
        sections = []
        for _, row in group.iterrows():
            sections.append({
                "section_label": row['groupedLabel'],
                "headline": row['headline'],
                "content": row['text']
            })
        
        # 데이터 포인트 생성
        data_point = {
            "stress_factor": factor_name,  # 추적용 컬럼 추가
            "instruction": instruction,     # 강화된 영어 지시문
            "input": {
                "title": title,
                "overview": overview
            },
            "reference_output": {
                "outline": [f"{s['section_label']}: {s['headline']}" for s in sections],
                "full_document": "\n\n".join([f"## {s['headline']}\n{s['content']}" for s in sections])
            }
        }
        dataset.append(data_point)

    # 3. JSONL 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\n성공! 총 {len(dataset)}개의 스트레스 테스트 데이터가 생성되었습니다.")
    print(f"저장 위치: {output_file}")
    
    # 요인별 개수 출력 (균등 배분 확인)
    from collections import Counter
    counts = Counter([d['stress_factor'] for d in dataset])
    print("\n[Factor Distribution]")
    for factor, count in counts.items():
        print(f"- {factor}: {count}")

# 실행
if os.path.exists(input_file):
    create_stress_dataset(input_file)