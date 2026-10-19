import pandas as pd
import json
import os

# 파일 경로 설정
input_file = './dataset/processed/wikihow_grouped.csv'
output_file = './dataset/processed/wikihow_prompt_schema.jsonl'

def create_prompt_schema(path):
    print("Prompt Schema 변환 시작...")
    df = pd.read_csv(path)
    
    # 1. Title을 기준으로 그룹화 (하나의 문서를 하나의 데이터 포인트로 변환)
    grouped = df.groupby('title')
    
    dataset = []
    
    for title, group in grouped:
        # 공통 정보 추출 (첫 번째 행 기준)
        overview = group['overview'].iloc[0]
        
        # 각 섹션의 정보를 리스트로 구성
        sections = []
        for _, row in group.iterrows():
            sections.append({
                "section_label": row['groupedLabel'],
                "headline": row['headline'],
                "content": row['text']
            })
        
        # 에이전트용 스키마 구성
        data_point = {
            "instruction": "Generate a comprehensive, structured guide based on the provided title and overview.",
            "input": {
                "title": title,
                "overview": overview
            },
            "output": {
                "outline": [s['section_label'] + ": " + s['headline'] for s in sections],
                "full_document": "\n\n".join([f"## {s['headline']}\n{s['content']}" for s in sections])
            }
        }
        dataset.append(data_point)

    # 2. JSONL 형식으로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\n변환 완료! 총 {len(dataset)}개의 문서 생성.")
    print(f"저장 위치: {output_file}")

# 실행
if os.path.exists(input_file):
    create_prompt_schema(input_file)
else:
    print("이전 단계의 파일이 없습니다.")