import json
import random
import os

# 경로 설정
input_file = './dataset/processed/wikihow_prompt_schema.jsonl'
output_dir = './dataset/processed/split/'

def split_dataset(path, train_ratio=0.8, dev_ratio=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 데이터 읽기
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 2. 데이터 셔플
    random.seed(42) # 재현성을 위한 시드 고정
    random.shuffle(lines)
    
    # 3. 영어 프롬프트로 업데이트 (Instruction 필드 수정)
    updated_lines = []
    for line in lines:
        data = json.loads(line)
        # 모든 프롬프트를 영어로 변경
        data['instruction'] = (
            "Based on the following title and overview, "
            "generate a structured guide document including an outline and the full content."
        )
        updated_lines.append(json.dumps(data, ensure_ascii=False))

    # 4. 분할 지점 계산
    total = len(updated_lines)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    train_data = updated_lines[:train_end]
    dev_data = updated_lines[train_end:dev_end]
    test_data = updated_lines[dev_end:]
    
    # 5. 파일 저장
    split_files = {
        "train.jsonl": train_data,
        "dev.jsonl": dev_data,
        "test.jsonl": test_data
    }
    
    for filename, data in split_files.items():
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(item + '\n')
                
    print(f"데이터 분할 완료!")
    print(f"- Train: {len(train_data)} samples")
    print(f"- Dev: {len(dev_data)} samples")
    print(f"- Test: {len(test_data)} samples")
    print(f"저장 위치: {output_dir}")

# 실행
if os.path.exists(input_file):
    split_dataset(input_file)
else:
    print("Input file not found. Please run the previous step first.")