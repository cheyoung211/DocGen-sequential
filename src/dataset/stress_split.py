import json
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# 경로 설정
input_file = './dataset/processed/stress_prompt_schema.jsonl'
output_dir = './dataset/processed/stress_split/'

def stratified_split_jsonl(path, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 데이터 읽기
    data = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
            labels.append(obj['stress_factor']) # 층화 추출의 기준이 될 라벨

    print(f"전체 데이터 개수: {len(data)}")

    # 2. 첫 번째 분할: Train과 나머지(Dev + Test)
    # test_size는 (dev_ratio + test_ratio)가 됩니다.
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, 
        test_size=(dev_ratio + test_ratio), 
        stratify=labels, 
        random_state=42
    )

    # 3. 두 번째 분할: 나머지 데이터를 Dev와 Test로 반분 (1:1)
    relative_dev_size = dev_ratio / (dev_ratio + test_ratio)
    dev_data, test_data = train_test_split(
        temp_data, 
        test_size=(1 - relative_dev_size), 
        stratify=temp_labels, 
        random_state=42
    )

    # 4. 저장 및 결과 보고
    split_sets = {
        "train.jsonl": train_data,
        "dev.jsonl": dev_data,
        "test.jsonl": test_data
    }

    print("\n[Split Result Status]")
    for name, content in split_sets.items():
        # 파일 저장
        save_path = os.path.join(output_dir, name)
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in content:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 비율 확인
        counts = Counter([d['stress_factor'] for d in content])
        print(f"파일: {name} | 개수: {len(content)}")
        for factor, count in counts.items():
            print(f"  - {factor}: {count} ({count/len(content)*100:.1f}%)")

    print(f"\n모든 파일이 {output_dir}에 성공적으로 저장되었습니다.")

# 실행
if os.path.exists(input_file):
    stratified_split_jsonl(input_file)
else:
    print("입력 파일을 찾을 수 없습니다. 이전 단계의 스크립트를 먼저 실행해주세요.")