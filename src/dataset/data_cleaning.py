import pandas as pd
import os

# 경로 설정
input_path = './dataset/raw/wikihow/wikihowSep.csv'
output_dir = './dataset/processed/'
output_file = os.path.join(output_dir, 'wikihow_cleaned.csv')

def clean_wikihow_data(path):
    # 1. 데이터 로드
    print("데이터 로딩 중...")
    df = pd.read_csv(path, low_memory=False)
    initial_count = len(df)
    
    # --- Step 1: 결측치 제거 ---
    df = df.dropna().reset_index(drop=True)
    after_dropna = len(df)
    print(f"결측치 제거 완료: {initial_count - after_dropna}개 삭제됨.")

    # --- Step 2: 텍스트 길이 하위 20% 제거 ---
    # 각 row의 text 길이 계산
    df['text_len'] = df['text'].str.len()
    # 하위 20%에 해당하는 임계값 계산
    threshold = df['text_len'].quantile(0.2)
    
    df = df[df['text_len'] > threshold].copy()
    # 임시로 만든 길이 컬럼은 삭제
    df = df.drop(columns=['text_len'])
    after_len_filter = len(df)
    print(f"텍스트 길이 필터링 완료 (기준: {threshold:.1f}자 이하 제거): {after_dropna - after_len_filter}개 삭제됨.")

    # --- Step 3: 중복 Title 제거 ---
    df = df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)
    after_dedup = len(df)
    print(f"중복 Title 제거 완료: {after_len_filter - after_dedup}개 삭제됨.")

    # --- Step 4: 저장 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"폴더 생성 완료: {output_dir}")
        
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n최종 정제 데이터 저장 완료: {output_file}")
    print(f"최종 데이터 개수: {len(df)}개 (전체 대비 {len(df)/initial_count*100:.1f}% 보존)")

    return df

cleaned_df = clean_wikihow_data(input_path)