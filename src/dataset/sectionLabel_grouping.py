import pandas as pd
import os

# 파일 경로 설정
input_file = './dataset/processed/wikihow_cleaned.csv'
output_file = './dataset/processed/wikihow_grouped.csv'

def group_section_labels_refined(df):
    print("스크린샷 기반 Refined Grouping 시작...")
    
    def transform_label(label):
        label = str(label).lower().strip()
        
        # 1. 핵심 절차 (Steps)
        if any(x in label for x in ['steps', 'method', 'action', 'forward', 'procedure', 'part']):
            return 'Steps'
        
        # 2. 시작 및 준비 (Preparation)
        elif any(x in label for x in ['getting started', 'preparation', 'preparing', 'setting up', 'tutorial']):
            return 'Preparation'
        
        # 3. IT/플랫폼 환경 (Platform-specific)
        elif any(x in label for x in ['windows', 'android', 'mac', 'ios', 'linux']):
            return 'Environment'
        
        # 4. 건강 및 라이프스타일 (Lifestyle)
        elif any(x in label for x in ['lifestyle', 'yourself', 'diet', 'remedies', 'game']):
            return 'Lifestyle'
        
        # 5. 문제 해결 및 도움말 (Support)
        elif any(x in label for x in ['troubleshooting', 'mistakes', 'help', 'q&a', 'warning']):
            return 'Support'
        
        else:
            return 'Others'

    # 새로운 컬럼 생성
    df['groupedLabel'] = df['sectionLabel'].apply(transform_label)
    
    # 결과 확인
    print("\n[업데이트된 그룹화 분포]")
    print(df['groupedLabel'].value_counts())
    
    return df

# 실행 로직
if os.path.exists(input_file):
    df = pd.read_csv(input_file)
    grouped_df = group_section_labels_refined(df)
    
    # 저장
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    grouped_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n파일 저장 완료: {output_file}")
else:
    print("입력 파일을 찾을 수 없습니다.")