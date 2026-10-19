import pandas as pd
from pathlib import Path

# DATA_DIR = Path("dataset/raw/wikihow")

# df_all = pd.read_csv(DATA_DIR / "wikihowAll.csv")
# df_sep = pd.read_csv(DATA_DIR / "wikihowSep.csv")

# print("\n===== wikihowAll sample =====")
# row = df_all.iloc[0]
# print("\ntitle:\n", row["title"])
# print("\nheadline:\n", row["headline"][:1000])
# print("\ntext:\n", row["text"][:2000])

# print("\n===== wikihowSep sample =====")
# row = df_sep.iloc[0]
# print("title:\n", row["title"])
# if "overview" in df_sep.columns:
#     print("\noverview:\n", str(row["overview"])[:1000])
# print("\nheadline:\n", row["headline"][:1000])
# print("\ntext:\n", row["text"][:2000])
# if "sectionLabel" in df_sep.columns:
#     print("\nsectionLabel:\n", row["sectionLabel"][:2000])

# # 3. 결측치(Missing Values) 확인
# print("\n컬럼별 결측치 개수:")
# missing_values = df_sep.isnull().sum()
# print(missing_values)

# 4. sectionLabel 분포 확인 (상위 20개)
print("\n주요 sectionLabel 분포 (Top 20):")
section_counts = df_sep['sectionLabel'].value_counts()
print(section_counts)