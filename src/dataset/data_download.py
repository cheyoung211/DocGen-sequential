from pathlib import Path
import requests
import pandas as pd

# =========================
# 1. 저장 경로 설정
# =========================
DATA_DIR = Path("data/raw/wikihow")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 다운로드 링크
#    GitHub README에 있는 Box 링크
# =========================
URLS = {
    "wikihowAll.csv": "https://ucsb.box.com/shared/static/ap23l8gafpezf4tq3wapr6u8241zz358.csv",
    "wikihowSep.csv": "https://ucsb.box.com/shared/static/7yq601ijl1lzvlfu4rjdbbxforzd2oag.csv",
}

def download_file(url: str, save_path: Path, chunk_size: int = 1024 * 1024):
    """파일을 스트리밍 방식으로 다운로드."""
    if save_path.exists():
        print(f"[Skip] Already exists: {save_path}")
        return

    print(f"[Download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print(f"[Saved] {save_path}")


def inspect_csv(csv_path: Path, n_preview: int = 3):
    """CSV 기본 정보 확인."""
    print("\n" + "=" * 80)
    print(f"Inspecting: {csv_path}")
    print("=" * 80)

    # 파일 크기
    size_mb = csv_path.stat().st_size / (1024 ** 2)
    print(f"File size: {size_mb:.2f} MB")

    # 로드
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # 결측치 개수
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # 앞부분 샘플 출력
    print("\nPreview rows:")
    print(df.head(n_preview))

    # 텍스트 길이 대략 확인
    for col in ["Title", "Overview", "Headline", "Text"]:
        if col in df.columns:
            lengths = df[col].fillna("").astype(str).str.len()
            print(f"\n[{col}] length stats")
            print(lengths.describe())


# =========================
# 3. 다운로드 실행
# =========================
for filename, url in URLS.items():
    save_path = DATA_DIR / filename
    download_file(url, save_path)

# =========================
# 4. 내용 확인
# =========================
for filename in URLS.keys():
    inspect_csv(DATA_DIR / filename)