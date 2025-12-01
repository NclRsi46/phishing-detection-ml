"""
build_dataset.py

Builds a labeled dataset for phishing URL detection by combining:
- PhishTank verified phishing URLs (label = 1)
- Benign URLs (label = 0)

Input files (expected in data/raw/):
    - verified_online.csv  (from PhishTank)
    - benign_urls.csv      (your own list of legitimate URLs)

Output file:
    - data/raw/phishing_raw.csv  (with columns: url, label)
"""

import os
import pandas as pd

RAW_DIR = "data/raw"

PHISH_FILE = "verified_online.csv"   # PhishTank CSV
BENIGN_FILE = "benign_urls.csv"      # Your benign URLs CSV
OUTPUT_FILE = "phishing_raw.csv"     # Final merged dataset

URL_COLUMN_NAME = "url"              # We will rename URL column to this
LABEL_COLUMN_NAME = "label"          # 1 = phishing, 0 = legitimate


def load_phishing_data() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, PHISH_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phishing file not found at {path}")

    df = pd.read_csv(path)

    # Try to find the URL column
    candidate_cols = ["url", "URL", "phish_url", "phish_site"]
    url_col = None
    for col in candidate_cols:
        if col in df.columns:
            url_col = col
            break

    if url_col is None:
        raise ValueError(
            f"Could not find a URL column in {PHISH_FILE}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[[url_col]].copy()
    df.rename(columns={url_col: URL_COLUMN_NAME}, inplace=True)

    # Clean
    df[URL_COLUMN_NAME] = df[URL_COLUMN_NAME].fillna("").astype(str)
    df = df[df[URL_COLUMN_NAME] != ""]
    df.drop_duplicates(subset=[URL_COLUMN_NAME], inplace=True)

    # Label as phishing
    df[LABEL_COLUMN_NAME] = 1

    print(f"Loaded {len(df)} phishing URLs from {PHISH_FILE}")
    return df


def load_benign_data() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, BENIGN_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benign file not found at {path}")

    df = pd.read_csv(path)

    candidate_cols = ["url", "URL", "domain", "site"]
    url_col = None
    for col in candidate_cols:
        if col in df.columns:
            url_col = col
            break

    if url_col is None:
        raise ValueError(
            f"Could not find a URL column in {BENIGN_FILE}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[[url_col]].copy()
    df.rename(columns={url_col: URL_COLUMN_NAME}, inplace=True)

    df[URL_COLUMN_NAME] = df[URL_COLUMN_NAME].fillna("").astype(str)
    df = df[df[URL_COLUMN_NAME] != ""]
    df.drop_duplicates(subset=[URL_COLUMN_NAME], inplace=True)

    df[LABEL_COLUMN_NAME] = 0

    print(f"Loaded {len(df)} benign URLs from {BENIGN_FILE}")
    return df


def build_dataset():
    phishing_df = load_phishing_data()
    benign_df = load_benign_data()

    # Balance classes (optional but recommended)
    min_count = min(len(phishing_df), len(benign_df))
    phishing_sample = phishing_df.sample(min_count, random_state=42)
    benign_sample = benign_df.sample(min_count, random_state=42)

    combined = pd.concat([phishing_sample, benign_sample], ignore_index=True)

    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_path = os.path.join(RAW_DIR, OUTPUT_FILE)
    combined.to_csv(out_path, index=False)

    print(f"\nFinal dataset saved to: {out_path}")
    print(f"Total rows: {len(combined)}")
    print(combined[LABEL_COLUMN_NAME].value_counts())


def main():
    print("=== Building labeled phishing dataset ===")
    build_dataset()
    print("Done.")


if __name__ == "__main__":
    main()
