"""
preprocess.py

Loads the raw phishing URL dataset, performs cleaning and feature extraction,
and saves the processed data to data/processed/.
"""

import os
import pandas as pd

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
RAW_FILENAME = "phishing_raw.csv"      # rename to your actual dataset filename
PROCESSED_FILENAME = "phishing_processed.csv"


def load_raw_data():
    raw_file = os.path.join(RAW_DATA_PATH, RAW_FILENAME)
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw data file not found at {raw_file}")
    return pd.read_csv(raw_file)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for real feature engineering.
    """
    processed_df = df.copy()
    return processed_df


def save_processed_data(df: pd.DataFrame):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    df.to_csv(out_path, index=False)
    print(f"Processed data saved to {out_path}")


def main():
    print("Loading raw dataset...")
    df = load_raw_data()
    print(f"Raw shape: {df.shape}")

    print("Engineering features...")
    processed = engineer_features(df)

    print("Saving processed dataset...")
    save_processed_data(processed)
    print("Done!")


if __name__ == "__main__":
    main()
