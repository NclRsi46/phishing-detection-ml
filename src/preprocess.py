"""
preprocess.py

Loads the raw phishing URL dataset, performs cleaning and feature extraction,
and saves the processed data to data/processed/.
"""

import os
import pandas as pd

# Paths and filenames
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

RAW_FILENAME = "phishing_raw.csv"      # from build_dataset.py
PROCESSED_FILENAME = "phishing_processed.csv"

URL_COLUMN = "url"      # MUST match build_dataset output
LABEL_COLUMN = "label"  # MUST match build_dataset output


def load_raw_data() -> pd.DataFrame:
    raw_file = os.path.join(RAW_DATA_PATH, RAW_FILENAME)
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw data file not found at {raw_file}")
    df = pd.read_csv(raw_file)

    if URL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected URL column '{URL_COLUMN}' not found. Columns: {df.columns.tolist()}"
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected label column '{LABEL_COLUMN}' not found. Columns: {df.columns.tolist()}"
        )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw dataframe with URL + label and build numeric features.
    """
    df = df.copy()

    # Ensure URL column is string and handle missing values
    df[URL_COLUMN] = df[URL_COLUMN].fillna("").astype(str)

    # Basic URL-based features
    df["url_length"] = df[URL_COLUMN].apply(len)
    df["num_dots"] = df[URL_COLUMN].str.count(r"\.")
    df["num_hyphens"] = df[URL_COLUMN].str.count("-")
    df["num_slashes"] = df[URL_COLUMN].str.count("/")
    df["num_question_marks"] = df[URL_COLUMN].str.count(r"\?")
    df["num_equals"] = df[URL_COLUMN].str.count("=")
    df["has_at_symbol"] = df[URL_COLUMN].str.contains("@").astype(int)
    df["uses_https"] = df[URL_COLUMN].str.lower().str.startswith("https").astype(int)

    # 👉 IMPORTANT: we do NOT keep the raw URL text here
    feature_columns = [
        "url_length",
        "num_dots",
        "num_hyphens",
        "num_slashes",
        "num_question_marks",
        "num_equals",
        "has_at_symbol",
        "uses_https",
    ]

    processed_df = df[feature_columns + [LABEL_COLUMN]].copy()
    return processed_df


def save_processed_data(df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    df.to_csv(out_path, index=False)
    print(f"Processed data saved to {out_path}")


def main():
    print("Loading raw dataset...")
    df_raw = load_raw_data()
    print(f"Raw shape: {df_raw.shape}")

    print("Engineering features...")
    df_processed = engineer_features(df_raw)
    print(f"Processed shape: {df_processed.shape}")

    print("Saving processed dataset...")
    save_processed_data(df_processed)
    print("Done!")


if __name__ == "__main__":
    main()
