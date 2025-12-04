import pandas as pd

INPUT_FILE = "data/raw/tranco.csv"     # rename your downloaded file to this
OUTPUT_FILE = "data/raw/benign_urls.csv"

def main():
    print("Loading Tranco list...")
    df = pd.read_csv(INPUT_FILE, header=None, names=["rank", "domain"])

    print(f"Total domains loaded: {len(df)}")

    # Convert to valid URLs
    df["url"] = "https://" + df["domain"]

    # Keep only the URL column for your dataset format
    benign = df[["url"]]

    # Save dataset
    benign.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(benign)} legitimate URLs → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
