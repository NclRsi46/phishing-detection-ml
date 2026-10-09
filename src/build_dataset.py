"""
build_dataset.py

Builds the labeled datasets from the raw sources:
    data/raw/verified_online.csv   PhishTank verified phishing URLs   (label 1)
    data/raw/tranco.csv            Tranco top 1M domains, no header   (label 0)
    data/raw/umbrella.csv          Cisco Umbrella top 1M hostnames    (label 0, optional)

Two datasets are written to data/processed/:
    dataset_domain.csv   PhishTank vs Tranco, one row per registered domain
    dataset_host.csv     PhishTank vs Umbrella, one row per hostname
                         (skipped if umbrella.csv is missing)

Each cleaning step prints how many rows it removed.
"""

import os

import pandas as pd

from features import hostname, registered_domain

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

PHISHTANK_FILE = os.path.join(RAW_DIR, "verified_online.csv")
TRANCO_FILE = os.path.join(RAW_DIR, "tranco.csv")
UMBRELLA_FILE = os.path.join(RAW_DIR, "umbrella.csv")

SEED = 42


def step(df: pd.DataFrame, new_df: pd.DataFrame, message: str) -> pd.DataFrame:
    print(f"  {message}: removed {len(df) - len(new_df):,}, left {len(new_df):,}")
    return new_df


def load_phishtank() -> pd.DataFrame:
    df = pd.read_csv(PHISHTANK_FILE, usecols=["url"])
    df["url"] = df["url"].fillna("").astype(str)
    print(f"PhishTank: {len(df):,} rows")
    df["host"] = df["url"].map(hostname)
    df["domain"] = df["url"].map(registered_domain)
    return step(df, df[df["domain"] != ""], "no valid domain")


def load_ranked_list(path: str, name: str) -> pd.DataFrame:
    # Tranco and Umbrella are both "rank,name" with no header.
    df = pd.read_csv(path, header=None, names=["rank", "url"])
    df["url"] = df["url"].fillna("").astype(str)
    print(f"{name}: {len(df):,} rows")
    df["host"] = df["url"].map(hostname)
    df["domain"] = df["url"].map(registered_domain)
    return step(df, df[df["host"] != ""], "no valid hostname")


def balance(phish: pd.DataFrame, benign: pd.DataFrame) -> pd.DataFrame:
    n = min(len(phish), len(benign))
    print(f"  Balancing to {n:,} per class")
    phish = phish.sample(n, random_state=SEED).assign(label=1)
    benign = benign.sample(n, random_state=SEED).assign(label=0)
    combined = pd.concat([phish, benign], ignore_index=True)
    return combined.sample(frac=1.0, random_state=SEED)[["url", "host", "domain", "label"]]


def build_domain_dataset(phish: pd.DataFrame, tranco: pd.DataFrame) -> pd.DataFrame:
    print("\n== Domain dataset (PhishTank vs Tranco) ==")
    tranco_domains = set(tranco["domain"])

    print("Phishing:")
    p = step(phish, phish.drop_duplicates("domain"), "duplicate registered domain")
    # PhishTank lists abused legitimate sites (docs.google.com, weebly.com,
    # bit.ly). Keeping them would teach the model that google.com is phishing.
    p = step(p, p[~p["domain"].isin(tranco_domains)], "registered domain is in Tranco")

    print("Benign:")
    b = step(tranco, tranco.drop_duplicates("domain"), "duplicate registered domain")
    b = step(b, b[~b["domain"].isin(set(p["domain"]))], "registered domain is in PhishTank")

    return balance(p, b)


def build_host_dataset(phish: pd.DataFrame, tranco: pd.DataFrame, umbrella: pd.DataFrame) -> pd.DataFrame:
    print("\n== Host dataset (PhishTank vs Umbrella) ==")
    tranco_domains = set(tranco["domain"])

    print("Phishing:")
    p = step(phish, phish.drop_duplicates("host"), "duplicate hostname")
    p = step(p, p[~p["domain"].isin(tranco_domains)], "registered domain is in Tranco")

    print("Benign:")
    b = step(umbrella, umbrella.drop_duplicates("host"), "duplicate hostname")
    # Umbrella ranks by DNS traffic, so it also contains phishing domains.
    b = step(b, b[~b["domain"].isin(set(p["domain"]))], "registered domain is in PhishTank")
    # A handful of domains (CDNs, Google, Microsoft) have thousands of
    # hostnames each. One random host per domain keeps them from dominating
    # and matches the phishing side, which is about one host per domain.
    one_per_domain = b.sample(frac=1.0, random_state=SEED).drop_duplicates("domain")
    b = step(b, one_per_domain, "extra hostnames of the same registered domain")

    return balance(p, b)


def save(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows to {path}")


def main():
    phish = load_phishtank()
    tranco = load_ranked_list(TRANCO_FILE, "Tranco")

    save(build_domain_dataset(phish, tranco), "dataset_domain.csv")

    if os.path.exists(UMBRELLA_FILE):
        umbrella = load_ranked_list(UMBRELLA_FILE, "Umbrella")
        save(build_host_dataset(phish, tranco, umbrella), "dataset_host.csv")
    else:
        print(f"\n{UMBRELLA_FILE} not found, skipping the host dataset.")


if __name__ == "__main__":
    main()
