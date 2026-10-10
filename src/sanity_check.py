"""
sanity_check.py

Scores a fixed list of real legitimate URLs and made-up phishing-style URLs
with both random forests. The first version of this project scored every
legitimate URL with a path as phishing; this list is how that was caught.

A "*" means that domain (or host) is in the model's training set, so its
score says little about generalization.

Usage:
    python src/sanity_check.py
"""

import os

import pandas as pd

from features import hostname, registered_domain
from predict import load_model, score

LEGITIMATE = [
    "https://www.amazon.com/gp/your-account/order-history?ref_=ya_d_c_yo",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.chase.com/personal/checking",
    "http://paypal.com",
    "https://github.com/NclRsi46",
    "https://www.pace.edu/admissions",
    "https://login.microsoftonline.com/common/oauth2/authorize",
    "https://www.irs.gov/refunds",
]

PHISHING_STYLE = [
    "https://paypal-login-secure.com",
    "http://secure-account-verify-amazon.xyz/login",
    "http://192.168.4.20/paypal/login.php",
    "http://allegrolokalnie.oferta-48213.sbs/",
    "http://apple-id-unlock-support.top/",
    "https://xn--pypal-4ve.com/signin",
    "http://wellsfargo-online-banking.info/",
    "http://paypal.com.account-check.xyz/login",
]

MODELS = {
    "domain": "models/domain_random_forest.pkl",
    "host": "models/host_random_forest.pkl",
}


def training_keys(level: str) -> set:
    path = f"data/processed/{level}_train.csv"
    column = "domain" if level == "domain" else "host"
    return set(pd.read_csv(path, usecols=[column])[column]) if os.path.exists(path) else set()


def main():
    bundles = {level: load_model(path) for level, path in MODELS.items() if os.path.exists(path)}
    seen = {level: training_keys(level) for level in bundles}

    print(f"{'expected':11}" + "".join(f"{level + ' RF':>11}" for level in bundles) + "   url")
    for expected, urls in (("legitimate", LEGITIMATE), ("phishing", PHISHING_STYLE)):
        for url in urls:
            cells = []
            for level, bundle in bundles.items():
                key = registered_domain(url) if level == "domain" else hostname(url)
                mark = "*" if key in seen[level] else " "
                cells.append(f"{score(bundle, url):10.2f}{mark}")
            print(f"{expected:11}" + "".join(cells) + f"   {url}")


if __name__ == "__main__":
    main()
