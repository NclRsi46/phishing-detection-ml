# Phishing Domain Detection with Machine Learning

A classifier that scores a URL as phishing or legitimate from its domain name alone, trained on PhishTank phishing reports and Tranco / Cisco Umbrella popular-domain lists.

This started as a team course project for CYB201/CIT251 at Pace University (fall 2025). I wrote the implementation. The first version reported 99% accuracy; that number was wrong, and the section [What I got wrong the first time](#what-i-got-wrong-the-first-time) explains why.

## How it works

1. **Build datasets** (`src/build_dataset.py`) from three raw sources:
   - PhishTank verified-online phishing URLs (label 1), snapshot from December 2025
   - Tranco top 1M domains (label 0), downloaded December 2025
   - Cisco Umbrella top 1M hostnames (label 0), downloaded October 2026

   Phishing URLs are deduplicated, and any whose registered domain is in Tranco is dropped. PhishTank lists abused legitimate sites like `docs.google.com`, `weebly.com`, and `bit.ly`; keeping them would teach the model that Google is phishing. Classes are then balanced by random sampling.

2. **Extract features** (`src/features.py`) from the domain name only. The path, query, and scheme are thrown away for every URL. Two experiments use two levels:

   | Experiment | Benign source | What gets scored | Why |
   |---|---|---|---|
   | **domain** (main) | Tranco | registered domain, e.g. `paypal-login-secure.com` | Tranco lists no subdomains, so subdomains are dropped for both classes |
   | **host** | Umbrella | full hostname, e.g. `paypal.com.account-check.xyz` | Umbrella has real subdomains (32.6% of benign hosts vs 40.5% of phishing) |

   Features (13 at domain level, 14 at host level):

   | Feature | Description |
   |---|---|
   | `length`, `num_hyphens`, `num_digits`, `digit_ratio` | Character counts of the name, without the public suffix |
   | `entropy` | Shannon entropy of the name (random-looking names score higher) |
   | `is_ip` | Host is a raw IPv4/IPv6 address |
   | `is_punycode` | Name contains `xn--` (internationalized, possible lookalike) |
   | `has_brand_word` | A brand like `paypal` appears, but the domain isn't the brand's own |
   | `has_lure_word` | Contains words like `login`, `verify`, `secure`, `account` |
   | `tld_com`, `tld_legacy`, `tld_country`, `tld_other` | TLD group: `.com`; `.net/.org/.edu/.gov/.mil/.int`; two-letter country codes; everything else (`.xyz`, `.top`, `.sbs`, ...) |
   | `subdomain_depth` | Host level only: number of labels before the registered domain |

3. **Split by campaign** (`src/preprocess.py`) with `GroupShuffleSplit` (80/20). The group is the registered domain with every run of digits replaced by `#`, so no domain is in both train and test, and neither are kit domains that differ only by a number (`pl-oferta48213.sbs` and `pl-oferta77120.sbs` both become `pl-oferta#.sbs`).

4. **Train** (`src/train.py`) three models for each experiment: a Logistic Regression baseline and a Random Forest on the features above, and a Logistic Regression on character n-grams of the scored name itself (TF-IDF over 1- to 5-character substrings of `paypal-login-secure.com`, suffix included). The n-gram model checks what the hand-built features miss. Its settings were fixed before the first run and not tuned.

5. **Evaluate** (`src/evaluate.py`) on the held-out test split only. Reports precision, recall, F1, and false positive rate at the default 0.5 threshold, plus recall at a 1% and 5% false positive rate and average precision. Writes `results/metrics.json`, a confusion matrix per model, and a precision-recall curve per experiment. `src/repeat_splits.py` repeats the split with 10 seeds to show how much the numbers move by chance, and `src/ngram_audit.py` checks where the n-gram model's gain comes from.

## Results

Mean and standard deviation over 10 campaign-grouped train/test splits (`results/repeated_splits.json`). Phishing is the positive class.

| Experiment | Model | Precision | Recall | F1 | False positive rate |
|---|---|---|---|---|---|
| domain (Tranco) | Logistic Regression | 71.0% ± 2.3 | 54.1% ± 1.6 | 61.4% ± 1.6 | 22.4% ± 1.3 |
| domain (Tranco) | Random Forest | 68.4% ± 2.2 | 56.4% ± 1.8 | 61.8% ± 1.7 | 26.4% ± 1.6 |
| domain (Tranco) | Char n-gram Logistic Regression | 72.2% ± 2.1 | 63.8% ± 1.4 | 67.7% ± 1.6 | 25.0% ± 1.4 |
| host (Umbrella) | Logistic Regression | 71.0% ± 2.6 | 63.3% ± 3.0 | 66.9% ± 2.4 | 25.4% ± 1.4 |
| host (Umbrella) | Random Forest | 69.2% ± 1.8 | 59.1% ± 2.2 | 63.7% ± 1.6 | 25.9% ± 1.3 |
| host (Umbrella) | Char n-gram Logistic Regression | 76.4% ± 2.3 | 67.6% ± 2.2 | 71.7% ± 1.7 | 20.6% ± 1.6 |

Operating points, over the same 10 splits:

| Experiment | Model | Recall at 1% FPR | Recall at 5% FPR | Average precision |
|---|---|---|---|---|
| domain (Tranco) | Logistic Regression | 14.4% ± 2.1 | 30.7% ± 2.7 | 73.9% ± 2.1 |
| domain (Tranco) | Random Forest | 15.4% ± 2.9 | 29.5% ± 2.6 | 73.2% ± 2.1 |
| domain (Tranco) | Char n-gram Logistic Regression | 28.2% ± 2.1 | 39.9% ± 2.4 | 80.8% ± 1.7 |
| host (Umbrella) | Logistic Regression | 11.1% ± 3.2 | 28.2% ± 3.9 | 74.5% ± 2.8 |
| host (Umbrella) | Random Forest | 13.6% ± 5.3 | 29.9% ± 2.7 | 74.4% ± 2.2 |
| host (Umbrella) | Char n-gram Logistic Regression | 33.6% ± 4.6 | 46.2% ± 2.6 | 84.0% ± 1.7 |

Recall at a fixed false positive rate is the best recall at any threshold whose false positive rate on the test split is at or below the target. The threshold is read off the test split's ROC curve, so this describes how well the model ranks domains; a threshold picked on one split won't land at exactly 1% on new data. With about 1,000 benign test rows, 1% allows about 10 false positives, so that column moves a lot between splits. Ties make it worse: a Random Forest score of 1.0 means all 100 trees agree, and many rows share it. In one host split, 12 benign and 128 phishing domains tie at 1.0, more benign rows than 1% allows, so no threshold can take any of them and that split's recall at 1% is 0. That one split accounts for most of the host forest's ± 5.3; the other nine range from 11.0% to 17.9%.

Precision and average precision depend on the class balance. The test sets are about 50/50; in real traffic phishing is a small fraction of domains, so precision at any threshold would be much lower. Recall at a fixed false positive rate doesn't depend on the balance.

The saved split that `predict.py`'s model and the confusion matrices and precision-recall curves below come from (`results/metrics.json`):

| Experiment | Model | Precision | Recall | F1 | False positive rate |
|---|---|---|---|---|---|
| domain (Tranco) | Logistic Regression | 72.0% | 56.9% | 63.6% | 22.4% |
| domain (Tranco) | Random Forest | 69.0% | 57.8% | 62.9% | 26.2% |
| domain (Tranco) | Char n-gram Logistic Regression | 71.1% | 65.6% | 68.3% | 26.9% |
| host (Umbrella) | Logistic Regression | 71.4% | 63.0% | 66.9% | 26.4% |
| host (Umbrella) | Random Forest | 69.0% | 59.1% | 63.7% | 27.7% |
| host (Umbrella) | Char n-gram Logistic Regression | 78.0% | 69.5% | 73.5% | 20.5% |

A single split can land a couple of points away from the mean, so differences smaller than that between models or experiments don't mean much.

Dataset sizes after cleaning:

| | Phishing | Benign | Train | Test |
|---|---|---|---|---|
| domain | 5,015 | 5,015 | 7,990 | 2,040 |
| host | 5,015 | 5,015 | 8,055 | 1,975 |

The Random Forest doesn't beat the linear baseline, which suggests the signal in these features is mostly simple (longer, more random-looking names with more hyphens and digits, on cheap TLDs). Entropy and length carry about 60% of the forest's feature importance in both experiments. At the default 0.5 threshold the false positive rate is above 20%, far too high for blocking anything. Ranked by score, the top slice of each test split that stays within a 1% false positive rate holds 11 to 15% of the phishing domains on average, but a threshold set to hit 1% on one split won't hit it exactly on new data. That's a narrow high-confidence slice, useful as one signal for triage, not as a filter. A third to over half of that slice is one brand: 32 to 55% of these models' catches at 1% FPR are Allegro-themed, while Allegro is about 8% of the phishing test rows (`results/ngram_audit.json`).

The character n-gram model beats all four hand-feature models on every ranking metric. Recall at 1% FPR goes from 14.4–15.4% to 28.2% on domains and from 11.1–13.6% to 33.6% on hosts, and average precision rises by 7 to 10 points. It reads the same name the hand features are computed from, so the 13 features miss signal that is in the name. At 0.5 its false positive rate is still 21 to 25%. `src/ngram_audit.py` reruns the 10 splits to see where the gain comes from:

- **Mostly the name, partly the TLD.** With the public suffix stripped, the same model still gets 24.0% (domain) and 29.1% (host) recall at 1% FPR. The TLD carries the rest: the hand features put `.cfd`, `.icu`, and `.io` in the same "other" group, while the n-gram model tells them apart, and a model on the suffix alone gets 15.7% and 14.2%, as much as the hand features. Many of the heaviest n-grams are TLDs (at domain level, `.cfd` and `.icu` toward phishing, `.ru`, `.jp`, and `.gov` toward benign; `results/metrics.json`). TLD abuse moves fast, so that part is the most likely to decay.
- **Not one kit.** All three models already catch 96 to 99% of Allegro-themed test domains at 0.5. Against the hand-feature Logistic Regression, the gain is on the other phishing rows: recall 50.1% to 60.6% on domains and 60.2% to 64.8% on hosts, and Allegro makes up less of the n-gram model's 1% slice (27% and 23%). Only Allegro was checked; other kits that vary letters can still sit on both sides of the split.
- **The known artifacts show up.** `.1`, from IP hosts that only phishing has, weighs toward phishing, and on hosts `cdn` and `api`, from Umbrella's infrastructure hostnames, weigh toward benign.

<p>
  <img src="results/confusion_domain_random_forest.png" width="380" alt="Confusion matrix, domain level, random forest">
  <img src="results/confusion_host_logistic_regression.png" width="380" alt="Confusion matrix, host level, logistic regression">
</p>

<p>
  <img src="results/pr_curve_domain.png" width="380" alt="Precision-recall curve, domain level">
  <img src="results/pr_curve_host.png" width="380" alt="Precision-recall curve, host level">
</p>

`src/sanity_check.py` scores a fixed list of real and made-up URLs. Real legitimate domains it scores high: `en.wikipedia.org` (0.63 on the domain model) and `login.microsoftonline.com` (0.53, pushed up because "microsoft" appears in a domain that isn't `microsoft.com`). The made-up phishing URLs all score high, but they were written to look like phishing, so they say little about recall. The test-set numbers above are the honest ones.

## What I got wrong the first time

The first version of this project ([`results/original_run.txt`](results/original_run.txt)) reported these held-out numbers for a Random Forest on 95,972 URLs:

| Accuracy | Precision | Recall | False positive rate |
|---|---|---|---|
| 99.46% | 99.80% | 99.12% | 0.20% |

They meant nothing. The benign URLs were built by putting `https://` in front of each Tranco domain, so every benign example was a bare homepage. Phishing URLs from PhishTank are full links:

| | Starts with `https` | Has `/` after the host | Has a real path | Has `?` query |
|---|---|---|---|---|
| Benign | 100% | 0% | 0% | 0% |
| Phishing | 95.3% | 97.0% | 67.0% | 20.9% |

So the model learned "anything after the domain = phishing." `num_slashes` alone carried 60% of the feature importance. Real legitimate URLs all scored as phishing, and an obvious phishing domain scored as safe:

```text
P(phish)=1.00  https://www.amazon.com/gp/your-account/order-history?ref_=ya_d_c_yo
P(phish)=1.00  https://en.wikipedia.org/wiki/Phishing
P(phish)=1.00  https://www.chase.com/personal/checking
P(phish)=0.99  http://paypal.com
P(phish)=0.11  https://paypal-login-secure.com
```

Three more problems:

- **Random row split.** PhishTank often has many URLs on one domain, so the same domain landed in both train and test. Only 5,875 of 47,994 phishing URLs have a distinct registered domain.
- **Evaluating on training data.** `evaluate.py` scored the model on the full dataset, training rows included.
- **The same leak one level down.** Switching to hostname features wasn't enough on its own: Tranco has no subdomains (0%), while 84% of PhishTank hostnames have one, so subdomain count would have given the label away the same way. That's why the domain experiment scores the registered domain only, and why the host experiment uses Umbrella, which does have subdomains.

A review of the fixed version found two smaller issues, now fixed:

- **Kit domains on both sides of the split.** Kits register domains that differ only by a number. Grouping by registered domain alone let 74 of these into the domain test set while their siblings were in training. The Random Forest caught 96% of them, against 52.5% of the other phishing test rows. The split is now grouped by the digit-collapsed pattern. Averaged over 10 splits, the change moved recall by less than one standard deviation, so the effect on the headline numbers was small, but the old split was still measuring memorization for those rows.
- **Unequal deduplication in the host experiment.** Benign hosts were capped at one per registered domain and phishing hosts weren't, so a few site builders contributed dozens of phishing rows each. Both sides are now capped the same way.

The lesson: when two classes come from different sources, check every column for differences in how the data was collected before trusting any score. A near-perfect result on a hard problem is a reason to look for leakage, not to celebrate.

## Limitations

- **Domain-only features can't see path tricks.** A phishing page on a compromised legitimate site, or on a free host like `weebly.com` or `web.app`, is invisible to this model, and those were dropped from training (85% of PhishTank rows had a registered domain in Tranco).
- **PhishTank is skewed.** 8.3% of phishing domains (418 of 5,015) imitate one brand, Allegro, mostly from a few kits. The campaign grouping only merges domains that differ by digits; kits that vary letters (`298278sdhsprywanie.cfd`) can still sit on both sides of the split.
- **Benign means popular.** Tranco and Umbrella are popularity lists. The model partly learns "popular vs obscure," and Umbrella also includes CDN and API hostnames that don't look like sites people visit.
- **`is_ip` only appears in phishing.** Neither benign list has IP hosts, so the model treats any IP as phishing.
- **The data snapshots are 10 months apart.** PhishTank is from December 2025 and Umbrella from October 2026. Naming and TLD trends that shifted in between can look like a phishing signal, and Umbrella's overlap with these phishing domains is understated because most of them stopped getting DNS traffic long before the Umbrella snapshot.
- **Lexical models decay.** Attackers change naming patterns and TLDs, and a model trained on one snapshot gets worse over time.
- **Small dataset.** About 10,000 rows per experiment after deduplication.

## How to run

Tested with Python 3.9.6.

```bash
git clone https://github.com/NclRsi46/phishing-detection-ml.git
cd phishing-detection-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `data/raw/` (`mkdir -p data/raw`) and put the raw data in it (not committed):

- `verified_online.csv`: PhishTank's verified online phishing CSV (phishtank.org)
- `tranco.csv`: Tranco top 1M list, `rank,domain` with no header (tranco-list.eu)
- `umbrella.csv` (optional, for the host experiment): Cisco Umbrella top 1M, `rank,hostname` with no header, from `https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip`

Run the pipeline from the project root:

```bash
python src/build_dataset.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/repeat_splits.py
python src/ngram_audit.py
python src/sanity_check.py
python tests/test_features.py
python tests/test_split.py
python tests/test_metrics.py
python tests/test_train.py
python tests/test_predict.py
```

Score a single URL with the domain-level Random Forest:

```bash
python src/predict.py "https://paypal-login-secure.com/x?y=1"
```

```text
URL:          https://paypal-login-secure.com/x?y=1
Scored:       paypal-login-secure.com (domain level, path and query ignored)
P(phishing):  0.990
Prediction:   Phishing
```

Use `--model models/host_random_forest.pkl` (or any other model in `models/`) to score with a different one.

## Project structure

```text
phishing-detection-ml/
├── data/
│   ├── raw/            source CSVs (not committed)
│   └── processed/      datasets and train/test splits (not committed)
├── models/             trained models (not committed)
├── results/
│   ├── metrics.json
│   ├── repeated_splits.json
│   ├── ngram_audit.json
│   ├── confusion_*.png
│   ├── pr_curve_*.png
│   └── original_run.txt
├── src/
│   ├── features.py      feature extraction shared by preprocess and predict
│   ├── build_dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── repeat_splits.py
│   ├── ngram_audit.py
│   └── sanity_check.py
├── tests/
│   ├── test_features.py
│   ├── test_split.py
│   ├── test_metrics.py
│   ├── test_train.py
│   └── test_predict.py
└── requirements.txt
```
