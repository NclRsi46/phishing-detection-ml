# Phishing Detection Using Machine Learning

This project is part of the CIT251/CYB201 course. The goal is to detect phishing URLs using machine learning techniques and propose strategies to help prevent phishing attacks.

## Team
- **Nicolas Rossi** (Project Manager)

## Dataset
We use phishing URLs obtained from the PhishTank “Verified Online” CSV feed, combined with a curated list of legitimate URLs.  

Data is stored in the following directories:
- `data/raw/` → original dataset files  
- `data/processed/` → cleaned and feature-engineered data  

## Project Structure
```text
phishing-detection-ml/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── proposal.pdf
│   ├── abstract.pdf
│   ├── final_report.pdf
│   └── gantt_chart.png
├── models/
│   └── phishing_model.pkl
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── build_dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── .gitignore
```

Usage

This section explains how to prepare the dataset, preprocess it, train the machine learning model, and evaluate the results. All commands should be run from the project’s root directory.

1. Set Up the Environment

Clone the repository and create a Python virtual environment:

git clone <your-repository-url>
cd phishing-detection-ml

python3 -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt

2. Add Raw Data Files

Place the required files into:

data/raw/


Required files:

verified_online.csv – downloaded from the PhishTank “Verified Online” CSV feed

benign_urls.csv – contains legitimate URLs with a single column named url

After adding them, the directory should contain:

data/raw/verified_online.csv
data/raw/benign_urls.csv

3. Build the Labeled Dataset

Run the dataset builder:

python src/build_dataset.py


This generates:

data/raw/phishing_raw.csv


With the columns:

url – the URL

label – 1 for phishing, 0 for legitimate

4. Preprocess the Dataset

Convert each URL into numeric features:

python src/preprocess.py


This generates:

data/processed/phishing_processed.csv


containing:

url_length

num_dots

num_hyphens

num_slashes

num_question_marks

num_equals

has_at_symbol

uses_https

label

5. Train the Machine Learning Model

Train the Random Forest classifier:

python src/train.py


This step loads the processed dataset, performs a train/test split, trains the model, prints evaluation metrics, and saves the trained model to:

models/phishing_model.pkl

6. Evaluate the Model

Evaluate the model on the full processed dataset:

python src/evaluate.py


This prints precision, recall, f1-score, and accuracy for both phishing and legitimate URLs.

7. Retraining the Model

If the raw data changes, rerun the steps:

python src/build_dataset.py
python src/preprocess.py
python src/train.py
python src/evaluate.py


This regenerates the dataset, extracts features, retrains the model, and re-evaluates performance.

Prediction Tool (predict.py)

This project includes a command-line prediction tool that allows a user to check whether a specific URL is phishing or legitimate. The script loads the trained machine learning model and applies the same feature extraction used during preprocessing.

Using the Prediction Tool

To check a single URL, run:

python src/predict.py "https://example.com"


Example legitimate URL:

python src/predict.py "https://www.google.com"


Example phishing URL:

python src/predict.py "http://paypal-security-alert.com/verify-account"


The tool prints:

The URL being analyzed

The model’s prediction (Legitimate or Phishing)

Estimated probability for each class

This allows users to quickly test suspicious links without modifying any code.