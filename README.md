# Phishing Detection Using Machine Learning

This project is part of the CIT251/CYB201 course. The goal is to detect phishing URLs using machine learning techniques and propose strategies to help prevent phishing attacks.

## 👥 Team

- **Nicolas Rossi** (Project Manager)

## 🧾 Dataset

We use a phishing URL dataset (such as PhishTank).  
Data is stored in the following directories:

- `data/raw/` → original dataset files  
- `data/processed/` → cleaned and feature-engineered data  

## 🏗 Project Structure

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
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── .gitignore
