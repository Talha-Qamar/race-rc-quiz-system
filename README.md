# RACE Reading Comprehension Quiz System

Classical-ML based reading comprehension and quiz generation pipeline built on the RACE dataset.

This repository contains an end-to-end project that:
- verifies correct answers for MCQ items (Model A),
- generates distractors and hints (Model B),
- exposes both pipelines through a Streamlit interface.

## Why This Project

This project demonstrates practical ML engineering skills for NLP systems:
- text preprocessing and feature engineering,
- supervised, semi-supervised, and unsupervised experimentation,
- model evaluation, calibration, and comparison,
- reproducible training/inference scripts,
- interactive productization via Streamlit.

## Tech Stack

- Python 3.10+
- scikit-learn, scipy, numpy, pandas
- nltk, gensim
- matplotlib
- streamlit

## Repository Structure

```text
src/                  Core training, inference, and experiment scripts
ui/                   Streamlit app and UI helper components
data/                 Dataset placeholders and storage policy docs
models/               Model placeholders and artifact policy docs
notebooks/            EDA and exploratory work
scripts/              Utility scripts
report/               Report assets
```

## Key Components

### Model A: Answer Verification

Primary inference entrypoint:

```bash
python3 src/inference.py --model ensemble --split test
python3 src/inference.py --model random_forest --split val
```

### Model B: Distractor + Hint Generation

Core pipeline modules:
- `src/model_b_distractor.py`
- `src/model_b_hint.py`
- `src/model_b_inference.py`

### Streamlit Application

Run:

```bash
python3 -m streamlit run ui/app.py
```

## Getting Started

### 1. Clone

```bash
git clone https://github.com/Talha-Qamar/race-rc-quiz-system.git
cd race-rc-quiz-system
```

### 2. Create Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Data and Artifacts

This repo intentionally does not track large data/model binaries.

See:
- `data/README.md`
- `models/README.md`

### 4. Run

```bash
# example inference
python3 src/inference.py --model ensemble --split val

# launch UI
python3 -m streamlit run ui/app.py
```

## Reproducibility Notes

- Scripts assume the project root as working directory.
- Processed arrays (`.npz`, `.npy`) and trained artifacts (`.joblib`, `.pkl`) are expected in local `data/` and `models/` directories.
- Keep generated artifacts out of Git and publish them via external artifact storage.

## Handling Large Files (Recommended)

For a CV-ready and collaboration-friendly repository:

1. Keep this code repository lightweight (code + docs only).
2. Store datasets in an external dataset host (Kaggle Dataset, Google Drive, or S3).
3. Store trained models in release assets, Hugging Face Hub, or a dedicated model bucket.
4. If you need versioned large artifacts, adopt DVC for data/model lineage.

## Suggested Repository Rename

Current name is good, but this format is more portfolio-friendly:

`race-rc-quiz-system-ml`

Alternative:

`reading-comprehension-quiz-generation`

## License

MIT License (see `LICENSE`).

## Author

Talha Qamar
