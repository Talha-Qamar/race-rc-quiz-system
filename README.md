# RACE Reading Comprehension Quiz System

This project is a classical machine learning system for reading comprehension on the RACE dataset. It supports answer verification, distractor generation, hint generation, and an interactive Streamlit interface.

## Project Summary

The repository contains two core pipelines:

1. Model A verifies the correct answer for a multiple-choice reading-comprehension question.
2. Model B generates distractors and hints for the same question-answer setting.

The codebase uses classical NLP and ML techniques such as TF-IDF, handcrafted verification features, calibration, ranking, and supervised evaluation.

## Repository Map

```text
src/                  Training, inference, preprocessing, and evaluation code
ui/                   Streamlit app and UI helper components
data/                 Local data directory and storage notes
models/               Local model directory and artifact notes
notebooks/            EDA and experimentation notebooks
scripts/              Utility scripts
report/               Report files and project write-up assets
train_*.py            Training entry points for corrected/tuned models
run_unsupervised_methods.sh  Script for unsupervised experiments
```


## Project Architecture


Dataset
   │
Preprocessing
   │
TF-IDF + Feature Extraction
   │
Model A (Answer Verification)
Model B (Distractor + Hint Generation)
   │
Streamlit Interface



## Main Entry Points

- [src/inference.py](src/inference.py): Model A inference and evaluation CLI.
- [ui/app.py](ui/app.py): Streamlit application.
- [src/model_b_inference.py](src/model_b_inference.py): Model B generation workflow.
- [train_corrected_models.py](train_corrected_models.py): Retrain corrected models.
- [train_tuned_models.py](train_tuned_models.py): Retrain tuned models.

## Requirements

- Python 3.10 or newer
- pip
- A virtual environment

## Clone and Set Up

Run these commands from the repository root after cloning:

```bash
git clone https://github.com/Talha-Qamar/race-rc-quiz-system.git
cd race-rc-quiz-system

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Project Data and Artifacts

The repository keeps large datasets and trained model binaries out of Git so the project stays easy to clone and manage.

Place your local files in these folders:

```text
data/raw/
data/processed/
models/model_a/traditional/
models/model_b/traditional/
```

The folder policies are documented in:
- [data/README.md](data/README.md)
- [models/README.md](models/README.md)

If you are starting from a fresh clone, make sure the processed arrays and trained model files are copied into those paths before running inference or the UI.

## How To Use the Project

### 1. Run Model A inference

```bash
python3 src/inference.py --model ensemble --split val
python3 src/inference.py --model random_forest --split test
```

### 2. Launch the Streamlit app

```bash
python3 -m streamlit run ui/app.py
```

### 3. Retrain models

```bash
python3 train_corrected_models.py
python3 train_tuned_models.py
```

### 4. Run unsupervised experiments

```bash
bash run_unsupervised_methods.sh
```

## Folder Roles

### `src/`

Contains the project logic for preprocessing, feature engineering, inference, training, and evaluation.

### `ui/`

Contains the Streamlit interface used to demonstrate the project interactively.

### `data/`

Contains local dataset files and processed arrays. This folder is excluded from Git to avoid large commits.

### `models/`

Contains trained model artifacts, evaluation outputs, and model cards. Large binaries should remain local or be published separately.

### `notebooks/`

Contains exploratory notebooks used during development.

## Working With Your Own Data

If you want to adapt this project for another dataset, keep the folder structure and replace the data files with your own CSV or processed artifacts. Then retrain the models using the training scripts and update the evaluation outputs accordingly.

## Common Issues

- Run commands from the repository root so relative paths resolve correctly.
- If a script cannot find data, confirm that the files exist in `data/processed/`.
- If a model cannot load, confirm that the artifact exists in the matching `models/.../traditional/` folder.
- If Streamlit fails to start, confirm that the active environment has `streamlit` installed.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).

## Author

Talha Qamar
Taaha Zaman Khan
