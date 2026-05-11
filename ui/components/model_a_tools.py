"""Utilities shared by the Streamlit UI and the Model A inference path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

try:
	from src.features import build_verification_feature_matrix
except ImportError:  # pragma: no cover - fallback when run from src scripts
	from features import build_verification_feature_matrix


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "model_a" / "traditional"
UNSUPERVISED_DIR = ROOT_DIR / "models" / "model_a" / "unsupervised"

MODEL_FILES = {
	"random_forest": [MODELS_DIR / "model_a_random_forest.pkl", MODELS_DIR / "model_a_random_forest.joblib", MODELS_DIR / "model_a_random_forest_intensive_tuned.pkl"],
	"logistic_regression": [MODELS_DIR / "model_a_logistic_regression.pkl", MODELS_DIR / "model_a_logistic_regression.joblib", MODELS_DIR / "model_a_logistic_regression_intensive_tuned.pkl"],
	"svm_calibrated": [MODELS_DIR / "model_a_svm_calibrated.pkl", MODELS_DIR / "model_a_svm_calibrated.joblib", MODELS_DIR / "model_a_linearsvc_calibrated.joblib"],
	"ensemble": [MODELS_DIR / "model_a_ensemble.pkl", MODELS_DIR / "model_a_ensemble.joblib", MODELS_DIR / "model_a_ensemble_voting.joblib"],
}

METADATA_PATH = MODELS_DIR / "model_metadata.json"

UNSUPERVISED_FILES = {
	"kmeans": UNSUPERVISED_DIR / "kmeans_results.json",
	"gmm": UNSUPERVISED_DIR / "gmm_results.json",
	"label_propagation": UNSUPERVISED_DIR / "label_propagation_results.json",
}


def _first_existing(paths: list[Path]) -> Path:
	"""Return the first existing path from a list of candidate files."""
	for path in paths:
		if path.exists():
			return path
	raise FileNotFoundError(f"Missing artifact. Checked: {', '.join(str(path) for path in paths)}")


def normalize_text(value: object) -> str:
	"""Normalize text for simple display and option construction."""
	if pd.isna(value):
		return ""
	text = str(value).lower()
	for symbol in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
		text = text.replace(symbol, " ")
	return " ".join(text.split())


def load_metadata() -> dict[str, Any]:
	"""Load the model metadata JSON if it exists."""
	if METADATA_PATH.exists():
		try:
			return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
		except Exception:
			pass
	return {}


def load_vectorizer():
	"""Load the fitted TF-IDF vectorizer used by Model A."""
	candidate = _first_existing([
		ROOT_DIR / "models" / "model_a" / "tfidf_vectorizer.pkl",
		PROCESSED_DIR / "artifacts" / "tfidf_vectorizer.joblib",
	])
	return joblib.load(candidate)


def load_question_ranker():
	"""Load the question ranker used by the generation tab."""
	candidate = MODELS_DIR / "question_ranker.pkl"
	if not candidate.exists():
		raise FileNotFoundError(f"Missing question ranker: {candidate}")
	return joblib.load(candidate)


def load_model(model_name: str):
	"""Load a trained Model A estimator by name."""
	model_name = model_name.lower().strip()
	if model_name not in MODEL_FILES:
		raise ValueError("model_name must be random_forest, logistic_regression, svm_calibrated, or ensemble")
	model_path = _first_existing(MODEL_FILES[model_name])
	return joblib.load(model_path)


def load_sample_frame(split: str = "test") -> pd.DataFrame:
	"""Load a cleaned split from the processed data directory."""
	split = split.lower().strip()
	frame_path = PROCESSED_DIR / f"{split}_clean.csv"
	if not frame_path.exists():
		raise FileNotFoundError(f"Missing split file: {frame_path}")
	return pd.read_csv(frame_path)


def load_unsupervised_results(result_name: str) -> dict[str, Any]:
	"""Load one of the unsupervised or semi-supervised result payloads."""
	result_name = result_name.lower().strip()
	if result_name not in UNSUPERVISED_FILES:
		raise ValueError(f"result_name must be one of: {', '.join(sorted(UNSUPERVISED_FILES))}")
	result_path = UNSUPERVISED_FILES[result_name]
	if not result_path.exists():
		raise FileNotFoundError(f"Missing unsupervised results: {result_path}")
	return json.loads(result_path.read_text(encoding="utf-8"))


def build_option_frame(article: str, question: str, options: dict[str, str]) -> pd.DataFrame:
	"""Create the four-row option frame used by Model A scoring.

	Args:
		article (str): Passage text.
		question (str): Question text.
		options (dict[str, str]): Mapping from option labels to answer text.

	Returns:
		pd.DataFrame: One row per answer option.
	"""
	article_clean = normalize_text(article)
	question_clean = normalize_text(question)
	rows: list[dict[str, Any]] = []
	for label in ("A", "B", "C", "D"):
		option_text = options.get(label, "")
		option_clean = normalize_text(option_text)
		rows.append(
			{
				"option_label": label,
				"option_text": option_text,
				"option_clean": option_clean,
				"article": article,
				"question": question,
				"article_clean": article_clean,
				"question_clean": question_clean,
				"combined_text": " ".join(part for part in [article_clean, question_clean, option_clean] if part),
			}
		)
	return pd.DataFrame(rows)


def build_prediction_matrix(option_frame: pd.DataFrame, vectorizer) -> csr_matrix:
	"""Build the combined sparse+dense feature matrix for inference."""
	text_matrix = vectorizer.transform(option_frame["combined_text"].fillna("").tolist())
	dense_matrix = build_verification_feature_matrix(option_frame, vectorizer)
	return hstack([text_matrix, csr_matrix(dense_matrix)], format="csr")


def predict_option_probabilities(model_name: str, article: str, question: str, options: dict[str, str]) -> pd.DataFrame:
	"""Score the four answer options with a trained Model A estimator."""
	vectorizer = load_vectorizer()
	model = load_model(model_name)
	option_frame = build_option_frame(article, question, options)
	matrix = build_prediction_matrix(option_frame, vectorizer)
	proba = model.predict_proba(matrix)[:, 1]
	result = option_frame.copy()
	result["prob_correct"] = proba
	result["predicted_label"] = (result["prob_correct"] >= 0.49).astype(int)
	return result.sort_values("prob_correct", ascending=False).reset_index(drop=True)


def summarize_prediction(result: pd.DataFrame) -> dict[str, Any]:
	"""Summarize a ranked Model A prediction table."""
	top_row = result.iloc[0]
	return {
		"predicted_option": str(top_row["option_label"]),
		"predicted_text": str(top_row["option_text"]),
		"confidence": float(top_row["prob_correct"]),
		"ranked_options": [
			{
				"option": str(row.option_label),
				"text": str(row.option_text),
				"prob_correct": float(row.prob_correct),
			}
			for row in result.itertuples(index=False)
		],
	}
