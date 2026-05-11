"""Inference utilities for Model A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from features import build_verification_feature_matrix


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "model_a" / "traditional"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

MODEL_FILES = {
	"random_forest": [MODELS_DIR / "model_a_random_forest.pkl", MODELS_DIR / "model_a_random_forest.joblib"],
	"logistic_regression": [MODELS_DIR / "model_a_logistic_regression.pkl", MODELS_DIR / "model_a_logistic_regression.joblib"],
	"svm_calibrated": [MODELS_DIR / "model_a_svm_calibrated.pkl", MODELS_DIR / "model_a_svm_calibrated.joblib"],
	"ensemble": [MODELS_DIR / "model_a_ensemble.pkl", MODELS_DIR / "model_a_ensemble.joblib"],
}

DEFAULT_ENSEMBLE_THRESHOLD = 0.49


def first_existing(paths: list[Path]) -> Path:
	"""Return the first existing path from a candidate list."""
	for path in paths:
		if path.exists():
			return path
	raise FileNotFoundError(f"Missing file. Checked: {', '.join(str(path) for path in paths)}")


def load_split(split: str):
	"""Load a processed Model A split for evaluation."""
	split = split.lower().strip()
	if split not in {"val", "test"}:
		raise ValueError("split must be 'val' or 'test'")

	x_path = PROCESSED_DIR / f"model_a_{split}_X.npz"
	y_path = PROCESSED_DIR / f"y_{split}.npy"
	if not x_path.exists():
		raise FileNotFoundError(f"Missing feature matrix: {x_path}")
	if not y_path.exists():
		raise FileNotFoundError(f"Missing labels: {y_path}")
	return load_npz(x_path), np.load(y_path)


def load_model(name: str):
	"""Load a trained Model A estimator."""
	name = name.lower().strip()
	if name not in MODEL_FILES:
		raise ValueError("model must be one of: random_forest, logistic_regression, svm_calibrated, ensemble")
	model_path = first_existing(MODEL_FILES[name])
	return joblib.load(model_path)


def load_ensemble_threshold() -> float:
	"""Load the stored ensemble threshold if it exists."""
	if MODEL_METADATA_PATH.exists():
		try:
			with MODEL_METADATA_PATH.open("r", encoding="utf-8") as fh:
				metadata = json.load(fh)
			threshold = metadata.get("selected_models", {}).get("ensemble_threshold")
			if threshold is not None:
				return float(threshold)
		except Exception:
			pass
	return DEFAULT_ENSEMBLE_THRESHOLD


def load_vectorizer():
	"""Load the fitted TF-IDF vectorizer used by Model A."""
	for candidate in [
		ROOT_DIR / "models" / "model_a" / "tfidf_vectorizer.pkl",
		PROCESSED_DIR / "artifacts" / "tfidf_vectorizer.joblib",
	]:
		if candidate.exists():
			return joblib.load(candidate)
	raise FileNotFoundError("Missing TF-IDF vectorizer artifact")


def predict(model_name: str, x, threshold: float | None = None):
	"""Predict hard labels for a feature matrix."""
	model = load_model(model_name)
	if model_name == "ensemble":
		decision_threshold = load_ensemble_threshold() if threshold is None else float(threshold)
		return (model.predict_proba(x)[:, 1] >= decision_threshold).astype(int)
	return model.predict(x)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
	"""Compute the standard evaluation metrics for Model A."""
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"f1_macro": float(f1_score(y_true, y_pred, average="macro")),
		"precision_class_0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
		"recall_class_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
		"precision_class_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
		"recall_class_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
	}


def evaluate(model_name: str, split: str, threshold: float | None = None) -> dict[str, Any]:
	"""Evaluate a Model A estimator on the requested split."""
	x, y_true = load_split(split)
	y_pred = predict(model_name, x, threshold=threshold)
	return {
		"model": model_name,
		"split": split,
		"n_samples": int(len(y_true)),
		"metrics": compute_metrics(y_true, y_pred),
	}


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for the inference CLI."""
	parser = argparse.ArgumentParser(description="Run Model A inference on val/test splits")
	parser.add_argument(
		"--model",
		choices=["random_forest", "logistic_regression", "svm_calibrated", "ensemble"],
		default="random_forest",
		help="Which model to run; defaults to the strongest single model",
	)
	parser.add_argument("--split", choices=["val", "test"], default="test")
	parser.add_argument(
		"--threshold",
		type=float,
		default=None,
		help="Optional decision threshold for the ensemble (overrides saved metadata)",
	)
	parser.add_argument(
		"--json",
		action="store_true",
		help="Print results as JSON instead of a human-readable table",
	)
	return parser.parse_args()


def main() -> int:
	"""Run the CLI evaluation workflow."""
	args = parse_args()
	result = evaluate(args.model, args.split, threshold=args.threshold)
	if args.model == "ensemble":
		result["ensemble_threshold"] = load_ensemble_threshold() if args.threshold is None else float(args.threshold)
	if args.json:
		print(json.dumps(result, indent=2))
	else:
		metrics = result["metrics"]
		print(f"Model: {result['model']}")
		print(f"Split: {result['split']} ({result['n_samples']} samples)")
		if args.model == "ensemble":
			print(f"Ensemble threshold:  {result['ensemble_threshold']:.2f}")
		print(f"Accuracy:           {metrics['accuracy']:.4f}")
		print(f"Balanced accuracy:   {metrics['balanced_accuracy']:.4f}")
		print(f"F1 macro:            {metrics['f1_macro']:.4f}")
		print(f"Precision class 0:   {metrics['precision_class_0']:.4f}")
		print(f"Recall class 0:      {metrics['recall_class_0']:.4f}")
		print(f"Precision class 1:   {metrics['precision_class_1']:.4f}")
		print(f"Recall class 1:      {metrics['recall_class_1']:.4f}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
