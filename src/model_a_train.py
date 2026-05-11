"""Model A training pipeline for answer verification.

This script trains the classical ML models used in the AL2002 RACE project,
now using sparse TF-IDF plus handcrafted verification features, a calibrated
soft-vote ensemble, and a lightweight question-generation ranker.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from evaluate import full_evaluation_report, save_evaluation_results
from question_generation import train_question_ranker


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "model_a" / "traditional"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split: str):
	"""Load a processed Model A split from disk.

	Args:
		split (str): One of ``train``, ``val``, or ``test``.

	Returns:
		tuple: Feature matrix and label vector.
	"""
	x_path = DATA_DIR / f"model_a_{split}_X.npz"
	y_path = DATA_DIR / f"y_{split}.npy"
	if not x_path.exists():
		raise FileNotFoundError(f"Missing features: {x_path}")
	if not y_path.exists():
		raise FileNotFoundError(f"Missing labels: {y_path}")
	return load_npz(x_path), np.load(y_path)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
	"""Compute the core validation and test metrics.

	Args:
		y_true: Ground-truth labels.
		y_pred: Predicted labels.

	Returns:
		dict[str, float]: Metric values used in the comparison table.
	"""
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"f1_macro": float(f1_score(y_true, y_pred, average="macro")),
		"f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
		"precision_correct": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
		"recall_correct": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
	}


def save_model_artifacts(model, stem_name: str) -> dict[str, str]:
	"""Save a fitted estimator under both .joblib and .pkl names.

	Args:
		model: Fitted estimator.
		stem_name (str): Base filename without extension.

	Returns:
		dict[str, str]: Saved artifact paths.
	"""
	joblib_path = MODELS_DIR / f"{stem_name}.joblib"
	pkl_path = MODELS_DIR / f"{stem_name}.pkl"
	joblib.dump(model, joblib_path)
	joblib.dump(model, pkl_path)
	return {"joblib": str(joblib_path), "pkl": str(pkl_path)}


def load_tfidf_vectorizer() -> object:
	"""Load the fitted TF-IDF vectorizer used for question ranker training."""
	for candidate in [
		ROOT_DIR / "models" / "model_a" / "tfidf_vectorizer.pkl",
		DATA_DIR / "artifacts" / "tfidf_vectorizer.joblib",
	]:
		if candidate.exists():
			return joblib.load(candidate)
	raise FileNotFoundError("Could not locate a fitted TF-IDF vectorizer.")


def build_search(model, param_grid: dict, scoring, n_iter: int):
	"""Create a randomized hyperparameter search wrapper.

	Args:
		model: Base estimator.
		param_grid (dict): Hyperparameter search space.
		scoring: Scikit-learn scorer.
		n_iter (int): Number of random samples.

	Returns:
		RandomizedSearchCV: Configured search object.
	"""
	return RandomizedSearchCV(
		model,
		param_grid,
		n_iter=n_iter,
		cv=3,
		scoring=scoring,
		random_state=42,
		n_jobs=-1,
		verbose=1,
	)


def main() -> int:
	"""Train the Model A classical ML stack and save all artifacts."""
	print("=" * 100)
	print("MODEL A: TRADITIONAL ML TRAINING PIPELINE")
	print("=" * 100)

	print("\n[1/5] Loading corrected, properly-split data...")
	start_time = time.time()
	X_train, y_train = load_split("train")
	X_val, y_val = load_split("val")
	X_test, y_test = load_split("test")
	print("✅ Data loaded successfully")
	print(f"   Train: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")
	print(f"   Val:   {X_val.shape[0]:,} samples")
	print(f"   Test:  {X_test.shape[0]:,} samples")
	class_counts = np.bincount(y_train)
	print(f"   Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
	print(f"   Imbalance ratio: {class_counts[0] / class_counts[1]:.2f}:1")

	scorer = make_scorer(f1_score, average="macro")

	print("\n" + "=" * 100)
	print("MODEL 1: LOGISTIC REGRESSION (class_weight='balanced')")
	print("=" * 100)
	search_lr = build_search(
		LogisticRegression(solver="saga", max_iter=1500, class_weight="balanced", random_state=42, n_jobs=-1),
		{"C": np.logspace(-2, 3, 20)},
		scorer,
		n_iter=12,
	)
	search_lr.fit(X_train, y_train)
	model_lr = search_lr.best_estimator_
	y_pred_lr_val = model_lr.predict(X_val)
	y_pred_lr_test = model_lr.predict(X_test)
	metrics_lr_val = compute_metrics(y_val, y_pred_lr_val)
	metrics_lr_test = compute_metrics(y_test, y_pred_lr_test)
	print(f"   Best C: {search_lr.best_params_['C']:.4f} | CV F1: {search_lr.best_score_:.4f}")
	print(f"   Test F1: {metrics_lr_test['f1_macro']:.4f} | Recall(Correct): {metrics_lr_test['recall_correct']:.4f}")
	save_model_artifacts(model_lr, "model_a_logistic_regression")

	print("\n" + "=" * 100)
	print("MODEL 2: LINEAR SVC (class_weight='balanced')")
	print("=" * 100)
	search_svc = build_search(
		LinearSVC(loss="squared_hinge", dual=False, max_iter=2500, class_weight="balanced", random_state=42),
		{"C": np.logspace(-1, 2, 15)},
		scorer,
		n_iter=10,
	)
	search_svc.fit(X_train, y_train)
	model_svc = search_svc.best_estimator_
	y_pred_svc_val = model_svc.predict(X_val)
	y_pred_svc_test = model_svc.predict(X_test)
	metrics_svc_val = compute_metrics(y_val, y_pred_svc_val)
	metrics_svc_test = compute_metrics(y_test, y_pred_svc_test)
	print(f"   Best C: {search_svc.best_params_['C']:.4f} | CV F1: {search_svc.best_score_:.4f}")
	print(f"   Test F1: {metrics_svc_test['f1_macro']:.4f} | Recall(Correct): {metrics_svc_test['recall_correct']:.4f}")
	save_model_artifacts(model_svc, "model_a_linearsvc")

	print("\n" + "=" * 100)
	print("MODEL 3: NAIVE BAYES")
	print("=" * 100)
	model_nb = MultinomialNB(alpha=0.1)
	model_nb.fit(X_train, y_train)
	y_pred_nb_val = model_nb.predict(X_val)
	y_pred_nb_test = model_nb.predict(X_test)
	metrics_nb_val = compute_metrics(y_val, y_pred_nb_val)
	metrics_nb_test = compute_metrics(y_test, y_pred_nb_test)
	print(f"   Test F1: {metrics_nb_test['f1_macro']:.4f} | Recall(Correct): {metrics_nb_test['recall_correct']:.4f}")
	save_model_artifacts(model_nb, "model_a_naive_bayes")

	print("\n" + "=" * 100)
	print("MODEL 4: RANDOM FOREST CLASSIFIER")
	print("=" * 100)
	search_rf = build_search(
		RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
		{
			"n_estimators": [100, 200, 300],
			"max_depth": [10, 20, 30],
			"min_samples_split": [5, 10],
			"min_samples_leaf": [2, 4],
		},
		scorer,
		n_iter=12,
	)
	search_rf.fit(X_train, y_train)
	model_rf = search_rf.best_estimator_
	y_pred_rf_val = model_rf.predict(X_val)
	y_pred_rf_test = model_rf.predict(X_test)
	metrics_rf_val = compute_metrics(y_val, y_pred_rf_val)
	metrics_rf_test = compute_metrics(y_test, y_pred_rf_test)
	print(f"   Best params: n_estimators={search_rf.best_params_['n_estimators']}, max_depth={search_rf.best_params_['max_depth']}")
	print(f"   Test F1: {metrics_rf_test['f1_macro']:.4f} | Recall(Correct): {metrics_rf_test['recall_correct']:.4f}")
	save_model_artifacts(model_rf, "model_a_random_forest")

	print("\n" + "=" * 100)
	print("ENSEMBLE: CALIBRATED SOFT VOTING")
	print("=" * 100)
	svm_calibrated = CalibratedClassifierCV(model_svc, method="isotonic", cv=3)
	svm_calibrated.fit(X_train, y_train)
	y_pred_svm_val = svm_calibrated.predict(X_val)
	y_pred_svm_test = svm_calibrated.predict(X_test)
	metrics_svm_val = compute_metrics(y_val, y_pred_svm_val)
	metrics_svm_test = compute_metrics(y_test, y_pred_svm_test)

	ensemble = VotingClassifier(
		estimators=[("rf", model_rf), ("lr", model_lr), ("svm", svm_calibrated)],
		voting="soft",
		weights=[2, 1, 1],
		n_jobs=-1,
	)
	ensemble.fit(X_train, y_train)
	y_pred_ensemble_val = ensemble.predict(X_val)
	y_pred_ensemble_test = ensemble.predict(X_test)
	y_proba_ensemble_test = ensemble.predict_proba(X_test)
	metrics_ensemble_val = compute_metrics(y_val, y_pred_ensemble_val)
	metrics_ensemble_test = compute_metrics(y_test, y_pred_ensemble_test)
	print(f"   Validation F1: {metrics_ensemble_val['f1_macro']:.4f}")
	print(f"   Test F1: {metrics_ensemble_test['f1_macro']:.4f} | Recall(Correct): {metrics_ensemble_test['recall_correct']:.4f}")
	save_model_artifacts(svm_calibrated, "model_a_svm_calibrated")
	save_model_artifacts(ensemble, "model_a_ensemble")

	print("\nTraining question ranker for generation support...")
	tfidf_vectorizer = load_tfidf_vectorizer()
	train_clean_frame = pd.read_csv(DATA_DIR / "train_clean.csv")
	ranker_frame = train_clean_frame.sample(n=min(len(train_clean_frame), 5000), random_state=42)
	question_ranker_path = MODELS_DIR / "question_ranker.pkl"
	train_question_ranker(ranker_frame, tfidf_vectorizer, question_ranker_path)

	print("\nEvaluating models on the test set...")
	evaluation_reports = {
		"random_forest": full_evaluation_report(y_test, y_pred_rf_test, model_rf.predict_proba(X_test), model_name="Random Forest"),
		"logistic_regression": full_evaluation_report(y_test, y_pred_lr_test, model_lr.predict_proba(X_test), model_name="Logistic Regression"),
		"linear_svc_calibrated": full_evaluation_report(y_test, y_pred_svm_test, svm_calibrated.predict_proba(X_test), model_name="Calibrated Linear SVC"),
		"naive_bayes": full_evaluation_report(y_test, y_pred_nb_test, model_nb.predict_proba(X_test), model_name="Naive Bayes"),
		"ensemble": full_evaluation_report(y_test, y_pred_ensemble_test, y_proba_ensemble_test, model_name="Soft-Vote Ensemble"),
	}
	evaluation_results_path = MODELS_DIR / "evaluation_results.json"
	save_evaluation_results(evaluation_reports, evaluation_results_path)

	models_data = [
		("Logistic Regression", metrics_lr_val, metrics_lr_test),
		("Calibrated Linear SVC", metrics_svm_val, metrics_svm_test),
		("Naive Bayes", metrics_nb_val, metrics_nb_test),
		("Random Forest", metrics_rf_val, metrics_rf_test),
		("Soft-Vote Ensemble", metrics_ensemble_val, metrics_ensemble_test),
	]
	best_model_name, best_f1 = max(((name, metrics["f1_macro"]) for name, metrics, _ in models_data), key=lambda item: item[1])

	results = {
		"timestamp": pd.Timestamp.now().isoformat(),
		"note": "Trained on corrected, properly-split data with sparse+dense features",
		"dataset_info": {
			"train_samples": int(X_train.shape[0]),
			"val_samples": int(X_val.shape[0]),
			"test_samples": int(X_test.shape[0]),
			"n_features": int(X_train.shape[1]),
			"class_distribution": {"0 (Incorrect)": int(class_counts[0]), "1 (Correct)": int(class_counts[1])},
			"imbalance_ratio": float(class_counts[0] / class_counts[1]),
		},
		"hyperparameters": {
			"logistic_regression": {"solver": "saga", "max_iter": 1500, "class_weight": "balanced", "best_C": float(search_lr.best_params_["C"]), "cv_f1_score": float(search_lr.best_score_)},
			"linear_svc": {"loss": "squared_hinge", "dual": False, "max_iter": 2500, "class_weight": "balanced", "best_C": float(search_svc.best_params_["C"]), "cv_f1_score": float(search_svc.best_score_)},
			"naive_bayes": {"alpha": 0.1},
			"random_forest": {"class_weight": "balanced", "best_n_estimators": int(search_rf.best_params_["n_estimators"]), "best_max_depth": int(search_rf.best_params_["max_depth"]), "cv_f1_score": float(search_rf.best_score_)},
			"ensemble": {"strategy": "soft_voting", "weights": [2, 1, 1], "members": ["random_forest", "logistic_regression", "calibrated_linear_svc"]},
		},
		"validation_results": {
			"logistic_regression": metrics_lr_val,
			"linear_svc_calibrated": metrics_svm_val,
			"naive_bayes": metrics_nb_val,
			"random_forest": metrics_rf_val,
			"ensemble": metrics_ensemble_val,
		},
		"test_results": {
			"logistic_regression": metrics_lr_test,
			"linear_svc_calibrated": metrics_svm_test,
			"naive_bayes": metrics_nb_test,
			"random_forest": metrics_rf_test,
			"ensemble": metrics_ensemble_test,
		},
		"evaluation_reports": evaluation_reports,
		"best_model": {"name": best_model_name, "validation_f1": float(best_f1)},
		"model_files": {
			"logistic_regression": str(MODELS_DIR / "model_a_logistic_regression.pkl"),
			"linear_svc": str(MODELS_DIR / "model_a_linearsvc.pkl"),
			"linear_svc_calibrated": str(MODELS_DIR / "model_a_svm_calibrated.pkl"),
			"naive_bayes": str(MODELS_DIR / "model_a_naive_bayes.pkl"),
			"random_forest": str(MODELS_DIR / "model_a_random_forest.pkl"),
			"ensemble": str(MODELS_DIR / "model_a_ensemble.pkl"),
			"question_ranker": str(question_ranker_path),
		},
	}

	results_json = MODELS_DIR / "results_all_models.json"
	results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

	comparison_df = pd.DataFrame([
		{"Model": "Logistic Regression", "Val_Accuracy": metrics_lr_val["accuracy"], "Val_Balanced_Accuracy": metrics_lr_val["balanced_accuracy"], "Val_F1_Macro": metrics_lr_val["f1_macro"], "Val_Recall_Correct": metrics_lr_val["recall_correct"], "Test_Accuracy": metrics_lr_test["accuracy"], "Test_Balanced_Accuracy": metrics_lr_test["balanced_accuracy"], "Test_F1_Macro": metrics_lr_test["f1_macro"], "Test_Recall_Correct": metrics_lr_test["recall_correct"]},
		{"Model": "Calibrated Linear SVC", "Val_Accuracy": metrics_svm_val["accuracy"], "Val_Balanced_Accuracy": metrics_svm_val["balanced_accuracy"], "Val_F1_Macro": metrics_svm_val["f1_macro"], "Val_Recall_Correct": metrics_svm_val["recall_correct"], "Test_Accuracy": metrics_svm_test["accuracy"], "Test_Balanced_Accuracy": metrics_svm_test["balanced_accuracy"], "Test_F1_Macro": metrics_svm_test["f1_macro"], "Test_Recall_Correct": metrics_svm_test["recall_correct"]},
		{"Model": "Naive Bayes", "Val_Accuracy": metrics_nb_val["accuracy"], "Val_Balanced_Accuracy": metrics_nb_val["balanced_accuracy"], "Val_F1_Macro": metrics_nb_val["f1_macro"], "Val_Recall_Correct": metrics_nb_val["recall_correct"], "Test_Accuracy": metrics_nb_test["accuracy"], "Test_Balanced_Accuracy": metrics_nb_test["balanced_accuracy"], "Test_F1_Macro": metrics_nb_test["f1_macro"], "Test_Recall_Correct": metrics_nb_test["recall_correct"]},
		{"Model": "Random Forest", "Val_Accuracy": metrics_rf_val["accuracy"], "Val_Balanced_Accuracy": metrics_rf_val["balanced_accuracy"], "Val_F1_Macro": metrics_rf_val["f1_macro"], "Val_Recall_Correct": metrics_rf_val["recall_correct"], "Test_Accuracy": metrics_rf_test["accuracy"], "Test_Balanced_Accuracy": metrics_rf_test["balanced_accuracy"], "Test_F1_Macro": metrics_rf_test["f1_macro"], "Test_Recall_Correct": metrics_rf_test["recall_correct"]},
		{"Model": "Soft-Vote Ensemble", "Val_Accuracy": metrics_ensemble_val["accuracy"], "Val_Balanced_Accuracy": metrics_ensemble_val["balanced_accuracy"], "Val_F1_Macro": metrics_ensemble_val["f1_macro"], "Val_Recall_Correct": metrics_ensemble_val["recall_correct"], "Test_Accuracy": metrics_ensemble_test["accuracy"], "Test_Balanced_Accuracy": metrics_ensemble_test["balanced_accuracy"], "Test_F1_Macro": metrics_ensemble_test["f1_macro"], "Test_Recall_Correct": metrics_ensemble_test["recall_correct"]},
	])
	results_csv = MODELS_DIR / "results_all_models.csv"
	comparison_df.to_csv(results_csv, index=False)

	metadata = {
		"project": "race-rc-quiz-system",
		"model_family": "Model A",
		"purpose": "Answer verification for multiple-choice reading comprehension",
		"selected_models": {
			"default_single_model": "random_forest",
			"backup_model": "logistic_regression",
			"ensemble": ["random_forest", "logistic_regression", "calibrated_linear_svc"],
			"ensemble_threshold": 0.49,
		},
		"artifacts": {
			"random_forest_pkl": "model_a_random_forest.pkl",
			"logistic_regression_pkl": "model_a_logistic_regression.pkl",
			"linear_svc_calibrated_pkl": "model_a_svm_calibrated.pkl",
			"ensemble_pkl": "model_a_ensemble.pkl",
			"question_ranker_pkl": "question_ranker.pkl",
			"tfidf_vectorizer_pkl": "tfidf_vectorizer.pkl",
		},
		"metrics_test": {
			"random_forest": {"accuracy": metrics_rf_test["accuracy"], "balanced_accuracy": metrics_rf_test["balanced_accuracy"], "f1_macro": metrics_rf_test["f1_macro"], "recall_class_1": metrics_rf_test["recall_correct"], "exact_match": evaluation_reports["random_forest"]["exact_match"]},
			"logistic_regression": {"accuracy": metrics_lr_test["accuracy"], "balanced_accuracy": metrics_lr_test["balanced_accuracy"], "f1_macro": metrics_lr_test["f1_macro"], "recall_class_1": metrics_lr_test["recall_correct"], "exact_match": evaluation_reports["logistic_regression"]["exact_match"]},
			"linear_svc_calibrated": {"accuracy": metrics_svm_test["accuracy"], "balanced_accuracy": metrics_svm_test["balanced_accuracy"], "f1_macro": metrics_svm_test["f1_macro"], "recall_class_1": metrics_svm_test["recall_correct"], "exact_match": evaluation_reports["linear_svc_calibrated"]["exact_match"]},
			"ensemble": {"accuracy": metrics_ensemble_test["accuracy"], "balanced_accuracy": metrics_ensemble_test["balanced_accuracy"], "f1_macro": metrics_ensemble_test["f1_macro"], "recall_class_1": metrics_ensemble_test["recall_correct"], "exact_match": evaluation_reports["ensemble"]["exact_match"]},
		},
		"notes": [
			"Random Forest is the default single model.",
			"Logistic Regression remains the backup model.",
			"The ensemble now uses calibrated soft voting over Random Forest, Logistic Regression, and Linear SVC.",
		],
	}
	metadata_path = MODELS_DIR / "model_metadata.json"
	metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

	print("\n" + "=" * 100)
	print("✅ TRAINING PIPELINE COMPLETE!")
	print("=" * 100)
	print(f"Best Model: {best_model_name}")
	print(f"Validation F1: {best_f1:.4f}")
	print(f"Evaluation JSON: {evaluation_results_path}")
	print(f"Results JSON: {results_json}")
	print(f"Results CSV: {results_csv}")
	print(f"Metadata: {metadata_path}")
	print(f"Elapsed: {time.time() - start_time:.1f}s")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
