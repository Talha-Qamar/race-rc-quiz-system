#!/usr/bin/env python3
"""
Semi-Supervised Learning: Label Propagation for Answer Verification

This script explores semi-supervised learning for the assignment requirement.
It trains on a small labeled set, propagates labels to the remaining unlabeled
data, and compares the result against supervised baselines.

Usage:
    python3 src/semi_supervised_label_prop.py

Output:
    - models/model_a/unsupervised/label_propagation_results.json
    - models/model_a/unsupervised/label_prop_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "model_a"
UNSUPERVISED_DIR = MODELS_DIR / "unsupervised"

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 3000
TEST_SAMPLE_SIZE = 1000
LABELED_FRACTION = 0.2
SVD_COMPONENTS = 50

# Create output directory
UNSUPERVISED_DIR.mkdir(parents=True, exist_ok=True)


def load_data(split: str = "train") -> tuple[csr_matrix, np.ndarray]:
    """Load TF-IDF features and labels."""
    X_path = DATA_DIR / f"model_a_{split}_X.npz"
    y_path = DATA_DIR / f"y_{split}.npy"

    X = load_npz(X_path)  # Keep sparse for efficiency
    y = np.load(y_path)
    return X, y


def sample_rows(X: csr_matrix, y: np.ndarray, sample_size: int) -> tuple[csr_matrix, np.ndarray]:
    """Take a reproducible random sample of rows."""
    sample_size = min(sample_size, X.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    return X[indices], y[indices]


def reduce_features(X_train: csr_matrix, X_test: csr_matrix, n_components: int = SVD_COMPONENTS) -> tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    """Compress sparse TF-IDF into dense low-dimensional features."""
    n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1] - 1)
    n_components = max(2, n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    # Normalize the dense representation so K-Means and label propagation operate
    # on comparable feature scales.
    train_norm = np.linalg.norm(X_train_reduced, axis=1, keepdims=True)
    test_norm = np.linalg.norm(X_test_reduced, axis=1, keepdims=True)
    X_train_reduced = X_train_reduced / np.clip(train_norm, 1e-12, None)
    X_test_reduced = X_test_reduced / np.clip(test_norm, 1e-12, None)
    return X_train_reduced, X_test_reduced, svd


def create_labeled_unlabeled_split(
    X: np.ndarray, y: np.ndarray, labeled_fraction: float = LABELED_FRACTION
) -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, np.ndarray]:
    """
    Create labeled and unlabeled splits.

    Args:
        X: Feature matrix (sparse)
        y: Labels
        labeled_fraction: Fraction of data to use as labeled (rest is unlabeled)

    Returns:
        X_train_labeled, y_train_labeled, X_train_unlabeled, y_train_unlabeled, indices
    """
    n_samples = X.shape[0]
    n_labeled = int(n_samples * labeled_fraction)

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=RANDOM_STATE)
    labeled_idx, unlabeled_idx = next(splitter.split(np.zeros(n_samples), y))

    X_labeled = X[labeled_idx]
    y_labeled = y[labeled_idx]
    X_unlabeled = X[unlabeled_idx]
    y_unlabeled = y[unlabeled_idx]

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (labeled_idx, unlabeled_idx)


def run_label_propagation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Run label propagation on training data and evaluate on test set.

    Args:
        X_train: Training features (includes unlabeled with -1 labels)
        y_train: Training labels (-1 for unlabeled)
        X_test: Test features
        y_test: Test labels

    Returns:
        Results dictionary
    """
    print("Running Label Propagation...")

    # Create label propagation model
    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=100)

    # Fit on train (with unlabeled samples marked as -1)
    print("  Fitting LabelPropagation...")
    lp.fit(X_train, y_train)

    # Predict on test
    y_pred = lp.predict(X_test)

    # Get prediction probabilities
    y_proba = lp.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    recall_per_class = precision_recall_fscore_support(y_test, y_pred, average=None)[1]

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Weighted F1: {f1:.4f}")
    print(f"  Recall (class 0): {recall_per_class[0]:.4f}")
    print(f"  Recall (class 1): {recall_per_class[1]:.4f}")

    return {
        "model": "LabelPropagation",
        "accuracy": float(accuracy),
        "weighted_f1": float(f1),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "recall_per_class": recall_per_class.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def run_self_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Run self-training semi-supervised learning.

    Args:
        X_train: Training features
        y_train: Training labels (-1 for unlabeled)
        X_test: Test features
        y_test: Test labels

    Returns:
        Results dictionary
    """
    print("Running Self-Training...")

    from sklearn.linear_model import LogisticRegression

    # Use a balanced Logistic Regression base learner for the dense reduced features.
    base_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    st = SelfTrainingClassifier(base_model, verbose=1, threshold=0.75, max_iter=10)

    # Fit
    print("  Fitting SelfTraining...")
    st.fit(X_train, y_train)

    # Predict on test
    y_pred = st.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    recall_per_class = precision_recall_fscore_support(y_test, y_pred, average=None)[1]

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Weighted F1: {f1:.4f}")
    print(f"  Recall (class 0): {recall_per_class[0]:.4f}")
    print(f"  Recall (class 1): {recall_per_class[1]:.4f}")

    return {
        "model": "SelfTraining",
        "accuracy": float(accuracy),
        "weighted_f1": float(f1),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "recall_per_class": recall_per_class.tolist(),
        "y_pred": y_pred.tolist(),
    }


def compare_baselines(
    X_labeled_only: np.ndarray,
    y_train_labeled: np.ndarray,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Compare supervised baselines vs semi-supervised methods.

    Args:
        X_train: Full training features
        y_train_labeled: Labels for labeled subset only
        y_train_full: Full training labels (including unlabeled as -1)
        X_test: Test features
        y_test: Test labels

    Returns:
        Results dictionary
    """
    print("\nComparing Baselines...")

    from sklearn.linear_model import LogisticRegression

    results = {}

    # Baseline 1: Train only on labeled data
    print("\n  Baseline 1: Supervised (labeled only)...")
    y_labeled_only = y_train_labeled

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_labeled_only, y_labeled_only)
    y_pred_labeled = lr.predict(X_test)
    acc_labeled = accuracy_score(y_test, y_pred_labeled)
    _, _, f1_labeled, _ = precision_recall_fscore_support(
        y_test, y_pred_labeled, average="weighted"
    )
    print(f"    Accuracy: {acc_labeled:.4f}, F1: {f1_labeled:.4f}")

    results["baseline_labeled_only"] = {
        "name": "Supervised (labeled only)",
        "accuracy": float(acc_labeled),
        "weighted_f1": float(f1_labeled),
    }

    # Baseline 2: Train on all data (oracle - cheating!)
    print("  Baseline 2: Supervised (all data - oracle)...")
    lr_all = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_all.fit(X_train_full, y_train_full)
    y_pred_all = lr_all.predict(X_test)
    acc_all = accuracy_score(y_test, y_pred_all)
    _, _, f1_all, _ = precision_recall_fscore_support(
        y_test, y_pred_all, average="weighted"
    )
    print(f"    Accuracy: {acc_all:.4f}, F1: {f1_all:.4f}")

    results["baseline_all_data"] = {
        "name": "Supervised (all data - oracle)",
        "accuracy": float(acc_all),
        "weighted_f1": float(f1_all),
    }

    return results


def visualize_comparison(results_all: dict) -> None:
    """Create comparison visualization."""
    print("\nCreating comparison visualization against supervised baselines...")

    models = list(results_all.keys())
    accuracies = [results_all[m].get("accuracy", 0) for m in models]
    f1_scores = [results_all[m].get("weighted_f1", 0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x, accuracies, width, alpha=0.8, color="steelblue")
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", " ") for m in models], rotation=45, ha="right")
    ax1.set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

    # F1 Score
    ax2.bar(x, f1_scores, width, alpha=0.8, color="coral")
    ax2.set_ylabel("Weighted F1-Score", fontsize=12)
    ax2.set_title("F1-Score Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("_", " ") for m in models], rotation=45, ha="right")
    ax2.set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    output_path = UNSUPERVISED_DIR / "label_propagation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved visualization: {output_path}")
    plt.close()


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("Semi-Supervised Learning: Label Propagation for Answer Verification")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")
    print(f"  ✓ Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  ✓ Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  ✓ Class balance (train): {y_train.mean():.2%} correct answers")

    # Sample down to keep label propagation tractable.
    X_train, y_train = sample_rows(X_train, y_train, TRAIN_SAMPLE_SIZE)
    X_test, y_test = sample_rows(X_test, y_test, TEST_SAMPLE_SIZE)
    print(f"  ✓ Using train sample: {X_train.shape[0]} rows")
    print(f"  ✓ Using test sample:  {X_test.shape[0]} rows")

    # Reduce sparse TF-IDF to dense features before semi-supervised learning.
    X_train, X_test, svd = reduce_features(X_train, X_test, n_components=SVD_COMPONENTS)
    print(f"  ✓ Reduced to {X_train.shape[1]} dense components using TruncatedSVD")

    # Create labeled/unlabeled split
    print("\n" + "-" * 70)
    print("Creating labeled/unlabeled split (20% labeled, 80% unlabeled)...")
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, (labeled_idx, unlabeled_idx) = (
        create_labeled_unlabeled_split(X_train, y_train, labeled_fraction=LABELED_FRACTION)
    )
    print(f"  ✓ Labeled: {len(labeled_idx)} samples ({y_labeled.mean():.2%} correct)")
    print(
        f"  ✓ Unlabeled: {len(unlabeled_idx)} samples ({y_unlabeled.mean():.2%} correct)"
    )

    # Create training set with unlabeled marked as -1
    print("\n" + "-" * 70)
    print("Preparing training data with unlabeled samples (-1)...")
    y_train_split = np.copy(y_train)
    y_train_split[unlabeled_idx] = -1  # Mark unlabeled samples
    print(f"  ✓ Training labels: {(y_train_split >= 0).sum()} labeled, "
          f"{(y_train_split < 0).sum()} unlabeled")

    # Run semi-supervised methods
    print("\n" + "-" * 70)
    results_lp = run_label_propagation(X_train, y_train_split, X_test, y_test)

    print("\n" + "-" * 70)
    results_st = run_self_training(X_train, y_train_split, X_test, y_test)

    print("\n" + "-" * 70)
    results_baselines = compare_baselines(
        X_labeled, y_labeled, X_train, y_train, X_test, y_test
    )

    # Combine all results
    all_results = {
        "label_propagation": results_lp,
        "self_training": results_st,
        "baselines": results_baselines,
        "configuration": {
            "labeled_fraction": LABELED_FRACTION,
            "labeled_samples": int(len(labeled_idx)),
            "unlabeled_samples": int(len(unlabeled_idx)),
            "kernel": "knn",
            "n_neighbors": 7,
            "train_sample_size": TRAIN_SAMPLE_SIZE,
            "test_sample_size": TEST_SAMPLE_SIZE,
            "svd_components": int(X_train.shape[1]),
        },
        "interpretation": {
            "key_finding": (
                "Compare semi-supervised (Label Prop, Self-Training) vs "
                "supervised baselines (labeled-only and all-data oracle)"
            ),
            "good_result": (
                "Semi-supervised accuracy close to 'all-data oracle' using only 20% labels"
            ),
            "expected": (
                "Label Propagation uses a k-nearest-neighbors graph after TruncatedSVD and "
                "stratified label sampling to reduce majority-class collapse. Self-Training "
                "with balanced Logistic Regression remains the practical comparison method "
                "for this dataset."
            ),
        },
    }

    # Save results
    print("\n" + "-" * 70)
    print("Saving results...")
    output_path = UNSUPERVISED_DIR / "label_propagation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Saved results: {output_path}")

    # Visualize comparison
    print("\n" + "-" * 70)
    comparison_data = {
        "label_propagation": results_lp,
        "self_training": results_st,
        "supervised_labeled_only": results_baselines["baseline_labeled_only"],
        "supervised_all_data": results_baselines["baseline_all_data"],
    }
    visualize_comparison(comparison_data)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Label Propagation:     {results_lp['accuracy']:.4f} accuracy, "
        f"{results_lp['weighted_f1']:.4f} F1"
    )
    print(
        f"Self-Training:         {results_st['accuracy']:.4f} accuracy, "
        f"{results_st['weighted_f1']:.4f} F1"
    )
    print(
        f"Supervised (20% only): {results_baselines['baseline_labeled_only']['accuracy']:.4f} accuracy"
    )
    print(
        f"Supervised (all data): {results_baselines['baseline_all_data']['accuracy']:.4f} accuracy"
    )

    if results_lp["accuracy"] > results_baselines["baseline_labeled_only"]["accuracy"]:
        print("\n✓ Label Propagation improved over labeled-only baseline")
    else:
        print("\n✗ Label Propagation did not improve (sparse features may be issue)")

    print("\nCompare with Model A supervised accuracy: 64.72%")
    print("=" * 70)


if __name__ == "__main__":
    main()
