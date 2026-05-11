#!/usr/bin/env python3
"""
Unsupervised Learning: K-Means Clustering for Answer Verification

This script applies K-Means clustering to the answer verification task to
explore natural groupings in the TF-IDF feature space and compare the cluster
structure against the gold labels, as required by the assignment.

Usage:
    python3 src/unsupervised_kmeans.py [--visualize]

Output:
    - models/model_a/unsupervised/kmeans_results.json (metrics & analysis)
    - models/model_a/unsupervised/kmeans_labels_test.npy (cluster assignments)
    - models/model_a/unsupervised/kmeans_pca_visualization.png (2D SVD scatter plot)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "model_a"
UNSUPERVISED_DIR = MODELS_DIR / "unsupervised"

# Create output directory
UNSUPERVISED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLE_SIZE = 2000
TEST_SAMPLE_SIZE = 1000
K_SEARCH_SAMPLE_SIZE = 1000
RANDOM_STATE = 42


def sample_rows(X: csr_matrix, y: np.ndarray, sample_size: int) -> tuple[csr_matrix, np.ndarray]:
    """Take a reproducible random sample of rows."""
    sample_size = min(sample_size, X.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    return X[indices], y[indices]


def reduce_features(
    X_train: csr_matrix, X_test: csr_matrix, n_components: int = 50
) -> tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    """Reduce sparse TF-IDF features to a compact dense representation."""
    n_components = min(n_components, X_train.shape[1] - 1, X_train.shape[0] - 1)
    n_components = max(2, n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    train_norm = np.linalg.norm(X_train_reduced, axis=1, keepdims=True)
    test_norm = np.linalg.norm(X_test_reduced, axis=1, keepdims=True)
    X_train_reduced = X_train_reduced / np.clip(train_norm, 1e-12, None)
    X_test_reduced = X_test_reduced / np.clip(test_norm, 1e-12, None)
    return X_train_reduced, X_test_reduced, svd


def load_data(split: str = "test") -> tuple[csr_matrix, np.ndarray]:
    """Load TF-IDF features and labels for a given split (sparse format)."""
    X_path = DATA_DIR / f"model_a_{split}_X.npz"
    y_path = DATA_DIR / f"y_{split}.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing features: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing labels: {y_path}")

    X = load_npz(X_path)  # Keep sparse
    y = np.load(y_path)
    return X, y


def find_optimal_k(X: np.ndarray, y: np.ndarray, max_k: int = 6) -> dict:
    """
    Find optimal number of clusters using MiniBatchKMeans (memory efficient).

    Args:
        X: Feature matrix (sparse, n_samples x n_features)
        y: Labels
        max_k: Maximum number of clusters to test

    Returns:
        Dictionary with scores for each k
    """
    print("Finding optimal K on reduced sample...")
    results = {}

    for k in range(2, max_k + 1):
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=RANDOM_STATE, n_init=3, batch_size=128, verbose=0
        )
        labels = kmeans.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        inertia = kmeans.inertia_

        results[k] = {
            "silhouette": float(silhouette),
            "davies_bouldin": float(davies_bouldin),
            "calinski_harabasz": float(calinski),
            "inertia": float(inertia),
        }

        print(f"  k={k}: silhouette={silhouette:.4f}, davies_bouldin={davies_bouldin:.4f}")

    return results


def run_kmeans(X: np.ndarray, y: np.ndarray, n_clusters: int = 2) -> tuple[dict, np.ndarray]:
    """
    Run K-Means clustering on features (MiniBatchKMeans for memory efficiency).

    Args:
        X: Feature matrix (sparse, n_samples x n_features)
        y: True labels (n_samples,)
        n_clusters: Number of clusters to find

    Returns:
        Results dictionary with metrics
    """
    print(f"\nRunning K-Means with k={n_clusters} on reduced features...")

    # Fit KMeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=3,
        batch_size=128,
        verbose=0,
    )
    labels = kmeans.fit_predict(X)

    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)

    print(f"  Silhouette Score:      {silhouette:.4f} (range: [-1, 1], higher is better)")
    print(f"  Davies-Bouldin Index:  {davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz:     {calinski:.4f} (higher is better)")

    # Analyze cluster composition
    print(f"\nCluster Composition:")
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_size = cluster_mask.sum()
        correct_in_cluster = y[cluster_mask].sum()
        correct_pct = 100 * correct_in_cluster / cluster_size if cluster_size > 0 else 0

        print(
            f"  Cluster {cluster_id}: {cluster_size} samples, "
            f"{correct_in_cluster} correct ({correct_pct:.1f}%)"
        )

    # Check if clusters separate correct/incorrect
    cluster_correct_ratio = np.array(
        [y[labels == i].mean() if (labels == i).sum() > 0 else 0 for i in range(n_clusters)]
    )
    cluster_separation = (
        cluster_correct_ratio.max() - cluster_correct_ratio.min()
    )

    print(f"\nCluster Separation:")
    print(f"  Correct answer % per cluster: {cluster_correct_ratio}")
    print(f"  Separation score: {cluster_separation:.4f} (0=no sep, 1=perfect sep)")

    return {
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette),
        "davies_bouldin_index": float(davies_bouldin),
        "calinski_harabasz_score": float(calinski),
        "cluster_separation": float(cluster_separation),
        "cluster_correct_ratios": cluster_correct_ratio.tolist(),
        "labels": labels.tolist(),
    }, labels


def visualize_clusters(X: np.ndarray, labels: np.ndarray, y: np.ndarray) -> None:
    """
    Visualize clusters using the first two TruncatedSVD components.

    Args:
        X: Feature matrix (sparse)
        labels: Cluster assignments
        y: True labels for coloring
    """
    print("\nVisualizing clusters in reduced space...")
    X_reduced = X[:, :2] if X.shape[1] >= 2 else np.column_stack([X[:, 0], np.zeros(X.shape[0])])
    print("  Using the first two TruncatedSVD components for visualization")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Clusters
    scatter1 = axes[0].scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")
    axes[0].set_title("K-Means Clusters")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    # Plot 2: True labels
    scatter2 = axes[1].scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=y,
        cmap="RdYlGn",
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")
    axes[1].set_title("True Labels (0=Incorrect, 1=Correct)")
    plt.colorbar(scatter2, ax=axes[1], label="Ground Truth")

    plt.tight_layout()
    output_path = UNSUPERVISED_DIR / "kmeans_pca_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved visualization: {output_path}")
    plt.close()


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("Unsupervised Learning: K-Means Clustering for Answer Verification")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    X_test, y_test = load_data("test")
    X_train, y_train = load_data("train")
    print(f"  ✓ Train: {X_train.shape[0]} samples, {X_train.shape[1]} features (sparse)")
    print(f"  ✓ Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features (sparse)")

    # Use manageable samples for a quick demo run.
    X_train, y_train = sample_rows(X_train, y_train, TRAIN_SAMPLE_SIZE)
    X_test, y_test = sample_rows(X_test, y_test, TEST_SAMPLE_SIZE)
    print(f"  ✓ Using train sample: {X_train.shape[0]} rows")
    print(f"  ✓ Using test sample:  {X_test.shape[0]} rows")

    # Reduce TF-IDF to a compact dense space before clustering.
    X_train_reduced, X_test_reduced, svd = reduce_features(X_train, X_test, n_components=50)
    print(f"  ✓ Reduced to {X_train_reduced.shape[1]} components using TruncatedSVD")

    # Find optimal K on a smaller reduced subset.
    print("\n" + "-" * 70)
    k_sample_size = min(K_SEARCH_SAMPLE_SIZE, X_train_reduced.shape[0])
    X_k_search = X_train_reduced[:k_sample_size]
    y_k_search = y_train[:k_sample_size]
    optimal_k_results = find_optimal_k(X_k_search, y_k_search, max_k=6)

    # Run K-Means with k=2 on reduced train data
    print("\n" + "-" * 70)
    results, train_labels = run_kmeans(X_train_reduced, y_train, n_clusters=2)

    # Apply to test set
    print("\nApplying K-Means to test set...")
    kmeans = MiniBatchKMeans(
        n_clusters=2, random_state=RANDOM_STATE, n_init=3, batch_size=128
    )
    kmeans.fit(X_train_reduced)
    test_labels = kmeans.predict(X_test_reduced)

    # Analyze test set
    test_separation = np.array([y_test[test_labels == i].mean() if (test_labels == i).sum() > 0 else 0 for i in range(2)])
    test_cluster_separation = test_separation.max() - test_separation.min()
    results["test_cluster_correct_ratios"] = test_separation.tolist()
    results["test_cluster_separation"] = float(test_cluster_separation)

    print(f"  Test cluster correct %: {test_separation}")
    print(f"  Test separation score: {test_cluster_separation:.4f}")

    # Visualize (use test set for visualization)
    print("\n" + "-" * 70)
    visualize_clusters(X_test_reduced, test_labels, y_test)

    # Save results
    print("\n" + "-" * 70)
    print("Saving results...")

    # Save metrics
    output_results = {
        "model": "K-Means (MiniBatchKMeans + TruncatedSVD)",
        "task": "Answer Verification (Unsupervised)",
        "train_results": results,
        "optimal_k_analysis": optimal_k_results,
        "interpretation": {
            "silhouette_score": "Range [-1, 1]. Higher is better. >0.5 is good.",
            "davies_bouldin_index": "Lower is better. Measures cluster separation.",
            "cluster_separation": "How well clusters separate correct/incorrect answers. 0-1 scale.",
            "observation": (
                "If separation < 0.2, clustering doesn't naturally split correct/incorrect. "
                "If separation > 0.5, unsupervised clustering aligns with true labels."
            ),
        },
    }

    output_path = UNSUPERVISED_DIR / "kmeans_results.json"
    with open(output_path, "w") as f:
        json.dump(output_results, f, indent=2)
    print(f"  ✓ Saved results: {output_path}")

    # Save cluster labels
    labels_path = UNSUPERVISED_DIR / "kmeans_labels_test.npy"
    np.save(labels_path, test_labels)
    print(f"  ✓ Saved test labels: {labels_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Silhouette Score (train): {results['silhouette_score']:.4f}")
    print(f"Cluster Separation (test): {results['test_cluster_separation']:.4f}")
    if test_cluster_separation > 0.3:
        print("✓ Clusters show reasonable separation of correct/incorrect answers")
    else:
        print("✗ Clusters do NOT naturally separate correct/incorrect (expected for raw features)")
    print("\nNext: Compare with supervised Model A results (64.72% accuracy)")
    print("=" * 70)


if __name__ == "__main__":
    main()
