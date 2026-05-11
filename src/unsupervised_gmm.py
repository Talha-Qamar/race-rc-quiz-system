"""Gaussian Mixture Model exploration for Model A.

This script mirrors the existing K-Means workflow, but adds soft cluster
assignments and model-selection scores such as BIC and AIC.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "model_a"
UNSUPERVISED_DIR = MODELS_DIR / "unsupervised"

UNSUPERVISED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 3000
TEST_SAMPLE_SIZE = 1000


def sample_rows(X: csr_matrix, y: np.ndarray, sample_size: int) -> tuple[csr_matrix, np.ndarray]:
	"""Return a reproducible subsample of rows."""
	sample_size = min(sample_size, X.shape[0])
	rng = np.random.default_rng(RANDOM_STATE)
	indices = rng.choice(X.shape[0], size=sample_size, replace=False)
	return X[indices], y[indices]


def reduce_features(X_train: csr_matrix, X_test: csr_matrix, n_components: int = 50) -> tuple[np.ndarray, np.ndarray, TruncatedSVD]:
	"""Reduce sparse features with TruncatedSVD and L2-normalize the output."""
	n_components = min(n_components, X_train.shape[1] - 1, X_train.shape[0] - 1)
	n_components = max(2, n_components)
	svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
	X_train_reduced = normalize(svd.fit_transform(X_train))
	X_test_reduced = normalize(svd.transform(X_test))
	return X_train_reduced, X_test_reduced, svd


def load_data(split: str = "test") -> tuple[csr_matrix, np.ndarray]:
	"""Load a processed Model A split."""
	X_path = DATA_DIR / f"model_a_{split}_X.npz"
	y_path = DATA_DIR / f"y_{split}.npy"
	if not X_path.exists():
		raise FileNotFoundError(f"Missing features: {X_path}")
	if not y_path.exists():
		raise FileNotFoundError(f"Missing labels: {y_path}")
	return load_npz(X_path), np.load(y_path)


def run_gmm_clustering(
	X_train_tfidf,
	X_test_tfidf,
	n_components: int = 2,
	svd_components: int = 50,
	sample_n: int = 3000,
):
	"""Fit a Gaussian Mixture Model on reduced TF-IDF features.

	Args:
		X_train_tfidf: Sparse training matrix.
		X_test_tfidf: Sparse test matrix.
		n_components (int): Number of mixture components.
		svd_components (int): Number of SVD dimensions before clustering.
		sample_n (int): Maximum number of training rows to use.

	Returns:
		tuple: (results, train_labels, test_labels, gmm, svd, soft_labels)
	"""
	X_train_sample = X_train_tfidf[:sample_n]
	svd = TruncatedSVD(n_components=min(svd_components, X_train_sample.shape[1] - 1, X_train_sample.shape[0] - 1), random_state=RANDOM_STATE)
	X_reduced = normalize(svd.fit_transform(X_train_sample))
	gmm = GaussianMixture(
		n_components=n_components,
		covariance_type="diag",
		max_iter=200,
		random_state=RANDOM_STATE,
	)
	gmm.fit(X_reduced)
	labels = gmm.predict(X_reduced)
	soft_labels = gmm.predict_proba(X_reduced)
	sil_score = silhouette_score(X_reduced, labels, metric="cosine", sample_size=min(1000, X_reduced.shape[0]))
	bic_score = gmm.bic(X_reduced)
	aic_score = gmm.aic(X_reduced)
	X_test_reduced = normalize(svd.transform(X_test_tfidf[:1000]))
	test_labels = gmm.predict(X_test_reduced)
	results = {
		"n_components": n_components,
		"silhouette": float(sil_score),
		"bic": float(bic_score),
		"aic": float(aic_score),
		"cluster_sizes": [int((labels == index).sum()) for index in range(n_components)],
		"soft_label_mean_confidence": float(soft_labels.max(axis=1).mean()),
	}
	return results, labels, test_labels, gmm, svd, soft_labels


def plot_gmm_clusters(X_reduced_2d, labels, soft_labels, output_path: Path) -> None:
	"""Plot hard and soft GMM assignments in two dimensions."""
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	colors = ["#E8593C", "#3B8BD4", "#2E8B57", "#8C564B"]
	for cluster_id in range(len(set(labels))):
		mask = labels == cluster_id
		axes[0].scatter(
			X_reduced_2d[mask, 0],
			X_reduced_2d[mask, 1],
			c=colors[cluster_id % len(colors)],
			alpha=0.4,
			s=8,
			label=f"Cluster {cluster_id}",
		)
	axes[0].set_title("GMM Hard Assignments")
	axes[0].legend()
	confidence = soft_labels.max(axis=1)
	scatter = axes[1].scatter(
		X_reduced_2d[:, 0],
		X_reduced_2d[:, 1],
		c=confidence,
		cmap="RdYlGn",
		alpha=0.5,
		s=8,
		vmin=0.5,
		vmax=1.0,
	)
	plt.colorbar(scatter, ax=axes[1], label="Assignment confidence")
	axes[1].set_title("GMM Soft Assignment Confidence")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close()


def main() -> None:
	"""Run the Gaussian Mixture clustering experiment and persist artifacts."""
	print("=" * 70)
	print("Unsupervised Learning: Gaussian Mixture Clustering for Answer Verification")
	print("=" * 70)
	X_test, y_test = load_data("test")
	X_train, y_train = load_data("train")
	X_train, y_train = sample_rows(X_train, y_train, TRAIN_SAMPLE_SIZE)
	X_test, y_test = sample_rows(X_test, y_test, TEST_SAMPLE_SIZE)
	results, train_labels, test_labels, gmm, svd, soft_labels = run_gmm_clustering(
		X_train,
		X_test,
		n_components=2,
		svd_components=50,
		sample_n=X_train.shape[0],
	)
	X_test_reduced = normalize(svd.transform(X_test[: min(1000, X_test.shape[0])]))
	test_soft_labels = gmm.predict_proba(X_test_reduced)
	plot_path = UNSUPERVISED_DIR / "gmm_pca_visualization.png"
	plot_gmm_clusters(X_test_reduced[:, :2], test_labels[: X_test_reduced.shape[0]], test_soft_labels, plot_path)
	output = {
		"model": "GaussianMixture (diag covariance + TruncatedSVD)",
		"task": "Answer Verification (Unsupervised)",
		"results": results,
		"interpretation": {
			"silhouette_score": "Higher is better; values near zero indicate weak cluster separation.",
			"bic": "Lower is better; compares model fit penalized by complexity.",
			"aic": "Lower is better; compares model fit with a lighter complexity penalty than BIC.",
		},
	}
	with open(UNSUPERVISED_DIR / "gmm_results.json", "w", encoding="utf-8") as handle:
		json.dump(output, handle, indent=2)
	np.save(UNSUPERVISED_DIR / "gmm_labels_test.npy", test_labels)
	joblib.dump(gmm, UNSUPERVISED_DIR / "gmm_model.pkl")
	joblib.dump(svd, UNSUPERVISED_DIR / "gmm_svd.pkl")
	print("Saved GMM artifacts to", UNSUPERVISED_DIR)


if __name__ == "__main__":
	main()
