"""
Calibration analysis for the selected Model A candidates.
Generates reliability plots and calibration metrics for:
- Random Forest
- Logistic Regression

Outputs are written to models/model_a/traditional/calibration/
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "model_a" / "traditional"
OUT_DIR = MODELS_DIR / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "random_forest": MODELS_DIR / "model_a_random_forest_intensive_tuned.pkl",
    "logistic_regression": MODELS_DIR / "model_a_logistic_regression_intensive_tuned.pkl",
}


def load_split(split: str):
    x = load_npz(DATA_DIR / f"model_a_{split}_X.npz")
    y = np.load(DATA_DIR / f"y_{split}.npy")
    return x, y


def load_model(name: str):
    return joblib.load(MODEL_FILES[name])


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    total = len(y_true)
    ece = 0.0

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        prob_mean = float(np.mean(y_prob[mask]))
        true_mean = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / total) * abs(prob_mean - true_mean)

    return float(ece)


def plot_reliability(ax, y_true, y_prob, label: str):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="uniform",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_predicted_value, fraction_of_positives, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)


def analyze_split(split: str):
    x, y_true = load_split(split)
    results = {}

    for model_name in MODEL_FILES:
        model = load_model(model_name)
        y_prob = model.predict_proba(x)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results[model_name] = {
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "log_loss": float(log_loss(y_true, np.column_stack([1 - y_prob, y_prob]))),
            "ece_10bin": float(expected_calibration_error(y_true, y_prob, n_bins=10)),
            "accuracy_at_0_5": float(np.mean(y_pred == y_true)),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, model_name in zip(axes, MODEL_FILES):
        model = load_model(model_name)
        y_prob = model.predict_proba(x)[:, 1]
        plot_reliability(ax, y_true, y_prob, model_name.replace("_", " ").title())
        ax.set_title(f"{model_name.replace('_', ' ').title()} ({split})")
        ax.legend(loc="lower right")

    fig.suptitle(f"Calibration / Reliability Curves - {split.title()} Split")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = OUT_DIR / f"reliability_{split}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return results, png_path


def main() -> int:
    summary = {}
    artifacts = {}
    for split in ("val", "test"):
        results, png_path = analyze_split(split)
        summary[split] = results
        artifacts[split] = str(png_path)

    summary_path = OUT_DIR / "calibration_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump({"metrics": summary, "plots": artifacts}, fh, indent=2)

    print(f"Saved calibration summary: {summary_path}")
    for split, path in artifacts.items():
        print(f"Saved {split} plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
