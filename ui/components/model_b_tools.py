"""Utilities used by the Streamlit UI for Model B inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from src.model_b_inference import generate_full_quiz, load_model_b_artifacts
except ImportError:  # pragma: no cover - fallback for direct script execution
    from model_b_inference import generate_full_quiz, load_model_b_artifacts


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_B_DIR = ROOT_DIR / "models" / "model_b" / "traditional"


def get_model_b_artifact_status() -> dict[str, Any]:
    """Return Model B artifact presence status for UI display."""
    expected = [
        "distractor_best.pkl",
        "hint_classifier.pkl",
        "hint_regressor.pkl",
        "vocab.pkl",
        "candidate_bank.pkl",
    ]
    present = [name for name in expected if (MODEL_B_DIR / name).exists()]
    missing = [name for name in expected if name not in present]
    return {
        "model_dir": str(MODEL_B_DIR),
        "present": present,
        "missing": missing,
    }


def run_model_b_generation(article: str, question: str, answer_text: str) -> dict[str, Any]:
    """Run end-to-end Model B quiz generation."""
    artifacts = load_model_b_artifacts(ROOT_DIR)
    return generate_full_quiz(article=article, question=question, answer_text=answer_text, artifacts=artifacts)
