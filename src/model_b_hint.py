"""Hint generation for Model B.

The hint scorer uses sentence-level classical features to rank supporting
sentences and then emits three graduated hints from general to near-explicit.
"""

from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "model_b" / "traditional"
HINT_SCORER_PATH = MODEL_DIR / "hint_scorer.joblib"
HINT_SCORER_PKL_PATH = MODEL_DIR / "hint_scorer.pkl"

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "i", "you", "he", "she", "it", "we", "they", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "its",
}

FALLBACK_HINT = "Read the passage carefully."


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", str(text)) if token]


def _tokenize_filtered(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in STOPWORDS and len(token) > 1]


def _split_sentences(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", str(text).replace("\n", " ").strip()) if sentence.strip()]


def _sentence_features(sentence: str, question: str, correct_answer: str, sentence_index: int, total_sentences: int) -> np.ndarray:
    sentence_tokens = set(_tokenize_filtered(sentence))
    question_tokens = _tokenize_filtered(question)
    answer_tokens = _tokenize_filtered(correct_answer)

    keyword_overlap = len(sentence_tokens & set(question_tokens)) / (len(question_tokens) + 1)
    answer_overlap = len(sentence_tokens & set(answer_tokens)) / (len(answer_tokens) + 1)
    position_score = 1.0 - (sentence_index / max(total_sentences, 1))
    sentence_length_norm = min(len(sentence.split()) / 25.0, 1.0)
    contains_answer = 1.0 if _normalize_whitespace(correct_answer).lower() in _normalize_whitespace(sentence).lower() else 0.0

    return np.array(
        [keyword_overlap, answer_overlap, position_score, sentence_length_norm, contains_answer],
        dtype=np.float32,
    )


def _pad_feature_matrix(matrix: np.ndarray, expected_features: int) -> np.ndarray:
    if matrix.shape[1] == expected_features:
        return matrix
    if matrix.shape[1] > expected_features:
        return matrix[:, :expected_features]
    padding = np.zeros((matrix.shape[0], expected_features - matrix.shape[1]), dtype=np.float32)
    return np.hstack([matrix, padding])


def _build_training_rows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in frame.iterrows():
        article = str(row.get("article", ""))
        question = str(row.get("question", ""))
        answer_label = str(row.get("answer", "")).strip().upper()
        correct_answer = str(row.get(answer_label, "")).strip() if answer_label in {"A", "B", "C", "D"} else ""
        if not article.strip() or not question.strip() or not correct_answer.strip():
            continue

        sentences = [sentence for sentence in _split_sentences(article) if len(sentence.split()) >= 5]
        total_sentences = len(sentences)
        if total_sentences == 0:
            continue

        for sentence_index, sentence in enumerate(sentences):
            features = _sentence_features(sentence, question, correct_answer, sentence_index, total_sentences)
            label = 1 if (features[4] >= 1.0 or features[0] > 0.4) else 0
            feature_rows.append(features)
            labels.append(label)

    if not feature_rows:
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.vstack(feature_rows), np.asarray(labels, dtype=np.int32)


def train_hint_scorer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    save_path: Path | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train the hint scorer and report validation R² plus precision."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or HINT_SCORER_PATH

    X_train, y_train = _build_training_rows(train_df)
    if X_train.size == 0:
        raise ValueError("No training examples were generated for the hint scorer.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)

    joblib.dump(model, save_path)
    if save_path.suffix != ".pkl":
        joblib.dump(model, HINT_SCORER_PKL_PATH)

    metrics = {"r2": 0.0, "precision": 0.0}
    if val_df is not None and not val_df.empty:
        X_val, y_val = _build_training_rows(val_df)
        if X_val.size:
            expected_features = int(getattr(model, "n_features_in_", X_val.shape[1]))
            X_val = _pad_feature_matrix(X_val, expected_features)
            probabilities = model.predict_proba(X_val)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = {
                "r2": float(r2_score(y_val, probabilities)),
                "precision": float(precision_score(y_val, predictions, zero_division=0)),
            }

    print(f"Training hint scorer... R²: {metrics['r2']:.2f}, Precision: {metrics['precision']:.2f}")
    return model, metrics


def _mask_answer(sentence: str, correct_answer: str, partial: bool = False) -> str:
    sentence_clean = _normalize_whitespace(sentence)
    answer_clean = _normalize_whitespace(correct_answer)
    if not answer_clean:
        return sentence_clean

    if partial and len(answer_clean) > 3:
        visible = max(1, len(answer_clean) // 2)
        replacement = f"{answer_clean[:visible]}[...]"
    else:
        replacement = "[...]"

    masked = re.sub(re.escape(answer_clean), replacement, sentence_clean, flags=re.IGNORECASE)
    return _normalize_whitespace(masked)


def _sentence_relevance(sentence: str, question: str, correct_answer: str, sentence_index: int, total_sentences: int, model: Any) -> float:
    features = _sentence_features(sentence, question, correct_answer, sentence_index, total_sentences)
    expected_features = int(getattr(model, "n_features_in_", features.shape[0]))
    feature_matrix = _pad_feature_matrix(features.reshape(1, -1), expected_features)
    return float(model.predict_proba(feature_matrix)[:, 1][0])


def generate_hints(article: str, question: str, correct_answer: str, hint_scorer_model: Any) -> list[str]:
    """Generate three graduated hints from the supporting article sentences."""
    sentences = [sentence for sentence in _split_sentences(article) if len(sentence.split()) >= 5]
    if not sentences:
        return [FALLBACK_HINT, FALLBACK_HINT, FALLBACK_HINT]

    total_sentences = len(sentences)
    scored_sentences = []
    for sentence_index, sentence in enumerate(sentences):
        score = _sentence_relevance(sentence, question, correct_answer, sentence_index, total_sentences, hint_scorer_model)
        scored_sentences.append((sentence, score, sentence_index))

    scored_sentences.sort(key=lambda item: item[1], reverse=True)
    unique_sentences: list[tuple[str, float, int]] = []
    for sentence, score, sentence_index in scored_sentences:
        if any(_normalize_whitespace(sentence).lower() == _normalize_whitespace(existing_sentence).lower() for existing_sentence, _, _ in unique_sentences):
            continue
        unique_sentences.append((sentence, score, sentence_index))
        if len(unique_sentences) >= 3:
            break

    while len(unique_sentences) < 3:
        unique_sentences.append((sentences[min(len(unique_sentences), len(sentences) - 1)], 0.0, min(len(unique_sentences), len(sentences) - 1)))

    top_sentence = unique_sentences[0][0]
    second_sentence = unique_sentences[1][0]
    third_sentence = unique_sentences[2][0]

    hint1 = _mask_answer(top_sentence, correct_answer, partial=False)
    hint2 = _mask_answer(second_sentence, correct_answer, partial=True)
    hint3 = f"The answer can be found in this part: '{_normalize_whitespace(top_sentence)[:80]}...'"

    cleaned_hints = [hint1, hint2, hint3]
    return [hint if hint else FALLBACK_HINT for hint in cleaned_hints][:3]
