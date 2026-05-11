"""Template-based question generation helpers for Model A.

The pipeline is classical-ML only: it extracts candidate sentences from the
passage, turns them into WH-style questions, and ranks the generated questions
with a lightweight logistic-regression classifier.
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
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "model_a" / "traditional"
QUESTION_RANKER_PATH = MODEL_DIR / "question_ranker.joblib"
QUESTION_RANKER_PKL_PATH = MODEL_DIR / "question_ranker.pkl"

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

WH_TEMPLATES = {
    "who": ["Who {verb} {rest}?", "Who was responsible for {noun_phrase}?"],
    "what": ["What {verb} {rest}?", "What is the main idea about {noun_phrase}?"],
    "where": ["Where did {event} take place?", "Where {verb} {rest}?"],
    "when": ["When did {event} happen?", "When {verb} {rest}?"],
    "why": ["Why {verb} {rest}?", "Why did {noun_phrase} occur?"],
    "how": ["How did {noun_phrase} {verb}?", "How {verb} {rest}?"],
}

PERSON_HINTS = {"city", "country", "state", "province", "village", "town", "school", "university", "park", "river", "mountain", "sea", "ocean", "street", "road", "airport", "station"}
LOCATION_MARKERS = {"in", "at", "on", "from", "to", "into", "inside", "within", "near"}
TIME_MARKERS = {"year", "years", "month", "months", "day", "days", "week", "weeks", "hour", "hours", "minute", "minutes", "today", "yesterday", "tomorrow"}
MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
COMMON_VERBS = {
    "is", "are", "was", "were", "am", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "go", "goes", "went", "come", "came", "make", "made",
    "take", "took", "give", "gave", "find", "found", "say", "said", "play", "played",
    "live", "lived", "work", "worked", "study", "studied", "build", "built", "win", "won",
    "begin", "began", "become", "became", "happen", "happened", "occur", "occurred", "move", "moved",
}


def _split_sentences(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    normalized = str(text).replace("\n", " ").strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", str(text)) if token]


def _tokenize_no_stopwords(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in STOPWORDS]


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        cleaned = _normalize_whitespace(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(cleaned)
    return unique_values


def _normalize_whitespace(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _strip_answer(sentence: str, answer: str) -> str:
    if not answer:
        return _normalize_whitespace(sentence)
    cleaned = re.sub(re.escape(answer), "___", sentence, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _normalize_whitespace(cleaned)


def extract_candidate_sentences(article: str, answer: str) -> list[dict[str, Any]]:
    """Split a passage into scored candidate sentences.

    Args:
        article (str): Passage text.
        answer (str): Known correct answer text.

    Returns:
        list[dict[str, Any]]: Sentence records sorted by combined relevance.
    """
    sentences = _split_sentences(article)
    if not sentences:
        return []

    answer_tokens = set(_tokenize_no_stopwords(answer))
    total_sentences = max(len(sentences), 1)
    scored: list[dict[str, Any]] = []

    for index, sentence in enumerate(sentences):
        sentence_tokens = set(_tokenize_no_stopwords(sentence))
        overlap_score = len(sentence_tokens & answer_tokens) / (len(answer_tokens) + 1)
        position_score = 1.0 - (index / total_sentences)
        length_score = min(len(sentence.split()) / 20.0, 1.0)
        combined_score = 0.5 * overlap_score + 0.3 * position_score + 0.2 * length_score
        scored.append(
            {
                "sentence": sentence,
                "overlap_score": float(overlap_score),
                "position_score": float(position_score),
                "length_score": float(length_score),
                "combined_score": float(combined_score),
                "index": index,
            }
        )

    return sorted(scored, key=lambda item: item["combined_score"], reverse=True)


def _infer_wh_word(answer: str) -> str:
    answer_text = _normalize_whitespace(answer)
    answer_lower = answer_text.lower()
    answer_tokens = _tokenize(answer_text)

    if not answer_text:
        return "what"
    if re.search(r"\b\d{4}\b", answer_text) or any(char.isdigit() for char in answer_text):
        return "when"
    if any(month in answer_lower for month in MONTH_NAMES) or any(token in TIME_MARKERS for token in answer_tokens):
        return "when"
    if any(token in PERSON_HINTS for token in answer_tokens) or any(marker in answer_tokens for marker in LOCATION_MARKERS):
        return "where"
    if answer_text[:1].isupper() and len(answer_tokens) <= 4:
        return "who"
    if len(answer_tokens) >= 2 and all(token[:1].isupper() for token in answer_text.split() if token):
        return "who"
    if any(marker in answer_tokens for marker in LOCATION_MARKERS):
        return "where"
    return "what" if len(answer_tokens) % 2 == 0 else "how"


def _guess_verb(sentence: str) -> str:
    tokens = _tokenize(sentence)
    for token in tokens:
        if token in COMMON_VERBS:
            return token
    if len(tokens) >= 2:
        return tokens[1]
    return "is"


def apply_templates(sentence: str, answer: str) -> list[str]:
    """Generate a small candidate set of WH-style questions.

    Args:
        sentence (str): Candidate source sentence.
        answer (str): Known correct answer text.

    Returns:
        list[str]: 2-3 candidate questions.
    """
    sentence_clean = _normalize_whitespace(sentence)
    answer_clean = _normalize_whitespace(answer)
    sentence_without_answer = _strip_answer(sentence_clean, answer_clean)
    wh_word = _infer_wh_word(answer_clean)
    verb = _guess_verb(sentence_without_answer or sentence_clean)
    rest = sentence_without_answer.rstrip(".?!") if sentence_without_answer else sentence_clean.rstrip(".?!")
    noun_phrase = answer_clean or "the passage"
    event = answer_clean or rest

    templates = WH_TEMPLATES.get(wh_word, WH_TEMPLATES["what"])
    question_candidates = [
        f"What does '___' refer to in: {sentence_without_answer}?" if sentence_without_answer else f"What does '___' refer to in: {sentence_clean}?",
        f"According to the passage, what is '{answer_clean}'?" if answer_clean else f"According to the passage, what is being described?",
    ]

    template_map = {
        "verb": verb,
        "rest": rest,
        "noun_phrase": noun_phrase,
        "event": event,
    }
    for template in templates:
        question_candidates.append(template.format(**template_map))

    return _unique_in_order(question_candidates)


def _simple_bleu1(candidate: str, reference: str) -> float:
    candidate_tokens = _tokenize(candidate)
    reference_tokens = _tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    ref_counts = Counter(reference_tokens)
    matches = 0
    for token in candidate_tokens:
        if ref_counts[token] > 0:
            matches += 1
            ref_counts[token] -= 1

    precision = matches / len(candidate_tokens)
    if precision <= 0.0:
        return 0.0

    brevity_penalty = 1.0 if len(candidate_tokens) > len(reference_tokens) else float(np.exp(1.0 - (len(reference_tokens) / max(len(candidate_tokens), 1))))
    return float(brevity_penalty * precision)


def _question_features(article: str, question: str, answer: str) -> np.ndarray:
    article_tokens = _tokenize(article)
    question_tokens = _tokenize(question)
    answer_tokens = _tokenize(answer)
    article_set = set(article_tokens)
    question_set = set(question_tokens)

    q_length = len(question_tokens)
    q_has_wh = 1.0 if question_tokens and question_tokens[0] in {"who", "what", "where", "when", "why", "how"} else 0.0
    answer_in_article = 1.0 if answer.lower() in article.lower() else 0.0
    overlap_q_article = len(question_set & article_set) / (len(article_set) + 1)
    answer_length = float(len(answer_tokens))

    return np.array([q_length, q_has_wh, answer_in_article, overlap_q_article, answer_length], dtype=np.float32)


def _extract_answer_text(row: pd.Series | dict[str, Any]) -> str:
    answer_label = str(row.get("answer", "")).strip().upper()
    if answer_label in {"A", "B", "C", "D"}:
        return str(row.get(answer_label, "")).strip()
    return str(row.get("answer_text", "")).strip()


def _build_training_rows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in frame.iterrows():
        article = str(row.get("article", ""))
        gold_question = str(row.get("question", ""))
        answer_text = _extract_answer_text(row)
        if not article.strip() or not gold_question.strip() or not answer_text.strip():
            continue

        candidate_sentences = extract_candidate_sentences(article, answer_text)[:3]
        for sentence_item in candidate_sentences:
            for candidate_question in apply_templates(sentence_item["sentence"], answer_text):
                features.append(_question_features(article, candidate_question, answer_text))
                labels.append(1 if _simple_bleu1(candidate_question, gold_question) > 0.3 else 0)

    if not features:
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def _rank_candidates(article: str, answer: str, candidates: list[dict[str, Any]], model: Any | None) -> list[dict[str, Any]]:
    if not candidates:
        return []

    feature_matrix = np.vstack([
        _question_features(article, candidate["question"], answer)
        for candidate in candidates
    ]).astype(np.float32)

    expected_features = int(getattr(model, "n_features_in_", feature_matrix.shape[1])) if model is not None else feature_matrix.shape[1]
    if feature_matrix.shape[1] < expected_features:
        padding = np.zeros((feature_matrix.shape[0], expected_features - feature_matrix.shape[1]), dtype=np.float32)
        feature_matrix = np.hstack([feature_matrix, padding])
    elif feature_matrix.shape[1] > expected_features:
        feature_matrix = feature_matrix[:, :expected_features]

    if model is None:
        scores = np.linspace(1.0, 0.0, num=len(candidates), endpoint=True)
    else:
        scores = model.predict_proba(feature_matrix)[:, 1]

    ranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["score"] = float(score)
        ranked.append(item)
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def generate_question_details(article: str, answer: str, ranker_model: Any | None = None) -> dict[str, Any]:
    """Run the full question-generation pipeline and keep metadata."""
    candidate_sentences = extract_candidate_sentences(article, answer)[:3]
    candidates: list[dict[str, Any]] = []
    for sentence_item in candidate_sentences:
        for question_text in apply_templates(sentence_item["sentence"], answer):
            candidates.append(
                {
                    "question": question_text,
                    "source_sentence": sentence_item["sentence"],
                    "sentence_score": sentence_item["combined_score"],
                    "wh_word": _infer_wh_word(answer),
                }
            )

    ranked = _rank_candidates(article, answer, candidates, ranker_model)
    if not ranked:
        fallback_question = f"According to the passage, what is '{_normalize_whitespace(answer)}'?"
        return {
            "question": fallback_question,
            "source_sentence": candidate_sentences[0]["sentence"] if candidate_sentences else "",
            "wh_word": _infer_wh_word(answer),
            "candidates": [],
        }

    best = ranked[0]
    return {
        "question": best["question"],
        "source_sentence": best.get("source_sentence", ""),
        "wh_word": best.get("wh_word", _infer_wh_word(answer)),
        "candidates": ranked,
    }


def generate_question(article: str, answer: str, ranker_model: Any | None = None) -> str:
    """Return the best generated question for the passage-answer pair."""
    return generate_question_details(article, answer, ranker_model=ranker_model)["question"]


def train_question_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    save_path: Path | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train a logistic-regression question ranker and optionally evaluate on validation data."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or QUESTION_RANKER_PATH

    X_train, y_train = _build_training_rows(train_df)
    if X_train.size == 0:
        raise ValueError("No training examples were generated for the question ranker.")

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    joblib.dump(model, save_path)
    if save_path.suffix != ".pkl":
        joblib.dump(model, QUESTION_RANKER_PKL_PATH)

    metrics: dict[str, float] = {}
    if val_df is not None and not val_df.empty:
        total = 0
        correct = 0
        for _, row in val_df.iterrows():
            article = str(row.get("article", ""))
            gold_question = str(row.get("question", ""))
            answer_text = _extract_answer_text(row)
            if not article.strip() or not gold_question.strip() or not answer_text.strip():
                continue
            details = generate_question_details(article, answer_text, ranker_model=model)
            total += 1
            correct += int(_simple_bleu1(details["question"], gold_question) > 0.3)
        metrics["val_accuracy"] = float(correct / total) if total else 0.0

    return model, metrics


def load_question_ranker(model_path: Path | None = None) -> Any:
    """Load the persisted question ranker."""
    candidates = [
        model_path or QUESTION_RANKER_PATH,
        QUESTION_RANKER_PKL_PATH,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return joblib.load(candidate)
    raise FileNotFoundError(f"Missing question ranker artifact. Checked: {', '.join(str(path) for path in candidates if path)}")
