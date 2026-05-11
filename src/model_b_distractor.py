"""Classical ML distractor generation for Model B.

This module combines article-derived candidates, a one-hot cosine retrieval path,
Word2Vec nearest neighbours, and a lightweight ranker to produce plausible but
incorrect answer options.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "model_b" / "traditional"
VOCAB_PATH = MODEL_DIR / "vocab.joblib"
VOCAB_PKL_PATH = MODEL_DIR / "vocab.pkl"
DISTRACTOR_RANKER_PATH = MODEL_DIR / "distractor_ranker.joblib"
DISTRACTOR_RANKER_PKL_PATH = MODEL_DIR / "distractor_ranker.pkl"
WORD2VEC_CACHE_DIR = MODEL_DIR / "word2vec_cache"
WORD2VEC_CACHE_PATH = WORD2VEC_CACHE_DIR / "word2vec-google-news-300.kv"

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

_W2V_MODEL: Any | None = None


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", str(text).lower())
    return _normalize_whitespace(cleaned)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", str(text)) if token]


def _tokenize_filtered(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in STOPWORDS and len(token) > 1]


def _split_sentences(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", str(text).replace("\n", " ").strip()) if sentence.strip()]


def _one_hot(tokens: list[str], word2idx: dict[str, int], size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    for token in tokens:
        index = word2idx.get(token)
        if index is not None:
            vector[index] = 1.0
    return vector


def _cosine(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denom)


def _char_bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", " ", _clean_text(text))
    padded = f" {cleaned} "
    return {padded[index : index + 2] for index in range(max(len(padded) - 1, 0))}


def _first_token(text: str) -> str:
    tokens = _tokenize_filtered(text)
    return tokens[0] if tokens else ""


def _sentence_index_for_candidate(article: str, candidate: str) -> int:
    candidate_clean = _clean_text(candidate)
    if not candidate_clean:
        return 0
    for index, sentence in enumerate(_split_sentences(article)):
        if candidate_clean in _clean_text(sentence):
            return index
    return len(_split_sentences(article))


def _occurrence_count(article: str, candidate: str) -> int:
    article_clean = _clean_text(article)
    candidate_clean = _clean_text(candidate)
    if not article_clean or not candidate_clean:
        return 0
    if " " in candidate_clean:
        return max(article_clean.count(candidate_clean), 0)
    return Counter(_tokenize_filtered(article)).get(candidate_clean, 0)


def _answer_token_vector(correct_answer: str, word2idx: dict[str, int], size: int) -> np.ndarray:
    answer_tokens = [token for token in _tokenize_filtered(correct_answer) if token in word2idx]
    return _one_hot(answer_tokens, word2idx, size)


def _candidate_features(
    candidate: str,
    question: str,
    correct_answer: str,
    article: str,
    vocab: list[str],
    word2idx: dict[str, int],
    w2v_model: Any | None,
    source_is_w2v: int,
) -> np.ndarray:
    candidate_tokens = [token for token in _tokenize_filtered(candidate) if token in word2idx]
    answer_tokens = [token for token in _tokenize_filtered(correct_answer) if token in word2idx]
    size = max(len(vocab), 1)

    candidate_vec = _one_hot(candidate_tokens, word2idx, size)
    answer_vec = _one_hot(answer_tokens, word2idx, size)
    cosine_sim_to_answer = _cosine(candidate_vec, answer_vec)

    freq_in_article = float(_occurrence_count(article, candidate))
    char_length_diff = float(abs(len(candidate) - len(correct_answer)))
    is_in_question = 1.0 if _clean_text(candidate) in _clean_text(question) else 0.0
    position_first_occurrence = float(_sentence_index_for_candidate(article, candidate))

    w2v_similarity = 0.0
    if w2v_model is not None:
        candidate_token = _first_token(candidate)
        answer_token = _first_token(correct_answer)
        if candidate_token and answer_token:
            if candidate_token in getattr(w2v_model, "key_to_index", {}) and answer_token in getattr(w2v_model, "key_to_index", {}):
                try:
                    w2v_similarity = float(w2v_model.similarity(candidate_token, answer_token))
                except Exception:
                    w2v_similarity = 0.0

    return np.array(
        [
            cosine_sim_to_answer,
            w2v_similarity,
            freq_in_article,
            char_length_diff,
            is_in_question,
            position_first_occurrence,
            float(source_is_w2v),
        ],
        dtype=np.float32,
    )


def _candidate_pool_from_article(article: str, vocab: list[str], top_n: int = 120) -> list[str]:
    article_tokens = _tokenize_filtered(article)
    token_counts = Counter(token for token in article_tokens if token in vocab)

    candidates: list[str] = [token for token, _ in token_counts.most_common(top_n)]

    sentence_candidates: list[str] = []
    for sentence in _split_sentences(article):
        sentence_tokens = [token for token in _tokenize_filtered(sentence) if token in vocab]
        for index in range(len(sentence_tokens) - 1):
            sentence_candidates.append(f"{sentence_tokens[index]} {sentence_tokens[index + 1]}")
        for index in range(len(sentence_tokens) - 2):
            sentence_candidates.append(
                f"{sentence_tokens[index]} {sentence_tokens[index + 1]} {sentence_tokens[index + 2]}"
            )

    return list(dict.fromkeys(candidates + sentence_candidates))


def _candidate_pool_from_gold_options(row: pd.Series | dict[str, Any]) -> list[str]:
    answer_label = str(row.get("answer", "")).strip().upper()
    options = [str(row.get(label, "")).strip() for label in ["A", "B", "C", "D"] if label != answer_label]
    return [option for option in options if option]


def _label_candidate(candidate: str, gold_distractors: list[str]) -> int:
    candidate_clean = _clean_text(candidate)
    if not candidate_clean:
        return 0
    for gold in gold_distractors:
        gold_clean = _clean_text(gold)
        if not gold_clean:
            continue
        if candidate_clean == gold_clean or candidate_clean in gold_clean or gold_clean in candidate_clean:
            return 1
    return 0


def _pad_feature_matrix(matrix: np.ndarray, expected_features: int) -> np.ndarray:
    if matrix.shape[1] == expected_features:
        return matrix
    if matrix.shape[1] > expected_features:
        return matrix[:, :expected_features]
    padding = np.zeros((matrix.shape[0], expected_features - matrix.shape[1]), dtype=np.float32)
    return np.hstack([matrix, padding])


def _score_candidates(
    candidates: list[str],
    article: str,
    question: str,
    correct_answer: str,
    vocab: list[str],
    word2idx: dict[str, int],
    ranker_model: Any,
    w2v_model: Any | None,
    source_is_w2v: int,
) -> list[tuple[str, float, dict[str, float]]]:
    if not candidates:
        return []

    feature_rows = []
    metadata_rows: list[dict[str, float]] = []
    for candidate in candidates:
        features = _candidate_features(
            candidate=candidate,
            question=question,
            correct_answer=correct_answer,
            article=article,
            vocab=vocab,
            word2idx=word2idx,
            w2v_model=w2v_model,
            source_is_w2v=source_is_w2v,
        )
        feature_rows.append(features)
        metadata_rows.append(
            {
                "candidate": candidate,
                "cosine_sim_to_answer": float(features[0]),
                "w2v_similarity": float(features[1]),
                "freq_in_article": float(features[2]),
                "char_length_diff": float(features[3]),
                "is_in_question": float(features[4]),
                "position_first_occurrence": float(features[5]),
                "source_is_w2v": float(features[6]),
            }
        )

    feature_matrix = np.vstack(feature_rows).astype(np.float32)
    expected_features = int(getattr(ranker_model, "n_features_in_", feature_matrix.shape[1]))
    feature_matrix = _pad_feature_matrix(feature_matrix, expected_features)
    scores = ranker_model.predict_proba(feature_matrix)[:, 1]

    return [
        (
            candidate,
            float(score) + (0.2 if source_is_w2v else 0.0),
            metadata,
        )
        for candidate, score, metadata in zip(candidates, scores, metadata_rows)
    ]


def build_vocab_and_cooccurrence(train_df: pd.DataFrame, vocab_size: int = 5000) -> tuple[list[str], dict[str, int], dict[str, Counter[str]]]:
    """Build the Model B vocabulary and a simple co-occurrence dictionary."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    token_counts: Counter[str] = Counter()
    cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)

    for article in train_df["article"].fillna("").astype(str):
        tokens = _tokenize_filtered(article)
        token_counts.update(tokens)
        for index, token in enumerate(tokens):
            window_end = min(index + 6, len(tokens))
            for context_index in range(index + 1, window_end):
                context_token = tokens[context_index]
                if token == context_token:
                    continue
                cooccurrence[token][context_token] += 1
                cooccurrence[context_token][token] += 1

    vocab = [token for token, _ in token_counts.most_common(vocab_size)]
    word2idx = {token: index for index, token in enumerate(vocab)}

    payload = {
        "vocab": vocab,
        "word2idx": word2idx,
        "vocab_size": vocab_size,
    }
    joblib.dump(payload, VOCAB_PATH)
    if VOCAB_PATH.suffix != ".pkl":
        joblib.dump(payload, VOCAB_PKL_PATH)

    print(f"Vocab size: {len(vocab)}")
    return vocab, word2idx, cooccurrence


def get_distractor_candidates(article: str, correct_answer: str, vocab: list[str], word2idx: dict[str, int], top_n: int = 20) -> list[str]:
    """Retrieve one-hot cosine candidates from the passage vocabulary."""
    article_tokens = [token for token in _tokenize_filtered(article) if token in word2idx]
    if not article_tokens:
        return []

    answer_lower = _clean_text(correct_answer)
    filtered_tokens = []
    seen: set[str] = set()
    for token in article_tokens:
        if token in seen:
            continue
        if token in answer_lower:
            continue
        seen.add(token)
        filtered_tokens.append(token)

    if not filtered_tokens:
        return []

    answer_vec = _answer_token_vector(correct_answer, word2idx, len(vocab))
    max_freq = max(Counter(article_tokens).values()) if article_tokens else 1
    scored_candidates: list[tuple[str, float]] = []
    for token in filtered_tokens:
        token_vec = _one_hot([token], word2idx, len(vocab))
        cosine_sim = _cosine(answer_vec, token_vec)
        frequency = Counter(article_tokens).get(token, 0) / max_freq
        score = 0.4 * cosine_sim + 0.6 * frequency
        scored_candidates.append((token, float(score)))

    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    return [token for token, _ in scored_candidates[:top_n]]


def load_word2vec_model():
    """Load and cache the Google News Word2Vec vectors via Gensim."""
    global _W2V_MODEL
    if _W2V_MODEL is not None:
        return _W2V_MODEL

    try:
        from gensim.models import KeyedVectors
        import gensim.downloader as api
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("gensim is required to load Word2Vec embeddings") from exc

    WORD2VEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if WORD2VEC_CACHE_PATH.exists():
        _W2V_MODEL = KeyedVectors.load(str(WORD2VEC_CACHE_PATH), mmap="r")
        return _W2V_MODEL

    print("Loading Word2Vec model... (first run may take a few minutes)")
    model = api.load("word2vec-google-news-300")
    try:
        model.save(str(WORD2VEC_CACHE_PATH))
    except Exception:
        pass
    _W2V_MODEL = model
    return _W2V_MODEL


def get_word2vec_distractors(correct_answer: str, article: str, w2v_model: Any, top_n: int = 20) -> list[str]:
    """Retrieve nearest-neighbour distractors from a pre-trained Word2Vec model."""
    answer_tokens = [token for token in _tokenize_filtered(correct_answer)]
    article_lower = _clean_text(article)
    answer_lower = _clean_text(correct_answer)
    if not answer_tokens:
        return []

    candidates: dict[str, float] = {}
    vocab = getattr(w2v_model, "key_to_index", {})

    def add_similar(tokens: list[str]) -> None:
        usable_tokens = [token for token in tokens if token in vocab]
        if not usable_tokens:
            return
        try:
            if len(usable_tokens) == 1:
                similar = w2v_model.most_similar(usable_tokens[0], topn=30)
            else:
                similar = w2v_model.most_similar(positive=usable_tokens, topn=30)
        except Exception:
            return
        for word, score in similar:
            cleaned_word = _clean_text(word)
            if not cleaned_word or cleaned_word in STOPWORDS or len(cleaned_word) < 3:
                continue
            if cleaned_word in article_lower or cleaned_word in answer_lower:
                continue
            previous = candidates.get(cleaned_word)
            if previous is None or score > previous:
                candidates[cleaned_word] = float(score)

    for token in answer_tokens:
        add_similar([token])

    if len(answer_tokens) > 1:
        add_similar(answer_tokens)

    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _ in ranked[:top_n]]


def _merge_candidate_sources(
    word2vec_candidates: list[str],
    one_hot_candidates: list[str],
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for candidate in word2vec_candidates + one_hot_candidates:
        cleaned = _normalize_whitespace(candidate)
        key = _clean_text(cleaned)
        if not cleaned or key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged


def _apply_diversity_penalty(scored_candidates: list[tuple[str, float, dict[str, float]]]) -> list[tuple[str, float, dict[str, float]]]:
    adjusted = list(scored_candidates)
    for outer_index in range(len(adjusted)):
        candidate_outer, score_outer, metadata_outer = adjusted[outer_index]
        for inner_index in range(outer_index + 1, len(adjusted)):
            candidate_inner, score_inner, metadata_inner = adjusted[inner_index]
            bigrams_outer = _char_bigrams(candidate_outer)
            bigrams_inner = _char_bigrams(candidate_inner)
            if not bigrams_outer or not bigrams_inner:
                continue
            overlap = len(bigrams_outer & bigrams_inner) / max(len(bigrams_outer | bigrams_inner), 1)
            if overlap <= 0.5:
                continue
            if score_outer >= score_inner:
                adjusted[inner_index] = (candidate_inner, score_inner * 0.5, metadata_inner)
            else:
                adjusted[outer_index] = (candidate_outer, score_outer * 0.5, metadata_outer)
    return adjusted


def generate_distractors_combined(
    article: str,
    question: str,
    correct_answer: str,
    vocab: list[str],
    word2idx: dict[str, int],
    ranker_model: Any,
    w2v_model: Any,
    n: int = 3,
) -> list[str]:
    """Generate distractors by merging Word2Vec and one-hot retrieval paths."""
    word2vec_candidates = get_word2vec_distractors(correct_answer, article, w2v_model, top_n=20)
    one_hot_candidates = get_distractor_candidates(article, correct_answer, vocab, word2idx, top_n=20)
    merged_candidates = _merge_candidate_sources(word2vec_candidates, one_hot_candidates)

    if not merged_candidates:
        return []

    source_lookup = {candidate: 1 for candidate in word2vec_candidates}
    scored_word2vec = _score_candidates(
        [candidate for candidate in merged_candidates if source_lookup.get(candidate)],
        article,
        question,
        correct_answer,
        vocab,
        word2idx,
        ranker_model,
        w2v_model,
        source_is_w2v=1,
    )
    scored_one_hot = _score_candidates(
        [candidate for candidate in merged_candidates if not source_lookup.get(candidate)],
        article,
        question,
        correct_answer,
        vocab,
        word2idx,
        ranker_model,
        w2v_model,
        source_is_w2v=0,
    )
    scored_candidates = scored_word2vec + scored_one_hot
    scored_candidates = _apply_diversity_penalty(scored_candidates)
    scored_candidates.sort(key=lambda item: item[1], reverse=True)

    selected: list[str] = []
    answer_clean = _clean_text(correct_answer)
    for candidate, score, metadata in scored_candidates:
        if len(selected) >= n:
            break
        candidate_clean = _clean_text(candidate)
        if not candidate_clean or candidate_clean == answer_clean:
            continue
        if any(_char_bigrams(candidate) and _char_bigrams(existing) and len(_char_bigrams(candidate) & _char_bigrams(existing)) / max(len(_char_bigrams(candidate) | _char_bigrams(existing)), 1) > 0.5 for existing in selected):
            continue
        selected.append(candidate)

    if len(selected) < n:
        for candidate in one_hot_candidates:
            if len(selected) >= n:
                break
            candidate_clean = _clean_text(candidate)
            if not candidate_clean or candidate_clean == answer_clean:
                continue
            if candidate_clean in _clean_text(article):
                if candidate not in selected:
                    selected.append(candidate)

    if len(selected) < n:
        article_tokens = [token for token in _tokenize_filtered(article) if token not in _clean_text(correct_answer)]
        for token in article_tokens:
            if len(selected) >= n:
                break
            if token not in selected:
                selected.append(token)

    return selected[:n]


def generate_distractors(
    article: str,
    question: str,
    correct_answer: str,
    vocab: list[str],
    word2idx: dict[str, int],
    ranker_model: Any,
    w2v_model: Any,
    n: int = 3,
) -> list[str]:
    """Backward-compatible alias for the combined distractor generator."""
    return generate_distractors_combined(
        article=article,
        question=question,
        correct_answer=correct_answer,
        vocab=vocab,
        word2idx=word2idx,
        ranker_model=ranker_model,
        w2v_model=w2v_model,
        n=n,
    )


def _build_training_candidates(article: str, row: pd.Series, vocab: list[str]) -> list[str]:
    answer_label = str(row.get("answer", "")).strip().upper()
    correct_answer = str(row.get(answer_label, "")).strip() if answer_label in {"A", "B", "C", "D"} else ""
    gold_distractors = _candidate_pool_from_gold_options(row)

    article_candidates = _candidate_pool_from_article(article, vocab)
    combined = list(dict.fromkeys(article_candidates + gold_distractors))
    if correct_answer:
        combined = [candidate for candidate in combined if _clean_text(candidate) != _clean_text(correct_answer)]
    return combined


def train_distractor_ranker(
    train_df: pd.DataFrame,
    vocab: list[str],
    word2idx: dict[str, int],
    val_df: pd.DataFrame | None = None,
    w2v_model: Any | None = None,
    save_path: Path | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train a logistic-regression distractor ranker and report validation metrics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or DISTRACTOR_RANKER_PATH
    if w2v_model is None:
        try:
            w2v_model = load_word2vec_model()
        except Exception:
            w2v_model = None

    features: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in train_df.iterrows():
        article = str(row.get("article", ""))
        question = str(row.get("question", ""))
        answer_label = str(row.get("answer", "")).strip().upper()
        correct_answer = str(row.get(answer_label, "")).strip() if answer_label in {"A", "B", "C", "D"} else ""
        gold_distractors = _candidate_pool_from_gold_options(row)

        candidate_pool = _build_training_candidates(article, row, vocab)
        if not candidate_pool:
            continue

        for candidate in candidate_pool:
            features.append(
                _candidate_features(
                    candidate=candidate,
                    question=question,
                    correct_answer=correct_answer,
                    article=article,
                    vocab=vocab,
                    word2idx=word2idx,
                    w2v_model=w2v_model,
                    source_is_w2v=0,
                )
            )
            labels.append(_label_candidate(candidate, gold_distractors))

    if not features:
        raise ValueError("No training candidates were generated for the distractor ranker.")

    X_train = np.vstack(features).astype(np.float32)
    y_train = np.asarray(labels, dtype=np.int32)
    # Guard against NaN / inf in feature matrix
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Candidate estimators to try. Random Forest often handles messy features better.
    candidates = {
        "logistic": Pipeline(steps=[("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))]),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    }

    best_model = None
    best_metrics: dict[str, float] = {}
    # If we have a validation set, evaluate each candidate and pick best by F1
    for name, estimator in candidates.items():
        try:
            est = estimator
            # Fit on training data
            est.fit(X_train, y_train)
        except Exception:
            continue

        # Default metrics if no val_df
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        if val_df is not None and not val_df.empty:
            y_true: list[int] = []
            y_pred: list[int] = []
            for _, row in val_df.iterrows():
                article = str(row.get("article", ""))
                question = str(row.get("question", ""))
                answer_label = str(row.get("answer", "")).strip().upper()
                correct_answer = str(row.get(answer_label, "")).strip() if answer_label in {"A", "B", "C", "D"} else ""
                gold_distractors = _candidate_pool_from_gold_options(row)
                candidate_pool = _build_training_candidates(article, row, vocab)
                for candidate in candidate_pool:
                    feats = _candidate_features(
                        candidate=candidate,
                        question=question,
                        correct_answer=correct_answer,
                        article=article,
                        vocab=vocab,
                        word2idx=word2idx,
                        w2v_model=w2v_model,
                        source_is_w2v=0,
                    )
                    feats = np.nan_to_num(feats.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
                    prob = est.predict_proba(feats)[:, 1][0] if hasattr(est, "predict_proba") else float(est.predict(feats)[0])
                    y_true.append(_label_candidate(candidate, gold_distractors))
                    y_pred.append(int(prob >= 0.5))

            if y_true:
                metrics = {
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                }

        # Choose best by F1
        if best_model is None or metrics.get("f1", 0.0) > best_metrics.get("f1", 0.0):
            best_model = est
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError("Failed to train any distractor ranker candidate")

    # Persist the selected best model
    joblib.dump(best_model, save_path)
    if save_path.suffix != ".pkl":
        joblib.dump(best_model, DISTRACTOR_RANKER_PKL_PATH)

    metrics = best_metrics

    metrics: dict[str, float] = {}
    if val_df is not None and not val_df.empty:
        y_true: list[int] = []
        y_pred: list[int] = []
        for _, row in val_df.iterrows():
            article = str(row.get("article", ""))
            question = str(row.get("question", ""))
            answer_label = str(row.get("answer", "")).strip().upper()
            correct_answer = str(row.get(answer_label, "")).strip() if answer_label in {"A", "B", "C", "D"} else ""
            gold_distractors = _candidate_pool_from_gold_options(row)
            candidate_pool = _build_training_candidates(article, row, vocab)
            for candidate in candidate_pool:
                features = _candidate_features(
                    candidate=candidate,
                    question=question,
                    correct_answer=correct_answer,
                    article=article,
                    vocab=vocab,
                    word2idx=word2idx,
                    w2v_model=w2v_model,
                    source_is_w2v=0,
                )
                y_true.append(_label_candidate(candidate, gold_distractors))
                y_pred.append(int(model.predict_proba(features.reshape(1, -1))[:, 1][0] >= 0.5))

        if y_true:
            metrics = {
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
        else:
            metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    print(
        "Training distractor ranker... Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Accuracy: {accuracy:.2f}".format(
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1=metrics.get("f1", 0.0),
            accuracy=metrics.get("accuracy", 0.0),
        )
    )
    return model, metrics
