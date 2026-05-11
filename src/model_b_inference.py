"""Model B inference utilities for distractor and hint generation.

This module ports the teammate notebook logic into a stable runtime API.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import random
import re
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)
FALLBACK_HINT = "Read the passage carefully."

_WORDNET = None
try:  # Optional dependency: keep runtime robust if nltk/wordnet is missing.
    from nltk.corpus import wordnet as _WORDNET  # type: ignore
except Exception:  # pragma: no cover
    _WORDNET = None


@dataclass(frozen=True)
class ModelBArtifacts:
    """Loaded Model B assets."""

    distractor_model: Any
    hint_classifier: Any
    hint_regressor: Any | None
    vocab: dict[str, int]
    candidate_bank: dict[str, list[str]]
    model_dir: Path


def clean_text(value: object) -> str:
    """Normalize text for feature extraction."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def tokenize(value: object) -> list[str]:
    return clean_text(value).split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t and t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    normalized = str(text).strip()
    normalized = normalized.replace("No.", "No")
    chunks = re.split(r"(?<=[.!?])\s+", normalized)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _clean_phrase(text: str) -> str:
    phrase = re.sub(r"\s+", " ", str(text).strip())
    phrase = phrase.strip("-_,.;:!?\"'()[]{}")
    if not phrase:
        return ""
    return phrase[0].upper() + phrase[1:] if len(phrase) > 1 else phrase.upper()


def _candidate_token_count(text: str) -> int:
    return len(tokenize(text))


def _is_candidate_valid(candidate: str, answer_text: str) -> bool:
    cand = _clean_phrase(candidate)
    if not cand:
        return False

    tokens = tokenize(cand)
    if len(tokens) == 0 or len(tokens) > 7:
        return False
    if all(token in STOPWORDS for token in tokens):
        return False
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return False
    if any(any(ch.isdigit() for ch in tok) for tok in tokens) and not any(ch.isdigit() for ch in str(answer_text)):
        return False
    if any(len(tok) < 3 for tok in tokens if tok.isalpha()):
        return False
    if re.search(r"[^A-Za-z0-9\s-]", cand):
        return False
    if character_overlap(cand, answer_text) > 0.82:
        return False
    return True


def _dedupe_phrases(candidates: list[str], threshold: float = 0.9) -> list[str]:
    deduped: list[str] = []
    for cand in candidates:
        if any(character_overlap(cand, kept) > threshold for kept in deduped):
            continue
        deduped.append(cand)
    return deduped


def character_overlap(a: str, b: str) -> float:
    a_clean = clean_text(a)
    b_clean = clean_text(b)
    if not a_clean or not b_clean:
        return 0.0
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def one_hot_encode(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    unk_index = vocab.get("<UNK>", 0)
    for token in tokens:
        idx = vocab.get(token, unk_index)
        if idx < len(vec):
            vec[idx] = 1.0
    return vec


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _align_feature_matrix(features: np.ndarray, expected_features: int) -> np.ndarray:
    """Pad or trim feature columns to match serialized model expectations."""
    current = features.shape[1]
    if current == expected_features:
        return features
    if current > expected_features:
        return features[:, :expected_features]

    pad = np.zeros((features.shape[0], expected_features - current), dtype=np.float32)
    return np.hstack([features, pad])


def _patch_model_compat(model: Any) -> Any:
    """Patch common missing attributes on sklearn models across version mismatches."""
    class_name = model.__class__.__name__
    if class_name == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model


def _extract_title_phrases(article_raw: str, answer_set: set[str]) -> list[str]:
    phrases: list[str] = []
    runs: list[str] = []
    for raw in re.findall(r"[A-Za-z]+", article_raw):
        if raw[:1].isupper() and raw.lower() not in STOPWORDS:
            runs.append(raw.lower())
            continue
        if len(runs) >= 2:
            phrase = " ".join(runs)
            if not any(part in answer_set for part in runs):
                phrases.append(phrase)
        runs = []
    if len(runs) >= 2:
        phrase = " ".join(runs)
        if not any(part in answer_set for part in runs):
            phrases.append(phrase)
    return phrases


def extract_candidates(article_tokens: list[str], answer_tokens: list[str], top_n: int = 50, article_raw: str = "") -> list[str]:
    """Build a candidate pool from article content and answer-neighbouring terms."""
    content_tokens = remove_stopwords(article_tokens)
    freq = Counter(content_tokens)
    answer_set = set(answer_tokens)

    unigram_pool = [w for w, _ in freq.most_common(top_n * 2) if w not in answer_set]

    bigram_freq: Counter[tuple[str, str]] = Counter()
    trigram_freq: Counter[tuple[str, str, str]] = Counter()
    source_sents = split_sentences(article_raw) if article_raw else [" ".join(content_tokens)]
    for sentence in source_sents:
        sent_toks = remove_stopwords(tokenize(sentence))
        for i in range(len(sent_toks) - 1):
            w1, w2 = sent_toks[i], sent_toks[i + 1]
            if w1 not in answer_set and w2 not in answer_set:
                bigram_freq[(w1, w2)] += 1
        for i in range(len(sent_toks) - 2):
            w1, w2, w3 = sent_toks[i], sent_toks[i + 1], sent_toks[i + 2]
            if w1 not in answer_set and w2 not in answer_set and w3 not in answer_set:
                trigram_freq[(w1, w2, w3)] += 1

    bigram_pool = [f"{w1} {w2}" for (w1, w2), _ in bigram_freq.most_common(top_n)]
    trigram_pool = [f"{w1} {w2} {w3}" for (w1, w2, w3), _ in trigram_freq.most_common(max(top_n // 2, 1))]

    adjacent: list[str] = []
    for i, token in enumerate(content_tokens):
        if token not in answer_set:
            continue
        window = content_tokens[max(0, i - 3):i] + content_tokens[i + 1:i + 4]
        for w in window:
            if w not in answer_set and w not in adjacent:
                adjacent.append(w)
        for j in range(len(window) - 1):
            phrase = f"{window[j]} {window[j + 1]}"
            if phrase not in adjacent:
                adjacent.append(phrase)

    ordinal_map = {
        "first": "second",
        "second": "third",
        "third": "fourth",
        "fourth": "fifth",
        "fifth": "sixth",
        "sixth": "seventh",
        "seventh": "eighth",
        "eighth": "ninth",
        "ninth": "tenth",
        "tenth": "first",
    }
    numeric_neighbours: list[str] = []
    for token in answer_tokens:
        if token.isdigit():
            value = int(token)
            for delta in (-2, -1, 1, 2):
                cand = str(value + delta)
                if cand not in answer_set:
                    numeric_neighbours.append(cand)
        if token in ordinal_map:
            numeric_neighbours.append(ordinal_map[token])

    title_phrases = _extract_title_phrases(article_raw, answer_set) if article_raw else []

    combined = list(
        dict.fromkeys(title_phrases + bigram_pool + unigram_pool + adjacent + numeric_neighbours + trigram_pool)
    )
    return combined[:top_n]


def compute_candidate_features(candidate: str, answer_text: str, article_tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    cand_tokens = tokenize(candidate)
    ans_tokens = tokenize(answer_text)
    art_freq = Counter(article_tokens)
    cand_set = set(cand_tokens)
    ans_set = set(ans_tokens)
    ans_len = max(len(ans_tokens), 1)

    cand_vec = one_hot_encode(cand_tokens, vocab)
    ans_vec = one_hot_encode(ans_tokens, vocab)

    f0 = cosine_similarity(cand_vec, ans_vec)
    f1 = character_overlap(candidate, answer_text)
    f2 = sum(art_freq[t] for t in cand_tokens) / max(len(article_tokens), 1)
    f3 = len(cand_tokens) / ans_len
    f4 = 1.0 if (clean_text(candidate) in clean_text(answer_text) or clean_text(answer_text) in clean_text(candidate)) else 0.0

    cand_len = max(len(cand_tokens), 1)
    f5 = 1.0 / max(max(cand_len, ans_len) / min(cand_len, ans_len), 1)
    f6 = len(cand_set & ans_set) / max(len(ans_set), 1)
    f7 = 1.0 if len(cand_tokens) >= 2 else 0.0

    first_pos = next((i for i, tok in enumerate(article_tokens) if tok in cand_set), len(article_tokens))
    f8 = 1.0 - (first_pos / max(len(article_tokens), 1))
    f9 = 1.0 - f6

    return np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)


def wordnet_candidates(answer_text: str, n: int = 15) -> list[str]:
    if _WORDNET is None:
        return []

    results: list[str] = []
    seen = set(tokenize(answer_text))

    def add_name(name: str) -> None:
        cand = name.replace("_", " ").lower().strip()
        if cand and cand not in seen and len(cand) > 1:
            seen.add(cand)
            results.append(cand)

    for word in remove_stopwords(tokenize(answer_text)):
        try:
            synsets = _WORDNET.synsets(word)
        except LookupError:
            return []

        for syn in synsets:
            for hypernym in syn.hypernyms():
                for hyponym in hypernym.hyponyms():
                    for lemma in hyponym.lemmas():
                        add_name(lemma.name())
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    add_name(antonym.name())
            for lemma in syn.lemmas():
                add_name(lemma.name())
        if len(results) >= n:
            break
    return results[:n]


def query_candidate_bank(bank: dict[str, list[str]], article_tokens: list[str], answer_text: str, n: int = 20) -> list[str]:
    freq = Counter(remove_stopwords(article_tokens))
    top_words = [word for word, _ in freq.most_common(10)]

    retrieved: list[str] = []
    seen: set[str] = set()
    for word in top_words:
        for cand in bank.get(word, []):
            if cand in seen:
                continue
            if character_overlap(cand, clean_text(answer_text)) > 0.7:
                continue
            seen.add(cand)
            retrieved.append(cand)
        if len(retrieved) >= n:
            break
    return retrieved[:n]


def select_diverse_distractors(
    candidates: list[str],
    scores: list[float],
    answer_text: str,
    n: int = 3,
    diversity_threshold: float = 0.65,
    length_tolerance: float = 2.5,
) -> list[str]:
    ans_len = max(len(tokenize(answer_text)), 1)
    paired = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)

    selected: list[str] = []
    for cand, _ in paired:
        if len(selected) == n:
            break
        if character_overlap(cand, answer_text) > 0.8:
            continue
        if any(character_overlap(cand, existing) > diversity_threshold for existing in selected):
            continue
        cand_len = max(len(tokenize(cand)), 1)
        ratio = max(cand_len / ans_len, ans_len / cand_len)
        if ratio > length_tolerance:
            continue
        selected.append(cand)

    if len(selected) < n:
        for cand, _ in paired:
            if len(selected) == n:
                break
            if cand in selected:
                continue
            if character_overlap(cand, answer_text) > 0.8:
                continue
            if any(character_overlap(cand, existing) > 0.9 for existing in selected):
                continue
            selected.append(cand)

    while len(selected) < n:
        selected.append("[no candidate]")
    return selected


def generate_distractors(article: str, answer_text: str, artifacts: ModelBArtifacts, n: int = 3) -> list[str]:
    article_tokens = tokenize(article)
    answer_tokens = tokenize(answer_text)

    article_candidates = extract_candidates(article_tokens, answer_tokens, top_n=100, article_raw=article)
    bank_candidates = query_candidate_bank(artifacts.candidate_bank, article_tokens, answer_text, n=20)
    wn_candidates = wordnet_candidates(answer_text, n=15)

    bank_norm = {clean_text(c) for c in bank_candidates}
    wn_norm = {clean_text(c) for c in wn_candidates}

    all_candidates = list(dict.fromkeys(article_candidates + bank_candidates + wn_candidates))
    all_candidates = [_clean_phrase(cand) for cand in all_candidates]
    all_candidates = [cand for cand in all_candidates if _is_candidate_valid(cand, answer_text)]
    all_candidates = _dedupe_phrases(all_candidates, threshold=0.92)
    if not all_candidates:
        return ["[no candidate]"] * n

    features = np.array(
        [
            compute_candidate_features(cand, clean_text(answer_text), article_tokens, artifacts.vocab)
            for cand in all_candidates
        ],
        dtype=np.float32,
    )
    expected = int(getattr(artifacts.distractor_model, "n_features_in_", features.shape[1]))
    features = _align_feature_matrix(features, expected)
    raw_scores = artifacts.distractor_model.predict_proba(features)[:, 1]

    quality_scores: list[float] = []
    answer_len = max(_candidate_token_count(answer_text), 1)
    article_clean = clean_text(article)
    for cand, raw in zip(all_candidates, raw_scores):
        overlap_penalty = 0.45 * character_overlap(cand, answer_text)
        cand_len = max(_candidate_token_count(cand), 1)
        length_ratio = max(cand_len / answer_len, answer_len / cand_len)
        length_penalty = 0.08 * max(0.0, length_ratio - 2.0)
        source_bonus = 0.06 if clean_text(cand) in article_clean else 0.0
        bank_bonus = 0.11 if clean_text(cand) in bank_norm else 0.0
        wn_bonus = 0.03 if clean_text(cand) in wn_norm else 0.0
        quality_scores.append(float(raw - overlap_penalty - length_penalty + source_bonus + bank_bonus + wn_bonus))

    selected = select_diverse_distractors(all_candidates, quality_scores, answer_text, n=n)
    selected = [_clean_phrase(item) for item in selected if item != "[no candidate]"]
    selected = _dedupe_phrases(selected, threshold=0.9)

    if len(selected) < n:
        fallback_pool = [_clean_phrase(c) for c in bank_candidates if _is_candidate_valid(c, answer_text)]
        for cand in fallback_pool:
            if len(selected) >= n:
                break
            if cand in selected:
                continue
            if any(character_overlap(cand, existing) > 0.9 for existing in selected):
                continue
            selected.append(cand)

    while len(selected) < n:
        selected.append("None of the above")
    return selected[:n]


def _sentence_overlaps(sentence: str, answer_text: str, question: str) -> tuple[float, float]:
    sent_tokens = remove_stopwords(tokenize(sentence))
    ans_tokens = remove_stopwords(tokenize(answer_text))
    question_tokens = remove_stopwords(tokenize(question))

    sent_set = set(sent_tokens)
    ans_overlap = len(sent_set & set(ans_tokens)) / max(len(ans_tokens), 1)
    q_overlap = len(sent_set & set(question_tokens)) / max(len(question_tokens), 1)
    return ans_overlap, q_overlap


def sentence_features(
    sentence: str,
    question: str,
    answer_text: str,
    position: int,
    total_sentences: int,
    vocab: dict[str, int],
) -> np.ndarray:
    sent_tokens = remove_stopwords(tokenize(sentence))
    question_tokens = tokenize(question)
    q_content_tokens = remove_stopwords(question_tokens)
    ans_tokens = remove_stopwords(tokenize(answer_text))

    sent_set = set(sent_tokens)
    f0 = len(sent_set & set(q_content_tokens)) / max(len(q_content_tokens), 1)
    f1 = len(sent_set & set(ans_tokens)) / max(len(ans_tokens), 1)
    f2 = 1.0 - (position / max(total_sentences - 1, 1))
    f3 = min(len(sent_tokens), 30) / 30.0

    sent_vec = one_hot_encode(sent_tokens, vocab)
    q_vec = one_hot_encode(question_tokens, vocab)
    ans_vec = one_hot_encode(ans_tokens, vocab)

    f4 = cosine_similarity(sent_vec, q_vec)
    f5 = cosine_similarity(sent_vec, ans_vec)

    return np.array([f0, f1, f2, f3, f4, f5], dtype=np.float32)


def generate_hints(article: str, question: str, answer_text: str, artifacts: ModelBArtifacts, n_hints: int = 3) -> list[str]:
    sentences = split_sentences(article)
    total = len(sentences)
    if total == 0:
        return [FALLBACK_HINT] * n_hints

    features = np.array(
        [
            sentence_features(sentence, question, answer_text, idx, total, artifacts.vocab)
            for idx, sentence in enumerate(sentences)
        ],
        dtype=np.float32,
    )
    expected = int(getattr(artifacts.hint_classifier, "n_features_in_", features.shape[1]))
    features = _align_feature_matrix(features, expected)
    scores = artifacts.hint_classifier.predict_proba(features)[:, 1]

    candidates = [
        {
            "sentence": sentence,
            "score": float(scores[i]),
            "answer_overlap": float(features[i][1]),
            "length": len(tokenize(sentence)),
        }
        for i, sentence in enumerate(sentences)
    ]

    filtered = [item for item in candidates if item["score"] > 0.3 and item["length"] >= 7]
    if len(filtered) < n_hints:
        filtered = sorted(candidates, key=lambda item: item["score"], reverse=True)[:n_hints]

    def hint_quality(item: dict[str, float | str]) -> float:
        length = int(item["length"])
        length_reward = 0.0
        if 10 <= length <= 28:
            length_reward = 0.12
        elif 8 <= length <= 35:
            length_reward = 0.06
        return float(item["score"]) + length_reward

    filtered_sorted = sorted(filtered, key=lambda item: hint_quality(item), reverse=True)
    unique_candidates: list[dict[str, float | str]] = []
    for item in filtered_sorted:
        sentence = str(item["sentence"])
        if any(character_overlap(sentence, str(kept["sentence"])) > 0.92 for kept in unique_candidates):
            continue
        unique_candidates.append(item)

    if not unique_candidates:
        return [FALLBACK_HINT] * n_hints

    unique_candidates = sorted(unique_candidates, key=lambda item: float(item["answer_overlap"]))
    if len(unique_candidates) >= n_hints:
        step = max(1, len(unique_candidates) // n_hints)
        indices = [min(i * step, len(unique_candidates) - 1) for i in range(n_hints)]
        selected = [str(unique_candidates[i]["sentence"]) for i in indices]
    else:
        selected = [str(item["sentence"]) for item in unique_candidates]

    cleaned_selected = [_clean_hint_text(text) for text in selected]
    cleaned_selected = [text for text in cleaned_selected if text]

    while len(cleaned_selected) < n_hints:
        cleaned_selected.append(FALLBACK_HINT)
    return cleaned_selected[:n_hints]


def _clean_hint_text(sentence: str, max_words: int = 32) -> str:
    text = re.sub(r"\s+", " ", str(sentence).strip())
    text = text.replace('"', "")
    text = text.replace("'", "")
    text = text.strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;:") + "..."
    if text and text[-1] not in ".!?":
        text = text + "."
    return text


def build_quiz_options(correct_answer: str, distractors: list[str]) -> dict[str, str]:
    labels = ["A", "B", "C", "D"]
    normalized_distractors = [
        _clean_phrase(d) for d in distractors if d and d != "[no candidate]" and _is_candidate_valid(d, correct_answer)
    ]
    normalized_distractors = _dedupe_phrases(normalized_distractors, threshold=0.9)

    while len(normalized_distractors) < 3:
        normalized_distractors.append("None of the above")

    choices = [_clean_phrase(correct_answer)] + normalized_distractors[:3]
    random.shuffle(choices)

    option_map = {label: choices[i] for i, label in enumerate(labels)}
    return option_map


def generate_full_quiz(article: str, question: str, answer_text: str, artifacts: ModelBArtifacts) -> dict[str, Any]:
    distractors = generate_distractors(article, answer_text, artifacts, n=3)
    hints = generate_hints(article, question, answer_text, artifacts, n_hints=3)
    options = build_quiz_options(answer_text, distractors)

    correct_label = ""
    for label, text in options.items():
        if clean_text(text) == clean_text(answer_text):
            correct_label = label
            break

    diagnostics = {
        "distractor_diversity": float(
            1.0
            - np.mean(
                [
                    character_overlap(distractors[i], distractors[j])
                    for i in range(len(distractors))
                    for j in range(i + 1, len(distractors))
                ]
            )
            if len(distractors) > 1
            else 1.0
        ),
        "answer_overlap_max": float(max(character_overlap(item, answer_text) for item in distractors)) if distractors else 0.0,
    }

    return {
        "question": question,
        "answer_text": answer_text,
        "distractors": distractors,
        "hints": hints,
        "options": options,
        "correct_option": correct_label,
        "diagnostics": diagnostics,
    }


def resolve_model_b_dir(base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else Path(__file__).resolve().parents[1]
    return root / "models" / "model_b" / "traditional"


def load_model_b_artifacts(base_dir: Path | None = None) -> ModelBArtifacts:
    model_dir = resolve_model_b_dir(base_dir)
    required = {
        "distractor_model": model_dir / "distractor_best.pkl",
        "hint_classifier": model_dir / "hint_classifier.pkl",
        "hint_regressor": model_dir / "hint_regressor.pkl",
        "vocab": model_dir / "vocab.pkl",
        "candidate_bank": model_dir / "candidate_bank.pkl",
    }

    missing = [str(path.name) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Model B artifacts in {model_dir}: {', '.join(missing)}")

    distractor_model = _patch_model_compat(joblib.load(required["distractor_model"]))
    hint_classifier = _patch_model_compat(joblib.load(required["hint_classifier"]))
    hint_regressor = _patch_model_compat(joblib.load(required["hint_regressor"]))

    return ModelBArtifacts(
        distractor_model=distractor_model,
        hint_classifier=hint_classifier,
        hint_regressor=hint_regressor,
        vocab=joblib.load(required["vocab"]),
        candidate_bank=joblib.load(required["candidate_bank"]),
        model_dir=model_dir,
    )
