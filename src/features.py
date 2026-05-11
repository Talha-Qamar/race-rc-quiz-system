"""Shared feature engineering helpers for Model A.

This module keeps the answer-verification feature logic in one place so the
training pipeline and the Streamlit UI can build identical feature vectors.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(value: object) -> str:
	"""Normalize free text for lightweight lexical comparisons.

	Args:
		value (object): Input text-like value.

	Returns:
		str: Lowercased, punctuation-stripped, whitespace-normalized text.

	Example:
		>>> normalize_text("Hello,  World!")
		'hello world'
	"""
	if value is None:
		return ""
	text = str(value).lower()
	for symbol in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
		text = text.replace(symbol, " ")
	return " ".join(text.split())


def tokenize_text(value: object) -> list[str]:
	"""Tokenize normalized text using whitespace splitting.

	Args:
		value (object): Input text-like value.

	Returns:
		list[str]: Normalized tokens.

	Example:
		>>> tokenize_text("Fast, reliable text")
		['fast', 'reliable', 'text']
	"""
	cleaned = normalize_text(value)
	return [token for token in cleaned.split(" ") if token]


def _safe_jaccard(left_tokens: Iterable[str], right_tokens: Iterable[str]) -> float:
	left_set = set(left_tokens)
	right_set = set(right_tokens)
	union = left_set | right_set
	if not union:
		return 0.0
	return len(left_set & right_set) / len(union)


def _rowwise_cosine(left_matrix: csr_matrix, right_matrix: csr_matrix) -> np.ndarray:
	numerator = left_matrix.multiply(right_matrix).sum(axis=1).A1
	left_norm = np.sqrt(left_matrix.multiply(left_matrix).sum(axis=1)).A1
	right_norm = np.sqrt(right_matrix.multiply(right_matrix).sum(axis=1)).A1
	denominator = np.maximum(left_norm * right_norm, 1e-12)
	return numerator / denominator


def build_verification_features(article: str, question: str, option: str, vectorizer) -> np.ndarray:
	"""Build dense verification features for one article/question/option triple.

	Args:
		article (str): Passage text.
		question (str): Question text.
		option (str): Candidate answer text.
		vectorizer: Fitted scikit-learn vectorizer used for cosine comparisons.

	Returns:
		np.ndarray: Dense array with 6 handcrafted verification features.

	Example:
		>>> features = build_verification_features(article, question, option, vectorizer)
	"""
	article_clean = normalize_text(article)
	question_clean = normalize_text(question)
	option_clean = normalize_text(option)

	v_article = vectorizer.transform([article_clean])
	v_question = vectorizer.transform([question_clean])
	v_option = vectorizer.transform([option_clean])

	sim_art_opt = cosine_similarity(v_article, v_option)[0][0]
	sim_q_opt = cosine_similarity(v_question, v_option)[0][0]
	sim_art_q = cosine_similarity(v_article, v_question)[0][0]

	art_tokens = tokenize_text(article_clean)
	q_tokens = tokenize_text(question_clean)
	opt_tokens = tokenize_text(option_clean)

	jaccard_art_opt = _safe_jaccard(art_tokens, opt_tokens)
	jaccard_q_opt = _safe_jaccard(q_tokens, opt_tokens)
	len_ratio = len(opt_tokens) / max(len(art_tokens), 1)

	return np.array(
		[
			sim_art_opt,
			sim_q_opt,
			sim_art_q,
			jaccard_art_opt,
			jaccard_q_opt,
			len_ratio,
		],
		dtype=np.float32,
	)


def build_verification_feature_matrix(frame, vectorizer) -> np.ndarray:
	"""Build a dense feature matrix for a frame of answer candidates.

	Args:
		frame: DataFrame-like object with article_clean, question_clean, and option_clean columns.
		vectorizer: Fitted vectorizer used for cosine features.

	Returns:
		np.ndarray: Dense matrix of shape (n_rows, 6).

	Example:
		>>> dense_features = build_verification_feature_matrix(option_frame, vectorizer)
	"""
	article_values = frame["article_clean"].fillna("").astype(str).tolist()
	question_values = frame["question_clean"].fillna("").astype(str).tolist()
	option_values = frame["option_clean"].fillna("").astype(str).tolist()

	v_article = vectorizer.transform(article_values)
	v_question = vectorizer.transform(question_values)
	v_option = vectorizer.transform(option_values)

	art_tokens = [tokenize_text(value) for value in article_values]
	q_tokens = [tokenize_text(value) for value in question_values]
	opt_tokens = [tokenize_text(value) for value in option_values]

	return np.column_stack(
		[
			_rowwise_cosine(v_article, v_option),
			_rowwise_cosine(v_question, v_option),
			_rowwise_cosine(v_article, v_question),
			np.array([_safe_jaccard(left, right) for left, right in zip(art_tokens, opt_tokens)], dtype=np.float32),
			np.array([_safe_jaccard(left, right) for left, right in zip(q_tokens, opt_tokens)], dtype=np.float32),
			np.array(
				[
					len(option.split()) / max(len(article.split()), 1)
					for article, option in zip(article_values, option_values)
				],
				dtype=np.float32,
			),
		]
	)


def combine_sparse_and_dense(X_sparse: csr_matrix, X_dense: np.ndarray) -> csr_matrix:
	"""Horizontally stack sparse TF-IDF features with dense handcrafted features.

	Args:
		X_sparse (csr_matrix): Sparse text feature matrix.
		X_dense (np.ndarray): Dense handcrafted feature matrix.

	Returns:
		csr_matrix: Combined sparse matrix suitable for scikit-learn estimators.

	Example:
		>>> X = combine_sparse_and_dense(X_tfidf, X_dense)
	"""
	return hstack([X_sparse, csr_matrix(X_dense)], format="csr")
