"""Template-based question generation helpers for Model A.

The generation pipeline is intentionally lightweight and classical-ML only: it
extracts promising sentences, turns them into WH-style questions, and ranks the
results with a small logistic regression classifier.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

try:
	from src.features import normalize_text, tokenize_text
except ImportError:  # pragma: no cover - script execution fallback
	from features import normalize_text, tokenize_text


WH_TEMPLATES = {
	"who": ["Who {verb} {rest}?", "Who was {rest}?"],
	"what": ["What {verb} {rest}?", "What did {rest}?"],
	"where": ["Where {verb} {rest}?", "Where did {rest}?"],
	"when": ["When {verb} {rest}?", "When did {rest}?"],
	"why": ["Why {verb} {rest}?", "Why did {rest}?"],
	"how": ["How {verb} {rest}?", "How did {rest}?"],
}

MONTH_NAMES = {
	"january", "february", "march", "april", "may", "june",
	"july", "august", "september", "october", "november", "december",
}

COMMON_PAST_TO_BASE = {
	"won": "win",
	"went": "go",
	"came": "come",
	"made": "make",
	"took": "take",
	"gave": "give",
	"found": "find",
	"held": "hold",
	"became": "become",
	"began": "begin",
	"wrote": "write",
	"met": "meet",
	"lived": "live",
	"worked": "work",
	"studied": "study",
	"played": "play",
	"built": "build",
	"won": "win",
}

LOCATION_HINTS = {
	"in", "at", "from", "to", "into", "inside", "within", "near", "on",
}

PLACE_HINTS = {
	"city", "country", "state", "province", "village", "town", "school", "university",
	"park", "river", "mountain", "sea", "ocean", "street", "road", "airport", "station",
}

PERSON_PRONOUNS = {"he", "she", "they", "we", "i", "you", "him", "her", "them", "his", "hers", "their"}

TIME_WORDS = {"year", "years", "month", "months", "day", "days", "week", "weeks", "hour", "hours", "minute", "minutes", "today", "yesterday", "tomorrow"}


def extract_candidate_sentences(article: str, correct_answer: str, vectorizer, top_k: int = 5) -> list[str]:
	"""Score article sentences against the answer and keep the strongest ones.

	Args:
		article (str): Passage text.
		correct_answer (str): Known correct answer text.
		vectorizer: Fitted vectorizer used for cosine similarity.
		top_k (int): Number of candidate sentences to keep.

	Returns:
		list[str]: Sentences sorted from most to least relevant.

	Example:
		>>> extract_candidate_sentences(article, answer, vectorizer, top_k=3)
	"""
	sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", article) if len(s.strip()) > 20]
	if not sentences:
		return []
	v_answer = vectorizer.transform([normalize_text(correct_answer)])
	v_sents = vectorizer.transform([normalize_text(sentence) for sentence in sentences])
	scores = cosine_similarity(v_answer, v_sents)[0]
	top_idx = np.argsort(scores)[::-1][:top_k]
	return [sentences[index] for index in top_idx]


def apply_wh_template(sentence: str, wh_word: str = "what") -> str:
	"""Convert a declarative sentence into a simple WH-style question.

	Args:
		sentence (str): Candidate source sentence.
		wh_word (str): WH-word to use in the generated question.

	Returns:
		str: Template-based question string.

	Example:
		>>> apply_wh_template("The child won the contest.", "who")
		'Who won the contest?'
	"""
	sentence = sentence.rstrip(".!?")
	tokens = sentence.split()
	if len(tokens) < 4:
		return f"{wh_word.capitalize()} happened regarding: {sentence}?"
	predicate = " ".join(tokens[1:])
	return f"{wh_word.capitalize()} {predicate}?"


def infer_question_family(sentence: str, correct_answer: str) -> str:
	"""Classify the answer into a broad WH family.

	The generator is not a seq2seq model; this function keeps it conservative by
	choosing the family from the answer itself instead of from vague sentence cues.
	"""
	answer = normalize_text(correct_answer)
	answer_tokens = tokenize_text(answer)
	sentence_clean = normalize_text(sentence)

	if not answer:
		return "what"

	if any(token.isdigit() for token in answer_tokens) or re.search(r"\b\d{4}\b", answer):
		return "when"
	if any(month in answer for month in MONTH_NAMES) or any(word in answer_tokens for word in TIME_WORDS):
		return "when"

	answer_text = str(correct_answer).strip()
	if any(pronoun == answer for pronoun in PERSON_PRONOUNS):
		return "who"
	if answer_text[:1].isupper() and any(token[:1].isupper() for token in answer_text.split()):
		return "who"
	if len(answer_tokens) >= 2 and answer_text[:1].isupper():
		return "who"

	if any(place in answer_tokens for place in PLACE_HINTS):
		return "where"
	if any(marker in sentence_clean for marker in [" in ", " at ", " from ", " to ", " into ", " within ", " near "]):
		# Only use WHERE when the answer itself looks like a location or the sentence
		# strongly frames it as one.
		if any(word in answer_tokens for word in PLACE_HINTS) or answer_text[:1].isupper():
			return "where"

	return "what"


def infer_wh_word(sentence: str, correct_answer: str) -> str:
	"""Backward-compatible alias for the generator family selector."""
	return infer_question_family(sentence, correct_answer)


def _strip_answer(sentence: str, correct_answer: str) -> str:
	cleaned_sentence = re.sub(re.escape(correct_answer), "", sentence, flags=re.IGNORECASE)
	cleaned_sentence = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_sentence)
	cleaned_sentence = re.sub(r"\s{2,}", " ", cleaned_sentence).strip()
	return cleaned_sentence.rstrip(".!?")


def build_question_from_sentence(sentence: str, correct_answer: str, wh_word: str) -> str:
	"""Turn a supporting sentence into a more readable WH question."""
	sentence = sentence.strip().rstrip(".!?")
	answer = str(correct_answer).strip()
	if not sentence:
		return f"{wh_word.capitalize()} is being asked about here?"

	stripped = _strip_answer(sentence, answer)
	parts = stripped.split()
	if not parts:
		return f"{wh_word.capitalize()} is mentioned in the passage?"

	lower_parts = [part.lower() for part in parts]
	for verb in ("is", "are", "was", "were"):
		if verb in lower_parts:
			verb_index = lower_parts.index(verb)
			subject = " ".join(parts[:verb_index]).strip()
			tail = " ".join(parts[verb_index + 1 :]).strip()
			if wh_word == "where":
				if tail:
					return f"Where {verb} {subject}?" if subject else f"Where {verb} it?"
				return f"Where {verb} {subject}?" if subject else f"Where is it?"
			if wh_word == "when":
				return f"When {verb} {subject}?" if subject else f"When {verb} it?"
			if wh_word == "who":
				return f"Who {verb} {tail}?" if tail else f"Who {verb} here?"
			return f"What {verb} {subject}?" if subject else f"What is mentioned?"

	for verb, base in COMMON_PAST_TO_BASE.items():
		pattern = rf"^(?P<subject>.+?)\s+{re.escape(verb)}\b(?P<rest>.*)$"
		match = re.match(pattern, stripped, flags=re.IGNORECASE)
		if match:
			subject = match.group("subject").strip()
			rest = match.group("rest").strip()
			if wh_word == "where":
				return f"Where did {subject} {base} {rest}?".replace("  ", " ").strip()
			if wh_word == "when":
				return f"When did {subject} {base} {rest}?".replace("  ", " ").strip()
			if wh_word == "who":
				return f"Who did {subject} {base} {rest}?".replace("  ", " ").strip()
			return f"What did {subject} {base} {rest}?".replace("  ", " ").strip()

	focus = " ".join(token for token in tokenize_text(stripped) if token not in {"the", "a", "an", "of", "and", "to", "in", "on", "at", "for"})
	if not focus:
		focus = stripped

	if wh_word == "where":
		return f"Where is the passage discussing {focus}?"
	if wh_word == "when":
		return f"When does the passage discuss {focus}?"
	if wh_word == "who":
		return f"Who is mentioned in relation to {focus}?"
	return f"What does the passage say about {focus}?"


def rank_questions(candidates: list[str], article: str, correct_answer: str, vectorizer, ranker_model) -> list[str]:
	"""Rank candidate questions using relevance-proxy features.

	Args:
		candidates (list[str]): Generated question strings.
		article (str): Source passage.
		correct_answer (str): Known correct answer text.
		vectorizer: Fitted vectorizer used for cosine features.
		ranker_model: Probability-based classifier used for ranking.

	Returns:
		list[str]: Candidate questions ordered from best to worst.

	Example:
		>>> rank_questions(candidates, article, answer, vectorizer, ranker)
	"""
	if not candidates:
		return []
	v_article = vectorizer.transform([normalize_text(article)])
	v_answer = vectorizer.transform([normalize_text(correct_answer)])
	features = []
	for question in candidates:
		v_question = vectorizer.transform([normalize_text(question)])
		features.append(
			[
				cosine_similarity(v_question, v_article)[0][0],
				cosine_similarity(v_question, v_answer)[0][0],
				len(tokenize_text(question)),
			]
		)
	# Ensure feature dimensionality matches the trained ranker model.
	X = np.asarray(features, dtype=np.float32)
	expected = getattr(ranker_model, "n_features_in_", None)
	if expected is not None:
		if X.shape[1] < expected:
			# Pad with zeros for missing features to preserve backward compatibility
			pad_width = expected - X.shape[1]
			X = np.hstack([X, np.zeros((X.shape[0], pad_width), dtype=X.dtype)])
		elif X.shape[1] > expected:
			# Truncate extra features if model expects fewer inputs
			X = X[:, :expected]
	scores = ranker_model.predict_proba(X)[:, 1]
	order = np.argsort(scores)[::-1]
	return [candidates[index] for index in order]


def generate_question(
	article: str,
	correct_answer: str,
	vectorizer,
	ranker_model=None,
	wh_words=("what", "who", "where", "when"),
) -> dict:
	"""Generate one question from a passage and an answer string.

	Args:
		article (str): Passage text.
		correct_answer (str): Known answer text.
		vectorizer: Fitted vectorizer used for sentence scoring.
		ranker_model: Optional probability-based question ranker.
		wh_words (tuple[str, ...]): WH-words to try during template generation.

	Returns:
		dict: Dictionary containing the generated question and provenance.

	Example:
		>>> generate_question(article, answer, vectorizer)
	"""
	candidate_sentences = extract_candidate_sentences(article, correct_answer, vectorizer, top_k=5)
	family = infer_question_family(candidate_sentences[0] if candidate_sentences else article, correct_answer)
	family_candidates = {
		"who": ("who", "what"),
		"when": ("when", "what"),
		"where": ("where", "what"),
		"what": ("what", "who"),
	}.get(family, ("what", "who"))
	allowed_wh_words = tuple(dict.fromkeys((*family_candidates, *wh_words)))
	all_questions: list[dict[str, str]] = []
	for sentence in candidate_sentences:
		for template_wh in allowed_wh_words:
			all_questions.append(
				{
					"question": build_question_from_sentence(sentence, correct_answer, template_wh),
					"source": sentence,
					"wh": template_wh,
				}
			)

	if not all_questions:
		return {
			"question": "What is the main topic of this passage?",
			"source_sentence": "",
			"wh_word": "what",
		}

	if ranker_model is not None:
		questions_text = [item["question"] for item in all_questions]
		ranked_questions = rank_questions(questions_text, article, correct_answer, vectorizer, ranker_model)
		best_question = ranked_questions[0] if ranked_questions else all_questions[0]["question"]
		best = next((item for item in all_questions if item["question"] == best_question), all_questions[0])
	else:
		best = all_questions[0]

	return {
		"question": best["question"],
		"source_sentence": best["source"],
		"wh_word": best["wh"],
		"family": family,
		"candidate_count": len(all_questions),
	}


def build_question_ranker_dataset(frame: pd.DataFrame, vectorizer, top_k_sentences: int = 3) -> tuple[np.ndarray, np.ndarray]:
	"""Build a small supervised dataset for the question ranker.

	Args:
		frame (pd.DataFrame): Training rows with article/question/answer information.
		vectorizer: Fitted vectorizer used for cosine features.
		top_k_sentences (int): Number of article sentences to use as negative candidates.

	Returns:
		tuple[np.ndarray, np.ndarray]: Feature matrix and binary labels.

	Example:
		>>> X, y = build_question_ranker_dataset(train_frame, vectorizer)
	"""
	feature_rows: list[list[float]] = []
	labels: list[int] = []

	for _, row in frame.iterrows():
		article = str(row.get("article", ""))
		question = str(row.get("question", ""))
		answer_label = str(row.get("answer", "")).strip().upper()
		answer_text = str(row.get(answer_label, "")) if answer_label in {"A", "B", "C", "D"} else ""

		positive = _ranker_features(question, article, answer_text, vectorizer)
		feature_rows.append(positive)
		labels.append(1)

		for candidate_sentence in extract_candidate_sentences(article, answer_text, vectorizer, top_k=top_k_sentences):
			for wh_word in ("what", "who"):
				generated_question = apply_wh_template(candidate_sentence, wh_word)
				negative = _ranker_features(generated_question, article, answer_text, vectorizer)
				feature_rows.append(negative)
				labels.append(0)

	return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def _ranker_features(question: str, article: str, correct_answer: str, vectorizer) -> list[float]:
	question_clean = normalize_text(question)
	article_clean = normalize_text(article)
	answer_clean = normalize_text(correct_answer)

	v_question = vectorizer.transform([question_clean])
	v_article = vectorizer.transform([article_clean])
	v_answer = vectorizer.transform([answer_clean])
	return [
		float(cosine_similarity(v_question, v_article)[0][0]),
		float(cosine_similarity(v_question, v_answer)[0][0]),
		float(len(tokenize_text(question_clean))),
	]


def train_question_ranker(frame: pd.DataFrame, vectorizer, output_path: Path) -> LogisticRegression:
	"""Train and persist the question ranker.

	Args:
		frame (pd.DataFrame): Training rows containing article/question/answer data.
		vectorizer: Fitted vectorizer used to build cosine features.
		output_path (Path): Destination path for the fitted model.

	Returns:
		LogisticRegression: Fitted question ranker model.

	Example:
		>>> model = train_question_ranker(train_frame, vectorizer, Path("models/.../question_ranker.pkl"))
	"""
	X, y = build_question_ranker_dataset(frame, vectorizer)
	ranker = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
	ranker.fit(X, y)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(ranker, output_path)
	return ranker
