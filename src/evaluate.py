"""Evaluation helpers for Model A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	r2_score,
)


def exact_match_score(y_true, y_pred) -> float:
	"""Compute exact match for hard classification labels.

	Args:
		y_true: Ground-truth labels.
		y_pred: Predicted labels.

	Returns:
		float: Fraction of samples with an exact label match.

	Example:
		>>> exact_match_score([0, 1], [0, 1])
		1.0
	"""
	y_true_arr = np.asarray(y_true)
	y_pred_arr = np.asarray(y_pred)
	if y_true_arr.shape != y_pred_arr.shape:
		raise ValueError("y_true and y_pred must have the same shape")
	return float((y_true_arr == y_pred_arr).mean())


def full_evaluation_report(y_true, y_pred, y_proba=None, model_name: str = "Model") -> dict[str, Any]:
	"""Build a full classification report with per-class and aggregate metrics.

	Args:
		y_true: Ground-truth labels.
		y_pred: Predicted labels.
		y_proba: Optional probability matrix for AUC computation.
		model_name (str): Human-readable model name.

	Returns:
		dict[str, Any]: JSON-serializable evaluation summary.

	Example:
		>>> report = full_evaluation_report(y_true, y_pred, y_proba, model_name="RF")
	"""
	report = {
		"model": model_name,
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
		"weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
		"precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
		"recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
		"precision_class_0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
		"recall_class_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
		"precision_class_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
		"recall_class_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
		"exact_match": exact_match_score(y_true, y_pred),
		"confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
	}
	if y_proba is not None:
		try:
			report["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
		except Exception:
			pass
	print(classification_report(y_true, y_pred, target_names=["Incorrect", "Correct"], zero_division=0))
	return report


def save_evaluation_results(results: dict[str, Any], output_path: Path) -> Path:
	"""Persist evaluation results as JSON.

	Args:
		results (dict[str, Any]): JSON-serializable report payload.
		output_path (Path): Destination file path.

	Returns:
		Path: The saved output path.

	Example:
		>>> save_evaluation_results(report, Path("models/.../evaluation_results.json"))
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
	return output_path


def _clean_text(value: object) -> str:
	if value is None:
		return ""
	text = str(value).lower()
	for symbol in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
		text = text.replace(symbol, " ")
	return " ".join(text.split())


def _tokenize(value: object) -> list[str]:
	return [token for token in _clean_text(value).split() if token]


def _row_correct_answer(row: pd.Series) -> str:
	answer_label = str(row.get("answer", "")).strip().upper()
	if answer_label in {"A", "B", "C", "D"}:
		return str(row.get(answer_label, "")).strip()
	return str(row.get("correct_answer_text", "")).strip()


def _match_generated_to_gold(generated: list[str], gold: list[str]) -> int:
	generated_clean = [_clean_text(item) for item in generated if _clean_text(item)]
	gold_clean = [_clean_text(item) for item in gold if _clean_text(item)]
	if not generated_clean or not gold_clean:
		return 0
	count = 0
	for candidate in generated_clean:
		for gold_item in gold_clean:
			if candidate == gold_item or candidate in gold_item or gold_item in candidate:
				count += 1
				break
	return count


def _score_distractor_set(generated: list[str], gold_wrong: list[str], correct_answer: str) -> dict[str, float]:
	generated_clean = [_clean_text(item) for item in generated if _clean_text(item)]
	gold_clean = [_clean_text(item) for item in gold_wrong if _clean_text(item)]
	if not generated_clean:
		return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

	matches = _match_generated_to_gold(generated_clean, gold_clean)
	precision = matches / max(len(generated_clean), 1)
	recall = matches / max(len(gold_clean), 1)
	f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision or recall) else 0.0
	accuracy = 1.0 if _clean_text(generated_clean[0]) != _clean_text(correct_answer) else 0.0
	return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": float(accuracy)}


def _extract_answer_sentence(article: str, correct_answer: str) -> str:
	article_sentences = [sentence.strip() for sentence in str(article).replace("\n", " ").split(". ") if sentence.strip()]
	if not article_sentences:
		return ""
	answer_clean = _clean_text(correct_answer)
	for sentence in article_sentences:
		if answer_clean and answer_clean in _clean_text(sentence):
			return sentence
	answer_tokens = set(_tokenize(correct_answer))
	best_sentence = article_sentences[0]
	best_overlap = -1.0
	for sentence in article_sentences:
		sentence_tokens = set(_tokenize(sentence))
		overlap = len(sentence_tokens & answer_tokens)
		if overlap > best_overlap:
			best_overlap = overlap
		best_sentence = sentence
	return best_sentence


def _sentence_overlap_score(left: str, right: str) -> float:
	left_tokens = set(_tokenize(left))
	right_tokens = set(_tokenize(right))
	if not left_tokens or not right_tokens:
		return 0.0
	return len(left_tokens & right_tokens) / max(len(right_tokens), 1)


def evaluate_distractor_approaches(
	val_df: pd.DataFrame,
	vocab: list[str],
	word2idx: dict[str, int],
	distractor_ranker,
	w2v_model=None,
	save_path: Path | None = None,
) -> dict[str, dict[str, float]]:
	"""Compare one-hot-only and Word2Vec-combined distractor generation."""
	try:
		from model_b_distractor import get_distractor_candidates, generate_distractors_combined
	except ImportError:  # pragma: no cover
		from src.model_b_distractor import get_distractor_candidates, generate_distractors_combined

	if save_path is None:
		save_path = Path(__file__).resolve().parents[1] / "models" / "model_b" / "evaluation_results.json"

	if w2v_model is None:
		word2vec_results = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
	else:
		word2vec_results = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
	one_hot_results = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
	word2vec_totals = {"matches": 0, "generated": 0, "gold": 0, "top1_correct": 0}
	one_hot_totals = {"matches": 0, "generated": 0, "gold": 0, "top1_correct": 0}
	rows_used = 0

	for _, row in val_df.iterrows():
		article = str(row.get("article", ""))
		question = str(row.get("question", ""))
		correct_answer = _row_correct_answer(row)
		if not article.strip() or not correct_answer.strip():
			continue
		gold_wrong = [str(row.get(label, "")).strip() for label in ["A", "B", "C", "D"] if label != str(row.get("answer", "")).strip().upper()]
		gold_wrong = [item for item in gold_wrong if item]
		rows_used += 1

		one_hot_candidates = get_distractor_candidates(article, correct_answer, vocab, word2idx, top_n=20)
		one_hot_selected = _rank_and_select_distractors(
			article=article,
			question=question,
			correct_answer=correct_answer,
			candidates=one_hot_candidates,
			distractor_ranker=distractor_ranker,
			vocab=vocab,
			word2idx=word2idx,
			w2v_model=None,
			source_is_w2v=0,
			n=3,
		)
		if w2v_model is None:
			word2vec_selected = list(one_hot_selected)
		else:
			word2vec_selected = generate_distractors_combined(
				article=article,
				question=question,
				correct_answer=correct_answer,
				vocab=vocab,
				word2idx=word2idx,
				ranker_model=distractor_ranker,
				w2v_model=w2v_model,
				n=3,
			)

		one_hot_totals["matches"] += _match_generated_to_gold(one_hot_selected, gold_wrong)
		one_hot_totals["generated"] += len(one_hot_selected)
		one_hot_totals["gold"] += len(gold_wrong)
		one_hot_totals["top1_correct"] += int(bool(one_hot_selected) and _clean_text(one_hot_selected[0]) != _clean_text(correct_answer))

		word2vec_totals["matches"] += _match_generated_to_gold(word2vec_selected, gold_wrong)
		word2vec_totals["generated"] += len(word2vec_selected)
		word2vec_totals["gold"] += len(gold_wrong)
		word2vec_totals["top1_correct"] += int(bool(word2vec_selected) and _clean_text(word2vec_selected[0]) != _clean_text(correct_answer))

	def _finalize(totals: dict[str, int]) -> dict[str, float]:
		precision = totals["matches"] / max(totals["generated"], 1)
		recall = totals["matches"] / max(totals["gold"], 1)
		f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision or recall) else 0.0
		accuracy = totals["top1_correct"] / max(rows_used, 1)
		return {
			"precision": float(precision),
			"recall": float(recall),
			"f1": float(f1),
			"accuracy": float(accuracy),
		}

	comparison = {
		"one_hot_only": _finalize(one_hot_totals),
		"word2vec_combined": _finalize(word2vec_totals),
	}

	print("| Approach          | Precision | Recall | F1   | Accuracy |")
	print("|-------------------|-----------|--------|------|----------|")
	for label in ["one_hot_only", "word2vec_combined"]:
		metrics = comparison[label]
		name = "One-Hot Only" if label == "one_hot_only" else "Word2Vec Combined"
		print(
			f"| {name:<17} |   {metrics['precision']:.2f}    |  {metrics['recall']:.2f}  | {metrics['f1']:.2f} |   {metrics['accuracy']:.2f}   |"
		)

	try:
		payload = {}
		if save_path.exists():
			payload = json.loads(save_path.read_text(encoding="utf-8"))
		payload.update(comparison)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		save_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
	except Exception:
		pass

	return comparison


def _rank_and_select_distractors(
	article: str,
	question: str,
	correct_answer: str,
	candidates: list[str],
	distractor_ranker,
	vocab: list[str],
	word2idx: dict[str, int],
	w2v_model=None,
	source_is_w2v: int = 0,
	n: int = 3,
) -> list[str]:
	try:
		from model_b_distractor import _candidate_features, _pad_feature_matrix
	except ImportError:  # pragma: no cover
		from src.model_b_distractor import _candidate_features, _pad_feature_matrix

	if not candidates:
		return []
	feature_rows = [
		_candidate_features(
			candidate=candidate,
			question=question,
			correct_answer=correct_answer,
			article=article,
			vocab=vocab,
			word2idx=word2idx,
			w2v_model=w2v_model,
			source_is_w2v=source_is_w2v,
		)
		for candidate in candidates
	]
	feature_matrix = np.vstack(feature_rows).astype(np.float32)
	expected_features = int(getattr(distractor_ranker, "n_features_in_", feature_matrix.shape[1]))
	feature_matrix = _pad_feature_matrix(feature_matrix, expected_features)
	scores = distractor_ranker.predict_proba(feature_matrix)[:, 1]
	ordered = [candidate for candidate, _ in sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)]
	selected: list[str] = []
	correct_clean = _clean_text(correct_answer)
	for candidate in ordered:
		if len(selected) >= n:
			break
		if _clean_text(candidate) == correct_clean:
			continue
		selected.append(candidate)
	return selected[:n]


def _hint_precision_at_1(article: str, question: str, correct_answer: str, hint_scorer) -> float:
	try:
		from model_b_hint import _sentence_features, _split_sentences
	except ImportError:  # pragma: no cover
		from src.model_b_hint import _sentence_features, _split_sentences

	sentences = [sentence for sentence in _split_sentences(article) if len(sentence.split()) >= 5]
	if not sentences:
		return 0.0
	total_sentences = len(sentences)
	feature_rows = [
		_sentence_features(sentence, question, correct_answer, index, total_sentences)
		for index, sentence in enumerate(sentences)
	]
	feature_matrix = np.vstack(feature_rows).astype(np.float32)
	scores = hint_scorer.predict_proba(feature_matrix)[:, 1]
	top_sentence = sentences[int(np.argmax(scores))]
	answer_sentence = _extract_answer_sentence(article, correct_answer)
	return float(_sentence_overlap_score(top_sentence, answer_sentence) > 0.2)


def evaluate_model_b(
	val_df: pd.DataFrame,
	vocab: list[str],
	word2idx: dict[str, int],
	distractor_ranker,
	hint_scorer,
	w2v_model=None,
) -> dict[str, float]:
	"""Evaluate Model B distractor and hint outputs on the validation split."""
	try:
		from model_b_distractor import get_distractor_candidates, generate_distractors_combined
	except ImportError:  # pragma: no cover
		from src.model_b_distractor import get_distractor_candidates, generate_distractors_combined

	if w2v_model is None:
		try:
			from model_b_distractor import load_word2vec_model
		except ImportError:  # pragma: no cover
			from src.model_b_distractor import load_word2vec_model
		try:
			w2v_model = load_word2vec_model()
		except Exception:
			w2v_model = None

	distractor_totals = {"matches": 0, "generated": 0, "gold": 0, "top1_correct": 0}
	hint_precision_hits = 0
	hint_r2_true: list[int] = []
	hint_r2_scores: list[float] = []
	rows_used = 0

	for _, row in val_df.iterrows():
		article = str(row.get("article", ""))
		question = str(row.get("question", ""))
		correct_answer = _row_correct_answer(row)
		if not article.strip() or not correct_answer.strip():
			continue
		rows_used += 1
		gold_wrong = [str(row.get(label, "")).strip() for label in ["A", "B", "C", "D"] if label != str(row.get("answer", "")).strip().upper()]
		gold_wrong = [item for item in gold_wrong if item]
		generated = _rank_and_select_distractors(
			article=article,
			question=question,
			correct_answer=correct_answer,
			candidates=(
				generate_distractors_combined(article, question, correct_answer, vocab, word2idx, distractor_ranker, w2v_model, n=20)
				if w2v_model is not None
				else get_distractor_candidates(article, correct_answer, vocab, word2idx, top_n=20)
			),
			distractor_ranker=distractor_ranker,
			vocab=vocab,
			word2idx=word2idx,
			w2v_model=w2v_model,
			source_is_w2v=1,
			n=3,
		)
		distractor_totals["matches"] += _match_generated_to_gold(generated, gold_wrong)
		distractor_totals["generated"] += len(generated)
		distractor_totals["gold"] += len(gold_wrong)
		distractor_totals["top1_correct"] += int(bool(generated) and _clean_text(generated[0]) != _clean_text(correct_answer))

		hint_precision_hits += int(_hint_precision_at_1(article, question, correct_answer, hint_scorer))

		try:
			from model_b_hint import _sentence_features, _split_sentences
		except ImportError:  # pragma: no cover
			from src.model_b_hint import _sentence_features, _split_sentences
		sentences = [sentence for sentence in _split_sentences(article) if len(sentence.split()) >= 5]
		if sentences:
			total_sentences = len(sentences)
			feature_rows = [_sentence_features(sentence, question, correct_answer, index, total_sentences) for index, sentence in enumerate(sentences)]
			feature_matrix = np.vstack(feature_rows).astype(np.float32)
			hint_probs = hint_scorer.predict_proba(feature_matrix)[:, 1]
			hint_predictions = (hint_probs >= 0.5).astype(int)
			hint_r2_true.extend((feature_matrix[:, 4] > 0.0).astype(int).tolist())
			hint_r2_scores.extend(hint_probs.tolist())

	distractor_precision = distractor_totals["matches"] / max(distractor_totals["generated"], 1)
	distractor_recall = distractor_totals["matches"] / max(distractor_totals["gold"], 1)
	distractor_f1 = (2 * distractor_precision * distractor_recall / max(distractor_precision + distractor_recall, 1e-12)) if (distractor_precision or distractor_recall) else 0.0
	distractor_accuracy = distractor_totals["top1_correct"] / max(rows_used, 1)
	hint_r2 = float(r2_score(hint_r2_true, hint_r2_scores)) if hint_r2_true and hint_r2_scores else 0.0
	hint_precision_at_1 = hint_precision_hits / max(rows_used, 1)

	return {
		"distractor_precision": float(distractor_precision),
		"distractor_recall": float(distractor_recall),
		"distractor_f1": float(distractor_f1),
		"distractor_accuracy": float(distractor_accuracy),
		"hint_precision_at_1": float(hint_precision_at_1),
		"hint_r2": float(hint_r2),
	}
