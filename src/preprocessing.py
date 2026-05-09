from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk


RAW_COLUMNS = ["id", "article", "question", "A", "B", "C", "D", "answer"]
OPTION_COLUMNS = ["A", "B", "C", "D"]
SPLIT_ALIASES = {
	"train": ("train.csv",),
	"dev": ("dev.csv", "val.csv"),
	"val": ("val.csv", "dev.csv"),
	"test": ("test.csv",),
}

_PUNCT_TRANSLATION = str.maketrans({symbol: " " for symbol in string.punctuation})
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class PreprocessConfig:
	raw_dir: Path
	processed_dir: Path
	vectorizer_kind: str = "tfidf"
	max_features: int = 15000
	ngram_max: int = 2
	cosine_max_features: int = 20000


def normalize_text(value: object) -> str:
	if pd.isna(value):
		return ""
	text = str(value).lower().translate(_PUNCT_TRANSLATION)
	return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize_text(value: object) -> list[str]:
	cleaned = normalize_text(value)
	return [token for token in cleaned.split(" ") if token]


def sentence_split(text: object) -> list[str]:
	cleaned = normalize_text(text)
	if not cleaned:
		return []
	try:
		# Ensure punkt is available; download if missing
		nltk.data.find("tokenizers/punkt")
	except LookupError:
		nltk.download("punkt")
	parts = nltk.sent_tokenize(str(text))
	sentences = [normalize_text(part) for part in parts if normalize_text(part)]
	return sentences or [cleaned]


def jaccard_similarity(left_tokens: Iterable[str], right_tokens: Iterable[str]) -> float:
	left_set = set(left_tokens)
	right_set = set(right_tokens)
	union = left_set | right_set
	if not union:
		return 0.0
	return len(left_set & right_set) / len(union)


def overlap_count(left_tokens: Iterable[str], right_tokens: Iterable[str]) -> int:
	return len(set(left_tokens) & set(right_tokens))


def rowwise_cosine(left_matrix, right_matrix) -> np.ndarray:
	numerator = left_matrix.multiply(right_matrix).sum(axis=1).A1
	left_norm = np.sqrt(left_matrix.multiply(left_matrix).sum(axis=1)).A1
	right_norm = np.sqrt(right_matrix.multiply(right_matrix).sum(axis=1)).A1
	denominator = np.maximum(left_norm * right_norm, 1e-12)
	return numerator / denominator


def add_cosine_feature(
	frame: pd.DataFrame,
	left_column: str,
	right_column: str,
	output_column: str,
	vectorizer,
) -> None:
	"""Compute rowwise cosine using a pre-fitted vectorizer (fit on train only).

	The vectorizer should be fitted once on train data and passed here to avoid
	repeated fitting and leakage.
	"""
	left_values = frame[left_column].fillna("").astype(str).tolist()
	right_values = frame[right_column].fillna("").astype(str).tolist()
	left_matrix = vectorizer.transform(left_values)
	right_matrix = vectorizer.transform(right_values)
	frame[output_column] = rowwise_cosine(left_matrix, right_matrix)


def add_option_cosine_features(frame: pd.DataFrame, vectorizer) -> pd.DataFrame:
	with_cosine = frame.copy()
	add_cosine_feature(
		with_cosine,
		left_column="question_clean",
		right_column="option_clean",
		output_column="question_option_cosine",
		vectorizer=vectorizer,
	)
	add_cosine_feature(
		with_cosine,
		left_column="article_clean",
		right_column="option_clean",
		output_column="article_option_cosine",
		vectorizer=vectorizer,
	)
	return with_cosine


def add_sentence_cosine_features(frame: pd.DataFrame, vectorizer) -> pd.DataFrame:
	with_cosine = frame.copy()
	add_cosine_feature(
		with_cosine,
		left_column="question_clean",
		right_column="sentence_clean",
		output_column="question_sentence_cosine",
		vectorizer=vectorizer,
	)
	add_cosine_feature(
		with_cosine,
		left_column="correct_answer_clean",
		right_column="sentence_clean",
		output_column="answer_sentence_cosine",
		vectorizer=vectorizer,
	)
	return with_cosine


def load_split(raw_dir: Path, split: str) -> pd.DataFrame:
	split = split.lower()
	for filename in SPLIT_ALIASES[split]:
		candidate = raw_dir / filename
		if candidate.exists():
			frame = pd.read_csv(candidate)
			missing = [column for column in RAW_COLUMNS if column not in frame.columns]
			if missing:
				raise ValueError(f"{candidate} is missing required columns: {missing}")
			return frame
	raise FileNotFoundError(f"Could not find a CSV for split '{split}' in {raw_dir}")


def enrich_clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
	enriched = frame.copy()
	for column in ["article", "question", *OPTION_COLUMNS]:
		enriched[f"{column}_clean"] = enriched[column].map(normalize_text)
		enriched[f"{column}_tokens"] = enriched[column].map(tokenize_text)

	enriched["article_token_count"] = enriched["article_tokens"].map(len)
	enriched["question_token_count"] = enriched["question_tokens"].map(len)
	enriched["article_char_count"] = enriched["article_clean"].map(len)
	enriched["question_char_count"] = enriched["question_clean"].map(len)
	return enriched


def build_option_level_frame(frame: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for _, row in frame.iterrows():
		answer = normalize_text(row["answer"]).upper()
		question_tokens = row["question_tokens"]
		article_tokens = row["article_tokens"]

		for option_label in OPTION_COLUMNS:
			option_text = row[option_label]
			option_clean = row[f"{option_label}_clean"]
			option_tokens = row[f"{option_label}_tokens"]

			rows.append(
				{
					"id": row["id"],
					"answer": answer,
					"option_label": option_label,
					"option_text": option_text,
					"option_clean": option_clean,
					"is_correct": int(option_label == answer),
					"article": row["article"],
					"question": row["question"],
					"article_clean": row["article_clean"],
					"question_clean": row["question_clean"],
					"article_token_count": row["article_token_count"],
					"question_token_count": row["question_token_count"],
					"article_char_count": row["article_char_count"],
					"question_char_count": row["question_char_count"],
					"option_token_count": len(option_tokens),
					"option_char_count": len(option_clean),
					"question_option_overlap": overlap_count(question_tokens, option_tokens),
					"article_option_overlap": overlap_count(article_tokens, option_tokens),
					"question_option_jaccard": jaccard_similarity(question_tokens, option_tokens),
					"article_option_jaccard": jaccard_similarity(article_tokens, option_tokens),
					"combined_text": " ".join(
						part for part in [row["article_clean"], row["question_clean"], option_clean] if part
					),
				}
			)

	option_frame = pd.DataFrame(rows)
	option_frame["label"] = option_frame["is_correct"].astype(int)
	return option_frame


def build_sentence_level_frame(frame: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for _, row in frame.iterrows():
		correct_label = normalize_text(row["answer"]).upper()
		correct_answer_text = row[f"{correct_label}"] if correct_label in OPTION_COLUMNS else ""
		correct_answer_clean = normalize_text(correct_answer_text)
		question_tokens = row["question_tokens"]
		answer_tokens = tokenize_text(correct_answer_text)

		sentences = sentence_split(row["article"])
		for sentence_index, sentence in enumerate(sentences):
			sentence_tokens = tokenize_text(sentence)
			rows.append(
				{
					"id": row["id"],
					"answer": correct_label,
					"question_clean": row["question_clean"],
					"correct_answer_clean": correct_answer_clean,
					"sentence_index": sentence_index,
					"sentence_text": sentence,
					"sentence_clean": normalize_text(sentence),
					"sentence_token_count": len(sentence_tokens),
					"sentence_char_count": len(sentence),
					"question_sentence_overlap": overlap_count(question_tokens, sentence_tokens),
					"answer_sentence_overlap": overlap_count(answer_tokens, sentence_tokens),
					"question_sentence_jaccard": jaccard_similarity(question_tokens, sentence_tokens),
					"answer_sentence_jaccard": jaccard_similarity(answer_tokens, sentence_tokens),
					"sentence_contains_answer": int(bool(set(answer_tokens) & set(sentence_tokens))),
					"sentence_order_ratio": sentence_index / max(len(sentences), 1),
				}
			)

	return pd.DataFrame(rows)


def vectorizer_for(kind: str, max_features: int, ngram_max: int):
	kind = kind.lower().strip()
	if kind == "tfidf":
		return TfidfVectorizer(
			max_features=max_features,
			stop_words="english",
			ngram_range=(1, ngram_max),
			min_df=1,
			max_df=0.95,
			sublinear_tf=True,
		)

	if kind in {"onehot", "count", "binary"}:
		return CountVectorizer(
			binary=True,
			max_features=max_features,
			stop_words="english",
			ngram_range=(1, ngram_max),
			min_df=1,
		)

	raise ValueError("vectorizer_kind must be either 'onehot' or 'tfidf'")


def save_split_outputs(frame: pd.DataFrame, processed_dir: Path, split_name: str) -> Path:
	output_path = processed_dir / f"{split_name}_clean.csv"
	frame.to_csv(output_path, index=False)
	return output_path


def save_vectorized_outputs(option_frames: dict[str, pd.DataFrame], config: PreprocessConfig) -> dict[str, Path]:
	artifacts_dir = config.processed_dir / "artifacts"
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	# Fit both TF-IDF (preferred) and BOW on train text; save both vectorizers.
	train_text = option_frames["train"]["combined_text"].fillna("").tolist()
	tfidf_vec = vectorizer_for("tfidf", config.max_features, config.ngram_max)
	tfidf_vec.fit(train_text)
	joblib.dump(tfidf_vec, artifacts_dir / "tfidf_vectorizer.joblib")

	bow_vec = vectorizer_for("onehot", config.max_features, config.ngram_max)
	bow_vec.fit(train_text)
	joblib.dump(bow_vec, artifacts_dir / "bow_vectorizer.joblib")

	# Use config.vectorizer_kind to select primary matrix type
	primary_vec = tfidf_vec if config.vectorizer_kind == "tfidf" else bow_vec

	# Save train matrix
	save_npz(config.processed_dir / "model_a_train_X.npz", primary_vec.transform(train_text))

	saved_paths = {
		"tfidf_vectorizer": artifacts_dir / "tfidf_vectorizer.joblib",
		"bow_vectorizer": artifacts_dir / "bow_vectorizer.joblib",
		"model_a_train_matrix": config.processed_dir / "model_a_train_X.npz",
	}

	for split_name, option_frame in option_frames.items():
		matrix = primary_vec.transform(option_frame["combined_text"].fillna("").tolist())
		matrix_path = config.processed_dir / f"model_a_{split_name}_X.npz"
		save_npz(matrix_path, matrix)

		# Save features CSV with clearer name
		option_path = config.processed_dir / f"model_a_{split_name}_features.csv"
		# Ensure token columns are not present on disk
		cols_to_drop = [c for c in option_frame.columns if c.endswith("_tokens")]
		frame_to_save = option_frame.drop(columns=cols_to_drop, errors="ignore")
		frame_to_save.to_csv(option_path, index=False)

		# Save labels separately as npy
		if "label" in option_frame.columns:
			y = option_frame["label"].to_numpy(dtype=np.int8)
			np.save(str(config.processed_dir / f"y_{split_name}.npy"), y)
			saved_paths[f"y_{split_name}"] = config.processed_dir / f"y_{split_name}.npy"

		saved_paths[f"model_a_{split_name}_matrix"] = matrix_path
		saved_paths[f"model_a_{split_name}_features"] = option_path

	return saved_paths


def run_preprocessing(config: PreprocessConfig) -> dict[str, str]:
	config.processed_dir.mkdir(parents=True, exist_ok=True)

	cleaned_splits: dict[str, pd.DataFrame] = {}
	option_frames: dict[str, pd.DataFrame] = {}
	sentence_frames: dict[str, pd.DataFrame] = {}
	outputs: dict[str, str] = {}

	# Collect frames in-memory first
	for split in ("train", "val", "test"):
		raw_frame = load_split(config.raw_dir, split)
		cleaned_frame = enrich_clean_columns(raw_frame)
		option_frame = build_option_level_frame(cleaned_frame)
		sentence_frame = build_sentence_level_frame(cleaned_frame)

		cleaned_splits[split] = cleaned_frame
		option_frames[split] = option_frame
		sentence_frames[split] = sentence_frame

	# Fit TF-IDF on train combined_text for cosine features only once
	train_text_for_cosine = option_frames["train"]["combined_text"].fillna("").tolist()
	cosine_vec = vectorizer_for("tfidf", config.cosine_max_features, config.ngram_max)
	cosine_vec.fit(train_text_for_cosine)

	# Now compute cosine features using the pre-fitted cosine_vec and save outputs
	for split in ("train", "val", "test"):
		cleaned_frame = cleaned_splits[split]
		# drop token columns before saving cleaned CSVs to disk
		cols_to_drop = [c for c in cleaned_frame.columns if c.endswith("_tokens")]
		cleaned_frame_to_save = cleaned_frame.drop(columns=cols_to_drop, errors="ignore")
		outputs[f"{split}_clean"] = str(save_split_outputs(cleaned_frame_to_save, config.processed_dir, split))

		# Add cosine features to frames using the pre-fitted vectorizer
		option_with_cosine = add_option_cosine_features(option_frames[split], cosine_vec)
		sentence_with_cosine = add_sentence_cosine_features(sentence_frames[split], cosine_vec)

		# Save candidate and sentence CSVs (candidates derived from option frame)
		candidate_df = option_with_cosine[["id", "option_label", "option_text", "option_clean"]].copy()
		candidate_path = config.processed_dir / f"model_b_{split}_candidates.csv"
		candidate_df.to_csv(candidate_path, index=False)
		sentence_path = config.processed_dir / f"model_b_{split}_sentences.csv"
		sentence_with_cosine.to_csv(sentence_path, index=False)

		outputs[f"{split}_model_b_candidates"] = str(candidate_path)
		outputs[f"{split}_model_b_sentences"] = str(sentence_path)

	# Save vectorized outputs (matrices, feature CSVs, and y arrays)
	outputs.update({key: str(path) for key, path in save_vectorized_outputs(option_frames, config).items()})

	manifest = {
		"vectorizer_kind": config.vectorizer_kind,
		"raw_dir": str(config.raw_dir),
		"processed_dir": str(config.processed_dir),
		"splits": {
			split: {
				"clean_rows": len(cleaned_splits[split]),
				"model_a_rows": len(option_frames[split]),
				"model_b_sentence_rows": len(sentence_frames[split]),
			}
			for split in cleaned_splits
		},
		"files": outputs,
	}

	manifest_path = config.processed_dir / "preprocessing_manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
	outputs["manifest"] = str(manifest_path)
	return outputs


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Preprocess the RACE dataset for model training.")
	parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing train/dev/test CSVs")
	parser.add_argument(
		"--processed-dir",
		type=Path,
		default=Path("data/processed"),
		help="Directory where cleaned and feature-engineered files are written",
	)
	parser.add_argument(
		"--vectorizer-kind",
		choices=("onehot", "tfidf"),
		default="tfidf",
		help="Primary classical representation to build for the option-level training data",
	)
	parser.add_argument("--max-features", type=int, default=15000, help="Maximum vocabulary size for vectorization")
	parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram size for the vectorizer")
	parser.add_argument(
		"--cosine-max-features",
		type=int,
		default=20000,
		help="Maximum vocabulary size for cosine-similarity feature computation",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	config = PreprocessConfig(
		raw_dir=args.raw_dir,
		processed_dir=args.processed_dir,
		vectorizer_kind=args.vectorizer_kind,
		max_features=args.max_features,
		ngram_max=args.ngram_max,
		cosine_max_features=args.cosine_max_features,
	)
	outputs = run_preprocessing(config)
	print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
	main()
