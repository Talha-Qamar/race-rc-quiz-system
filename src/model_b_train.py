"""Main training script for Model B.

Run with: python src/model_b_train.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
	from evaluate import evaluate_distractor_approaches, evaluate_model_b
	from model_b_distractor import (
		DISTRACTOR_RANKER_PATH,
		VOCAB_PATH,
		build_vocab_and_cooccurrence,
		load_word2vec_model,
		train_distractor_ranker,
	)
	from model_b_hint import HINT_SCORER_PATH, train_hint_scorer
except ImportError:  # pragma: no cover - fallback when executed as a package
	from src.evaluate import evaluate_distractor_approaches, evaluate_model_b
	from src.model_b_distractor import (
		DISTRACTOR_RANKER_PATH,
		VOCAB_PATH,
		build_vocab_and_cooccurrence,
		load_word2vec_model,
		train_distractor_ranker,
	)
	from src.model_b_hint import HINT_SCORER_PATH, train_hint_scorer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
MODEL_B_DIR = ROOT_DIR / "models" / "model_b"
EVALUATION_RESULTS_PATH = MODEL_B_DIR / "evaluation_results.json"
TRAIN_SAMPLE_SIZE = 4000
VAL_SAMPLE_SIZE = 1000


def _load_split(split: str) -> pd.DataFrame:
	candidate = DATA_DIR / f"{split}.csv"
	if not candidate.exists():
		raise FileNotFoundError(f"Missing split file: {candidate}")
	frame = pd.read_csv(candidate)
	required = {"article", "question", "A", "B", "C", "D", "answer"}
	missing = sorted(required - set(frame.columns))
	if missing:
		raise ValueError(f"{candidate} is missing required columns: {missing}")
	return frame


def _attach_answer_text(frame: pd.DataFrame) -> pd.DataFrame:
	enriched = frame.copy()
	enriched["correct_answer_text"] = enriched.apply(
		lambda row: str(row.get(str(row.get("answer", "")).strip().upper(), "")).strip(),
		axis=1,
	)
	return enriched


def _print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
	print(f"Train rows: {len(train_df):,}")
	print(f"Val rows:   {len(val_df):,}")


def _sample_frame(frame: pd.DataFrame, sample_size: int) -> pd.DataFrame:
	if len(frame) <= sample_size:
		return frame.reset_index(drop=True)
	return frame.sample(n=sample_size, random_state=42).reset_index(drop=True)


def main() -> int:
	print("=" * 88)
	print("MODEL B: TRADITIONAL GENERATION TRAINING PIPELINE")
	print("=" * 88)

	train_df = _attach_answer_text(_load_split("train"))
	val_df = _attach_answer_text(_load_split("val"))
	train_df = _sample_frame(train_df, TRAIN_SAMPLE_SIZE)
	val_df = _sample_frame(val_df, VAL_SAMPLE_SIZE)
	_print_split_summary(train_df, val_df)

	print("\n[1/4] Building vocabulary and co-occurrence statistics...")
	vocab, word2idx, _cooccurrence = build_vocab_and_cooccurrence(train_df, vocab_size=5000)
	print(f">>> Vocab size: {len(vocab)}")

	try:
		w2v_model = load_word2vec_model()
	except Exception as exc:  # pragma: no cover - keep training robust without internet access
		print(f"Word2Vec could not be loaded: {exc}")
		w2v_model = None

	print("\n[2/4] Training distractor ranker...")
	distractor_ranker, distractor_metrics = train_distractor_ranker(
		train_df=train_df,
		vocab=vocab,
		word2idx=word2idx,
		val_df=val_df,
		w2v_model=w2v_model,
		save_path=DISTRACTOR_RANKER_PATH,
	)

	print("\n[3/4] Training hint scorer...")
	hint_scorer, hint_metrics = train_hint_scorer(
		train_df=train_df,
		val_df=val_df,
		save_path=HINT_SCORER_PATH,
	)

	print("\n[4/4] Evaluating Model B on validation data...")
	comparison_metrics = evaluate_distractor_approaches(
		val_df=val_df,
		vocab=vocab,
		word2idx=word2idx,
		distractor_ranker=distractor_ranker,
		w2v_model=w2v_model,
	)
	combined_metrics = evaluate_model_b(
		val_df=val_df,
		vocab=vocab,
		word2idx=word2idx,
		distractor_ranker=distractor_ranker,
		hint_scorer=hint_scorer,
		w2v_model=w2v_model,
	)

	MODEL_B_DIR.mkdir(parents=True, exist_ok=True)
	evaluation_results: dict[str, Any] = {
		"distractor_validation": distractor_metrics,
		"hint_validation": hint_metrics,
		"one_hot_only": comparison_metrics.get("one_hot_only", {}),
		"word2vec_combined": comparison_metrics.get("word2vec_combined", {}),
		"combined_validation": combined_metrics,
		"artifacts": {
			"vocab": str(VOCAB_PATH),
			"distractor_ranker": str(DISTRACTOR_RANKER_PATH),
			"hint_scorer": str(HINT_SCORER_PATH),
		},
	}
	EVALUATION_RESULTS_PATH.write_text(json.dumps(evaluation_results, indent=2, sort_keys=True), encoding="utf-8")

	print("Models saved.")
	print(f"Evaluation summary: {EVALUATION_RESULTS_PATH}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
