"""Training script for the Model A question ranker.

Run with: python src/model_a_train_generator.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
	from model_a_generate import QUESTION_RANKER_PATH, train_question_ranker
except ImportError:  # pragma: no cover
	from src.model_a_generate import QUESTION_RANKER_PATH, train_question_ranker

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"


def _load_split(split: str) -> pd.DataFrame:
	frame_path = DATA_DIR / f"{split}.csv"
	if not frame_path.exists():
		raise FileNotFoundError(f"Missing split file: {frame_path}")
	frame = pd.read_csv(frame_path)
	required = {"article", "question", "A", "B", "C", "D", "answer"}
	missing = sorted(required - set(frame.columns))
	if missing:
		raise ValueError(f"{frame_path} is missing required columns: {missing}")
	return frame


def main() -> int:
	train_df = _load_split("train")	
	val_df = _load_split("val")
	print("Training question ranker...")
	_, metrics = train_question_ranker(train_df=train_df, val_df=val_df, save_path=QUESTION_RANKER_PATH)
	print(f">>> Training question ranker... Accuracy on val: {metrics.get('val_accuracy', 0.0):.2f}")
	print("Model saved.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
