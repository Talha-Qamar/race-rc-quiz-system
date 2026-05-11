# Model A Card

## Summary
Model A is the answer-verification classifier for the RACE quiz system. The final selected models are:

- Random Forest: best overall macro F1 / accuracy
- Logistic Regression: better class-1 recall and used as a backup / ensemble partner

## Training Data
- Corrected, non-overlapping `train/val/test` splits rebuilt from the merged RACE source.
- Stratified 70/15/15 split.
- Vectorized with TF-IDF on the corrected processed pipeline.

## Best Tuned Artifacts
- `model_a_random_forest_intensive_tuned.joblib`
- `model_a_random_forest_intensive_tuned.pkl`
- `model_a_logistic_regression_intensive_tuned.joblib`
- `model_a_logistic_regression_intensive_tuned.pkl`

## Observed Test Metrics
- Random Forest: accuracy 0.6472, balanced accuracy 0.5057, macro F1 0.5051
- Logistic Regression: accuracy 0.5329, balanced accuracy 0.5128, macro F1 0.4878

## Recommendation
Use Random Forest as the default single model. Use the Random Forest + Logistic Regression pair for comparison, backup, and ensemble-style evaluation.

## Ensemble decision rule
- Soft-vote ensemble over the two tuned models.
- Decision threshold: 0.49 on the averaged positive-class probability.
- This threshold was selected from the validation split to better support correct-answer detection.
