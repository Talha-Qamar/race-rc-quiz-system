# race-rc-quiz-system
AI powered Reading Comprehension and Quiz Generation System using Classical Machine Learning, TF-IDF, SVM, Logistic Regression, and Streamlit on the RACE dataset.

## Model A status

The final tuned top-2 models are stored in `models/model_a/traditional/`.

- Default single model: Random Forest
- Backup / high-recall model: Logistic Regression
- Inference entrypoint: `src/inference.py`

Quick checks:

```bash
python3 src/inference.py --model ensemble --split test
python3 src/inference.py --model random_forest --split val
```

## Streamlit UI

The Model A tester UI lives in `ui/app.py`.

Run it with:

```bash
python3 -m streamlit run ui/app.py
```

The UI includes:

- a custom Model A tester for article/question/option input
- a sample browser using corrected RACE data
- a model summary panel with deployed metrics
- a short explanation of how Model A works
