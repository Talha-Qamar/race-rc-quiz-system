"""
Smart Intensive Tuning: Balanced GridSearchCV + RandomizedSearchCV
Thorough hyperparameter search on all 4 models in reasonable time (45-60 min).
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, make_scorer
)
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("SMART INTENSIVE TUNING: ALL 4 MODELS")
print("=" * 100)

# Setup & Data Loading
DATA_DIR = Path('data/processed')
MODELS_DIR = Path('models/model_a/traditional')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("\n[LOADING DATA]")
X_train = load_npz(DATA_DIR / 'model_a_train_X.npz')
y_train = np.load(DATA_DIR / 'y_train.npy')
X_val = load_npz(DATA_DIR / 'model_a_val_X.npz')
y_val = np.load(DATA_DIR / 'y_val.npy')
X_test = load_npz(DATA_DIR / 'model_a_test_X.npz')
y_test = np.load(DATA_DIR / 'y_test.npy')

print(f"✅ Data loaded: {X_train.shape[0]:,} train samples × {X_train.shape[1]:,} features")

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_incorrect': precision_score(y_true, y_pred, pos_label=0),
        'recall_incorrect': recall_score(y_true, y_pred, pos_label=0),
        'precision_correct': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_correct': recall_score(y_true, y_pred, pos_label=1),
    }

scorer = make_scorer(f1_score, average='macro')
all_results = {}

# MODEL 1: LOGISTIC REGRESSION
print("\n" + "=" * 100)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 100)

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'max_iter': [1000, 5000],
    'class_weight': ['balanced'],
}

print(f"GridSearchCV: 12 combinations × 5 folds = 60 fits")
start = time.time()

lr_search = GridSearchCV(
    LogisticRegression(solver='saga', penalty='l2', random_state=42, n_jobs=-1),
    param_grid_lr,
    cv=5,
    scoring=scorer,
    n_jobs=-1,
    verbose=0
)

lr_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"✅ Complete in {elapsed:.1f}s")
print(f"   Best C: {lr_search.best_params_['C']:.4f}")
print(f"   Best CV F1: {lr_search.best_score_:.4f}")

model_lr = lr_search.best_estimator_
metrics_lr_val = compute_metrics(y_val, model_lr.predict(X_val))
metrics_lr_test = compute_metrics(y_test, model_lr.predict(X_test))

print(f"   Val F1: {metrics_lr_val['f1_macro']:.4f} | Test F1: {metrics_lr_test['f1_macro']:.4f}")

joblib.dump(model_lr, MODELS_DIR / 'model_a_logistic_regression_intensive_tuned.joblib')
all_results['logistic_regression'] = {
    'best_params': dict(lr_search.best_params_),
    'cv_f1': float(lr_search.best_score_),
    'val_f1': float(metrics_lr_val['f1_macro']),
    'test_f1': float(metrics_lr_test['f1_macro']),
}

# MODEL 2: LINEAR SVC
print("\n" + "=" * 100)
print("MODEL 2: LINEAR SVC")
print("=" * 100)

param_grid_svc = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'loss': ['squared_hinge', 'hinge'],
    'max_iter': [2000, 5000],
    'class_weight': ['balanced'],
}

print(f"GridSearchCV: 24 combinations × 5 folds = 120 fits")
start = time.time()

svc_search = GridSearchCV(
    LinearSVC(dual=False, random_state=42, verbose=0),
    param_grid_svc,
    cv=5,
    scoring=scorer,
    n_jobs=-1,
    verbose=0
)

svc_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"✅ Complete in {elapsed:.1f}s")
print(f"   Best C: {svc_search.best_params_['C']:.4f}")
print(f"   Best CV F1: {svc_search.best_score_:.4f}")

model_svc = svc_search.best_estimator_
metrics_svc_val = compute_metrics(y_val, model_svc.predict(X_val))
metrics_svc_test = compute_metrics(y_test, model_svc.predict(X_test))

print(f"   Val F1: {metrics_svc_val['f1_macro']:.4f} | Test F1: {metrics_svc_test['f1_macro']:.4f}")

joblib.dump(model_svc, MODELS_DIR / 'model_a_linearsvc_intensive_tuned.joblib')
all_results['linear_svc'] = {
    'best_params': dict(svc_search.best_params_),
    'cv_f1': float(svc_search.best_score_),
    'val_f1': float(metrics_svc_val['f1_macro']),
    'test_f1': float(metrics_svc_test['f1_macro']),
}

# MODEL 3: NAIVE BAYES
print("\n" + "=" * 100)
print("MODEL 3: MULTINOMIAL NAIVE BAYES")
print("=" * 100)

param_grid_nb = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'fit_prior': [True, False],
}

print(f"GridSearchCV: 10 combinations × 5 folds = 50 fits")
start = time.time()

nb_search = GridSearchCV(
    MultinomialNB(),
    param_grid_nb,
    cv=5,
    scoring=scorer,
    n_jobs=-1,
    verbose=0
)

nb_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"✅ Complete in {elapsed:.1f}s")
print(f"   Best alpha: {nb_search.best_params_['alpha']}")
print(f"   Best CV F1: {nb_search.best_score_:.4f}")

model_nb = nb_search.best_estimator_
metrics_nb_val = compute_metrics(y_val, model_nb.predict(X_val))
metrics_nb_test = compute_metrics(y_test, model_nb.predict(X_test))

print(f"   Val F1: {metrics_nb_val['f1_macro']:.4f} | Test F1: {metrics_nb_test['f1_macro']:.4f}")

joblib.dump(model_nb, MODELS_DIR / 'model_a_naive_bayes_intensive_tuned.joblib')
all_results['naive_bayes'] = {
    'best_params': dict(nb_search.best_params_),
    'cv_f1': float(nb_search.best_score_),
    'val_f1': float(metrics_nb_val['f1_macro']),
    'test_f1': float(metrics_nb_test['f1_macro']),
}

# MODEL 4: RANDOM FOREST
print("\n" + "=" * 100)
print("MODEL 4: RANDOM FOREST CLASSIFIER")
print("=" * 100)

param_dist_rf = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [10, 15, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample', None],
}

print(f"RandomizedSearchCV: 25 random combinations × 5 folds = 125 fits")
start = time.time()

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0),
    param_dist_rf,
    n_iter=25,
    cv=5,
    scoring=scorer,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"✅ Complete in {elapsed:.1f}s")
print(f"   Best n_estimators: {rf_search.best_params_['n_estimators']}")
print(f"   Best max_depth: {rf_search.best_params_['max_depth']}")
print(f"   Best CV F1: {rf_search.best_score_:.4f}")

model_rf = rf_search.best_estimator_
metrics_rf_val = compute_metrics(y_val, model_rf.predict(X_val))
metrics_rf_test = compute_metrics(y_test, model_rf.predict(X_test))

print(f"   Val F1: {metrics_rf_val['f1_macro']:.4f} | Test F1: {metrics_rf_test['f1_macro']:.4f}")

joblib.dump(model_rf, MODELS_DIR / 'model_a_random_forest_intensive_tuned.joblib')
all_results['random_forest'] = {
    'best_params': {k: v if not isinstance(v, np.integer) else int(v) 
                   for k, v in rf_search.best_params_.items()},
    'cv_f1': float(rf_search.best_score_),
    'val_f1': float(metrics_rf_val['f1_macro']),
    'test_f1': float(metrics_rf_test['f1_macro']),
}

# RANKING
print("\n" + "=" * 100)
print("🏆 MODEL RANKING")
print("=" * 100)

ranking = []
for name, metrics_val, metrics_test in [
    ('Logistic Regression', metrics_lr_val, metrics_lr_test),
    ('Linear SVC', metrics_svc_val, metrics_svc_test),
    ('Naive Bayes', metrics_nb_val, metrics_nb_test),
    ('Random Forest', metrics_rf_val, metrics_rf_test),
]:
    ranking.append({
        'Model': name,
        'Val_F1': metrics_val['f1_macro'],
        'Test_F1': metrics_test['f1_macro'],
        'Avg_F1': (metrics_val['f1_macro'] + metrics_test['f1_macro']) / 2,
    })

ranking_df = pd.DataFrame(ranking).sort_values('Avg_F1', ascending=False).reset_index(drop=True)
ranking_df.index = ranking_df.index + 1

print("\n" + ranking_df.to_string())

top_2 = ranking_df.head(2)['Model'].tolist()

print(f"\n🥇 TOP 2 MODELS SELECTED:")
for i, model_name in enumerate(top_2, 1):
    print(f"   {i}. {model_name}")

# SAVE RESULTS
print("\n[SAVING RESULTS]")

results = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'method': 'Smart intensive tuning: GridSearchCV + RandomizedSearchCV',
    'ranking': ranking_df.to_dict('records'),
    'top_2_selected': top_2,
}

results_json = MODELS_DIR / 'results_intensive_tuning_final.json'
with open(results_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved: {results_json}")

ranking_csv = MODELS_DIR / 'ranking_intensive_tuning_final.csv'
ranking_df.to_csv(ranking_csv)
print(f"✅ Saved: {ranking_csv}")

print("\n" + "=" * 100)
print("✅ INTENSIVE TUNING COMPLETE!")
print("=" * 100)
print(f"\n🏆 Top 2 Selected:")
for i, model_name in enumerate(top_2, 1):
    print(f"   {i}. {model_name}")
