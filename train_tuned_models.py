"""
Train models with hyperparameter tuning on corrected splits.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, make_scorer
)
import json

print("="*80)
print("HYPERPARAMETER TUNING ON CORRECTED SPLITS")
print("="*80)

# Load corrected data
DATA_DIR = Path('data/processed')
X_train = load_npz(DATA_DIR / 'model_a_train_X.npz')
y_train = np.load(DATA_DIR / 'y_train.npy')
X_val = load_npz(DATA_DIR / 'model_a_val_X.npz')
y_val = np.load(DATA_DIR / 'y_val.npy')
X_test = load_npz(DATA_DIR / 'model_a_test_X.npz')
y_test = np.load(DATA_DIR / 'y_test.npy')

print(f"\n✅ Data loaded: Train {X_train.shape[0]:,} | Val {X_val.shape[0]:,} | Test {X_test.shape[0]:,}")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION WITH TUNING
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION - HYPERPARAMETER TUNING")
print("="*80)

print("\nSearching for best C parameter...")
param_dist = {
    'C': np.logspace(-3, 3, 20),  # Search wider range: 0.001 to 1000
}

# Use F1 macro as scoring metric (better for imbalanced data)
scorer = make_scorer(f1_score, average='macro')

lr_base = LogisticRegression(
    solver='saga',  # Better for sparse data
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

search_lr = RandomizedSearchCV(
    lr_base,
    param_dist,
    n_iter=15,
    cv=3,
    scoring=scorer,
    random_state=42,
    n_jobs=-1
)

search_lr.fit(X_train, y_train)
print(f"Best C: {search_lr.best_params_['C']:.4f}")
print(f"Best CV F1 Score: {search_lr.best_score_:.4f}")

model_lr_tuned = search_lr.best_estimator_
y_pred_lr = model_lr_tuned.predict(X_val)

metrics_lr = {
    'accuracy': accuracy_score(y_val, y_pred_lr),
    'f1_macro': f1_score(y_val, y_pred_lr, average='macro'),
    'f1_weighted': f1_score(y_val, y_pred_lr, average='weighted'),
    'precision_incorrect': precision_score(y_val, y_pred_lr, pos_label=0),
    'recall_incorrect': recall_score(y_val, y_pred_lr, pos_label=0),
    'precision_correct': precision_score(y_val, y_pred_lr, pos_label=1, zero_division=0),
    'recall_correct': recall_score(y_val, y_pred_lr, pos_label=1)
}

print(f"\nValidation Metrics:")
print(f"  Accuracy: {metrics_lr['accuracy']:.4f}")
print(f"  F1 (macro): {metrics_lr['f1_macro']:.4f}")
print(f"  Recall (Correct): {metrics_lr['recall_correct']:.4f}")
print(f"  Precision (Correct): {metrics_lr['precision_correct']:.4f}")

# Save
model_path = Path('models/model_a/traditional/model_a_lr_tuned_corrected.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_lr_tuned, model_path)
print(f"✅ Saved: {model_path}")

# ============================================================================
# MODEL 2: LINEAR SVC WITH TUNING
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: LINEAR SVC - HYPERPARAMETER TUNING")
print("="*80)

print("\nSearching for best C parameter...")
param_dist_svc = {
    'C': np.logspace(-2, 2, 15),  # Search range: 0.01 to 100
}

svc_base = LinearSVC(
    loss='squared_hinge',
    dual=False,
    max_iter=2000,
    class_weight='balanced',
    random_state=42,
    verbose=0
)

search_svc = RandomizedSearchCV(
    svc_base,
    param_dist_svc,
    n_iter=12,
    cv=3,
    scoring=scorer,
    random_state=42,
    n_jobs=-1
)

search_svc.fit(X_train, y_train)
print(f"Best C: {search_svc.best_params_['C']:.4f}")
print(f"Best CV F1 Score: {search_svc.best_score_:.4f}")

model_svc_tuned = search_svc.best_estimator_
y_pred_svc = model_svc_tuned.predict(X_val)

metrics_svc = {
    'accuracy': accuracy_score(y_val, y_pred_svc),
    'f1_macro': f1_score(y_val, y_pred_svc, average='macro'),
    'f1_weighted': f1_score(y_val, y_pred_svc, average='weighted'),
    'precision_incorrect': precision_score(y_val, y_pred_svc, pos_label=0),
    'recall_incorrect': recall_score(y_val, y_pred_svc, pos_label=0),
    'precision_correct': precision_score(y_val, y_pred_svc, pos_label=1, zero_division=0),
    'recall_correct': recall_score(y_val, y_pred_svc, pos_label=1)
}

print(f"\nValidation Metrics:")
print(f"  Accuracy: {metrics_svc['accuracy']:.4f}")
print(f"  F1 (macro): {metrics_svc['f1_macro']:.4f}")
print(f"  Recall (Correct): {metrics_svc['recall_correct']:.4f}")
print(f"  Precision (Correct): {metrics_svc['precision_correct']:.4f}")

# Save
model_path = Path('models/model_a/traditional/model_a_svc_tuned_corrected.joblib')
joblib.dump(model_svc_tuned, model_path)
print(f"✅ Saved: {model_path}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON - VALIDATION SET")
print("="*80)

comparison = pd.DataFrame([
    {
        'Model': 'Tuned LR',
        'Accuracy': metrics_lr['accuracy'],
        'F1 (macro)': metrics_lr['f1_macro'],
        'Recall (Correct)': metrics_lr['recall_correct'],
        'Precision (Correct)': metrics_lr['precision_correct'],
    },
    {
        'Model': 'Tuned SVC',
        'Accuracy': metrics_svc['accuracy'],
        'F1 (macro)': metrics_svc['f1_macro'],
        'Recall (Correct)': metrics_svc['recall_correct'],
        'Precision (Correct)': metrics_svc['precision_correct'],
    }
])

print("\n" + comparison.to_string(index=False))

# Test set evaluation
print("\n" + "="*80)
print("TEST SET EVALUATION")
print("="*80)

y_pred_lr_test = model_lr_tuned.predict(X_test)
y_pred_svc_test = model_svc_tuned.predict(X_test)

test_results = {
    'LR (tuned)': {
        'Accuracy': accuracy_score(y_test, y_pred_lr_test),
        'F1 (macro)': f1_score(y_test, y_pred_lr_test, average='macro'),
        'Recall (Correct)': recall_score(y_test, y_pred_lr_test, pos_label=1),
        'Precision (Correct)': precision_score(y_test, y_pred_lr_test, pos_label=1, zero_division=0),
    },
    'SVC (tuned)': {
        'Accuracy': accuracy_score(y_test, y_pred_svc_test),
        'F1 (macro)': f1_score(y_test, y_pred_svc_test, average='macro'),
        'Recall (Correct)': recall_score(y_test, y_pred_svc_test, pos_label=1),
        'Precision (Correct)': precision_score(y_test, y_pred_svc_test, pos_label=1, zero_division=0),
    }
}

test_df = pd.DataFrame(test_results).T
print("\n" + test_df.to_string())

# Save final results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

final_results = {
    'timestamp': str(pd.Timestamp.now()),
    'note': 'Tuned hyperparameters on corrected, properly-split data',
    'data_info': {
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'features': int(X_train.shape[1]),
        'class_ratio': 'Approximately 3:1 (incorrect:correct)',
    },
    'hyperparameters': {
        'logistic_regression': {
            'solver': 'saga',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'best_C': float(search_lr.best_params_['C']),
            'cv_f1_score': float(search_lr.best_score_),
        },
        'linear_svc': {
            'loss': 'squared_hinge',
            'dual': False,
            'max_iter': 2000,
            'class_weight': 'balanced',
            'best_C': float(search_svc.best_params_['C']),
            'cv_f1_score': float(search_svc.best_score_),
        }
    },
    'validation_results': {
        'logistic_regression': {k: float(v) for k, v in metrics_lr.items()},
        'linear_svc': {k: float(v) for k, v in metrics_svc.items()},
    },
    'test_results': {
        'logistic_regression': {k: float(v) for k, v in test_results['LR (tuned)'].items()},
        'linear_svc': {k: float(v) for k, v in test_results['SVC (tuned)'].items()},
    }
}

results_path = Path('models/model_a/traditional/results_tuned_corrected_splits.json')
with open(results_path, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n✅ Saved: {results_path}")

# Summary
print("\n" + "="*80)
print("✅ HYPERPARAMETER TUNING COMPLETE!")
print("="*80)

best_model_name = 'Tuned LR' if metrics_lr['f1_macro'] > metrics_svc['f1_macro'] else 'Tuned SVC'
best_f1 = max(metrics_lr['f1_macro'], metrics_svc['f1_macro'])

print(f"\nBest Model: {best_model_name} (Validation F1: {best_f1:.4f})")
print(f"\nModels saved to: models/model_a/traditional/")
print(f"Results saved to: {results_path}")
