"""
Train top 2 models on corrected, disjoint train/val/test splits.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
import json

print("="*80)
print("TRAINING TOP 2 MODELS ON CORRECTED SPLITS")
print("="*80)

# Load corrected data
DATA_DIR = Path('data/processed')
X_train = load_npz(DATA_DIR / 'model_a_train_X.npz')
y_train = np.load(DATA_DIR / 'y_train.npy')
X_val = load_npz(DATA_DIR / 'model_a_val_X.npz')
y_val = np.load(DATA_DIR / 'y_val.npy')
X_test = load_npz(DATA_DIR / 'model_a_test_X.npz')
y_test = np.load(DATA_DIR / 'y_test.npy')

print(f"\n✅ Data loaded:")
print(f"   Train: {X_train.shape[0]:,} samples")
print(f"   Val:   {X_val.shape[0]:,} samples")
print(f"   Test:  {X_test.shape[0]:,} samples")

# ============================================================================
# MODEL 1: BALANCED LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION (class_weight='balanced')")
print("="*80)

model_lr_balanced = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    max_iter=500,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\nTraining on train set...")
model_lr_balanced.fit(X_train, y_train)
y_pred_lr = model_lr_balanced.predict(X_val)

metrics_lr = {
    'accuracy': accuracy_score(y_val, y_pred_lr),
    'f1_macro': f1_score(y_val, y_pred_lr, average='macro'),
    'f1_weighted': f1_score(y_val, y_pred_lr, average='weighted'),
    'precision_incorrect': precision_score(y_val, y_pred_lr, pos_label=0),
    'recall_incorrect': recall_score(y_val, y_pred_lr, pos_label=0),
    'precision_correct': precision_score(y_val, y_pred_lr, pos_label=1, zero_division=0),
    'recall_correct': recall_score(y_val, y_pred_lr, pos_label=1)
}

print(f"Validation Results:")
print(f"  Accuracy: {metrics_lr['accuracy']:.4f}")
print(f"  F1 (macro): {metrics_lr['f1_macro']:.4f}")
print(f"  Recall (Correct): {metrics_lr['recall_correct']:.4f}")
print(f"  Precision (Correct): {metrics_lr['precision_correct']:.4f}")

# Save model
model_path = Path('models/model_a/traditional/model_a_lr_balanced_corrected.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_lr_balanced, model_path)
print(f"\n✅ Saved: {model_path}")

# ============================================================================
# MODEL 2: LINEAR SVC
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: LINEAR SVC (class_weight='balanced')")
print("="*80)

model_svc = LinearSVC(
    C=1.0,
    loss='squared_hinge',
    dual=False,
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    verbose=0
)

print("\nTraining on train set...")
model_svc.fit(X_train, y_train)
y_pred_svc = model_svc.predict(X_val)

metrics_svc = {
    'accuracy': accuracy_score(y_val, y_pred_svc),
    'f1_macro': f1_score(y_val, y_pred_svc, average='macro'),
    'f1_weighted': f1_score(y_val, y_pred_svc, average='weighted'),
    'precision_incorrect': precision_score(y_val, y_pred_svc, pos_label=0),
    'recall_incorrect': recall_score(y_val, y_pred_svc, pos_label=0),
    'precision_correct': precision_score(y_val, y_pred_svc, pos_label=1, zero_division=0),
    'recall_correct': recall_score(y_val, y_pred_svc, pos_label=1)
}

print(f"Validation Results:")
print(f"  Accuracy: {metrics_svc['accuracy']:.4f}")
print(f"  F1 (macro): {metrics_svc['f1_macro']:.4f}")
print(f"  Recall (Correct): {metrics_svc['recall_correct']:.4f}")
print(f"  Precision (Correct): {metrics_svc['precision_correct']:.4f}")

# Save model
model_path = Path('models/model_a/traditional/model_a_linearsvc_corrected.joblib')
joblib.dump(model_svc, model_path)
print(f"\n✅ Saved: {model_path}")

# ============================================================================
# COMPARISON & TEST SET EVALUATION
# ============================================================================
print("\n" + "="*80)
print("COMPARISON ON VALIDATION SET")
print("="*80)

comparison = pd.DataFrame([
    {
        'Model': 'Balanced LR',
        'Accuracy': metrics_lr['accuracy'],
        'F1 (macro)': metrics_lr['f1_macro'],
        'Recall (Correct)': metrics_lr['recall_correct'],
        'Precision (Correct)': metrics_lr['precision_correct'],
    },
    {
        'Model': 'LinearSVC',
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

y_pred_lr_test = model_lr_balanced.predict(X_test)
y_pred_svc_test = model_svc.predict(X_test)

metrics_lr_test = {
    'accuracy': accuracy_score(y_test, y_pred_lr_test),
    'f1_macro': f1_score(y_test, y_pred_lr_test, average='macro'),
    'precision_correct': precision_score(y_test, y_pred_lr_test, pos_label=1, zero_division=0),
    'recall_correct': recall_score(y_test, y_pred_lr_test, pos_label=1),
}

metrics_svc_test = {
    'accuracy': accuracy_score(y_test, y_pred_svc_test),
    'f1_macro': f1_score(y_test, y_pred_svc_test, average='macro'),
    'precision_correct': precision_score(y_test, y_pred_svc_test, pos_label=1, zero_division=0),
    'recall_correct': recall_score(y_test, y_pred_svc_test, pos_label=1),
}

test_comparison = pd.DataFrame([
    {'Model': 'Balanced LR (Test)', **metrics_lr_test},
    {'Model': 'LinearSVC (Test)', **metrics_svc_test},
])

print("\n" + test_comparison.to_string(index=False))

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'timestamp': str(pd.Timestamp.now()),
    'note': 'Trained on corrected, properly-split data (70% train, 15% val, 15% test)',
    'data_info': {
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'features': int(X_train.shape[1]),
    },
    'models': {
        'balanced_lr': {
            'validation': {k: float(v) for k, v in metrics_lr.items()},
            'test': {k: float(v) for k, v in metrics_lr_test.items()},
        },
        'linearsvc': {
            'validation': {k: float(v) for k, v in metrics_svc.items()},
            'test': {k: float(v) for k, v in metrics_svc_test.items()},
        }
    }
}

results_path = Path('models/model_a/traditional/results_corrected_splits.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved: {results_path}")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\nBest Model on Validation Set:")
if metrics_lr['f1_macro'] > metrics_svc['f1_macro']:
    print(f"  → Balanced LR (F1: {metrics_lr['f1_macro']:.4f})")
else:
    print(f"  → LinearSVC (F1: {metrics_svc['f1_macro']:.4f})")
