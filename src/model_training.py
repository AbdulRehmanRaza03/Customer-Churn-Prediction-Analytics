"""
model_training.py
Train, evaluate, and persist ML models for churn prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)


# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"


# ── Model definitions ──────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    """Return full metric dict for one model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ── Training pipeline ─────────────────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train all models, compare, return:
      (best_model, best_name, all_results_dict)
    """
    models = get_models()
    results = {}
    best_f1 = -1
    best_model = None
    best_name = ""

    for name, model in models.items():
        print(f"  Training: {name} …")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {"model": model, **metrics}

        print(f"    Accuracy={metrics['accuracy']}  F1={metrics['f1']}  AUC={metrics['roc_auc']}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_name = name

    print(f"\n  Best model → {best_name}  (F1={best_f1})")
    return best_model, best_name, results


# ── Persist ───────────────────────────────────────────────────────────────────
def save_artifacts(model, scaler, feature_names: list):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    print(f"  Saved model → {MODEL_PATH}")
    print(f"  Saved scaler → {SCALER_PATH}")
    print(f"  Saved feature names → {FEATURE_NAMES_PATH}")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, scaler, feature_names


# ── Feature importance ────────────────────────────────────────────────────────
def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Works for RF (feature_importances_) and LR (coef_)."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    return df.sort_values("Importance", ascending=False).head(20).reset_index(drop=True)
