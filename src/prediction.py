"""
prediction.py
Single-customer churn prediction from raw form input.
"""

import pandas as pd
import numpy as np
from .model_training import load_artifacts
from .data_preprocessing import NUMERIC_COLS


# ── All one-hot columns produced during training ───────────────────────────────
MULTICAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]


def build_input_row(raw: dict, feature_names: list) -> pd.DataFrame:
    """
    Convert flat form dict → one-hot encoded row matching training columns.
    raw keys (sample):
      gender, SeniorCitizen, Partner, Dependents, tenure,
      PhoneService, MultipleLines, InternetService, OnlineSecurity,
      OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
      StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
      MonthlyCharges, TotalCharges
    """
    # Binary encode
    binary_map = {
        "gender":           {"Male": 1, "Female": 0},
        "Partner":          {"Yes": 1, "No": 0},
        "Dependents":       {"Yes": 1, "No": 0},
        "PhoneService":     {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
    }

    row = {}
    for col, mapping in binary_map.items():
        row[col] = mapping.get(raw.get(col, "No"), 0)

    row["SeniorCitizen"] = int(raw.get("SeniorCitizen", 0))
    row["tenure"] = float(raw.get("tenure", 0))
    row["MonthlyCharges"] = float(raw.get("MonthlyCharges", 0))
    row["TotalCharges"] = float(raw.get("TotalCharges", 0))

    # One-hot encode multi-class cols
    for col in MULTICAT_COLS:
        val = raw.get(col, "")
        # Find matching dummies in feature_names
        for feat in feature_names:
            if feat.startswith(f"{col}_"):
                suffix = feat[len(col) + 1:]
                row[feat] = 1 if suffix == val else 0

    # Build DataFrame with ALL training columns in order
    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]


def predict_churn(raw: dict) -> dict:
    """
    Load artifacts, preprocess single row, return prediction dict.
    Returns:
      {
        "churn": bool,
        "probability": float,          # 0-100
        "label": "CHURN" | "NO CHURN",
        "confidence": str,
        "explanation": str
      }
    """
    model, scaler, feature_names = load_artifacts()
    X = build_input_row(raw, feature_names)

    # Scale numeric cols
    num_cols = [c for c in NUMERIC_COLS if c in X.columns]
    X[num_cols] = scaler.transform(X[num_cols])

    prob = model.predict_proba(X)[0][1]
    churn = prob >= 0.5

    # Confidence band
    if prob < 0.30:
        confidence = "Low Risk"
    elif prob < 0.50:
        confidence = "Moderate Risk"
    elif prob < 0.70:
        confidence = "High Risk"
    else:
        confidence = "Very High Risk"

    explanation = _generate_explanation(raw, prob)

    return {
        "churn": churn,
        "probability": round(prob * 100, 1),
        "label": "⚠️ LIKELY TO CHURN" if churn else "✅ LIKELY TO STAY",
        "confidence": confidence,
        "explanation": explanation,
    }


def _generate_explanation(raw: dict, prob: float) -> str:
    """Human-readable explanation of key churn drivers."""
    factors = []

    contract = raw.get("Contract", "")
    tenure = float(raw.get("tenure", 0))
    monthly = float(raw.get("MonthlyCharges", 0))
    internet = raw.get("InternetService", "")
    security = raw.get("OnlineSecurity", "No")

    if contract == "Month-to-month":
        factors.append("month-to-month contract (highest churn risk)")
    elif contract == "Two year":
        factors.append("two-year contract (strong retention signal)")

    if tenure < 12:
        factors.append(f"only {int(tenure)} months as customer (new customers churn more)")
    elif tenure > 36:
        factors.append(f"{int(tenure)} months tenure (loyal customer)")

    if monthly > 80:
        factors.append(f"high monthly charges (${monthly:.0f}/mo)")
    elif monthly < 35:
        factors.append(f"low monthly charges (${monthly:.0f}/mo)")

    if internet == "Fiber optic" and security == "No":
        factors.append("fiber optic without online security add-on")

    if not factors:
        factors.append("average risk profile across all features")

    factor_str = "; ".join(factors)
    direction = "increase" if prob >= 0.5 else "reduce"
    return (
        f"Churn probability: **{prob*100:.1f}%**. "
        f"Key factors that {direction} churn risk: {factor_str}."
    )
