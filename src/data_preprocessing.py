"""
data_preprocessing.py
Handles all data loading, cleaning, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os


# ── Column groups ──────────────────────────────────────────────────────────────
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn"
]

MULTICAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET = "Churn"


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV, do minimal dtype fixes."""
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused cols, fix missing values."""
    df = df.copy()

    # Drop customer ID - not a feature
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # TotalCharges NaN → median (new customers with 0 tenure)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop any remaining NaN rows
    df = df.dropna()

    return df


def encode_features(df: pd.DataFrame):
    """
    Encode categoricals.
    Returns: encoded df, label_encoders dict, scaler.
    """
    df = df.copy()
    label_encoders = {}

    # Binary cols → 0/1
    for col in BINARY_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Multi-class categoricals → one-hot
    df = pd.get_dummies(df, columns=MULTICAT_COLS, drop_first=True)

    return df, label_encoders


def scale_features(X_train, X_test):
    """Standard scale numeric features."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, scaler


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline → train/test splits."""
    df_clean = clean_data(df)
    df_encoded, label_encoders = encode_features(df_clean)

    X = df_encoded.drop(columns=[TARGET])
    y = df_encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, list(X.columns)


def generate_sample_dataset(save_path: str = "data/dataset.csv", n: int = 7043):
    """Generate synthetic Telco-style churn dataset."""
    np.random.seed(42)

    genders = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n)
    dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = np.random.randint(0, 73, n)
    phone = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multi_lines = np.where(phone == "No", "No phone service",
                           np.random.choice(["Yes", "No"], n))
    internet = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def internet_service(col_name, no_val="No internet service"):
        return np.where(internet == "No", no_val,
                        np.random.choice(["Yes", "No"], n))

    online_sec = internet_service("OnlineSecurity")
    online_bk = internet_service("OnlineBackup")
    device_prot = internet_service("DeviceProtection")
    tech_sup = internet_service("TechSupport")
    stream_tv = internet_service("StreamingTV")
    stream_mv = internet_service("StreamingMovies")
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    monthly = np.round(np.random.uniform(18, 119, n), 2)
    total = np.round(monthly * tenure + np.random.normal(0, 10, n), 2)
    total = np.clip(total, 0, None)

    # Churn logic — higher for month-to-month, fiber, high charges
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        + 0.10 * (monthly > 70)
        - 0.15 * (tenure > 24)
        - 0.10 * (online_sec == "Yes")
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = np.where(np.random.random(n) < churn_prob, "Yes", "No")

    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bk,
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mv,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved → {save_path}  ({n} rows)")
    return df
