"""
train.py
Run once to generate model.pkl, scaler.pkl, feature_names.pkl.
Usage:  python train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import generate_sample_dataset, load_data, get_train_test_split
from src.model_training import train_all_models, save_artifacts


DATA_PATH = "data/dataset.csv"


def main():
    print("=" * 55)
    print("  Customer Churn Model Training Pipeline")
    print("=" * 55)

    # 1. Dataset
    if not os.path.exists(DATA_PATH):
        print("\n[1/4] Generating synthetic dataset …")
        generate_sample_dataset(DATA_PATH)
    else:
        print(f"\n[1/4] Dataset found → {DATA_PATH}")

    # 2. Load & split
    print("\n[2/4] Loading and preprocessing …")
    df = load_data(DATA_PATH)
    print(f"  Shape: {df.shape}  |  Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")

    X_train, X_test, y_train, y_test, scaler, _, feature_names = get_train_test_split(df)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # 3. Train
    print("\n[3/4] Training models …")
    best_model, best_name, results = train_all_models(X_train, X_test, y_train, y_test)

    # 4. Save
    print("\n[4/4] Saving artifacts …")
    save_artifacts(best_model, scaler, feature_names)

    # Summary
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    for name, res in results.items():
        marker = " ← SELECTED" if name == best_name else ""
        print(f"\n  {name}{marker}")
        print(f"    Accuracy : {res['accuracy']}")
        print(f"    Precision: {res['precision']}")
        print(f"    Recall   : {res['recall']}")
        print(f"    F1 Score : {res['f1']}")
        print(f"    ROC-AUC  : {res['roc_auc']}")

    print("\n  Files created:")
    print("    model.pkl  |  scaler.pkl  |  feature_names.pkl")
    print("\n  Run app:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
