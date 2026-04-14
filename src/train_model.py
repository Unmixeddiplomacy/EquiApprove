# src/train_model.py

import os
import sys
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# ─── Ensure src folder is importable ─────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_prep import load_and_prepare_data

def train_and_save_model(data_path, model_path="results/model_xgb.pkl"):
    print(f"[📂] Loading dataset: {data_path}")
    try:
        (X_train, X_test, y_train, y_test), _ = load_and_prepare_data(data_path)
    except ValueError as ve:
        print(f"[❌] {ve}")
        return

    print("[⚙️] Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    print("[📈] Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"[✅] Accuracy: {acc:.4f}")
    print(f"[✅] AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[💾] Model saved to: {model_path}")

if __name__ == "__main__":
    train_and_save_model("data/loan_dataset.csv")
