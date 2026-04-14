# run_pipeline.py

import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ───────────── Config ─────────────
DATA_PATH        = "data/loan_dataset.csv"
MODEL_PATH       = "results/model_xgb.pkl"
ENCODERS_PATH    = "results/label_encoders.pkl"
PREDICTIONS_PATH = "results/baseline_predictions.csv"
RESULT_DIR       = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ───────────── Load Data ─────────────
print("📥  Loading dataset...")
df = pd.read_csv(DATA_PATH).dropna()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_") 
# ───────────── Encode Categorical Features ─────────────
print("🔤  Encoding categorical variables...")
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ───────────── Train/Test Split ─────────────
X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ───────────── Train Model ─────────────
print("🧠  Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# ───────────── Evaluate Model ─────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
print(f"✅  Accuracy: {acc:.4f}")

# ───────────── Save Artifacts ─────────────
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoders, ENCODERS_PATH)
print(f"💾  Model saved to        → {MODEL_PATH}")
print(f"💾  Label encoders saved → {ENCODERS_PATH}")

# ───────────── Save Predictions ─────────────
print("📊  Saving predictions for fairness audit...")
pred_df = X_test.copy()
pred_df["y_true"] = y_test.values
pred_df["y_pred"] = y_pred
pred_df["y_prob"] = y_prob
pred_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"📁  Predictions saved to → {PREDICTIONS_PATH}")
