# generate_submission.py

import pandas as pd
import joblib
import warnings
import os

# ───────────── Paths ─────────────
MODEL_PATH = "results/model_xgb.pkl"
ENCODERS_PATH = "results/label_encoders.pkl"
TEST_PATH = "data/test.csv"
SUBMISSION_PATH = "submission.csv"

# ───────────── Load Model & Encoders ─────────────
print("📦 Loading model and encoders...")
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

# ───────────── Load and Prepare Test Data ─────────────
print("📄 Loading test dataset...")
df = pd.read_csv(TEST_PATH)

# Convert PascalCase to snake_case for consistency with training
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename 'id' column if it was 'ID' in original
if 'id' not in df.columns and 'id' in [col.lower() for col in df.columns]:
    df.rename(columns={col: 'id' for col in df.columns if col.lower() == 'id'}, inplace=True)

# Save original IDs
ids = df["id"]

# ───────────── Encode Categorical Features ─────────────
print("🔁 Encoding categorical features...")
df_encoded = df.copy()

for col in df_encoded.columns:
    if col in encoders:
        le = encoders[col]
        try:
            df_encoded[col] = le.transform(df_encoded[col].astype(str))
        except Exception:
            warnings.warn(f"⚠️ Encoding issue in column '{col}' — unseen values filled with -1.")
            df_encoded[col] = df_encoded[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    elif df_encoded[col].dtype == object:
        warnings.warn(f"⚠️ No encoder found for column '{col}' — filling with -1.")
        df_encoded[col] = -1  # fallback for unknown categorical

# ───────────── Feature Alignment ─────────────
model_features = model.get_booster().feature_names
for col in model_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[model_features]

# ───────────── Generate Predictions ─────────────
print("🧠 Generating predictions...")
try:
    probs = model.predict_proba(df_encoded)[:, 1]
    preds = (probs >= 0.5).astype(int)
except Exception as e:
    print(f"[❌] Prediction error: {e}")
    exit(1)

# ───────────── Save Submission ─────────────
print(f"💾 Saving predictions to {SUBMISSION_PATH}...")
submission = pd.DataFrame({
    "ID": ids,
    "LoanApproved": preds
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ Submission file created: {SUBMISSION_PATH}")
