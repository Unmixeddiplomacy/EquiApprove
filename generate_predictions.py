# scripts/generate_predictions.py

import pandas as pd
import joblib
import os

print("📥 Loading trained model & dataset...")

# Load model
model = joblib.load("results/model_xgb.pkl")

# Load dataset
df = pd.read_csv("data/loan_dataset.csv").dropna()
df.columns = df.columns.str.strip().str.lower()

# Convert target to binary if needed
if df['loan_approved'].dtype == object:
    df['loan_approved'] = df['loan_approved'].str.strip().map({
        "Approved": 1,
        "Denied": 0
    })

# Final check
if df['loan_approved'].isnull().any():
    raise ValueError("❌ Target column 'loan_approved' contains unmapped values (must be 'Approved' or 'Denied')")

# Required features
features = ['age', 'income', 'loan_amount', 'credit_score', 'gender', 'race', 'zip_code_group']
missing_cols = [f for f in features if f not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required columns: {missing_cols}")

# Prepare features
X = df[features]
y = df["loan_approved"]

# One-hot encode
X_encoded = pd.get_dummies(X)

# Align with model input
model_features = model.get_booster().feature_names
for col in model_features:
    if col not in X_encoded:
        X_encoded[col] = 0
X_encoded = X_encoded[model_features]

# Predict
y_pred = model.predict(X_encoded)
y_prob = model.predict_proba(X_encoded)[:, 1]

# Save results
results = pd.DataFrame({
    "y_true": y,
    "y_pred": y_pred,
    "y_prob": y_prob
})

os.makedirs("results", exist_ok=True)
results.to_csv("results/baseline_predictions.csv", index=False)

print("✅ Predictions saved to: results/baseline_predictions.csv")
