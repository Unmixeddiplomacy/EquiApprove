# scripts/generate_shap_explainer.py

import pandas as pd
import joblib
import shap
import os

print("🔍 Generating SHAP explainer...")

# ───── Load trained model ─────
model_path = "results/model_xgb.pkl"
data_path = "data/loan_dataset.csv"
explainer_path = "results/shap_explainer.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}")

model = joblib.load(model_path)

# ───── Load and prepare data ─────
df = pd.read_csv(data_path).dropna()
df.columns = df.columns.str.strip().str.lower()

# Map target column if needed
if df['loan_approved'].dtype == object:
    df['loan_approved'] = df['loan_approved'].str.strip().map({
        "Approved": 1,
        "Denied": 0
    })

# Features used for training
features = ['age', 'income', 'loan_amount', 'credit_score', 'gender', 'race', 'zip_code_group']
if not all(f in df.columns for f in features):
    missing = [f for f in features if f not in df.columns]
    raise ValueError(f"❌ Missing required columns: {missing}")

X = df[features]
X_encoded = pd.get_dummies(X)

# Align with model's expected features
model_features = model.get_booster().feature_names
for col in model_features:
    if col not in X_encoded:
        X_encoded[col] = 0
X_encoded = X_encoded[model_features]

# ───── Generate SHAP TreeExplainer ─────
print("📊 Building SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model, X_encoded, feature_perturbation="tree_path_dependent")

# ───── Save explainer ─────
os.makedirs("results", exist_ok=True)
joblib.dump(explainer, explainer_path)
print(f"✅ SHAP explainer saved to: {explainer_path}")
