# src/generate_assets.py

"""
generate_assets.py
──────────────────
Re-trains the production model and creates perfectly matched
model_xgb.pkl + shap_explainer.pkl inside the results/ folder.
"""

import os
import joblib
import shap
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ───────────── Config ─────────────
DATA_PATH       = "data/loan_dataset.csv"
RESULT_DIR      = "results"
MODEL_PATH      = os.path.join(RESULT_DIR, "model_xgb.pkl")
EXPLAINER_PATH  = os.path.join(RESULT_DIR, "shap_explainer.pkl")

os.makedirs(RESULT_DIR, exist_ok=True)

# ───────────── Load and Preprocess ─────────────
print("📥  Loading dataset …")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()
df.fillna(method='ffill', inplace=True)

# Convert target to binary
if df["loan_approved"].dtype == object:
    df["loan_approved"] = df["loan_approved"].str.strip().map({
        "Approved": 1,
        "Denied": 0
    })

if df["loan_approved"].isnull().any():
    raise ValueError("[❌] 'loan_approved' column has invalid values. Use 'Approved' or 'Denied'.")

# ───────────── Feature Selection ─────────────
features = ['age', 'income', 'loan_amount', 'credit_score', 'gender', 'race', 'zip_code_group']
for col in features:
    if col not in df.columns:
        raise ValueError(f"[❌] Missing required column: {col}")

X = df[features]
y = df['loan_approved']

# ───────────── Encode Categorical ─────────────
X = pd.get_dummies(X, drop_first=False)

# ───────────── Split ─────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ───────────── Train Model ─────────────
print("🧠  Training XGBoost …")
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

# ───────────── Evaluate ─────────────
proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, proba)
print(f"✅  Accuracy: {acc:.4f} | AUC: {auc:.4f}")

# ───────────── SHAP Explainer ─────────────
print("🔍  Building SHAP TreeExplainer …")
explainer = shap.TreeExplainer(model, X_train)

# ───────────── Save ─────────────
joblib.dump(model, MODEL_PATH)
joblib.dump(explainer, EXPLAINER_PATH)
print(f"💾  Model saved        → {MODEL_PATH}")
print(f"💾  SHAP explainer     → {EXPLAINER_PATH}")

print("🎉  Assets regenerated and production-ready!")
