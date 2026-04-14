# scripts/generate_debiased_predictions.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Make project root importable so pickled custom estimators can be loaded.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ──────────────────────────────
# Paths
# ──────────────────────────────
DATA_PATH = Path("data/loan_dataset.csv")
MODEL_PATH = Path("results/model_debiased_xgb.pkl")
FEATS_PATH = Path("results/debiased_model_features.pkl")
OUT_PATH = Path("results/debiased_predictions.csv")
IDENTIFIER_COLUMNS = ["id", "name", "email", "phone", "address", "ssn"]

# ──────────────────────────────
# Load Dataset
# ──────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Generate debiased predictions with privacy-safe defaults.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to input dataset")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path to saved debiased model")
    parser.add_argument("--features-path", default=str(FEATS_PATH), help="Path to saved debiased feature list")
    parser.add_argument("--out-path", default=str(OUT_PATH), help="Path to output predictions CSV")
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="Include y_prob in output CSV (disabled by default to reduce leakage)",
    )
    parser.add_argument(
        "--minimization-mode",
        choices=["none", "drop_identifiers", "strict"],
        default="strict",
        help="Data minimization level. strict removes encoded gender/race/zip feature columns from model inputs.",
    )
    return parser.parse_args()


def apply_data_minimization(df_features, minimization_mode):
    cols_to_drop = []

    if minimization_mode in {"drop_identifiers", "strict"}:
        cols_to_drop.extend([c for c in df_features.columns if c.lower() in IDENTIFIER_COLUMNS])

    if minimization_mode == "strict":
        sensitive_prefixes = ("gender_", "race_", "zip_code_group_")
        cols_to_drop.extend([c for c in df_features.columns if c.startswith(sensitive_prefixes)])

    cols_to_drop = sorted(set(cols_to_drop))
    if cols_to_drop:
        df_features = df_features.drop(columns=cols_to_drop, errors="ignore")
        print(f"🧹 Data minimization removed {len(cols_to_drop)} columns: {cols_to_drop}")
    else:
        print("🧹 Data minimization did not remove any columns.")

    return df_features


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    feats_path = Path(args.features_path)
    out_path = Path(args.out_path)

    print("📥  Loading dataset...")
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    if "loan_approved" not in df.columns:
        raise ValueError("❌ 'loan_approved' column missing from dataset.")

    # Drop known direct identifiers before any transformation.
    df = df.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")

    # Save ground truth before encoding
    y = df["loan_approved"].map({"Denied": 0, "Approved": 1})


    # ──────────────────────────────
    # Encode Features
    # ──────────────────────────────
    print("🔤  Encoding categorical variables...")
    df_enc = pd.get_dummies(df, drop_first=True)
    X = df_enc.drop(columns=["loan_approved"], errors="ignore")

    X = apply_data_minimization(X, minimization_mode=args.minimization_mode)

    # ──────────────────────────────
    # Load Model + Features
    # ──────────────────────────────
    print("📦  Loading debiased model...")
    model = joblib.load(model_path)

    print("📄  Loading feature list used during training...")
    if not feats_path.exists():
        raise FileNotFoundError(f"❌ Feature list not found at {feats_path}")

    model_feats = joblib.load(feats_path)

    # ──────────────────────────────
    # Align Features
    # ──────────────────────────────
    print("📐  Aligning features...")
    for col in model_feats:
        if col not in X.columns:
            X[col] = 0
    X = X[model_feats]

    # ──────────────────────────────
    # Predict
    # ──────────────────────────────
    print("🧠  Making predictions...")
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"❌ Error during prediction: {e}")

    y_prob = None
    if args.include_probabilities:
        try:
            y_prob = model.predict_proba(X)[:, 1]
            print("✅  Probability export enabled.")
        except Exception:
            print("⚠️  Model does not support `predict_proba`; saving labels only.")
    else:
        print("🔒  Probability export disabled by default to reduce leakage.")

    # ──────────────────────────────
    # Save Results
    # ──────────────────────────────
    print("💾  Saving predictions...")
    result_dict = {
        "y_true": y.values,
        "y_pred": y_pred,
    }
    if y_prob is not None:
        result_dict["y_prob"] = y_prob

    df_out = pd.DataFrame(result_dict)
    df_out.dropna(inplace=True)  # prevent dashboard crash

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"✅  Saved debiased predictions to → {out_path}")


if __name__ == "__main__":
    main()
