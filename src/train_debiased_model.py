

# src/train_debiased_model.py

import os
import sys
import argparse
import joblib
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import accuracy_score, roc_auc_score

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_prep import load_and_prepare_data
from src.dp_logistic_regression import DPLogisticRegression


IDENTIFIER_COLUMNS = [
    "id",
    "name",
    "email",
    "phone",
    "address",
    "ssn",
]


def _select_sensitive_feature(X_train, requested_column=None):
    if requested_column and requested_column in X_train.columns:
        return X_train[requested_column], requested_column

    if "gender_Male" in X_train.columns:
        return X_train["gender_Male"], "gender_Male"

    if "gender" in X_train.columns:
        return X_train["gender"], "gender"

    gender_columns = [c for c in X_train.columns if c.startswith("gender_")]
    if gender_columns:
        chosen = sorted(gender_columns)[0]
        return X_train[chosen], chosen

    raise ValueError("Sensitive feature not found. Provide --sensitive-column or include a gender column.")


def _apply_data_minimization(X_train, X_test, minimization_mode):
    columns_to_drop = []

    if minimization_mode in {"drop_identifiers", "strict"}:
        columns_to_drop.extend([c for c in X_train.columns if c.lower() in IDENTIFIER_COLUMNS])

    if minimization_mode == "strict":
        sensitive_prefixes = ("gender_", "race_", "zip_code_group_")
        columns_to_drop.extend([c for c in X_train.columns if c.startswith(sensitive_prefixes)])

    columns_to_drop = sorted(set(columns_to_drop))
    if columns_to_drop:
        X_train = X_train.drop(columns=columns_to_drop, errors="ignore")
        X_test = X_test.drop(columns=columns_to_drop, errors="ignore")

    return X_train, X_test, columns_to_drop


def _build_base_estimator(privacy_mode, epsilon, max_iter, random_state):
    if privacy_mode == "dp":
        return DPLogisticRegression(
            epsilon=epsilon,
            epochs=max_iter,
            random_state=random_state,
        )

    return LogisticRegression(solver="liblinear", max_iter=max_iter, random_state=random_state)


def train_fair_model(
    data_path,
    model_path="results/model_debiased_xgb.pkl",
    features_path="results/debiased_model_features.pkl",
    privacy_mode="dp",
    epsilon=8.0,
    minimization_mode="strict",
    sensitive_column=None,
    max_iter=300,
    random_state=42,
):
    print("📥 Loading and preparing dataset...")
    (X_train, X_test, y_train, y_test), _ = load_and_prepare_data(data_path)

    sensitive_feature, sensitive_name = _select_sensitive_feature(X_train, requested_column=sensitive_column)
    print(f"👤 Using '{sensitive_name}' as sensitive feature.")

    X_train, X_test, dropped_columns = _apply_data_minimization(
        X_train,
        X_test,
        minimization_mode=minimization_mode,
    )
    if dropped_columns:
        print(f"🧹 Data minimization removed {len(dropped_columns)} columns: {dropped_columns}")
    else:
        print("🧹 Data minimization did not remove any columns.")

    print("🔒 Privacy mode:", privacy_mode)
    if privacy_mode == "dp":
        print(f"🔒 Differential privacy epsilon: {epsilon}")

    # ───── Fair + private model training ─────
    print("⚖️ Training debiased model with Demographic Parity constraint...")
    base_model = _build_base_estimator(
        privacy_mode=privacy_mode,
        epsilon=epsilon,
        max_iter=max_iter,
        random_state=random_state,
    )
    mitigator = ExponentiatedGradient(base_model, DemographicParity())
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_feature)

    # ───── Evaluate model ─────
    y_pred = mitigator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Debiased Accuracy: {acc:.4f}")

    try:
        y_prob = mitigator._pmf_predict(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"✅ Debiased AUC: {auc:.4f}")
    except Exception:
        print("⚠️ Could not calculate AUC — probabilities not available.")

    # ───── Save model and features ─────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(mitigator, model_path)
    joblib.dump(X_train.columns.tolist(), features_path)
    print(f"💾 Debiased model saved to: {model_path}")
    print(f"💾 Features saved to: {features_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train fairness-mitigated debiased model with privacy controls.")
    parser.add_argument("--data-path", default="data/loan_dataset.csv", help="Path to training dataset CSV")
    parser.add_argument("--model-path", default="results/model_debiased_xgb.pkl", help="Path to save debiased model")
    parser.add_argument(
        "--features-path",
        default="results/debiased_model_features.pkl",
        help="Path to save model feature list",
    )
    parser.add_argument(
        "--privacy-mode",
        choices=["dp", "standard"],
        default="dp",
        help="Use 'dp' for differential privacy logistic regression, or 'standard' for non-private logistic regression",
    )
    parser.add_argument("--epsilon", type=float, default=8.0, help="Differential privacy epsilon (used only when privacy-mode=dp)")
    parser.add_argument(
        "--minimization-mode",
        choices=["none", "drop_identifiers", "strict"],
        default="strict",
        help="Data minimization level: strict drops gender/race/zip encoded features from model inputs",
    )
    parser.add_argument("--sensitive-column", default=None, help="Optional encoded sensitive column to use for fairness constraint")
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum iterations for logistic regression optimizer")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_fair_model(
        data_path=args.data_path,
        model_path=args.model_path,
        features_path=args.features_path,
        privacy_mode=args.privacy_mode,
        epsilon=args.epsilon,
        minimization_mode=args.minimization_mode,
        sensitive_column=args.sensitive_column,
        max_iter=args.max_iter,
    )
