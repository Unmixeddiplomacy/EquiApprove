# src/fairness_utils.py

import pandas as pd
import json
import argparse

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate_difference,
    equal_opportunity_difference,
    false_positive_rate,
    false_negative_rate
)

from sklearn.metrics import accuracy_score


def compute_fairness_metrics(df, y_true_col, y_pred_col, sensitive_col):
    # Drop rows with missing values in required columns
    df = df.dropna(subset=[y_true_col, y_pred_col, sensitive_col])
    
    if df.empty:
        raise ValueError("No valid rows after dropping missing values.")

    y_true = df[y_true_col]
    y_pred = df[y_pred_col]
    sensitive = df[sensitive_col]

    # Define metrics
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "demographic_parity_difference": lambda yt, yp: demographic_parity_difference(
            yt, yp, sensitive_features=sensitive),
        "equal_opportunity_difference": lambda yt, yp: equal_opportunity_difference(
            yt, yp, sensitive_features=sensitive),
        "false_negative_rate_difference": lambda yt, yp: false_negative_rate_difference(
            yt, yp, sensitive_features=sensitive),
        "equalized_odds_difference": lambda yt, yp: equalized_odds_difference(
            yt, yp, sensitive_features=sensitive),
    }

    # Compute metric frame
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )

    return {
        "overall": metric_frame.overall.to_dict(),
        "by_group": metric_frame.by_group.to_dict()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV with y_true, y_pred, and sensitive attribute")
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    parser.add_argument("--target", default="y_true", help="Column name for true labels")
    parser.add_argument("--prediction", default="y_pred", help="Column name for predictions")
    parser.add_argument("--sensitive", default="gender", help="Sensitive attribute column (e.g., gender, race)")

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    try:
        results = compute_fairness_metrics(df, args.target, args.prediction, args.sensitive)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)

        print(f"[📊] Fairness report saved to: {args.output}")
    except Exception as e:
        print(f"[❌] Failed to compute fairness metrics: {e}")
