"""
generate_metrics_dashboard.py
─────────────────────────────
Creates ROC, PR, confusion‑matrix plots + a markdown report
for baseline and debiased predictions.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

# ───────────── Config ─────────────
BASELINE = "results/baseline_predictions.csv"
DEBIASED = "results/debiased_predictions.csv"
OUT_DIR = "results"
REPORT = os.path.join(OUT_DIR, "metrics_report.md")
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────── Helpers ─────────────
def load(path):
    df = pd.read_csv(path)

    # Normalize y_true
    if df["y_true"].dtype == object:
        df["y_true"] = df["y_true"].str.strip().map({"Approved": 1, "Denied": 0})
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype("Int64")

    # Normalize y_pred
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").astype("Int64")

    # Normalize y_prob
    if "y_prob" in df.columns:
        df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")

    return df["y_true"], df["y_pred"], df.get("y_prob")

def plot_roc(y, p, tag):
    if pd.isna(p).any():
        print(f"[⚠️] Skipping ROC plot for {tag} — contains NaNs in y_prob")
        return
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve · {tag}")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/roc_{tag.lower()}.png")
    plt.close()

def plot_pr(y, p, tag):
    if pd.isna(p).any():
        print(f"[⚠️] Skipping PR plot for {tag} — contains NaNs in y_prob")
        return
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve · {tag}")
    plt.savefig(f"{OUT_DIR}/pr_{tag.lower()}.png")
    plt.close()

def plot_cm(y, pred, tag):
    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(f"Confusion Matrix · {tag}")
    plt.savefig(f"{OUT_DIR}/cm_{tag.lower()}.png")
    plt.close()

def add_report(y, pred, tag):
    rep = classification_report(y, pred, digits=2)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(f"### {tag} Classification Report\n\n```\n{rep}\n```\n\n")

def run_for(tag, path):
    y, pred, prob = load(path)
    add_report(y, pred, tag)

    if prob is not None and prob.notna().all():
        plot_roc(y, prob, tag)
        plot_pr(y, prob, tag)
    else:
        print(f"[⚠️] Skipping ROC/PR for {tag} — y_prob is missing or contains NaNs")

    plot_cm(y, pred, tag)

# ───────────── Run Script ─────────────
if __name__ == "__main__":
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# 📈 Model Evaluation Report\n\n")
    for tag, path in [("Baseline", BASELINE), ("Debiased", DEBIASED)]:
        if os.path.exists(path) and os.path.getsize(path):
            run_for(tag, path)
    print(f"✅  Metrics + plots saved in {OUT_DIR}")
