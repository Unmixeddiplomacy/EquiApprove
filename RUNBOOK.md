# EquiApprove / FairLoans — Runbook (from scratch)

This file is a copy/paste runbook to set up the environment, train models, generate predictions + reports, and launch the dashboard.

## 0) Prerequisites

- Windows: **Python 3.11–3.13 recommended**. (Python 3.14 may cause `scipy` to build from source on Windows and fail.)
- `data/loan_dataset.csv` must exist.
- Optional: `data/test.csv` (only needed to create `submission.csv`).

---

## 1) Setup (Windows CMD)

From the project folder:

```bat
cd C:\Users\sinha\Downloads\EquiApprove

:: confirm available Pythons
py -0p

:: clean old venv if present
if exist .venv rmdir /s /q .venv

:: create venv using Python 3.11
py -3.11 -m venv .venv

:: activate venv
.venv\Scripts\activate

:: keep pip < 26 (avoids a known Windows launcher rollback issue)
python -m pip install -U "pip<26" setuptools wheel

:: install project deps
python -m pip install -r requirements.txt
```

Quick sanity check:

```bat
python -c "import sys; print(sys.version)"
python -c "import numpy, scipy, sklearn, xgboost, fairlearn, aif360, shap, streamlit; print('ok')"
```

---

## 2) (Optional) Clean outputs before re-running

If you previously ran the pipeline and want a fresh run:

```bat
if exist results rmdir /s /q results
if exist submission.csv del /q submission.csv
mkdir results
```

---

## 3) Train baseline model + save artifacts

This trains the baseline model and writes:
- `results/model_xgb.pkl`
- `results/label_encoders.pkl`
- `results/baseline_predictions.csv`

```bat
python run_pipeline.py
```

If you only need to regenerate the baseline predictions CSV (model already exists):

```bat
python generate_predictions.py
```

---

## 4) Train debiased model

This trains the fairness-mitigated model and writes:
- `results/model_debiased_xgb.pkl`
- `results/debiased_model_features.pkl`

Default behavior now includes privacy protections:
- Differentially private Logistic Regression (`privacy-mode=dp`)
- Fairness mitigation with Demographic Parity via `ExponentiatedGradient`
- Strict data minimization for model inputs (drops encoded `gender/race/zip` features)

```bat
python src\train_debiased_model.py --privacy-mode dp --epsilon 8.0 --minimization-mode strict
```

If you need a non-private ablation run for comparison:

```bat
python src\train_debiased_model.py --privacy-mode standard --minimization-mode strict
```

---

## 5) Generate debiased predictions

This writes:
- `results/debiased_predictions.csv`

Default behavior now exports only `y_true` and `y_pred` (no `y_prob`) to reduce leakage risk.

```bat
python scripts\generate_debiased_predictions.py
```

If you explicitly need probabilities for research/plots, opt in:

```bat
python scripts\generate_debiased_predictions.py --include-probabilities
```

---

## 6) Generate SHAP explainer (baseline)

This writes:
- `results/shap_explainer.pkl`

```bat
python scripts\generate_shap_explainer.py
```

Notes:
- If you see scikit-learn/XGBoost "inconsistent version" warnings when loading old `.pkl` files, delete `results/` and retrain (Step 2 + Step 3).

---

## 7) Generate metrics report + plots

This writes a markdown report and plots into `results/`:
- `results/metrics_report.md`
- `results/roc_*.png`, `results/pr_*.png`, `results/cm_*.png`

```bat
python generate_metrics_dashboard.py
```

---

## 8) (Optional) Export fairness metric JSON reports

Baseline fairness report by `gender`:

```bat
python src\fairness_utils.py --input results\baseline_predictions.csv --output results\fairness_baseline_gender.json --sensitive gender
```

Debiased fairness report by `gender`:

```bat
python src\fairness_utils.py --input results\debiased_predictions.csv --output results\fairness_debiased_gender.json --sensitive gender
```

You can replace `gender` with `race` or `region` if those columns exist in the predictions CSV (the dashboard can merge them from `data/loan_dataset.csv`).

---

## 9) Launch the Streamlit dashboard

```bat
streamlit run dashboard.py
```

In the dashboard sidebar, upload:
- `results/baseline_predictions.csv`
- `results/debiased_predictions.csv` (optional)

---

## 10) Generate submission file

This writes:
- `submission.csv`

Requirements:
- `data/test.csv` must exist and include an `id` column
- baseline artifacts must exist (`results/model_xgb.pkl`, `results/label_encoders.pkl`)

```bat
python generate_submission.py
```

---

## One-shot pipeline (macOS/Linux/WSL)

If you’re on macOS/Linux/WSL and already installed requirements:

```bash
bash run_all.sh
```

---

## Troubleshooting

### SciPy build errors on Windows

If you see errors about Meson/OpenBLAS/pkg-config while installing, you’re likely on Python 3.14.
- Use Python 3.11–3.13 and recreate the venv (Step 1).

### InconsistentVersionWarning while unpickling

That means you’re loading a model saved with a different library version.
- Fix: delete `results/` and rerun Steps 3–7.

### DP training is unstable or too noisy

If debiased DP training underperforms heavily, tune privacy/training knobs:

```bat
python src\train_debiased_model.py --privacy-mode dp --epsilon 12.0 --max-iter 500 --minimization-mode strict
```

Guideline:
- Higher `--epsilon` usually improves utility but weakens privacy.
- Higher `--max-iter` can improve convergence for the DP optimizer.

---

## 11) Privacy Governance Checklist (Responsible AI)

Use this checklist before sharing artifacts outside the project team.

1. Access control
- Keep raw data (`data/loan_dataset.csv`) restricted to approved team members only.
- Share only generated artifacts in `results/` unless raw data is explicitly required.

2. Data minimization
- Use debiased training with `--minimization-mode strict` in production-style runs.
- Do not add direct identifiers (name, email, phone, address, SSN) to model features.

3. Output minimization
- Keep default prediction export (`y_true`, `y_pred`) and avoid `--include-probabilities` unless required.

4. Retention
- Retain raw data only for the minimum period required for reproducibility/audit.
- Remove stale model artifacts and old prediction exports after reporting cycles.

5. Auditability
- Log the exact command used for training, including `--privacy-mode`, `--epsilon`, and `--minimization-mode`.
