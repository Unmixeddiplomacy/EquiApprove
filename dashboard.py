# dashboard.py · EquiApprove Interactive Fairness Dashboard

import os, joblib, shap, json
import pandas as pd, numpy as np, streamlit as st
import seaborn as sns, matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
)
from fairlearn.metrics import (
    MetricFrame, selection_rate, false_positive_rate, false_negative_rate,
    demographic_parity_difference, equal_opportunity_difference,
    false_negative_rate_difference, equalized_odds_difference
)

# ───────────── Setup ─────────────
st.set_page_config(page_title="EquiApprove Dashboard", layout="wide")
st.title("📊 EquiApprove – Bias & Performance Dashboard")

# ───────────── Sidebar ─────────────
st.sidebar.header("📂 Upload prediction CSVs")
base_csv = st.sidebar.file_uploader("Baseline predictions", type="csv")
deb_csv  = st.sidebar.file_uploader("Debiased predictions", type="csv")
st.sidebar.markdown("---")
enable_shap = st.sidebar.checkbox("🔍  Show SHAP explainability", value=False)
enable_sim  = st.sidebar.checkbox("🧪  Enable input simulator",   value=True)

# ───────────── Constants ─────────────
REQ = {"y_true", "y_pred"}

# ───────────── Load raw feature data ─────────────
DATA_PATH = Path("data/loan_dataset.csv")
if not DATA_PATH.exists():
    st.error("`data/loan_dataset.csv` not found.")
    st.stop()
raw_df = pd.read_csv(DATA_PATH)
raw_df.columns = raw_df.columns.str.strip().str.lower().str.replace(" ", "_")

# ───────────── CSV loader ─────────────
def load_preds(file, raw_df):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Ensure prediction columns
    missing = REQ - set(df.columns)
    if missing:
        st.error(f"❌ CSV missing column(s): {missing}")
        st.stop()

    # Ensure numeric
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    if "y_prob" in df.columns:
        df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")
    df.dropna(subset=list(REQ), inplace=True)

    # Merge sensitive features
    for col in ["gender", "race", "region"]:
        if col in raw_df.columns and col not in df.columns:
            df[col] = raw_df[col]

    return df

# ───────────── Tabs ─────────────
tab_over, tab_bias, tab_shap, tab_sim, tab_submit = st.tabs(
    ["🏠 Overview", "⚖️ Fairness", "🔍 SHAP", "🧪 Simulator", "📤 Generate Submission"]
)


# ═════════════ 🏠 OVERVIEW ═════════════
with tab_over:
    st.subheader("How to use this dashboard")
    st.markdown("""
1. Upload **baseline** and (optionally) **debiased** prediction CSVs ⬅️  
2. Pick a sensitive feature to inspect fairness metrics  
3. Toggle SHAP for feature‑level insights  
4. Use the Simulator to test a custom applicant
""")
    st.info("Upload files to begin!")

# ═════════════ ⚖️ FAIRNESS ═════════════
# ═════════════ ⚖️ FAIRNESS ═════════════
with tab_bias:
    st.header("Group‑wise Fairness Metrics")

    if base_csv is None:
        st.warning("Upload at least a *Baseline* CSV to begin.")
    else:
        df_base = load_preds(base_csv, raw_df)
        df_deb = load_preds(deb_csv, raw_df) if deb_csv else None

        sens_avail = [c for c in ["gender", "race", "region"] if c in df_base.columns]
        if not sens_avail:
            st.error("No sensitive features (`gender`, `race`, `region`) found.")
            st.stop()

        sens = st.radio("📌 Select Sensitive Feature", sens_avail, horizontal=True)

        def compute_metric_frame(df, sens):
            df = df.dropna(subset=[sens, "y_true", "y_pred"])
            if df.empty:
                return None
            return MetricFrame(
                metrics={
                    "Accuracy": accuracy_score,
                    "Selection Rate": selection_rate,
                    "FPR": false_positive_rate,
                    "FNR": false_negative_rate
                },
                y_true=df["y_true"],
                y_pred=df["y_pred"],
                sensitive_features=df[sens]
            )

        def compute_overall_fairness(df, sens):
            return {
                "Demographic Parity Diff": float(demographic_parity_difference(
                    y_true=df["y_true"], y_pred=df["y_pred"], sensitive_features=df[sens])),
                "Equal Opportunity Diff": float(equal_opportunity_difference(
                    y_true=df["y_true"], y_pred=df["y_pred"], sensitive_features=df[sens])),
                "FNR Difference": float(false_negative_rate_difference(
                    y_true=df["y_true"], y_pred=df["y_pred"], sensitive_features=df[sens])),
                "Equalized Odds Diff": float(equalized_odds_difference(
                    y_true=df["y_true"], y_pred=df["y_pred"], sensitive_features=df[sens]))
            }

        col1, col2 = st.columns(2)

        # ───── Baseline ─────
        with col1:
            st.markdown("### Baseline")
            mf_b = compute_metric_frame(df_base, sens)
            if mf_b is not None:
                st.dataframe(mf_b.by_group)
                fig, ax = plt.subplots(figsize=(6, 3))
                mf_b.by_group.T.plot(kind="bar", ax=ax)
                ax.set_title(f"Baseline by {sens}")
                ax.set_ylabel("Metric Value")
                st.pyplot(fig)
                st.markdown("#### 📋 Overall Baseline Metrics")
                st.write(compute_overall_fairness(df_base, sens))
            else:
                st.warning("No valid baseline rows.")

        # ───── Debiased ─────
        with col2:
            st.markdown("### Debiased")
            mf_d = compute_metric_frame(df_deb, sens) if df_deb is not None else None
            if mf_d is not None:
                st.dataframe(mf_d.by_group)
                fig, ax = plt.subplots(figsize=(6, 3))
                mf_d.by_group.T.plot(kind="bar", ax=ax, color="salmon")
                ax.set_title(f"Debiased by {sens}")
                ax.set_ylabel("Metric Value")
                st.pyplot(fig)
                st.markdown("#### 📋 Overall Debiased Metrics")
                st.write(compute_overall_fairness(df_deb, sens))
            else:
                st.info("Upload debiased CSV to compare.")

        # 📌 Summary Insights
        st.markdown("### 📌 Summary Insights")
        if mf_b is not None and mf_d is not None:
            try:
                sr_b = mf_b.by_group["Selection Rate"]
                sr_d = mf_d.by_group["Selection Rate"]

                # 🔐 Fix mixed-type indices (e.g., int vs str)
                sr_b.index = sr_b.index.astype(str)
                sr_d.index = sr_d.index.astype(str)

                sr_b = pd.to_numeric(sr_b, errors="coerce")
                sr_d = pd.to_numeric(sr_d, errors="coerce")

                acc_b = accuracy_score(df_base["y_true"], df_base["y_pred"])
                acc_d = accuracy_score(df_deb["y_true"], df_deb["y_pred"])

                dp_b = demographic_parity_difference(
                    y_true=df_base["y_true"], y_pred=df_base["y_pred"], sensitive_features=df_base[sens])
                dp_d = demographic_parity_difference(
                    y_true=df_deb["y_true"], y_pred=df_deb["y_pred"], sensitive_features=df_deb[sens])

                comparison_df = pd.DataFrame({
                    "Baseline Approval Rate": sr_b,
                    "Debiased Approval Rate": sr_d,
                    "Improvement": sr_d - sr_b
                })
                st.dataframe(comparison_df.style.format("{:.2f}"))

                sr_b_clean = sr_b.dropna()
                if not sr_b_clean.empty:
                    majority_group = sr_b_clean.idxmax()
                    minority_group = sr_b_clean.idxmin()

                    delta_acc = acc_d - acc_b
                    delta_dp = dp_b - dp_d

                    insight = f"""
                    After applying **ExponentiatedGradient**, the approval rate for **{minority_group}** applicants improved from **{sr_b[minority_group]:.0%}** to **{sr_d[minority_group]:.0%}**.
                    The demographic parity gap decreased by **{delta_dp:.2f}**, while overall accuracy changed from **{acc_b:.1%}** to **{acc_d:.1%}** ({delta_acc:+.1%}).
                    """
                    st.success(insight.strip())
                else:
                    st.warning("⚠️ No valid numeric selection rates to generate insights.")
            except Exception as e:
                st.warning(f"⚠️ Could not generate summary insights: {e}")

        # 📤 Export Fairness Report
        st.markdown("### 📤 Download Fairness Report")
        fairness_report = {
            "baseline": compute_overall_fairness(df_base, sens),
            "debiased": compute_overall_fairness(df_deb, sens) if df_deb is not None else {}
        }
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(fairness_report, indent=4),
            file_name="fairness_report.json",
            mime="application/json"
        )


        # ─── Confusion + ROC/PR ───
        st.markdown("---")
        st.subheader("Performance Curves")
        perf_cols = st.columns(2)

        for name, df_show, slot in [("Baseline", df_base, 0), ("Debiased", df_deb, 1)]:
            if df_show is None:
                continue
            with perf_cols[slot]:
                st.markdown(f"### {name}")
                cm = confusion_matrix(df_show["y_true"], df_show["y_pred"])
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_title(f"{name} · Confusion Matrix")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

                if "y_prob" in df_show.columns and df_show["y_prob"].notna().all():
                    fpr, tpr, _ = roc_curve(df_show["y_true"], df_show["y_prob"])
                    prec, rec, _ = precision_recall_curve(df_show["y_true"], df_show["y_prob"])

                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label="ROC")
                    ax_roc.plot([0, 1], [0, 1], "--")
                    ax_roc.set_title(f"{name} · ROC")
                    ax_roc.set_xlabel("FPR")
                    ax_roc.set_ylabel("TPR")
                    st.pyplot(fig_roc)

                    fig_pr, ax_pr = plt.subplots()
                    ax_pr.plot(rec, prec)
                    ax_pr.set_title(f"{name} · Precision-Recall")
                    ax_pr.set_xlabel("Recall")
                    ax_pr.set_ylabel("Precision")
                    st.pyplot(fig_pr)
                else:
                    st.warning(f"{name}: `y_prob` missing — ROC/PR skipped.")

# ═════════════ 🔍 SHAP ═════════════
with tab_shap:
    st.header("Global Feature Importance (SHAP)")
    if not enable_shap:
        st.info("Enable SHAP in sidebar.")
    else:
        raw_features = raw_df.drop(columns=["id", "loan_approved"], errors="ignore")
        df_enc_all = pd.get_dummies(raw_features, drop_first=False)

        if raw_features.empty:
            st.error("No rows available for SHAP explainability.")
        else:
            sample_n = min(200, len(raw_features))
            sample_idx = raw_features.sample(n=sample_n, random_state=42).index

            col_base, col_deb = st.columns(2)

            # ───── Baseline SHAP ─────
            with col_base:
                st.subheader("Baseline SHAP")
                try:
                    model_base = joblib.load("results/model_xgb.pkl")
                    if hasattr(model_base, "get_booster"):
                        base_feats = model_base.get_booster().feature_names
                    elif hasattr(model_base, "feature_names_in_"):
                        base_feats = list(model_base.feature_names_in_)
                    else:
                        raise ValueError("Could not determine baseline model feature names.")

                    # Build baseline feature matrix in the same style used during training.
                    if set(base_feats).issubset(set(raw_features.columns)):
                        X_base_full = raw_features.copy()
                        enc_path = Path("results/label_encoders.pkl")
                        encoders = joblib.load(enc_path) if enc_path.exists() else {}

                        for col in base_feats:
                            if col in encoders:
                                cls = encoders[col].classes_
                                mapping = {str(v): i for i, v in enumerate(cls)}
                                X_base_full[col] = X_base_full[col].astype(str).map(mapping).fillna(-1)
                            elif X_base_full[col].dtype == object:
                                X_base_full[col] = pd.to_numeric(X_base_full[col], errors="coerce").fillna(-1)

                        X_base_full = X_base_full.reindex(columns=base_feats, fill_value=0)
                    else:
                        X_base_full = pd.get_dummies(raw_features, drop_first=False).reindex(columns=base_feats, fill_value=0)

                    X_base = X_base_full.loc[sample_idx]
                    explainer_base = shap.Explainer(model_base)
                    shap_base = explainer_base(X_base)

                    fig_base = plt.figure(figsize=(8, 6))
                    shap.summary_plot(
                        shap_base,
                        X_base,
                        max_display=X_base.shape[1],
                        show=False,
                    )
                    st.pyplot(fig_base)
                    st.caption(f"Showing all {X_base.shape[1]} baseline model features.")
                except Exception as e:
                    st.error(f"Baseline SHAP failed: {e}")

            # ───── Debiased SHAP ─────
            with col_deb:
                st.subheader("Debiased SHAP")
                try:
                    deb_model_path = Path("results/model_debiased_xgb.pkl")
                    deb_feats_path = Path("results/debiased_model_features.pkl")

                    if not deb_model_path.exists():
                        raise FileNotFoundError("Debiased model file not found at results/model_debiased_xgb.pkl")

                    model_deb = joblib.load(deb_model_path)

                    if deb_feats_path.exists():
                        deb_feats = joblib.load(deb_feats_path)
                    elif hasattr(model_deb, "feature_names_in_"):
                        deb_feats = list(model_deb.feature_names_in_)
                    else:
                        raise ValueError("Debiased feature list not found. Please generate results/debiased_model_features.pkl")

                    X_deb_full = df_enc_all.reindex(columns=deb_feats, fill_value=0)
                    X_deb_sample = X_deb_full.loc[sample_idx]
                    bg_n = min(60, len(X_deb_sample))
                    X_bg = X_deb_sample.sample(n=bg_n, random_state=42)

                    def predict_debiased(X):
                        if isinstance(X, np.ndarray):
                            X = pd.DataFrame(X, columns=deb_feats)
                        else:
                            X = pd.DataFrame(X).reindex(columns=deb_feats, fill_value=0)

                        if hasattr(model_deb, "_pmf_predict"):
                            pmf = model_deb._pmf_predict(X)
                            if isinstance(pmf, np.ndarray) and pmf.ndim == 2 and pmf.shape[1] > 1:
                                return pmf[:, 1]

                        if hasattr(model_deb, "predict_proba"):
                            return model_deb.predict_proba(X)[:, 1]

                        return np.asarray(model_deb.predict(X), dtype=float)

                    explainer_deb = shap.Explainer(predict_debiased, X_bg)
                    shap_deb = explainer_deb(X_deb_sample)

                    fig_deb = plt.figure(figsize=(8, 6))
                    shap.summary_plot(
                        shap_deb,
                        X_deb_sample,
                        max_display=X_deb_sample.shape[1],
                        show=False,
                    )
                    st.pyplot(fig_deb)
                    st.caption(f"Showing all {X_deb_sample.shape[1]} debiased model features.")
                except Exception as e:
                    st.error(f"Debiased SHAP failed: {e}")

# ═════════════ 🧪 SIMULATOR ═════════════
with tab_sim:
    st.header("Loan‑Approval Simulator")
    if not enable_sim:
        st.info("Enable simulator in sidebar.")
    else:
        left, right = st.columns(2)
        with left:
            age = st.slider("Age", 18, 70, 30)
            income = st.number_input("Income", 10000, 200000, 50000, 1000)
            credit = st.slider("Credit Score", 300, 850, 650)
        with right:
            loan_amt = st.number_input("Loan Amount", 1000, 100000, 15000, 500)
            gender = st.radio("Gender", ["Male", "Female"])
            race = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])
            region = st.selectbox("Region", ["Urban", "Rural", "Suburban"])

        user = {
            "age": age, "income": income, "loan_amount": loan_amt,
            "credit_score": credit, f"gender_{gender}": 1,
            f"race_{race}": 1, f"region_{region}": 1
        }
        X_user = pd.DataFrame([user])

        # ───── Baseline model ─────
        try:
            model_base = joblib.load("results/model_xgb.pkl")
            base_feats = model_base.get_booster().feature_names
            for col in base_feats:
                if col not in X_user:
                    X_user[col] = 0
            X_user = X_user[base_feats]
            prob = model_base.predict_proba(X_user)[0, 1]
            st.success(f"Baseline → {'✅ Approved' if prob >= 0.5 else '❌ Rejected'}  (p={prob:.2f})")
        except Exception as e:
            st.error(f"Baseline model error: {e}")

        # ───── Debiased model ─────
        deb_path = Path("results/model_debiased_xgb.pkl")
        feats_path = Path("results/debiased_model_features.pkl")

        if deb_path.exists():
            try:
                model_deb = joblib.load(deb_path)
                if hasattr(model_deb, "get_booster"):
                    deb_feats = model_deb.get_booster().feature_names
                elif hasattr(model_deb, "feature_names_in_"):
                    deb_feats = list(model_deb.feature_names_in_)
                elif feats_path.exists():
                    deb_feats = joblib.load(feats_path)
                else:
                    raise ValueError("No feature list found for debiased model.")

                for col in deb_feats:
                    if col not in X_user:
                        X_user[col] = 0
                X_user_deb = X_user[deb_feats]

                if hasattr(model_deb, "predict_proba"):
                    prob_deb = model_deb.predict_proba(X_user_deb)[0, 1]
                    verdict = '✅ Approved' if prob_deb >= 0.5 else '❌ Rejected'
                    st.info(f"Debiased → {verdict}  (p={prob_deb:.2f})")
                else:
                    pred = model_deb.predict(X_user_deb)[0]
                    verdict = '✅ Approved' if pred == 1 else '❌ Rejected'
                    st.info(f"Debiased → {verdict}  (probability unavailable)")
            except Exception as e:
                st.error(f"Debiased model error: {e}")
        else:
            st.info("Debiased model not found.")

# ═════════════ 📤 GENERATE SUBMISSION ═════════════
with tab_submit:
    st.header("📤 Generate Submission File")
    try:
        model = joblib.load("results/model_xgb.pkl")
        encoders = joblib.load("results/label_encoders.pkl")
    except Exception as e:
        st.error(f"❌ Could not load model or encoders: {e}")
        st.stop()

    test_path = Path("data/test.csv")
    if not test_path.exists():
        st.error("❌ `data/test.csv` not found.")
        st.stop()

    test_df = pd.read_csv(test_path)
    test_df.columns = test_df.columns.str.strip().str.lower().str.replace(" ", "_")

    for col in test_df.columns:
        if col in encoders:
            test_df[col] = encoders[col].transform(test_df[col].astype(str))
        elif test_df[col].dtype == object:
            st.warning(f"⚠️ Unencoded column '{col}' — filling with -1")
            test_df[col] = -1

    X_test = test_df.copy()
    id_col = X_test["id"] if "id" in X_test.columns else np.arange(len(X_test))

    try:
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        submission = pd.DataFrame({
            "ID": id_col,
            "LoanApproved": preds
        })
        st.dataframe(submission.head())

        csv = submission.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Submission CSV",
            data=csv,
            file_name="submission.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")


