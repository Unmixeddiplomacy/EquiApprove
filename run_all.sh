#!/bin/bash
echo "▶️ Training baseline model..."
python run_pipeline.py

echo "▶️ Training debiased model..."
python src/train_debiased_model.py

echo "▶️ Generating debiased predictions..."
python scripts/generate_debiased_predictions.py

echo "▶️ Generating SHAP explainer..."
python scripts/generate_shap_explainer.py

echo "▶️ Generating metrics dashboard..."
python generate_metrics_dashboard.py

echo "✅ All steps complete. Launch the dashboard using: streamlit run dashboard.py"
