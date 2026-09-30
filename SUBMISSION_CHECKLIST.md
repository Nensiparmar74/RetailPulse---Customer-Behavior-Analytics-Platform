# RetailPulse Submission Checklist

## Required by the official brief
- [x] EDA notebook + quality audit
- [x] Data cleaning pipeline
- [x] Cleaned and engineered CSV outputs
- [x] RFM features
- [x] K-Means with elbow + silhouette
- [x] Four business segment profiles
- [x] Churn label engineering
- [x] Random Forest churn classifier
- [x] Cross-validation
- [x] Held-out ROC-AUC
- [x] SHAP beeswarm/bar/waterfall
- [x] Monthly revenue series
- [x] SARIMA model selection/backtest
- [x] 3-month forward forecast
- [x] 95% confidence interval
- [x] Cohort retention matrix
- [x] Cohort heatmap
- [x] Cohort AOV
- [x] t-test / chi-square / ANOVA
- [x] MLflow integration code
- [ ] Live MLflow tracking run generated in a local environment
- [x] Streamlit dashboard with 5 required views
- [x] Plotly charts
- [x] st.cache_data used
- [x] Serialized models
- [x] Executive PDF report generator
- [x] README
- [x] requirements.txt
- [x] Streamlit deployment configuration
- [x] 5-minute demo walkthrough

## Final local verification

The environment used to build this package did not have MLflow or Streamlit installed, and network access was unavailable, so live external runtime/deployment status is intentionally not claimed.
Run:

```powershell
python run_pipeline.py --source xlsx
python -m reports.create_executive_report
streamlit run dashboard/app.py
```

Then verify that these files exist:

```text
data/processed/retail_cleaned.csv
data/processed/customer_features.csv
data/processed/churn_training_table.csv
models/segmentation_model.pkl
models/churn_model.pkl
models/forecast_model.pkl
artifacts/metrics/data_quality_raw.json
artifacts/metrics/segmentation_metrics.json
artifacts/metrics/churn_metrics.json
artifacts/metrics/forecast_metrics.json
artifacts/metrics/statistical_tests.json
artifacts/shap/shap_beeswarm.png
artifacts/shap/shap_bar.png
artifacts/shap/shap_waterfall_high_risk.png
artifacts/forecasts/revenue_forecast.csv
artifacts/tables/cohort_retention.csv
artifacts/tables/cohort_summary.csv
artifacts/retailpulse_artifacts.json
reports/executive_report.pdf
```

## Methodological notes
The churn target is operational and temporal. The forecast excludes the incomplete final month from final training. K=4 is selected for the required four-profile business view while transparently documenting that silhouette peaks at K=2.
