# RetailPulse Final Status

## Completed project deliverables

- Canonical Online Retail II workbook and raw provenance
- Reproducible Python source pipeline
- Data-quality audit and documented cleaning policy
- Missing-description imputation policy
- Customer-ID exclusion policy (not imputed because it would fabricate identity)
- IQR analytical outlier capping helper fields (`Quantity_Capped`, `Price_Capped`) while preserving original revenue fields
- 7 Jupyter notebooks
- 12 EDA visualizations
- Customer-level RFM, lifecycle and CLV approximation features
- K-Means segmentation, elbow, silhouette and four business profiles
- Temporal churn label
- Random Forest churn classifier with held-out ROC-AUC 0.829 and 3-fold CV mean AUC 0.826
- SHAP bar, beeswarm and waterfall outputs
- SARIMA forecasting with three-month horizon and 95% confidence intervals
- Cohort retention matrix, heatmap and cohort AOV
- Welch t-test, chi-square and ANOVA
- Serialized segmentation, churn and forecast models
- Streamlit dashboard with seven pages including the five required analytical views
- Executive PDF report
- README, requirements and deployment instructions
- Five-minute demo walkthrough video

## Environment-bound actions

These are code-complete but require an external/local environment:

1. Install MLflow and rerun the pipeline to create live MLflow tracking runs.
2. Install Streamlit and run `streamlit run dashboard/app.py`.
3. Push to GitHub and deploy the Streamlit app to Streamlit Cloud.

The current execution environment cannot claim these external runtime/account actions were completed because `mlflow` and `streamlit` were not installed and network access is unavailable.
