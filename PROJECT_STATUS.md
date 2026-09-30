# RetailPulse Project Status

## Verified in this environment

- Canonical workbook loaded and validated: `data/raw/online_retail_II.xlsx`
- Raw transaction rows: 1,067,371
- Exact duplicate rows: 34,335
- Primary completed positive-sales rows: 779,425
- Tracked customers: 5,878
- Positive-sales revenue: GBP 17,374,804.27
- Orders: 36,969
- Average order value: GBP 469.98
- Repeat customer rate: 72.4%
- Churn held-out ROC-AUC: 0.829
- Churn 3-fold CV mean AUC: 0.826
- Forecast model: SARIMA(0,1,1)x(1,0,0,12)
- Forecast backtest MAPE: 3.86%
- Forecast horizon: 3 months
- Segmentation K: 4 with elbow + silhouette trade-off documented
- EDA visualizations: 12
- SHAP beeswarm, bar and waterfall: generated
- Cohort retention matrix and heatmap: generated
- Statistical tests: Welch t-test, chi-square, one-way ANOVA
- Executive PDF report: generated and rendered for visual QA
- Streamlit dashboard source: complete
- Seven reproducible Jupyter notebooks: present
- Serialized segmentation, churn and forecast models: present

## Runtime prerequisite not executed here

MLflow is included in `requirements.txt` and fully wired through `src/mlflow_tracking.py`, but the current execution environment did not have the `mlflow` package installed. The pipeline therefore recorded an explicit `mlflow_status.json` instead of pretending a tracking run existed.

After installing requirements locally, run the pipeline again to create actual MLflow runs.

## Official brief alignment

The project includes the major requirements from the Code A Nova Project 4 brief: EDA/data quality, cleaning, feature engineering, RFM + K-Means, churn prediction with held-out AUC, SHAP, SARIMA forecasting, three-month forecast with confidence intervals, cohort retention, statistical testing, MLflow integration, Streamlit dashboard, model artifacts, report generation, README and demo script.
