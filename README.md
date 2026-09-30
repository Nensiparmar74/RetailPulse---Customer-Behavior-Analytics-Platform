# RetailPulse - Customer Behavior Analytics & Revenue Intelligence Platform

This is the complete internship project implementation aligned to the Code A Nova Project 4 brief. It includes the Python data-science pipeline, reproducible Jupyter notebooks, ML artifacts, SHAP explainability, SARIMA forecasting, cohort analysis, statistical tests, MLflow integration, the official Streamlit dashboard, an executive PDF report generator, and the previous React presentation frontend under `frontend/`.

## Official scope

The project covers EDA/data quality, cleaning, RFM + K-Means segmentation, churn prediction with held-out ROC-AUC, SHAP, monthly revenue forecasting with a 3-month horizon and confidence intervals, cohort retention, hypothesis testing, MLflow tracking, and a five-tab Streamlit dashboard. The official brief requires all notebooks to be reproducible, random seeds fixed at 42, K selection justified with elbow + silhouette, held-out churn AUC > 0.80, SHAP, labelled/source-annotated visualizations, `st.cache_data`, serialized model files, and business recommendations.

## Dataset

Canonical source:
`data/raw/online_retail_II.xlsx`

The workbook contains:
- `Year 2009-2010`
- `Year 2010-2011`

Do not concatenate duplicate copies of the same dataset. A CSV cache is optional and is generated locally by `scripts/prepare_raw_cache.py`.

## Project structure

```text
RetailPulse_Final/
├── data/raw/online_retail_II.xlsx
├── data/processed/
├── notebooks/
├── src/
├── models/
├── artifacts/
├── dashboard/app.py
├── reports/create_executive_report.py
├── scripts/prepare_raw_cache.py
├── run_pipeline.py
├── requirements.txt
└── frontend/
```

## Windows setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the complete pipeline

The safest order is:

```powershell
python scripts/prepare_raw_cache.py
python run_pipeline.py --source csv
python -m reports.create_executive_report
streamlit run dashboard/app.py
```

The cache is optional. Without it, `python run_pipeline.py --source xlsx` reads both workbook sheets directly.

## MLflow

After installing requirements, start the local UI in another terminal:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

The pipeline logs churn and forecast runs when MLflow is installed and the tracking server/runtime is available.

## Notebooks

Run top-to-bottom in this order:

1. `01_EDA.ipynb`
2. `02_Data_Cleaning.ipynb`
3. `03_Feature_Engineering.ipynb`
4. `04_Segmentation.ipynb`
5. `05_Churn_Prediction.ipynb`
6. `06_Revenue_Forecasting.ipynb`
7. `07_Cohort_Analysis.ipynb`

The notebooks call the same reusable `src/` functions used by `run_pipeline.py`.

## Verified run results from the uploaded workbook

The full workbook was converted to a temporary CSV cache solely to validate the analysis without changing the canonical source. The verified audit results are preserved by the generated artifacts after the complete pipeline is run. The final report also records source-file SHA-256 provenance.


## Latest validated run (2026-09-29)

Using the canonical `online_retail_II.xlsx` workbook (validated via an exact derived CSV cache for runtime speed), the pipeline produced:

- Raw rows: 1,067,371
- Exact duplicate rows: 34,335
- Primary positive-sales rows after documented cleaning: 779,425
- Tracked customers: 5,878
- Positive-sales revenue: GBP 17,374,804.27
- Orders: 36,969
- Average order value: GBP 469.98
- Repeat-customer rate: 72.4%
- Held-out churn ROC-AUC: 0.829
- Churn cross-validation mean AUC: 0.826
- Forecast model: SARIMA(0,1,1)x(1,0,0,12)
- Three-month historical backtest MAPE: 3.86%
- Selected segmentation K: 4, with silhouette limitation explicitly documented because K=2 has the highest silhouette score.

These are regenerated artifacts, not hardcoded UI values. Re-running the pipeline recomputes them from the canonical dataset.

MLflow note: this environment did not have the `mlflow` package installed, so the analytics pipeline wrote an explicit `mlflow_status.json` indicating that MLflow tracking must be enabled after installing the requirements.

## Important methodological notes

### Cleaning
The primary positive-sales dataset excludes missing Customer ID, cancellation-style invoices, non-positive quantity, non-positive price, and invalid dates. Cancellation/return information is retained for behavioral features and audit reporting.

### Segmentation
RFM features are log-transformed and standardized. Candidate K values 2-8 are scored. K=4 is selected because the inertia curve begins flattening after K=4 while silhouette remains positive, and the official brief requires four business profiles. The higher K=2 silhouette score is disclosed as a limitation rather than hidden.

### Churn
Churn uses an observation cutoff of 2011-06-30 and a three-month prediction window ending 2011-09-30. The target is 1 when a tracked customer has zero completed purchase orders in the prediction window. Features use observation-period data only.

### Forecasting
The final December 2011 observation is incomplete (the workbook ends on 2011-12-09), so the final training series uses the last complete month, November 2011. SARIMA candidates are selected by a 3-month historical backtest and the final model produces a 3-month forward forecast with a 95% confidence interval.

## Executive report

Run:

```powershell
python -m reports.create_executive_report
```

Output:
`reports/executive_report.pdf`

## Official dashboard

Run:

```powershell
streamlit run dashboard/app.py
```

The main views are:
- Overview
- Segments
- Churn
- Forecast
- Cohorts

Additional views cover Data Quality and Executive Insights.

## React frontend

The prior React presentation dashboard is preserved under `frontend/`. It is optional for the official internship deliverable; Streamlit is the official dashboard technology specified by the brief.

## Final quality check

Before submission verify:
- notebooks run top-to-bottom
- real dataset is used
- no fake metrics
- churn AUC is held-out
- SHAP plots exist
- forecast intervals exist
- cohort heatmap exists
- MLflow tracking works after installation
- Streamlit runs
- executive PDF is generated
- README is included
