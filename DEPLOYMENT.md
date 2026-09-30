# RetailPulse Deployment & Runtime Guide

## Local setup

Use Python 3.12 as required by the project brief.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Regenerate the full analytics pipeline

From the project root:

```powershell
python run_pipeline.py --source xlsx
python -m reports.create_executive_report
```

The pipeline regenerates cleaned data, customer features, segmentation, churn/SHAP, forecasting, cohorts, statistics and dashboard artifacts.

## MLflow

After installing `mlflow` from `requirements.txt`, run the pipeline again. The pipeline will create two runs under the `RetailPulse` experiment when MLflow is importable.

Start the tracking UI locally:

```powershell
mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`.

If your installed MLflow uses a different command or storage backend, follow its local installation documentation.

## Streamlit

```powershell
streamlit run dashboard/app.py
```

The dashboard reads `artifacts/retailpulse_artifacts.json` and the CSV/PNG artifacts under `artifacts/`.

## Streamlit Cloud

1. Push the complete repository to GitHub.
2. In Streamlit Cloud, create a new app from the repository.
3. Select the repository and branch.
4. Set the main file path to:
   `dashboard/app.py`
5. Deploy.
6. Confirm the app loads the committed `artifacts/` files.
7. Confirm the five required views work: Overview, Segments, Churn, Forecast, Cohorts.

No personal Windows paths are required.

## Important submission note

The final Streamlit Cloud deployment and MLflow tracking page are account/environment actions. The code and configuration are included in this repository, but a live external deployment cannot be created without access to the destination account/repository.
