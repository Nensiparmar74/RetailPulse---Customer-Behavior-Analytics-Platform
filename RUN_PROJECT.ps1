$ErrorActionPreference = "Stop"

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py --source xlsx
python -m reports.create_executive_report
Write-Host "RetailPulse analytics and report regenerated."
Write-Host "Start Streamlit with: streamlit run dashboard/app.py"
Write-Host "Start MLflow with: mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000"
