$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1
if (-not (Get-Command mlflow -ErrorAction SilentlyContinue)) {
    throw "MLflow is not installed. Run: pip install -r requirements.txt"
}
mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
