from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
EDA_FIGURES = FIGURES / "eda"
CHURN_FIGURES = FIGURES / "churn"
FORECAST_FIGURES = FIGURES / "forecast"
COHORT_FIGURES = FIGURES / "cohort"
SEGMENT_FIGURES = FIGURES / "segmentation"
METRICS = ARTIFACTS / "metrics"
TABLES = ARTIFACTS / "tables"
SHAP_DIR = ARTIFACTS / "shap"
FORECASTS = ARTIFACTS / "forecasts"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
RANDOM_STATE = 42
CURRENCY = "£"
DATA_XLSX = RAW_DIR / "online_retail_II.xlsx"
DATA_CACHE_CSV = RAW_DIR / "combined_transactions.csv"

for path in [PROCESSED_DIR, EDA_FIGURES, CHURN_FIGURES, FORECAST_FIGURES, COHORT_FIGURES, SEGMENT_FIGURES, METRICS, TABLES, SHAP_DIR, FORECASTS, MODELS, REPORTS]:
    path.mkdir(parents=True, exist_ok=True)
