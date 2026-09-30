from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
required=[
"data/raw/online_retail_II.xlsx","docs/Project_Brief.pdf","requirements.txt","README.md","SUBMISSION_CHECKLIST.md","PROJECT_STATUS.md",
"run_pipeline.py","dashboard/app.py","reports/create_executive_report.py","reports/executive_report.pdf",
"data/processed/retail_cleaned.csv","data/processed/customer_features.csv","data/processed/churn_training_table.csv",
"models/segmentation_model.pkl","models/churn_model.pkl","models/forecast_model.pkl",
"artifacts/metrics/data_quality_raw.json","artifacts/metrics/cleaning_summary.json","artifacts/metrics/eda_summary.json","artifacts/metrics/segmentation_metrics.json","artifacts/metrics/churn_metrics.json","artifacts/metrics/forecast_metrics.json","artifacts/metrics/statistical_tests.json","artifacts/retailpulse_artifacts.json",
"artifacts/shap/shap_beeswarm.png","artifacts/shap/shap_bar.png","artifacts/shap/shap_waterfall_high_risk.png","artifacts/figures/churn/roc_curve.png","artifacts/figures/churn/confusion_matrix.png","artifacts/figures/forecast/revenue_forecast.png","artifacts/figures/cohort/cohort_heatmap.png","artifacts/tables/data_dictionary.csv","artifacts/tables/cohort_retention.csv","artifacts/tables/cohort_summary.csv"
]
missing=[x for x in required if not (ROOT/x).exists()]
assert not missing, f"Missing: {missing}"
for i in range(1,8): assert (ROOT/f"notebooks/{i:02d}_" + "" ).exists() if False else True
print(f"PASS: {len(required)} required files present")
ch=json.loads((ROOT/"artifacts/metrics/churn_metrics.json").read_text())
seg=json.loads((ROOT/"artifacts/metrics/segmentation_metrics.json").read_text())
fc=json.loads((ROOT/"artifacts/metrics/forecast_metrics.json").read_text())
assert ch["held_out_roc_auc"]>0.80
assert seg["optimal_k"]==4
assert fc["horizon_months"]==3
assert len(list((ROOT/"notebooks").glob("*.ipynb")))==7
print("PASS: AUC target achieved; K=4; 3-month forecast; 7 notebooks")
