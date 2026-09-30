from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from src.data_loader import load_data, audit_dataset
from src.cleaning import clean_transactions
from src.eda import run_eda
from src.features import build_customer_features
from src.segmentation import build_segmentation
from src.churn import build_churn_dataset, train_churn_model
from src.forecasting import monthly_revenue, fit_final_forecast
from src.cohorts import build_cohort_analysis
from src.statistics import run_statistical_tests
from src.mlflow_tracking import log_experiments
from src.pipeline_artifacts import save_json, build_dashboard_artifact


def sha256(path: Path) -> str:
    """Return SHA-256 for dataset provenance."""
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main(source: str="auto") -> None:
    print("[1/10] Loading data...")
    raw=load_data(source); audit=audit_dataset(raw); save_json(audit,METRICS/"data_quality_raw.json")
    print(json.dumps({"raw_rows":audit["raw_row_count"],"duplicates":audit["duplicate_rows"]},indent=2))

    print("[2/10] Cleaning...")
    sales,deduped,cleaning=clean_transactions(raw); sales.to_csv(PROCESSED_DIR/"retail_cleaned.csv",index=False); save_json(cleaning,METRICS/"cleaning_summary.json")

    print("[3/10] EDA...")
    eda=run_eda(sales,deduped,EDA_FIGURES); save_json(eda,METRICS/"eda_summary.json")

    print("[4/10] Customer features / RFM...")
    customers=build_customer_features(sales,deduped); customers.to_csv(PROCESSED_DIR/"customer_features.csv",index=False)
    data_dictionary=pd.DataFrame([
        {"column":"Invoice","description":"Invoice/order identifier; values beginning with C indicate cancellation-style invoices."},
        {"column":"StockCode","description":"Product/stock identifier."},
        {"column":"Description","description":"Product description."},
        {"column":"Quantity","description":"Units on the transaction line; positive values are completed purchases in the primary sales table."},
        {"column":"InvoiceDate","description":"Transaction timestamp used for temporal analysis, RFM, churn windows and cohorts."},
        {"column":"Price","description":"Unit price in GBP."},
        {"column":"Customer ID","description":"Customer identifier; missing values are excluded from longitudinal customer analysis."},
        {"column":"Country","description":"Customer country."},
        {"column":"Revenue","description":"Line revenue = Quantity * Price for primary positive-sales analysis."},
    ]); data_dictionary.to_csv(TABLES/"data_dictionary.csv",index=False)
    print("customers",len(customers))

    print("[5/10] Segmentation...")
    segmented,profiles,seg_payload,scaler,seg_model=build_segmentation(customers,MODELS)
    segmented.to_csv(TABLES/"customer_segments.csv",index=False); profiles.to_csv(TABLES/"segment_profiles.csv",index=False); save_json(seg_payload,METRICS/"segmentation_metrics.json")
    # Segmentation plots
    plt.figure(figsize=(8,5)); plt.plot([x["k"] for x in seg_payload["elbow"]],[x["inertia"] for x in seg_payload["elbow"]],marker="o"); plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow Method for K-Means"); plt.figtext(0.99,0.01,"Source: RetailPulse customer RFM features",ha="right",fontsize=7,color="gray"); plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig(SEGMENT_FIGURES/"elbow_method.png",dpi=160,bbox_inches="tight"); plt.close()
    plt.figure(figsize=(8,5)); plt.plot([x["k"] for x in seg_payload["silhouette"]],[x["silhouette"] for x in seg_payload["silhouette"]],marker="o"); plt.xlabel("K"); plt.ylabel("Silhouette Score"); plt.title("Silhouette Score by Cluster Count"); plt.figtext(0.99,0.01,"Source: RetailPulse customer RFM features",ha="right",fontsize=7,color="gray"); plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig(SEGMENT_FIGURES/"silhouette_scores.png",dpi=160,bbox_inches="tight"); plt.close()

    print("[6/10] Churn model...")
    churn_df,churn_meta=build_churn_dataset(sales,deduped); churn_df.to_csv(PROCESSED_DIR/"churn_training_table.csv",index=False); save_json(churn_meta,METRICS/"churn_definition.json")
    churn_metrics,churn_model,pred_table=train_churn_model(churn_df,MODELS,SHAP_DIR); save_json(churn_metrics,METRICS/"churn_metrics.json"); pred_table.to_csv(TABLES/"customer_churn_predictions.csv",index=False)
    print("held-out AUC",churn_metrics["held_out_roc_auc"])

    print("[7/10] Revenue forecasting...")
    monthly=monthly_revenue(sales); monthly.to_csv(TABLES/"monthly_revenue.csv",header=["revenue"])
    forecast_metrics,forecast_df,forecast_model,backtest=fit_final_forecast(monthly); forecast_df.to_csv(FORECASTS/"revenue_forecast.csv",index=False); backtest.to_csv(FORECASTS/"forecast_backtest.csv",index=False); joblib.dump(forecast_model,MODELS/"forecast_model.pkl"); save_json(forecast_metrics,METRICS/"forecast_metrics.json")
    # Forecast plot
    actual=forecast_df[forecast_df["actual_revenue"].notna()]; future=forecast_df[forecast_df["forecast_revenue"].notna()]
    plt.figure(figsize=(11,5)); plt.plot(pd.to_datetime(actual["month"]),actual["actual_revenue"],label="Actual"); plt.plot(pd.to_datetime(future["month"]),future["forecast_revenue"],label="Forecast"); plt.fill_between(pd.to_datetime(future["month"]),future["lower_confidence_bound"],future["upper_confidence_bound"],alpha=.2,label="95% CI"); plt.xlabel("Month"); plt.ylabel("Revenue (£)"); plt.title("Revenue Forecast - 3 Month Forward"); plt.legend(); plt.figtext(0.99,0.01,"Source: RetailPulse monthly revenue series",ha="right",fontsize=7,color="gray"); plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig(FORECAST_FIGURES/"revenue_forecast.png",dpi=160,bbox_inches="tight"); plt.close()

    print("[8/10] Cohorts...")
    cohort_ret,cohort_summary=build_cohort_analysis(sales); cohort_ret.to_csv(TABLES/"cohort_retention.csv",index=False); cohort_summary.to_csv(TABLES/"cohort_summary.csv",index=False)
    heat=cohort_ret.set_index("cohort_month"); numeric=[c for c in heat.columns if isinstance(c,int)]
    plt.figure(figsize=(12,7)); sns.heatmap(heat[numeric],annot=False,cmap="Blues",vmin=0,vmax=100); plt.xlabel("Relative Month"); plt.ylabel("Cohort Month"); plt.title("Monthly Cohort Retention Heatmap"); plt.figtext(0.99,0.01,"Source: RetailPulse completed purchase data",ha="right",fontsize=7,color="gray"); plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig(COHORT_FIGURES/"cohort_heatmap.png",dpi=160,bbox_inches="tight"); plt.close()

    print("[9/10] Statistical testing + MLflow...")
    stats=run_statistical_tests(customers,churn_df,segmented); save_json(stats,METRICS/"statistical_tests.json")
    mlflow_status=log_experiments(churn_metrics,forecast_metrics,str(MODELS),str(ARTIFACTS)); save_json(mlflow_status,METRICS/"mlflow_status.json")

    print("[10/10] Dashboard/report artifacts...")
    overview={
        "total_revenue":float(sales["Revenue"].sum()),"total_customers":int(sales["Customer ID"].nunique()),"total_orders":int(sales["Invoice"].nunique()),"total_products":int(sales["StockCode"].nunique()),"average_order_value":float(sales.groupby("Invoice")["Revenue"].sum().mean()),"repeat_customer_rate":float((sales.groupby("Customer ID")["Invoice"].nunique()>1).mean()*100),"currency_symbol":"£",
        "monthly_revenue":[{"month":idx.strftime("%Y-%m"),"revenue":float(val),"orders":int(sales.loc[sales["InvoiceDate"].dt.to_period("M")==idx.to_period("M"),"Invoice"].nunique())} for idx,val in monthly.items()],
        "top_countries":[{"country":str(k),"revenue":float(v),"customers":int(sales.loc[sales["Country"]==k,"Customer ID"].nunique()),"share":float(v/sales["Revenue"].sum()*100)} for k,v in sales.groupby("Country")["Revenue"].sum().nlargest(8).items()],
        "top_products":pd.read_csv(TABLES/"top_products.csv").rename(columns={"StockCode":"stockCode","Description":"description","Quantity":"quantity","Revenue":"revenue"}).to_dict("records"),
        "revenue_distribution":[]
    }
    # customer-level artifact records
    pred=pred_table[["Customer ID","churn_probability","risk_category"]].copy()
    seg_lookup=segmented.set_index("Customer ID")[["cluster","segment_name"]]
    cc=customers.set_index("Customer ID").join(seg_lookup,how="left").reset_index().merge(pred,left_on="Customer ID",right_on="Customer ID",how="left")
    customer_records=[]
    for _,r in cc.iterrows():
        customer_records.append({"customerId":str(r["Customer ID"]),"country":str(r.get("Country","Unknown")),"segmentId":int(r.get("cluster",-1)),"segmentName":str(r.get("segment_name","Unknown")),"recencyDays":int(r["Recency"]),"frequencyOrders":int(r["Frequency"]),"monetaryValue":float(r["Monetary"]),"avgOrderValue":float(r["AOV"]),"churnProbability":float(r["churn_probability"]) if pd.notna(r["churn_probability"]) else 0.0,"riskCategory":str(r["risk_category"]) if pd.notna(r["risk_category"]) else "Not scored","firstPurchase":str(r["first_purchase"]),"lastPurchase":str(r["last_purchase"]),"topProductCategory":"","shapContributions":[]})
    overview["customer_records"]=customer_records
    provenance={"generated_at":pd.Timestamp.utcnow().isoformat(),"source_sha256":sha256(DATA_XLSX) if DATA_XLSX.exists() else "","source_file":DATA_XLSX.name,"pipeline_version":"RetailPulse v3.0"}
    dashboard_payload=build_dashboard_artifact(audit,cleaning,overview,seg_payload,churn_metrics,forecast_metrics,cohort_ret,cohort_summary,stats,DATA_XLSX.name,provenance)
    save_json(dashboard_payload,ARTIFACTS/"retailpulse_artifacts.json")
    executive_insights={
        "revenue": {"total": float(overview["total_revenue"]), "top_country": max(overview["top_countries"], key=lambda x:x["revenue"])},
        "customers": {"tracked": int(overview["total_customers"]), "repeat_rate": float(overview["repeat_customer_rate"])},
        "segmentation": {"selected_k": int(seg_payload["optimal_k"]), "largest_segment": max(seg_payload["segments"], key=lambda x:x["customers"])},
        "churn": {"auc": float(churn_metrics["held_out_roc_auc"]), "churn_rate": float(churn_metrics["churn_rate"]), "top_shap_feature": churn_metrics["shap_summary"][0]["feature"] if churn_metrics["shap_summary"] else None},
        "forecast": {"model": forecast_metrics["model_name"], "next_month": float(forecast_metrics["next_month_forecast"]), "backtest_mape": float(forecast_metrics["backtest_mape"])},
        "cohorts": {"m1": float(cohort_ret[1].mean()) if 1 in cohort_ret.columns else None, "m6": float(cohort_ret[6].mean()) if 6 in cohort_ret.columns else None, "m12": float(cohort_ret[12].mean()) if 12 in cohort_ret.columns else None},
        "statistics": stats
    }
    save_json(executive_insights,METRICS/"executive_insights.json")
    save_json({"source_file":DATA_XLSX.name,"source_sha256":provenance["source_sha256"],"generated_at":provenance["generated_at"]},METRICS/"provenance.json")
    print(json.dumps({"raw_rows":len(raw),"cleaned_sales_rows":len(sales),"customers":len(customers),"revenue":overview["total_revenue"],"held_out_auc":churn_metrics["held_out_roc_auc"],"forecast_model":forecast_metrics["model_name"],"forecast_mape":forecast_metrics["backtest_mape"],"optimal_k":4},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--source",choices=["auto","xlsx","csv"],default="auto"); args=parser.parse_args(); main(args.source)
