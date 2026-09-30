from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

def save_json(data: dict, path: str | Path) -> None:
    """Write JSON with stable formatting and string conversion for non-JSON-native values."""
    Path(path).write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")

def build_dashboard_artifact(audit: dict, cleaning: dict, overview: dict, segmentation: dict, churn: dict, forecast: dict, cohorts: pd.DataFrame, cohort_summary: pd.DataFrame, stats: list[dict], source_dataset: str, provenance: dict) -> dict:
    """Assemble a single verified artifact document for dashboards and reports."""
    max_interval=max([c for c in cohorts.columns if isinstance(c,int)],default=0)
    retention_rows=[]
    for _,row in cohorts.iterrows():
        rates=[None if pd.isna(row.get(i)) else float(row.get(i)) for i in range(max_interval+1)]
        sm=cohort_summary.loc[cohort_summary["cohort_month"]==row["cohort_month"]]
        aov=float(sm["average_order_value"].iloc[0]) if not sm.empty else 0.0
        size=int(sm["cohort_size"].iloc[0]) if not sm.empty else 0
        retention_rows.append({"cohortMonth":row["cohort_month"],"cohortSize":size,"averageOrderValue":aov,"retentionRates":rates})
    def seg_convert(item):
        return {"id":int(item["cluster"]),"name":item["name"],"count":int(item["customers"]),"percentage":float(item["percentage"]),"avgRecencyDays":float(item["avg_recency"]),"avgFrequencyOrders":float(item["avg_frequency"]),"avgMonetaryValue":float(item["avg_monetary"]),"avgOrderValue":float(item["avg_aov"]),"behavioralDescription":"Profile derived from observed RFM behavior.","businessInterpretation":"Interpret using the displayed RFM statistics and lifecycle metrics.","recommendedAction":"Use differentiated retention, development, or reactivation actions appropriate to this segment."} 
    cust=overview.get("customer_records",[])
    return {
        "sourceDataset":source_dataset,"verified":True,"provenance":provenance,
        "dataQuality":{
          "rawRowCount":audit["raw_row_count"],"cleanedRowCount":cleaning["cleaned_sales_rows"],"duplicateRows":audit["duplicate_rows"],"missingCustomerIds":audit["missing_values"].get("Customer ID",0),"missingDescriptions":audit["missing_values"].get("Description",0),"cancelledInvoices":audit["cancellation_style_rows"],"negativeQuantities":audit["negative_quantity_rows"],"zeroOrNegativePrices":audit["zero_price_rows"]+audit["negative_price_rows"],"dateRange":{"start":audit["date_range"]["start"],"end":audit["date_range"]["end"]},"uniqueCountries":audit["unique_countries"],"uniqueProducts":audit["unique_products"],"uniqueInvoices":audit["unique_invoices"],"uniqueCustomers":audit["unique_customers_non_null"],"rulesApplied":cleaning["rules"]
        },
        "overview":overview,"segmentation":{**segmentation,"segments":[seg_convert(x) for x in segmentation["segments"]]},
        "churn":churn,"customers":cust,"forecast":forecast,
        "cohorts":{"retention_matrix":retention_rows,"average_m1_retention":float(cohorts[1].mean()) if 1 in cohorts.columns else None,"average_m6_retention":float(cohorts[6].mean()) if 6 in cohorts.columns else None,"average_m12_retention":float(cohorts[12].mean()) if 12 in cohorts.columns else None},
        "statistics":stats
    }
