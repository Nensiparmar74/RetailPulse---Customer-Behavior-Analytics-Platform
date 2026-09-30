from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts"

st.set_page_config(page_title="RetailPulse",page_icon="RP",layout="wide",initial_sidebar_state="expanded")

@st.cache_data
def load_json(path: Path):
    """Load a JSON artifact and cache it for the Streamlit session."""
    if not path.exists(): return None
    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_data
def load_csv(path: Path):
    """Load a CSV artifact and cache it."""
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path)

artifact=load_json(ART/"retailpulse_artifacts.json")

st.markdown("""<style>
body{background:#070b14}.block-container{padding-top:1.2rem}.metric-card{background:#0c1220;border:1px solid #1f2937;border-radius:14px;padding:16px}.small{color:#94a3b8;font-size:.78rem}.title{font-weight:800;letter-spacing:.04em}.pill{padding:.2rem .45rem;border-radius:999px;background:#111827;border:1px solid #334155;font-size:.7rem}
</style>""",unsafe_allow_html=True)

if not artifact or not artifact.get("verified"):
    st.title("RetailPulse")
    st.warning("No verified analytics artifact is loaded. Run the Python pipeline first: `python run_pipeline.py --source xlsx`.")
    st.stop()

source=artifact["sourceDataset"]
with st.sidebar:
    st.markdown("# RETAIL<span style='color:#818cf8'>PULSE</span>",unsafe_allow_html=True)
    st.caption("Executive BI Suite")
    page=st.radio("Navigate",["Overview","Segments","Churn","Forecast","Cohorts","Data Quality","Executive Insights"])
    st.divider(); st.caption(f"Source: {source}")
    prov=artifact.get("provenance",{})
    st.caption(f"Generated: {prov.get('generated_at','N/A')}")

over=artifact["overview"]; dq=artifact["dataQuality"]; seg=artifact["segmentation"]; ch=artifact["churn"]; fc=artifact["forecast"]; co=artifact["cohorts"]; stats=artifact.get("statistics",[])

if page=="Overview":
    st.title("Executive Business Overview")
    st.caption("Verified outputs from the RetailPulse analytics pipeline.")
    c=st.columns(6)
    vals=[(money:=f"{over['currency_symbol']}{over['total_revenue']:,.0f}","Total Revenue"),(f"{over['total_customers']:,}","Customers"),(f"{over['total_orders']:,}","Orders"),(f"{over['total_products']:,}","Products"),(f"{over['currency_symbol']}{over['average_order_value']:,.2f}","Average Order Value"),(f"{over['repeat_customer_rate']:.1f}%","Repeat Customer Rate")]
    for col,(v,label) in zip(c,vals): col.metric(label,v)
    mrev=pd.DataFrame(over["monthly_revenue"]); mrev["month"]=pd.to_datetime(mrev["month"])
    fig=px.line(mrev,x="month",y="revenue",markers=True,title="Monthly Revenue")
    st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left:
        country=pd.DataFrame(over["top_countries"]); st.plotly_chart(px.bar(country.sort_values("revenue"),x="revenue",y="country",orientation="h",title="Top Countries by Revenue"),use_container_width=True)
    with right:
        prod=pd.DataFrame(over["top_products"]); st.dataframe(prod.rename(columns={"stockCode":"Stock Code","description":"Description","quantity":"Quantity","revenue":"Revenue"}),use_container_width=True,hide_index=True)

elif page=="Segments":
    st.title("Customer Segmentation - RFM + K-Means")
    st.caption(seg["selection_note"])
    c=st.columns(3); c[0].metric("Customers",f"{seg['totalCustomers']:,}"); c[1].metric("Selected K",str(seg["optimal_k"])); c[2].metric("Profiles",str(len(seg["segments"])))
    e=pd.DataFrame(seg["elbow"]); s=pd.DataFrame(seg["silhouette"])
    a,b=st.columns(2); a.plotly_chart(px.line(e,x="k",y="inertia",markers=True,title="Elbow Method"),use_container_width=True); b.plotly_chart(px.line(s,x="k",y="silhouette",markers=True,title="Silhouette Score"),use_container_width=True)
    profiles=pd.DataFrame(seg["segments"]); st.dataframe(profiles,use_container_width=True,hide_index=True)
    st.subheader("Business Profiles")
    for row in seg["segments"]:
        with st.expander(f"Cluster {row['cluster']} - {row['name']}"):
            st.write(f"Customers: {row['customers']:,} ({row['percentage']:.1f}%)")
            st.write(f"Avg Recency: {row['avg_recency']:.1f} days | Avg Frequency: {row['avg_frequency']:.1f} | Avg Monetary: £{row['avg_monetary']:,.2f} | Avg AOV: £{row['avg_aov']:,.2f}")

elif page=="Churn":
    st.title("Churn Intelligence")
    c=st.columns(5); c[0].metric("Held-out AUC",f"{ch['held_out_roc_auc']:.3f}"); c[1].metric("Precision",f"{ch['precision']:.3f}"); c[2].metric("Recall",f"{ch['recall']:.3f}"); c[3].metric("F1",f"{ch['f1_score']:.3f}"); c[4].metric("Churn Rate",f"{ch['churn_rate']:.1f}%")
    roc=pd.DataFrame(ch["roc_curve"]); fig=go.Figure(); fig.add_trace(go.Scatter(x=roc.fpr,y=roc.tpr,mode="lines",name="Model ROC")); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Chance",line=dict(dash="dash"))); fig.update_layout(title="ROC Curve",xaxis_title="False Positive Rate",yaxis_title="True Positive Rate"); st.plotly_chart(fig,use_container_width=True)
    fi=pd.DataFrame(ch["feature_importance"]).sort_values("importance"); st.plotly_chart(px.bar(fi,x="importance",y="feature",orientation="h",title="Random Forest Feature Importance"),use_container_width=True)
    st.subheader("SHAP Explainability")
    shap_a, shap_b = st.columns(2)
    bees=ART/"shap"/"shap_beeswarm.png"; water=ART/"shap"/"shap_waterfall_high_risk.png"
    if bees.exists(): shap_a.image(str(bees),caption="SHAP Beeswarm",use_container_width=True)
    else: shap_a.info("SHAP beeswarm artifact not available.")
    if water.exists(): shap_b.image(str(water),caption="High-risk customer SHAP waterfall",use_container_width=True)
    else: shap_b.info("SHAP waterfall artifact not available.")
    st.dataframe(pd.DataFrame(ch["shap_summary"]),use_container_width=True,hide_index=True)
    pred=load_csv(ART/"tables"/"customer_churn_predictions.csv")
    if not pred.empty:
        q=st.text_input("Search customer ID")
        view=pred[pred["Customer ID"].astype(str).str.contains(q.strip(),na=False)] if q.strip() else pred.head(100)
        st.dataframe(view,use_container_width=True,hide_index=True)

elif page=="Forecast":
    st.title("Revenue Forecast")
    c=st.columns(4); c[0].metric("Model",fc["model_name"]); c[1].metric("Backtest MAPE",f"{fc['backtest_mape']:.2f}%"); c[2].metric("Next Month",f"£{fc['next_month_forecast']:,.0f}"); c[3].metric("3-Month Total",f"£{fc['next_three_months_forecast_total']:,.0f}")
    fdf=load_csv(ART/"forecasts"/"revenue_forecast.csv"); fdf["month"]=pd.to_datetime(fdf["month"])
    fig=go.Figure(); actual=fdf[fdf.actual_revenue.notna()]; future=fdf[fdf.forecast_revenue.notna()]; fig.add_trace(go.Scatter(x=actual.month,y=actual.actual_revenue,mode="lines+markers",name="Actual")); fig.add_trace(go.Scatter(x=future.month,y=future.forecast_revenue,mode="lines+markers",name="Forecast")); fig.add_trace(go.Scatter(x=future.month,y=future.upper_confidence_bound,mode="lines",line=dict(width=0),showlegend=False)); fig.add_trace(go.Scatter(x=future.month,y=future.lower_confidence_bound,mode="lines",fill="tonexty",line=dict(width=0),name="95% CI")); fig.update_layout(title="Historical Revenue + Three-Month Forecast",xaxis_title="Month",yaxis_title="Revenue (£)"); st.plotly_chart(fig,use_container_width=True); st.dataframe(fdf,use_container_width=True,hide_index=True)

elif page=="Cohorts":
    st.title("Cohort Retention")
    m1=co.get("average_m1_retention"); m6=co.get("average_m6_retention"); m12=co.get("average_m12_retention"); c=st.columns(3); c[0].metric("Avg M1 Retention",f"{m1:.1f}%" if m1 is not None else "N/A"); c[1].metric("Avg M6 Retention",f"{m6:.1f}%" if m6 is not None else "N/A"); c[2].metric("Avg M12 Retention",f"{m12:.1f}%" if m12 is not None else "N/A")
    ret=load_csv(ART/"tables"/"cohort_retention.csv"); numeric=[c for c in ret.columns if str(c).isdigit()]; long=[]
    for _,r in ret.iterrows():
        for col in numeric:
            long.append({"cohort":r["cohort_month"],"month":f"M{col}","retention":r.get(col)})
    st.plotly_chart(px.density_heatmap(pd.DataFrame(long),x="month",y="cohort",z="retention",color_continuous_scale="Blues",title="Monthly Cohort Retention Heatmap"),use_container_width=True)
    st.dataframe(load_csv(ART/"tables"/"cohort_summary.csv"),use_container_width=True,hide_index=True)

elif page=="Data Quality":
    st.title("Data Quality Audit")
    c=st.columns(6); metrics=[("Raw Rows",dq["rawRowCount"]),("Cleaned Rows",dq["cleanedRowCount"]),("Duplicates",dq["duplicateRows"]),("Missing Customer IDs",dq["missingCustomerIds"]),("Cancellations",dq["cancelledInvoices"]),("Negative Quantities",dq["negativeQuantities"])];
    for col,(label,val) in zip(c,metrics): col.metric(label,f"{val:,}")
    st.write("**Date range:**",dq["dateRange"]["start"],"to",dq["dateRange"]["end"]); st.write("**Countries:**",dq["uniqueCountries"],"**Products:**",dq["uniqueProducts"],"**Invoices:**",dq["uniqueInvoices"],"**Customers:**",dq["uniqueCustomers"]); st.dataframe(pd.DataFrame(dq["rulesApplied"]),use_container_width=True,hide_index=True)

elif page=="Executive Insights":
    st.title("Executive Insights")
    st.info("These statements should be regenerated from the latest verified artifacts after each pipeline run.")
    st.write("**Revenue:**",f"Primary positive-sales revenue is £{over['total_revenue']:,.0f}.")
    st.write("**Customers:**",f"{over['total_customers']:,} tracked customers with a {over['repeat_customer_rate']:.1f}% repeat-customer rate.")
    st.write("**Churn:**",f"Held-out ROC-AUC is {ch['held_out_roc_auc']:.3f}. See SHAP outputs for model explanations.")
    st.write("**Forecast:**",f"Selected {fc['model_name']} model with {fc['backtest_mape']:.2f}% historical backtest MAPE.")
    st.subheader("Statistical Tests")
    st.dataframe(pd.DataFrame(stats),use_container_width=True,hide_index=True)
