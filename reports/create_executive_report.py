from __future__ import annotations

from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from config import ARTIFACTS, REPORTS

def money(v): return f"£{v:,.0f}"

def load(name): return json.loads((ARTIFACTS/"metrics"/name).read_text(encoding="utf-8"))

def build() -> Path:
    """Create the executive BI PDF from the latest generated analytics artifacts."""
    aq=load("data_quality_raw.json"); cl=load("cleaning_summary.json"); eda=load("eda_summary.json"); seg=load("segmentation_metrics.json"); ch=load("churn_metrics.json"); fc=load("forecast_metrics.json"); stats=json.loads((ARTIFACTS/"metrics"/"statistical_tests.json").read_text(encoding="utf-8")); prov=load("provenance.json")
    out=REPORTS/"executive_report.pdf"
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name="Small",parent=styles["BodyText"],fontSize=8.5,leading=12)); styles.add(ParagraphStyle(name="Section",parent=styles["Heading2"],spaceBefore=12,spaceAfter=6,textColor=colors.HexColor("#1e293b")))
    doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=15*mm,bottomMargin=15*mm)
    story=[]
    story += [Paragraph("RetailPulse - Customer Behavior Analytics & Revenue Intelligence Platform",styles["Title"]),Paragraph("Executive Business Intelligence Report",styles["Heading2"]),Spacer(1,7*mm)]
    story.append(Paragraph(f"Source: {prov['source_file']} | Generated: {prov['generated_at']}",styles["Small"]))
    story.append(Spacer(1,5*mm))
    overview_text=(f"The analysis covers {aq['raw_row_count']:,} transaction rows. After exact-duplicate removal and the documented positive-sales rules, {cl['cleaned_sales_rows']:,} completed positive-sales rows were retained for the primary customer and revenue analyses. The primary revenue measure is Quantity multiplied by Price for positive completed sales.")
    story.append(Paragraph("Executive Summary",styles["Section"])); story.append(Paragraph(overview_text,styles["BodyText"]))
    story.append(Spacer(1,3*mm))
    kpi=[['Metric','Value'],['Raw transactions',f"{aq['raw_row_count']:,}"],['Cleaned sales lines',f"{cl['cleaned_sales_rows']:,}"],['Customers',f"{eda['unique_customers']:,}"],['Orders',f"{eda['total_orders']:,}"],['Revenue',money(eda['total_revenue'])],['AOV',money(eda['average_order_value'])],['Repeat customer rate',f"{eda['repeat_customer_rate']:.1f}%"]]
    t=Table(kpi,colWidths=[72*mm,70*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0f172a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#cbd5e1')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),8.5),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.append(t)
    story.append(Paragraph("1. Data Quality and Cleaning",styles["Section"]))
    story.append(Paragraph(f"The raw audit found {aq['duplicate_rows']:,} exact duplicate rows, {aq['missing_values']['Customer ID']:,} missing Customer ID values, {aq['missing_values']['Description']:,} missing descriptions, {aq['negative_quantity_rows']:,} negative-quantity rows, {aq['zero_price_rows']:,} zero-price rows, {aq['negative_price_rows']:,} negative-price rows, and {aq['cancellation_style_rows']:,} cancellation-style invoice rows. Customer-level analyses exclude missing customer identifiers because lifecycle tracking is otherwise impossible.",styles["BodyText"]))
    if 'outlier_capping' in cl:
        oc=cl['outlier_capping']; imp=cl.get('imputation',{})
        story.append(Paragraph(f"Imputation: missing descriptions are filled with 'Unknown Product Description' ({imp.get('rows_imputed_in_sales',0):,} primary-sales rows). Customer IDs are deliberately not imputed. Robustness: IQR 1.5x helper caps are applied to Quantity and Price for analytical/modeling use, with {oc.get('quantity_values_capped',0):,} quantity values and {oc.get('price_values_capped',0):,} price values capped; original fields remain intact for headline revenue. Quantity cap: {oc.get('quantity_lower_bound')} to {oc.get('quantity_upper_bound')}; Price cap: {oc.get('price_lower_bound')} to {oc.get('price_upper_bound')}.",styles["Small"]))
    story.append(Paragraph("2. Exploratory Data Analysis",styles["Section"]))
    top_country=max(eda['top_countries'].items(),key=lambda kv:kv[1]); story.append(Paragraph(f"EDA generated {eda['visualization_count']} meaningful visualizations covering revenue, orders, quantity, price, countries, products, customer monetary value, cancellations, missingness and correlation. The top revenue country was {top_country[0]} with {money(top_country[1])} of positive-sales revenue.",styles["BodyText"]))
    story.append(Paragraph("3. Customer Segmentation",styles["Section"]))
    story.append(Paragraph(seg['selection_note'],styles["BodyText"]))
    seg_table=[['Cluster','Profile','Customers','%','Avg Recency','Avg Frequency','Avg Monetary','Avg AOV']]
    for r in sorted(seg['segments'],key=lambda x:x['cluster']):
        seg_table.append([str(r['cluster']),str(r['name']),f"{r['customers']:,}",f"{r['percentage']:.1f}%",f"{r['avg_recency']:.1f}",f"{r['avg_frequency']:.1f}",money(r['avg_monetary']),money(r['avg_aov'])])
    st=Table(seg_table,colWidths=[14*mm,45*mm,18*mm,14*mm,18*mm,20*mm,24*mm,20*mm]); st.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0f172a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#cbd5e1')),('FONTSIZE',(0,0),(-1,-1),7),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.append(st)
    img=ARTIFACTS/'figures'/'segmentation'/'elbow_method.png';
    if img.exists(): story += [Spacer(1,3*mm),Image(str(img),width=135*mm,height=55*mm)]
    img2=ARTIFACTS/'figures'/'segmentation'/'silhouette_scores.png';
    if img2.exists(): story += [Spacer(1,2*mm),Image(str(img2),width=135*mm,height=55*mm)]
    story.append(Paragraph("4. Churn Prediction",styles["Section"]))
    story.append(Paragraph(f"A temporal churn target was created using an observation window ending 2011-06-30 and a three-month prediction window ending 2011-09-30. The Random Forest held-out test ROC-AUC was {ch['held_out_roc_auc']:.3f}; cross-validation mean AUC was {ch['cv_auc_mean']:.3f}. Precision was {ch['precision']:.3f}, recall {ch['recall']:.3f}, F1 {ch['f1_score']:.3f}. The >0.80 target was {'achieved' if ch['held_out_roc_auc']>0.8 else 'not achieved'} on the held-out set.",styles["BodyText"]))
    for fname in ['roc_curve.png','confusion_matrix.png']:
        p=ARTIFACTS/'figures'/'churn'/fname
        if p.exists(): story += [Spacer(1,3*mm),Image(str(p),width=150*mm,height=65*mm)]
    story.append(Paragraph("5. SHAP Explainability",styles["Section"]))
    story.append(Paragraph("SHAP summaries explain which features most influence churn predictions. The results are predictive associations, not causal effects. The strongest mean-absolute SHAP feature is shown in the dashboard and saved under artifacts/shap.",styles["BodyText"]))
    p=ARTIFACTS/'shap'/'shap_beeswarm.png'
    if p.exists(): story += [Spacer(1,3*mm),Image(str(p),width=145*mm,height=58*mm)]
    story.append(Paragraph("The SHAP waterfall for a representative high-risk test customer is retained under artifacts/shap/shap_waterfall_high_risk.png and is also shown in the Streamlit dashboard.",styles["Small"]))
    story.append(PageBreak())
    story.append(Paragraph("6. Revenue Forecasting",styles["Section"]))
    story.append(Paragraph(f"A SARIMA model was selected using a three-month historical backtest. Selected model: {fc['model_name']}. Backtest MAPE was {fc['backtest_mape']:.2f}%, RMSE was {money(fc['backtest_rmse'])}. The final model was fitted through the last complete observed month and generated a three-month forward forecast with 95% confidence intervals. Forecast values are {money(fc['next_month_forecast'])} for the first month and {money(fc['next_three_months_forecast_total'])} cumulatively across the three-month horizon.",styles["BodyText"]))
    p=ARTIFACTS/'figures'/'forecast'/'revenue_forecast.png';
    if p.exists(): story += [Spacer(1,3*mm),Image(str(p),width=155*mm,height=70*mm)]
    story.append(Paragraph("7. Cohort Retention",styles["Section"]))
    story.append(Paragraph("Monthly cohorts are defined from the customer's first purchase month. Retention is the share of the original cohort that is active in each relative month. The full retention matrix is saved in artifacts/tables/cohort_retention.csv.",styles["BodyText"]))
    p=ARTIFACTS/'figures'/'cohort'/'cohort_heatmap.png';
    if p.exists(): story += [Spacer(1,3*mm),Image(str(p),width=155*mm,height=88*mm)]
    story.append(PageBreak())
    story.append(Paragraph("8. Statistical Testing",styles["Section"]))
    rows=[['Test','Statistic','p-value','Significant?']]
    for r in stats: rows.append([r['test_name'],f"{r['statistic_value']:.3g}",f"{r['p_value']:.3g}","Yes" if r['is_statistically_significant'] else "No"])
    tt=Table(rows,colWidths=[90*mm,25*mm,25*mm,25*mm]); tt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0f172a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#cbd5e1')),('FONTSIZE',(0,0),(-1,-1),7.5)])); story.append(tt)
    story.append(Paragraph("For every test, the null and alternative hypotheses, test statistic and p-value are also retained in artifacts/metrics/statistical_tests.json. Statistical significance is not treated as causation.",styles["BodyText"]))
    story.append(Paragraph("9. Business Recommendations",styles["Section"]))
    recs=[
      "Use RFM segments to differentiate retention, development and reactivation campaigns rather than applying one message to all customers.",
      "Prioritize churn interventions around customers whose recency has moved materially beyond their normal purchase cadence.",
      "Use SHAP explanations with model-risk safeguards so retention teams can see the behavioral features behind an individual prediction.",
      "Use the revenue forecast as a planning input for inventory, staffing and cash-flow preparation, while monitoring actual-versus-forecast error each month.",
      "Track cohort retention after acquisition campaigns so marketing teams can compare customer quality beyond first-order revenue."
    ]
    for r in recs: story.append(Paragraph("- "+r,styles["BodyText"]))
    story.append(Paragraph("10. Reproducibility and Runtime Notes",styles["Section"]))
    story.append(Paragraph("The repository contains the seven notebooks plus reusable source modules and a single run_pipeline.py entry point. After installing requirements, the pipeline regenerates the numerical artifacts and the report. MLflow logging is implemented in src/mlflow_tracking.py and requires the MLflow package at runtime. Streamlit dashboard code is implemented in dashboard/app.py and requires Streamlit at runtime. These are environment/account actions and are not silently represented as completed when the package is unavailable in the build environment.",styles["BodyText"]))
    story.append(Paragraph("11. Limitations",styles["Section"]))
    story.append(Paragraph(f"The churn label is operational rather than a directly observed business churn field. The dataset's final month is incomplete, so the forecasting history excludes that partial month from the final training series. K=4 is selected for the required four-profile business view even though the global silhouette maximum is K=2; this trade-off is explicitly documented. The current held-out churn AUC is {ch['held_out_roc_auc']:.3f}.",styles["BodyText"]))
    story.append(Spacer(1,5*mm)); story.append(Paragraph("Generated from the canonical uploaded Online Retail II workbook. Re-run run_pipeline.py to regenerate all numerical artifacts.",styles["Small"]))
    doc.build(story)
    return out

if __name__=="__main__": print(build())
