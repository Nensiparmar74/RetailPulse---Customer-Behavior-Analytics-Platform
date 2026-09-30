from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def savefig(path: Path, title: str) -> None:
    """Apply layout and save a matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.title(title)
    plt.xlabel(plt.gca().get_xlabel() or "Metric")
    plt.figtext(0.99, 0.01, "Source: RetailPulse cleaned transaction data", ha="right", fontsize=7, color="gray")
    plt.tight_layout(rect=[0,0.03,1,1])
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def run_eda(sales: pd.DataFrame, raw: pd.DataFrame, out_dir: str | Path) -> dict:
    """Generate business-focused EDA charts and summary tables."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    monthly = sales.set_index("InvoiceDate")["Revenue"].resample("MS").sum()
    monthly_orders = sales.set_index("InvoiceDate")["Invoice"].resample("MS").nunique()
    order_values = sales.groupby("Invoice")["Revenue"].sum()
    customer_values = sales.groupby("Customer ID")["Revenue"].sum()

    # 1 Monthly revenue
    plt.figure(figsize=(11,4.5)); monthly.plot(); plt.xlabel("Month"); plt.ylabel("Revenue (£)"); savefig(out/"01_monthly_revenue.png","Monthly Revenue")
    # 2 Monthly orders
    plt.figure(figsize=(11,4.5)); monthly_orders.plot(); plt.xlabel("Month"); plt.ylabel("Orders"); savefig(out/"02_monthly_orders.png","Monthly Order Count")
    # 3 Revenue distribution
    plt.figure(figsize=(9,4.5)); order_values.clip(upper=order_values.quantile(.99)).hist(bins=60); plt.xlabel("Order Revenue (£)"); plt.ylabel("Orders"); savefig(out/"03_order_revenue_distribution.png","Order Revenue Distribution (display capped at 99th percentile)")
    # 4 Quantity distribution
    plt.figure(figsize=(9,4.5)); sales["Quantity"].clip(upper=sales["Quantity"].quantile(.99)).hist(bins=60); plt.xlabel("Quantity"); plt.ylabel("Lines"); savefig(out/"04_quantity_distribution.png","Quantity Distribution (display capped at 99th percentile)")
    # 5 Price distribution
    plt.figure(figsize=(9,4.5)); sales["Price"].clip(upper=sales["Price"].quantile(.99)).hist(bins=60); plt.xlabel("Unit Price (£)"); plt.ylabel("Lines"); savefig(out/"05_price_distribution.png","Unit Price Distribution (display capped at 99th percentile)")
    # 6 Top countries
    topc=sales.groupby("Country")["Revenue"].sum().nlargest(10).sort_values(); plt.figure(figsize=(9,5)); topc.plot.barh(); plt.xlabel("Revenue (£)"); plt.ylabel("Country"); savefig(out/"06_top_countries.png","Top 10 Countries by Revenue")
    # 7 Top products
    top_products=sales.groupby(["StockCode","Description"],dropna=False).agg(Quantity=("Quantity","sum"),Revenue=("Revenue","sum")).nlargest(10,"Revenue").reset_index(); plt.figure(figsize=(9,5)); top_products.sort_values("Revenue").plot.barh(x="StockCode",y="Revenue",legend=False); plt.xlabel("Revenue (£)"); plt.ylabel("Stock Code"); savefig(out/"07_top_products.png","Top 10 Products by Revenue")
    # 8 Customer revenue distribution
    plt.figure(figsize=(9,4.5)); customer_values.clip(upper=customer_values.quantile(.99)).hist(bins=60); plt.xlabel("Customer Monetary Value (£)"); plt.ylabel("Customers"); savefig(out/"08_customer_revenue_distribution.png","Customer Monetary Value Distribution (display capped at 99th percentile)")
    # 9 Average order value by country (top 8 customers volume)
    country_aov=sales.groupby("Country").apply(lambda g:g.groupby("Invoice")["Revenue"].sum().mean()).dropna().nlargest(8).sort_values(); plt.figure(figsize=(9,4.5)); country_aov.plot.barh(); plt.xlabel("Average Order Value (£)"); plt.ylabel("Country"); savefig(out/"09_country_aov.png","Top 8 Countries by Average Order Value")
    # 10 returns / cancellations counts
    cancellation = raw["IsCancellation"].value_counts().rename(index={False:"Normal",True:"Cancellation"}); plt.figure(figsize=(7,4)); cancellation.plot.bar(); plt.xlabel("Invoice Type"); plt.ylabel("Transaction Rows"); savefig(out/"10_cancellation_rows.png","Cancellation-Style Transaction Rows")
    # 11 missing values
    missing=raw.isna().sum().sort_values(ascending=False); missing=missing[missing>0]; plt.figure(figsize=(8,4.5)); missing.plot.bar(); plt.xlabel("Column"); plt.ylabel("Missing Values"); savefig(out/"11_missing_values.png","Missing Value Audit")
    # 12 correlation heatmap
    corr=sales[["Quantity","Price","Revenue"]].corr(); plt.figure(figsize=(6,5)); sns.heatmap(corr,annot=True,fmt=".2f",cmap="Blues",square=True); plt.xlabel("Metric"); plt.ylabel("Metric"); savefig(out/"12_correlation_heatmap.png","Transaction Metric Correlation Heatmap")

    top_products.to_csv(out.parent.parent/"tables"/"top_products.csv",index=False)
    country_aov.rename("average_order_value").reset_index().to_csv(out.parent.parent/"tables"/"country_aov.csv",index=False)
    topc.rename("revenue").reset_index().to_csv(out.parent.parent/"tables"/"top_countries.csv",index=False)
    pd.DataFrame({"monthly_revenue":monthly,"monthly_orders":monthly_orders}).reset_index().rename(columns={"InvoiceDate":"month"}).to_csv(out.parent.parent/"tables"/"monthly_kpis.csv",index=False)

    summary={
        "sales_rows":int(len(sales)),
        "total_revenue":float(sales["Revenue"].sum()),
        "total_orders":int(sales["Invoice"].nunique()),
        "unique_customers":int(sales["Customer ID"].nunique()),
        "unique_products":int(sales["StockCode"].nunique()),
        "average_order_value":float(order_values.mean()),
        "repeat_customer_rate":float((sales.groupby("Customer ID")["Invoice"].nunique()>1).mean()*100),
        "top_countries":topc.to_dict(),
        "top_products":top_products.to_dict("records"),
        "visualization_count":12,
        "revenue_currency":"GBP",
    }
    return summary
