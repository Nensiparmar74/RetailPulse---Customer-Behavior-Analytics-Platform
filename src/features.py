from __future__ import annotations

import numpy as np
import pandas as pd


def build_customer_features(sales: pd.DataFrame, deduped_raw: pd.DataFrame | None = None, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build customer-level RFM, lifecycle, cadence, CLV approximation, and return features."""
    if sales.empty:
        raise ValueError("Sales table is empty.")
    ref = reference_date or (sales["InvoiceDate"].max() + pd.Timedelta(days=1))
    g=sales.groupby("Customer ID")
    out=g.agg(
        last_purchase=("InvoiceDate","max"),
        first_purchase=("InvoiceDate","min"),
        Frequency=("Invoice","nunique"),
        Monetary=("Revenue","sum"),
        TotalQuantity=("Quantity","sum"),
        UniqueProducts=("StockCode","nunique"),
    ).reset_index()
    out["Recency"]=(ref-out["last_purchase"]).dt.days.astype(int)
    out["LifetimeDays"]=(out["last_purchase"]-out["first_purchase"]).dt.days.clip(lower=1)
    out["AOV"]=out["Monetary"]/out["Frequency"].replace(0,np.nan)
    out["AvgDaysBetweenPurchases"]=np.where(out["Frequency"]>1,out["LifetimeDays"]/(out["Frequency"]-1),out["LifetimeDays"])
    out["PurchaseFrequencyPerYear"]=out["Frequency"]/(out["LifetimeDays"]/365.25)
    out["ActiveMonths"]=sales.assign(_month=sales["InvoiceDate"].dt.to_period("M")).groupby("Customer ID")["_month"].nunique().reindex(out["Customer ID"]).fillna(0).astype(int).to_numpy()
    out["WeekendPurchaseRatio"]=sales.assign(_weekend=sales["InvoiceDate"].dt.dayofweek>=5).groupby("Customer ID")["_weekend"].mean().reindex(out["Customer ID"]).fillna(0).to_numpy()
    out["DistinctWeekdays"]=sales.groupby("Customer ID")["InvoiceDate"].apply(lambda s:s.dt.dayofweek.nunique()).reindex(out["Customer ID"]).fillna(0).astype(int).to_numpy()
    out["MeanPurchaseHour"]=sales.groupby("Customer ID")["InvoiceDate"].apply(lambda s:s.dt.hour.mean()).reindex(out["Customer ID"]).fillna(0).to_numpy()
    country=sales.sort_values("InvoiceDate").groupby("Customer ID")["Country"].last()
    out["Country"]=out["Customer ID"].map(country)
    out["CLVApprox"]=out["AOV"]*out["PurchaseFrequencyPerYear"]*np.maximum(out["LifetimeDays"]/365.25,1)
    if deduped_raw is not None:
        raw=deduped_raw[deduped_raw["Customer ID"].notna()].copy()
        ret=raw.groupby("Customer ID").agg(
            ReturnQuantity=("Quantity",lambda s: float(s[s<0].abs().sum())),
            PositiveQuantity=("Quantity",lambda s: float(s[s>0].sum())),
            CancellationRows=("IsCancellation","sum"),
        ).reset_index()
        ret["ReturnCancellationRatio"]=(ret["ReturnQuantity"]/ret["PositiveQuantity"].replace(0,np.nan)).fillna(0)
        out=out.merge(ret[["Customer ID","ReturnQuantity","PositiveQuantity","CancellationRows","ReturnCancellationRatio"]],on="Customer ID",how="left")
    else:
        out["ReturnCancellationRatio"]=0.0
    return out.replace([np.inf,-np.inf],np.nan)
