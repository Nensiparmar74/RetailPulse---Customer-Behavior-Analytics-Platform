from __future__ import annotations

import pandas as pd
import numpy as np

def build_cohort_analysis(sales: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Build monthly cohort retention matrix and cohort AOV summary."""
    d=sales.copy(); d["activity_month"]=d["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    first=d.groupby("Customer ID")["activity_month"].min().rename("cohort_month")
    d=d.join(first,on="Customer ID")
    d["period_number"]=(d["activity_month"].dt.year-d["cohort_month"].dt.year)*12+(d["activity_month"].dt.month-d["cohort_month"].dt.month)
    active=d.groupby(["cohort_month","period_number"])["Customer ID"].nunique().unstack(fill_value=0)
    cohort_size=active[0]
    retention=active.divide(cohort_size,axis=0)*100
    retention.index=retention.index.strftime("%Y-%m")
    retention.columns=[int(c) for c in retention.columns]
    retention=retention.reset_index().rename(columns={"cohort_month":"cohort_month"})
    aov=d.groupby(["cohort_month","Invoice"])["Revenue"].sum().groupby(level=0).mean()
    summary=pd.DataFrame({"cohort_month":cohort_size.index.strftime("%Y-%m"),"cohort_size":cohort_size.values,"average_order_value":aov.reindex(cohort_size.index).values})
    return retention,summary
