from __future__ import annotations

import pandas as pd


def add_transaction_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add cancellation, return, missingness, and line-revenue indicators without deleting rows."""
    out = df.copy()
    out["InvoiceStr"] = out["Invoice"].astype("string")
    out["IsCancellation"] = out["InvoiceStr"].str.upper().str.startswith("C", na=False)
    out["Revenue"] = out["Quantity"] * out["Price"]
    out["IsReturn"] = out["Quantity"] < 0
    out["DescriptionWasMissing"] = out["Description"].isna()
    out["CustomerIdWasMissing"] = out["Customer ID"].isna()
    out["Description"] = out["Description"].fillna("Unknown Product Description")
    return out


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Return IQR-based lower and upper bounds for an analytical outlier rule."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _add_capped_features(sales: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add IQR-capped helper fields while preserving original transaction values for revenue."""
    out = sales.copy()
    q_lo, q_hi = _iqr_bounds(out["Quantity"])
    p_lo, p_hi = _iqr_bounds(out["Price"])
    # Positive completed sales are already filtered, so lower bounds only need a positive floor.
    q_lo = max(1.0, q_lo)
    p_lo = max(0.01, p_lo)
    out["Quantity_Capped"] = out["Quantity"].clip(q_lo, q_hi)
    out["Price_Capped"] = out["Price"].clip(p_lo, p_hi)
    details = {
        "method": "IQR 1.5x rule",
        "purpose": "Robust analytical/modeling helper fields; original Quantity, Price and Revenue are preserved for headline commercial metrics.",
        "quantity_lower_bound": q_lo,
        "quantity_upper_bound": q_hi,
        "price_lower_bound": p_lo,
        "price_upper_bound": p_hi,
        "quantity_values_capped": int((out["Quantity"] != out["Quantity_Capped"]).sum()),
        "price_values_capped": int((out["Price"] != out["Price_Capped"]).sum()),
    }
    return out, details


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create the positive completed-sales table with documented imputation and outlier-cap helpers."""
    flagged = add_transaction_flags(df)
    raw_rows = len(flagged)
    duplicate_rows = int(flagged.duplicated().sum())
    deduped = flagged.drop_duplicates().copy()
    sales = deduped[
        deduped["Customer ID"].notna()
        & (~deduped["IsCancellation"])
        & (deduped["Quantity"] > 0)
        & (deduped["Price"] > 0)
        & deduped["InvoiceDate"].notna()
    ].copy()

    sales, outliers = _add_capped_features(sales)

    imputed_description_rows = int(sales["DescriptionWasMissing"].sum())
    invalid_date_rows = int(deduped["InvoiceDate"].isna().sum())
    rules = [
        {"issue":"Missing Customer ID","count":int(deduped["Customer ID"].isna().sum()),"action":"Exclude from customer-level RFM, churn and cohort datasets; do not impute IDs","status":"Excluded","justification":"A synthetic customer ID would fabricate longitudinal customer history."},
        {"issue":"Missing Description","count":int(deduped["DescriptionWasMissing"].sum()),"action":"Impute missing descriptions as 'Unknown Product Description' in the analysis table","status":"Cleaned","justification":"Preserves transaction rows while making the text field explicit and auditable."},
        {"issue":"Exact duplicate rows","count":duplicate_rows,"action":"Drop exact duplicates","status":"Cleaned","justification":"Prevents double-counting identical transaction records."},
        {"issue":"Cancellation-style invoices","count":int(deduped["IsCancellation"].sum()),"action":"Exclude from positive-sales table; preserve return/cancellation signals","status":"Reviewed","justification":"Cancellation rows are not completed positive sales but remain useful for customer behavior features."},
        {"issue":"Non-positive quantity","count":int((deduped["Quantity"] <= 0).sum()),"action":"Exclude from positive-sales table","status":"Excluded","justification":"RFM and primary positive commercial revenue use completed positive-quantity purchases."},
        {"issue":"Non-positive unit price","count":int((deduped["Price"] <= 0).sum()),"action":"Exclude from positive-sales table","status":"Excluded","justification":"Zero or negative unit-price rows do not represent positive commercial revenue."},
        {"issue":"Invalid dates","count":invalid_date_rows,"action":"Exclude from time-based analyses","status":"Excluded","justification":"Time-series, churn windows and cohorts require valid dates."},
        {"issue":"Transaction outliers","count":int(max(outliers["quantity_values_capped"], outliers["price_values_capped"])),"action":"Create IQR-capped helper fields Quantity_Capped and Price_Capped; preserve originals","status":"Reviewed","justification":"Caps reduce leverage of extreme line-item values in robust analytical/modeling use without altering the primary revenue definition."},
    ]
    summary = {
        "raw_rows": int(raw_rows),
        "deduped_rows": int(len(deduped)),
        "cleaned_sales_rows": int(len(sales)),
        "rows_removed_from_sales": int(raw_rows - len(sales)),
        "imputation": {"Description": "Unknown Product Description", "rows_imputed_in_sales": imputed_description_rows, "Customer ID": "Not imputed; excluded from longitudinal customer analysis"},
        "outlier_capping": outliers,
        "rules": rules,
        "revenue_definition": "Primary positive-sales revenue = original Quantity * original Price for valid completed positive sales. IQR-capped helper fields are not used to rewrite headline revenue.",
    }
    return sales, deduped, summary
