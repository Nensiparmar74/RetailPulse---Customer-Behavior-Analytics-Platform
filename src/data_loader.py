from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country"
]
SHEETS = ["Year 2009-2010", "Year 2010-2011"]


def _normalise_customer_id(series: pd.Series) -> pd.Series:
    """Normalise customer IDs so Excel floats such as 12345.0 become '12345'."""
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.round(0).astype("Int64").astype("string")
    return result


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a descriptive error when required transaction columns are missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_xlsx(path: str | Path) -> pd.DataFrame:
    """Load the two canonical Online Retail II workbook sheets."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frames = [pd.read_excel(path, sheet_name=sheet, engine="openpyxl") for sheet in SHEETS]
    df = pd.concat(frames, ignore_index=True)
    return clean_dtypes(df)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a previously generated raw CSV cache of the workbook."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV cache not found: {path}")
    df = pd.read_csv(path, dtype={"Invoice": "string", "StockCode": "string", "Customer ID": "string"})
    return clean_dtypes(df)


def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the core transaction dtypes."""
    validate_schema(df)
    out = df.copy()
    out["Invoice"] = out["Invoice"].astype("string")
    out["StockCode"] = out["StockCode"].astype("string")
    out["Description"] = out["Description"].astype("string")
    out["Country"] = out["Country"].astype("string")
    out["Customer ID"] = _normalise_customer_id(out["Customer ID"])
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    return out


def load_data(source: str = "auto") -> pd.DataFrame:
    """Load canonical data, preferring a generated CSV cache when present."""
    from config import DATA_XLSX, DATA_CACHE_CSV
    if source == "csv":
        return load_csv(DATA_CACHE_CSV)
    if source == "xlsx":
        return load_xlsx(DATA_XLSX)
    if DATA_CACHE_CSV.exists():
        return load_csv(DATA_CACHE_CSV)
    return load_xlsx(DATA_XLSX)


def audit_dataset(df: pd.DataFrame) -> dict:
    """Return comprehensive raw data-quality statistics."""
    invoice_text = df["Invoice"].astype("string")
    cancellation = invoice_text.str.upper().str.startswith("C", na=False)
    return {
        "raw_row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing_values": {str(c): int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_invoices": int(df["Invoice"].nunique(dropna=True)),
        "unique_products": int(df["StockCode"].nunique(dropna=True)),
        "unique_customers_non_null": int(df["Customer ID"].nunique(dropna=True)),
        "unique_countries": int(df["Country"].nunique(dropna=True)),
        "negative_quantity_rows": int((df["Quantity"] < 0).sum()),
        "zero_quantity_rows": int((df["Quantity"] == 0).sum()),
        "zero_price_rows": int((df["Price"] == 0).sum()),
        "negative_price_rows": int((df["Price"] < 0).sum()),
        "cancellation_style_rows": int(cancellation.sum()),
        "invalid_dates": int(df["InvoiceDate"].isna().sum()),
        "date_range": {
            "start": df["InvoiceDate"].min().isoformat() if df["InvoiceDate"].notna().any() else "",
            "end": df["InvoiceDate"].max().isoformat() if df["InvoiceDate"].notna().any() else "",
        },
    }
