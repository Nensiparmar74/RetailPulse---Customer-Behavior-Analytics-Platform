from pathlib import Path
import pandas as pd
from config import DATA_XLSX, DATA_CACHE_CSV
from src.data_loader import load_xlsx

def main() -> None:
    """Convert the canonical workbook into a local CSV cache for faster repeated runs."""
    df=load_xlsx(DATA_XLSX)
    df.to_csv(DATA_CACHE_CSV,index=False)
    print(f"Wrote {len(df):,} rows to {DATA_CACHE_CSV}")

if __name__=="__main__": main()
