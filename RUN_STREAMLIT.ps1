$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1
if (-not (Get-Command streamlit -ErrorAction SilentlyContinue)) {
    throw "Streamlit is not installed. Run: pip install -r requirements.txt"
}
streamlit run dashboard/app.py
