# THTDA — E-Commerce Events Analytics Dashboard (Streamlit)

Multi-page Streamlit dashboard built from an e-commerce events dataset (view → cart → purchase), covering funnel & cart abandonment, RFM segmentation, cohort retention, and CLV analysis.

## Features
- **Executive Overview**: headline KPIs, revenue trend, funnel, abandonment, retention & CLV snapshots  
- **EDA**: event trends/mix, top categories & brands, distributions  
- **Funnel & Cart Abandonment**: conversion ratios, abandonment drivers, top abandoned categories  
- **RFM Segmentation**: customer segments, revenue/customers by segment, score distribution  
- **Cohort & CLV**: cohort matrix, M1 retention trend, CLV by segment, top CLV users  

## Repository structure
.
├─ app.py
├─ pages/
├─ src/
├─ preprocess.py
├─ requirements.txt
├─ data/
│ ├─ raw/ # ignored (not committed)
│ └─ processed/ # ignored (not committed)
├─ sql/ # optional: SQL queries
└─ .gitignore

## Quickstart (local)
> Python 3.10+ recommended

```bash
# Create venv
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -U pip
pip install -r requirements.txt
Preprocess data
Put raw file(s) in data/raw/ and run:

bash
Copy code
python preprocess.py
This outputs Parquet tables to data/processed/.

Run the app
bash
Copy code
streamlit run app.py
Notes
Large data files (e.g., .xlsx, .parquet) are intentionally ignored via .gitignore to keep the repo lightweight.

For Streamlit Cloud deployment, ensure requirements.txt is included and data loading is compatible (local vs external storage).

Author
Ghania Aisyah
