# THTDA — E-Commerce Events Analytics Dashboard (Streamlit)

Multi-page Streamlit dashboard built from an e-commerce events dataset (view → cart → purchase), covering funnel & cart abandonment, RFM segmentation, cohort retention, and CLV analysis.

## Features
- **Executive Overview**: headline KPIs, revenue trend, funnel, abandonment, retention & CLV snapshots
- **EDA**: event trends/mix, top categories & brands, distributions
- **Funnel & Cart Abandonment**: conversion ratios, abandonment drivers, top abandoned categories
- **RFM Segmentation**: customer segments, revenue/customers by segment, score distribution
- **Cohort & CLV**: cohort matrix, M1 retention trend, CLV by segment, top CLV users

## Repository structure
```text
THTDA/
  app.py
  pages/
  src/
  preprocess.py
  requirements.txt
  data/
    raw/          # ignored (not committed)
    processed/    # ignored (not committed)
  sql/            # optional: SQL queries
  .gitignore
