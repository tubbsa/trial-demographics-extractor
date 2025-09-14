# trial-demographics-extractor

A Streamlit app that bulk-extracts **baseline demographics** from **ClinicalTrials.gov** Results by uploading a CSV/XLSX of **NCT IDs**. It exports a clean, wide Excel containing only the demographics you select: **Age**, **Sex/Gender**, **Race**, and **Ethnicity**.

## Features
- Parses **Results → Baseline Characteristics** only (no eligibility/design/locations).
- Captures **all age data**:
  - Continuous stats: `age_mean`, `age_sd`, `age_median`, `age_min`, `age_max`, `age_iqr` (+ fallbacks).
  - **Every** categorical bucket as `agecat_<bucket>_n` and/or `agecat_<bucket>_pct` (e.g., `ge_65_years`, `65_to_74_years`).
- Aggregates counts across arms/parts and computes percents when possible.
- Sidebar lets you **choose which demographics** to include in the export.
- Download a single Excel: `Export_Wide` sheet.

## Quickstart (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Usage
1. Upload a file (CSV/XLSX/TSV/TXT) containing NCT IDs **or** paste them in the sidebar.
2. Pick demographics to export: **Age**, **Sex/Gender**, **Race**, **Ethnicity**.
3. Click **Run Extraction** → Review the preview table → **Download Excel**.

**Notes**
- Counts are summed across arms/parts per category.
- Percents are computed from counts when denominators are available; otherwise the first reported percent per category is kept.
- Age “continuous” stats are taken as first occurrence per stat (mean/SD/…); accurate weighting across arms needs per-arm N (not always provided).

## Deploy (Streamlit Community Cloud)
1. Make the repo **public** on GitHub.
2. Go to https://share.streamlit.io → **New app** → select this repo, branch `main`, file `app.py`.
3. Deploy. It auto-redeploys on commits.

## Data source & disclaimer
- Pulls public Results data from **ClinicalTrials.gov**. No PHI is collected or stored.
- Always verify extracted numbers against the source study page.

## Cite
If you use this tool in research, please cite the software and a release DOI if available.
- **Repo:** https://github.com/tubbsa/trial-demographics-extractor
- **DOI:** (add after enabling Zenodo + creating a GitHub Release)

## License
MIT (see `LICENSE`).