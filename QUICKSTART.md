# Quick Start (No Coding Required)

This guide shows how to extract baseline demographics (race, ethnicity, sex/gender, age) from ClinicalTrials.gov for a user-defined cohort of trials using **Clinical Trial Demographics Extractor (v10)**.

## What you need
- A list of ClinicalTrials.gov trial identifiers (**NCT IDs**)  
- The web app (recommended) OR local installation (optional)

**Web app:** https://tubbstrialfeatureextractor.streamlit.app  
**Code repository:** https://github.com/tubbsa/trial-demographics-extractor  
**License:** MIT  
**Support:** abigail.tubbs@und.edu

---

## 1) Create an NCT ID list from ClinicalTrials.gov (recommended workflow)

1. Go to ClinicalTrials.gov
2. Use **Advanced Search** and apply your filters (examples):
   - Condition/Disease (e.g., cardiovascular disease)
   - Study type
   - Status
   - **Results posted** (recommended if you want baseline demographics)
   - Date range (e.g., after 2017-01-18)
3. Run the search.
4. Export results and keep the **NCT Number** column.

### Important note on results
Baseline demographics are typically available only for trials with **posted results**. Trials without results may return missing demographics.

---

## 2) Prepare the input file (accepted formats + exact structure)

The app accepts:
- **CSV** (`.csv`)
- **Excel** (`.xlsx`)
- **Text** (`.txt`) — one NCT ID per line
- Or paste NCT IDs directly into the app

### Required structure (CSV/XLSX)
Create a file with **one column** containing NCT IDs. The header can be any of:
- `NCTId`
- `NCT ID`
- `NCT Number`
- `nct_id`
- `NCT`

Example CSV:

NCTId  
NCT01234567  
NCT07654321  
NCT00000000  

**Do not include URLs.**  
**Do not include commas/spaces inside the IDs.**

---

## 3) Run the web app (fastest path)

1. Open the app: https://tubbstrialfeatureextractor.streamlit.app
2. Upload your file (**CSV/XLSX/TXT**) OR paste NCT IDs into the text box
3. Select extraction domains (recommended: Race, Ethnicity, Sex/Gender, Age)
4. Click **Start Extraction**
5. Preview the results table in the app
6. Download the **.xlsx** output

---

## 4) Minimal working example (for reproducibility)

Use the included example file in this repo:

- `examples/minimal_nct_ids.csv`

Workflow:
1. Upload `examples/minimal_nct_ids.csv`
2. Select **Race, Ethnicity, Sex/Gender, Age**
3. Run extraction
4. Download the output `.xlsx`
5. Confirm output columns appear (examples):
   - `race_white_n`, `race_white_pct`
   - `ethnicity_hispanic_or_latino_n`, `ethnicity_hispanic_or_latino_pct`
   - `gender_female_n`, `gender_female_pct`
   - `age_mean`, `age_sd`, `age_min`, `age_max` (if reported)

Expected behavior:
- Some NCT IDs may return missing values if results are not posted or demographics are not reported.

---

## 5) Feature selection guidance (what to choose and why)

- **Race / Ethnicity / Sex-Gender:** Use these to assess representation patterns and reporting completeness.
- **Age:** Use summary stats to evaluate population coverage (mean, SD, min/max where available).
- **Study design / Eligibility:** Useful for downstream analyses linking protocol design to representation.

Recommended for first-time users:
- Select **Race, Ethnicity, Sex/Gender, Age** only.

---

## 6) Troubleshooting (common errors)

### A) “No demographics returned”
Most common causes:
- Trial has **no posted results**
- Baseline characteristics module is missing
- Demographics reported in an unexpected structure

What to do:
- Confirm the trial page shows a **Results** tab on ClinicalTrials.gov
- Try a smaller set of known-results trials first (use the minimal example)
- Re-run with API throttling enabled (if available in settings)

### B) “Invalid NCT ID format”
- Ensure each ID matches `NCT` + 8 digits (example: `NCT01234567`)
- Remove spaces, punctuation, or URLs

### C) “API timeout / rate limiting”
- Reduce batch size (e.g., 50–200 trials)
- Enable throttling / delays (if available)
- Re-run later if ClinicalTrials.gov is temporarily slow

### D) “Upload read error (CSV/XLSX)”
- Ensure the file is not password protected
- Ensure only one sheet (for XLSX) or that the first sheet contains IDs
- Ensure the NCT ID column has no mixed data types (no extra text)

---

## 7) Local installation (optional)

If you prefer to run locally:

1. Clone the repo:
   - `git clone https://github.com/tubbsa/trial-demographics-extractor.git`
2. Create environment and install dependencies:
   - `pip install -r requirements.txt`
3. Start the app:
   - `streamlit run app.py` (or your main entrypoint)

See `README.md` for full installation details.

---

## 8) Outputs (what you get)

The tool produces:
- A harmonized Excel file (`.xlsx`) with standardized column names
- Raw counts and computed percentages (where available)
- Optional error log export for transparency

---


