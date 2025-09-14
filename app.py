#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit — ClinicalTrials.gov  Baseline Demographics Extractor 
Exports ONLY: Age, Sex/Gender, Race, Ethnicity (from Results: Baseline Characteristics)
NOW WITH: user-selectable demographics (Age, Sex/Gender, Race, Ethnicity)

Key features:
- Exports ALL age data:
  - Continuous stats: mean, sd, median, min, max, iqr (plus fallback columns)
  - ALL categorical buckets as: agecat_<bucket>_n and/or agecat_<bucket>_pct
- Preserves value types (count vs percent vs continuous)
- Only the four demographics are parsed from ClinicalTrials.gov Results
- User can choose which demographics to include in the final Excel
"""

from __future__ import annotations

import io
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util import Retry  # for robust retries if available
except Exception:
    Retry = None

import streamlit as st

# =============================== Constants ===================================

API_STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{nct}?format=json"
NCT_RE = re.compile(r"^NCT\d{8}$", re.I)

# Keep ONLY baseline demographics measures
KEEP_RE = re.compile(r"(?i)\bage\b|\bsex\b|\bgender\b|\brace\b|\bethnic|\bhispanic|\blatino")

_DEF_OVERALL_LABELS = {"overall", "total", "all participants", "all-participants", "all"}
_DEF_UNIT_CANDIDATE_KEYS = ("units", "unit", "unitOfMeasure", "baselineMeasureUnit", "measureUnit")
_DEF_LABEL_KEYS = (
    "category", "categoryTitle", "baselineCategoryTitle",
    "classTitle", "baselineClassTitle", "groupTitle",
    "title", "measureTitle", "baselineMeasureTitle",
    "name", "label", "param", "dispersion", "statistic",
)
_DEF_VALUE_KEYS = (
    "count", "percentage", "percent", "proportion",
    "value", "measurementValue", "number",
    "mean", "median", "min", "max", "standardDeviation", "stdDev", "sd",
    "iqr", "interquartileRange", "lowerLimit", "upperLimit",
)

_RX_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Normalize common statistic keys
_STAT_NORM = {
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
    "standarddeviation": "sd", "stddev": "sd", "sd": "sd",
    "iqr": "iqr", "interquartilerange": "iqr",
    "lowerlimit": "lower", "upperlimit": "upper",
    "value": "value",   # generic fallback for continuous
    "count": "count", "number": "count",
    "percent": "percent", "percentage": "percent", "proportion": "percent",
}

# Canonical mapping for age buckets (best-effort)
_CAN_AGE = [
    (re.compile(r"(?i)^\s*<\s*18"), "lt_18_years"),
    (re.compile(r"(?i)\b0\s*[-–]\s*17\b"), "lt_18_years"),
    (re.compile(r"(?i)18\s*[-–]\s*6?5\b"), "18_to_65_years"),
    (re.compile(r"(?i)\b(?:≥|>=)\s*6?5\b|\b6?5\s*(and\s*older|plus|\+)\b"), "ge_65_years"),
    (re.compile(r"(?i)65\s*[-–]\s*74\b"), "65_to_74_years"),
    (re.compile(r"(?i)\b(?:≥|>=)\s*75\b|\b75\s*(and\s*older|plus|\+)\b"), "ge_75_years"),
    (re.compile(r"(?i)\bunder\s*18\b"), "lt_18_years"),
]


# ============================== HTTP Session =================================

def _make_session(total_retries: int = 5, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "ctgov-baseline-demographics-extractor/streamlit/1.1",
        "Accept": "application/json",
    })
    if Retry is not None:
        r = Retry(
            total=total_retries,
            connect=total_retries,
            read=total_retries,
            status=total_retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=r, pool_connections=16, pool_maxsize=32)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    return s

@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
    return _make_session()


# ============================== Utilities ====================================

def _num_like(x: Any) -> bool:
    if x is None: return False
    s = str(x).strip().replace(",", "")
    return bool(_RX_NUM.search(s))

def _to_float(x: Any) -> Optional[float]:
    if x is None: return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
        return float(x)
    s = str(x).strip().replace(",", "").rstrip("%")
    m = _RX_NUM.search(s)
    return float(m.group(0)) if m else None

def _merge_units(node: Dict[str, Any], unit_hint: Optional[str]) -> Optional[str]:
    for k in _DEF_UNIT_CANDIDATE_KEYS:
        if k in node and node[k]:
            return node[k]
    return unit_hint

def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s[:64]

def canonical_age_bucket(label: str) -> str:
    lab = label or ""
    for rx, tag in _CAN_AGE:
        if rx.search(lab):
            return tag
    return slugify(lab or "bucket")

def measure_kind(title: str) -> str:
    t = (title or "").lower()
    if re.search(r"\brace\b|race/ethnicity|racial", t): return "race"
    if re.search(r"\bethnic|hispanic|latino", t): return "ethnicity"
    if re.search(r"\bsex\b|\bgender\b|male|female", t): return "gender"
    if re.search(r"\bage\b", t): return "age"
    return "other"


# ============================= API + Normalize ===============================

@st.cache_data(show_spinner=False)
def fetch_study(nct_id: str, timeout: int = 60) -> Dict[str, Any]:
    session = get_http_session()
    url = API_STUDY_URL.format(nct=nct_id)
    r = session.get(url, timeout=timeout)
    # Respect Retry-After
    if r.status_code in (429, 503) and r.headers.get("Retry-After"):
        try:
            time.sleep(float(r.headers["Retry-After"]))
            r = session.get(url, timeout=timeout)
        except Exception:
            pass
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected payload for {nct_id}")
    # Normalize possible envelopes
    if isinstance(data.get("study"), dict): return data["study"]
    if isinstance(data.get("studies"), list) and data["studies"]: return data["studies"][0]
    return data


# ==================== Baseline (Results) demographics walker =================

def flatten_baseline_numbers_deep(study: Dict[str, Any], nct_id: str) -> pd.DataFrame:
    """Walk the Results -> Baseline Characteristics and extract demographics."""
    bc = (study.get("resultsSection") or {}).get("baselineCharacteristicsModule") or {}
    measures = (
        (bc.get("baselineMeasureList") or {}).get("baselineMeasure")
        or bc.get("baselineMeasures") or bc.get("measures") or []
    )
    pop = bc.get("baselinePopulationDescription") or bc.get("populationDescription")
    rows: List[Dict[str, Any]] = []

    def record(measure_title: str, label_path: List[str], val_key: str, val_raw: Any, unit_hint: Optional[str]):
        v = _to_float(val_raw)
        if v is None:
            return
        label_candidates = [
            x for x in label_path
            if x and str(x).strip().lower() not in _DEF_OVERALL_LABELS
        ]
        category = label_candidates[-1] if label_candidates else None

        vk = val_key.strip().lower().replace(" ", "")
        stat = _STAT_NORM.get(vk, "value")

        if stat == "count":
            vtype = "count"; unit = "participants"
        elif stat == "percent":
            vtype = "percent"; unit = "%"
        else:
            vtype = "continuous"; unit = unit_hint

        if not unit and "age" in (measure_title or "").lower():
            unit = "Years"

        rows.append({
            "nct_id": nct_id,
            "measure_title": measure_title,
            "measure": measure_kind(measure_title),
            "category": category,
            "value": v,
            "unit": unit,
            "value_type": vtype,     # count | percent | continuous
            "stat": stat,            # mean, median, sd, ...
            "label_path": " / ".join([str(x) for x in label_candidates if x]),
            "population_desc": pop,
        })

    def walk(node: Any, label_path: List[str], unit_hint: Optional[str], measure_title: str):
        if isinstance(node, dict):
            local_labels = list(label_path)
            for k in _DEF_LABEL_KEYS:
                if k in node and node[k] not in (None, ""):
                    local_labels.append(str(node[k]))
            u = _merge_units(node, unit_hint)
            for vk in _DEF_VALUE_KEYS:
                if vk in node and _num_like(node[vk]):
                    record(measure_title, local_labels, vk, node[vk], u)
            for _, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, local_labels, u, measure_title)
        elif isinstance(node, list):
            for it in node:
                walk(it, label_path, unit_hint, measure_title)

    for m in measures:
        mtitle = (m.get("title") or m.get("measureTitle") or m.get("baselineMeasureTitle") or m.get("name") or "")
        if not KEEP_RE.search(mtitle or ""):
            continue
        munit = (m.get("units") or m.get("unitOfMeasure") or m.get("baselineMeasureUnit") or m.get("measureUnit"))
        walk(m, [mtitle], munit, mtitle)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["measure"].isin(["age", "gender", "race", "ethnicity"])].copy()
        df = df.drop_duplicates(subset=["measure_title","category","value","value_type","stat"], keep="first")
        df = df.sort_values(["measure", "measure_title", "category", "stat"], kind="stable").reset_index(drop=True)
    return df


# ======================== Long -> Wide (aggregate) ===========================

def _aggregate_counts(long_df: pd.DataFrame, measure_name: str) -> Dict[str, float]:
    """Sum counts across arms/parts for a given measure (race/ethnicity/gender/agecat)."""
    sub = long_df[(long_df["measure"] == measure_name) & (long_df["value_type"] == "count")]
    if sub.empty:
        return {}
    g = sub.groupby(sub["category"].fillna("")).agg(total=("value", "sum"))
    return {k: float(v) for k, v in g["total"].items()}

def _first_percent_map(long_df: pd.DataFrame, measure_name: str) -> Dict[str, float]:
    """Take the first available percent per category (used only if counts unavailable)."""
    sub = long_df[(long_df["measure"] == measure_name) & (long_df["value_type"] == "percent")]
    out: Dict[str, float] = {}
    for _, r in sub.iterrows():
        cat = r.get("category") or ""
        if cat not in out:
            out[cat] = float(r["value"])
    return out

def make_export_wide_from_long(nct_id: str, long_df: pd.DataFrame) -> pd.Series:
    """
    Build a single wide row per NCT:
    - Age continuous: age_mean, age_sd, age_median, age_min, age_max, age_iqr, plus fallback
    - Age categorical: agecat_<bucket>_{n|pct}
    - Gender/Race/Ethnicity: <measure>_<category>_{n|pct}
    """
    out: Dict[str, Any] = {"nct_id": nct_id}

    # ------- AGE: continuous -------
    age_cont = long_df[(long_df["measure"] == "age") & (long_df["value_type"] == "continuous")]
    for stat_key, col in [
        ("mean", "age_mean"),
        ("sd", "age_sd"),
        ("median", "age_median"),
        ("min", "age_min"),
        ("max", "age_max"),
        ("iqr", "age_iqr"),
    ]:
        s = age_cont[age_cont["stat"].str.lower() == stat_key]
        if not s.empty:
            out[col] = float(s.iloc[0]["value"])
    for _, r in age_cont.iterrows():
        stat = (r.get("stat") or "").lower()
        if stat not in {"mean","sd","median","min","max","iqr"}:
            col = f"age_stat_{slugify(stat or (r.get('category') or 'value'))}"
            if col not in out:
                out[col] = float(r["value"])

    # ------- AGE: categorical (counts + pct) -------
    age_counts = _aggregate_counts(long_df[long_df["measure"] == "age"].copy(), "age")
    age_p1 = _first_percent_map(long_df[long_df["measure"] == "age"].copy(), "age")

    if age_counts:
        denom = sum(v for v in age_counts.values() if v is not None)
        for cat_raw, n in age_counts.items():
            bucket = canonical_age_bucket(cat_raw)
            out[f"agecat_{bucket}_n"] = float(n)
            if denom and denom > 0:
                out[f"agecat_{bucket}_pct"] = float(n) * 100.0 / float(denom)

    if age_p1:
        for cat_raw, pct in age_p1.items():
            bucket = canonical_age_bucket(cat_raw)
            col = f"agecat_{bucket}_pct"
            if col not in out:
                out[col] = float(pct)

    # ------- Gender / Race / Ethnicity -------
    for measure in ["gender", "race", "ethnicity"]:
        counts = _aggregate_counts(long_df, measure)
        p1 = _first_percent_map(long_df, measure)

        denom = sum(v for v in counts.values()) if counts else 0.0
        for cat_raw, n in counts.items():
            coln = f"{measure}_{slugify(cat_raw)}_n"
            out[coln] = float(n)
            if denom and denom > 0:
                out[f"{measure}_{slugify(cat_raw)}_pct"] = float(n) * 100.0 / float(denom)

        for cat_raw, pct in p1.items():
            colp = f"{measure}_{slugify(cat_raw)}_pct"
            if colp not in out:
                out[colp] = float(pct)

    for c in ["age_mean", "age_sd", "age_median"]:
        out.setdefault(c, None)

    return pd.Series(out)


# =============================== NCT parsing =================================

def _extract_ncts_from_series(s: pd.Series, allow_scan: bool) -> List[str]:
    vals = s.dropna().astype(str).str.upper()
    out: List[str] = []
    for v in vals:
        v_strip = v.strip()
        if NCT_RE.match(v_strip):
            out.append(v_strip)
        elif allow_scan:
            for tok in re.findall(r"NCT\d{8}", v_strip, flags=re.I):
                out.append(tok.upper())
    seen, uniq = set(), []
    for z in out:
        if z not in seen:
            seen.add(z); uniq.append(z)
    return uniq

def load_ncts_from_upload(uploaded, column: Optional[str], allow_scan: bool) -> List[str]:
    if uploaded is None:
        return []
    ext = os.path.splitext(uploaded.name or "")[1].lower()
    try:
        if ext in {".xlsx", ".xls"}:
            df = pd.read_excel(uploaded, dtype=str, engine="openpyxl")
        elif ext in {".csv", ".txt"}:
            df = pd.read_csv(uploaded, dtype=str, engine="python")
        elif ext == ".tsv":
            df = pd.read_csv(uploaded, dtype=str, sep="\t")
        else:
            df = pd.read_csv(uploaded, dtype=str, engine="python")
    except Exception:
        try:
            text = uploaded.read().decode("utf-8", errors="ignore")
            ids = re.findall(r"NCT\d{8}", text, flags=re.I)
            seen, uniq = set(), []
            for z in [i.upper() for i in ids]:
                if z not in seen:
                    seen.add(z); uniq.append(z)
            return uniq
        except Exception:
            return []

    if df is None or df.empty:
        return []

    if column:
        cols_ci = {str(c).lower(): c for c in df.columns}
        col_actual = cols_ci.get(column.lower())
        if col_actual is None:
            for c in df.columns:
                if str(c).strip().lower() == column.lower():
                    col_actual = c; break
        if col_actual is None:
            return []
        return _extract_ncts_from_series(df[col_actual], allow_scan)

    for c in df.columns:
        if re.search(r"\bnct(_?id)?\b", str(c), re.I):
            ids = _extract_ncts_from_series(df[c], allow_scan)
            if ids:
                return ids

    tokens: List[str] = []
    for c in df.columns:
        tokens.extend(_extract_ncts_from_series(df[c], allow_scan))
    seen, uniq = set(), []
    for z in tokens:
        if z not in seen:
            seen.add(z); uniq.append(z)
    return uniq


# ================================ Streamlit UI ===============================

st.set_page_config(page_title="CT.gov v2 — Baseline Demographics Extractor", page_icon="📊", layout="wide")
st.title("📊 ClinicalTrials.gov v2 — Baseline Demographics Extractor (Results)")
st.caption("Upload NCT IDs and download a wide Excel with selected demographics (Age, Sex/Gender, Race, Ethnicity).")

with st.sidebar:
    st.header("1) Input NCT IDs")
    uploaded = st.file_uploader("Upload CSV/XLSX/TSV/TXT with NCT IDs", type=["csv","xlsx","xls","tsv","txt"])
    colname = st.text_input("Column name to read (optional)", value="")
    allow_scan = st.checkbox("Scan other cells for NCT IDs if needed", value=True,
                             help="When off, only the specified column (or NCT-like column) is read.")
    st.divider()

    st.header("2) Demographics to export")
    demo_choices = ["Age", "Sex/Gender", "Race", "Ethnicity"]
    selected_demos = st.multiselect("Choose demographics", demo_choices, default=demo_choices,
                                    help="Only the selected domains will be included in the Excel export.")

    st.divider()
    st.header("3) Options")
    pasted = st.text_area("Or paste NCT IDs (one per line)", height=120, placeholder="NCT01234567\nNCT07654321")
    process_all = st.checkbox("Process all NCT IDs", value=True)
    with st.expander("Advanced", expanded=False):
        limit = 0 if process_all else st.number_input("Max NCTs", min_value=1, max_value=5000, value=200, step=1)
        sleep_sec = st.number_input("Politeness delay (sec)", min_value=0.0, max_value=5.0, value=0.25, step=0.05)
        timeout_sec = st.number_input("Request timeout (sec)", min_value=5, max_value=120, value=60, step=5)

    run_btn = st.button("Run Extraction", type="primary", use_container_width=True)

# Resolve NCT list
ncts_from_file: List[str] = load_ncts_from_upload(uploaded, colname or None, allow_scan)
ncts_from_paste: List[str] = []
if pasted.strip():
    lines = [ln.strip().upper() for ln in pasted.strip().splitlines()]
    ncts_from_paste = [ln for ln in lines if NCT_RE.match(ln)]

# Combine & de-dupe in order
seen = set()
resolved_ncts: List[str] = []
for n in (ncts_from_paste + ncts_from_file):
    n = n.upper().strip()
    if NCT_RE.match(n) and (n not in seen):
        seen.add(n)
        resolved_ncts.append(n)

if limit and limit > 0:
    resolved_ncts = resolved_ncts[: int(limit)]

st.write(f"**Resolved NCT IDs:** {len(resolved_ncts)}")
if resolved_ncts:
    st.code(", ".join(resolved_ncts[:10]) + (" ..." if len(resolved_ncts) > 10 else ""))

submitted = bool(run_btn)

if submitted and not resolved_ncts:
    st.info("To get started:\n1) Upload a file **or** paste NCT IDs\n2) Choose demographics\n3) Click **Run Extraction**")
    st.stop()


# ============================== Helpers (filter) =============================

def filter_by_selected_demos(df: pd.DataFrame, selected: List[str]) -> pd.DataFrame:
    """Keep only nct_id and columns matching the selected demographics."""
    if df.empty: 
        return df
    keep = ["nct_id"]
    cols = df.columns.tolist()

    selected = selected or []  # if user deselects all, show only nct_id

    if "Age" in selected:
        keep += [c for c in cols if c.startswith("age_") or c.startswith("agecat_")]
    if "Sex/Gender" in selected:
        keep += [c for c in cols if c.startswith("gender_")]
    if "Race" in selected:
        keep += [c for c in cols if c.startswith("race_")]
    if "Ethnicity" in selected:
        keep += [c for c in cols if c.startswith("ethnicity_")]

    keep = [c for c in keep if c in cols]
    # Put some common columns first if present
    core_first = [c for c in ["nct_id", "age_mean", "age_sd", "age_median"] if c in keep]
    others = [c for c in keep if c not in core_first]
    return df[core_first + others].copy()


# ================================== Run ======================================

def write_wide_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Export_Wide")
    buf.seek(0)
    return buf.getvalue()

if submitted:
    prog = st.progress(0)
    status = st.empty()
    wide_rows: List[pd.Series] = []
    errors: List[Tuple[str, str]] = []

    for i, nct in enumerate(resolved_ncts, 1):
        status.write(f"Fetching **{nct}** ({i}/{len(resolved_ncts)}) …")
        try:
            study = fetch_study(nct, timeout=int(timeout_sec))
            demo_df = flatten_baseline_numbers_deep(study, nct)
            if demo_df is None or demo_df.empty:
                raise RuntimeError("No baseline demographics found in Results")

            row = make_export_wide_from_long(nct, demo_df)
            row["nct_id"] = nct
            wide_rows.append(row)

        except Exception as e:
            errors.append((nct, str(e)))

        prog.progress(i / len(resolved_ncts))
        if sleep_sec and float(sleep_sec) > 0:
            time.sleep(float(sleep_sec))

    if not wide_rows:
        st.error("No rows produced. Check errors below or try a different set of NCT IDs.")
        if errors:
            with st.expander("Errors"):
                for nct, msg in errors[:200]:
                    st.write(f"**{nct}** — {msg}")
        st.stop()

    wide_df = pd.DataFrame(wide_rows)

    # Apply user-selected demographics filter
    filtered_df = filter_by_selected_demos(wide_df, selected_demos)

    st.success(f"Done. Extracted {len(filtered_df)} records.")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Processed", len(resolved_ncts))
    c2.metric("Successful", len(filtered_df))
    c3.metric("Failed", len(errors))

    # Download Excel (only selected demographics)
    xlsx_bytes = write_wide_excel(filtered_df)
    st.download_button(
        label="⬇️ Download Excel (Export_Wide)",
        data=xlsx_bytes,
        file_name="extracted_baseline_demographics_selected.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # Failures CSV (if any)
    if errors:
        err_df = pd.DataFrame(errors, columns=["nct_id", "error"])
        st.download_button(
            "⬇️ Download failures (.csv)",
            err_df.to_csv(index=False).encode(),
            file_name="failures.csv",
            use_container_width=True,
        )
        with st.expander("Some records failed (showing up to 200)"):
            for nct, msg in errors[:200]:
                st.write(f"**{nct}** — {msg}")
