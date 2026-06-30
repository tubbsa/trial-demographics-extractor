#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter

try:
    from urllib3.util import Retry
except Exception:
    Retry = None

# =============================== Constants ===================================

API_STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{nct}?format=json"
NCT_RE = re.compile(r"^NCT\d{8}$", re.I)

KEEP_RE = re.compile(r"(?i)\bage\b|\bsex\b|\bgender\b|\brace\b|\bethnic|\bhispanic|\blatino")

_DEF_LABEL_KEYS = (
    "category",
    "categoryTitle",
    "baselineCategoryTitle",
    "classTitle",
    "baselineClassTitle",
    "groupTitle",
    "title",
    "measureTitle",
    "baselineMeasureTitle",
    "name",
    "label",
    "param",
    "dispersion",
    "statistic",
)

_DEF_VALUE_KEYS = (
    "count",
    "percentage",
    "percent",
    "proportion",
    "value",
    "measurementValue",
    "number",
    "mean",
    "median",
    "min",
    "max",
    "standardDeviation",
    "stdDev",
    "sd",
    "iqr",
    "interquartileRange",
    "lowerLimit",
    "upperLimit",
)

_RX_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# ================= HTTP Session =================================

def _make_session(total_retries: int = 5, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "ctgov-demog-selector/streamlit/1.0",
            "Accept": "application/json",
        }
    )
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

# ================= Utilities =================================

def _num_like(x: Any) -> bool:
    if x is None:
        return False
    s = str(x).strip().replace(",", "")
    return bool(_RX_NUM.search(s))

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    s = str(x).strip().replace(",", "").rstrip("%")
    m = _RX_NUM.search(s)
    return float(m.group(0)) if m else None

# ================= API + Normalize =================

@st.cache_data(show_spinner=False)
def fetch_study(nct_id: str, timeout: int = 60) -> Dict[str, Any]:
    session = get_http_session()
    url = API_STUDY_URL.format(nct=nct_id)
    r = session.get(url, timeout=timeout)
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
    return data

def normalize_study(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data.get("study"), dict):
        return data["study"]
    if isinstance(data.get("studies"), list) and data["studies"]:
        return data["studies"][0]
    return data

# ============== Baseline demographics walker =============

def flatten_baseline_numbers_deep(study: Dict[str, Any], nct_id: str) -> pd.DataFrame:
    bc = (study.get("resultsSection") or {}).get("baselineCharacteristicsModule") or {}
    measures = (
        (bc.get("baselineMeasureList") or {}).get("baselineMeasure")
        or bc.get("baselineMeasures")
        or bc.get("measures")
        or []
    )
    rows: List[Dict[str, Any]] = []

    def record(
        measure_title: str,
        label_path: List[str],
        val_key: str,
        val_raw: Any,
        unit_hint: Optional[str],
    ):
        v = _to_float(val_raw)
        if v is None:
            return
        # filter out "overall / total / all" in labels
        label_candidates = [
            x
            for x in label_path
            if x
            and str(x).strip().lower()
            not in {"overall", "total", "all participants", "all", "all-participants"}
        ]
        category = label_candidates[-1] if label_candidates else None
        key_l = val_key.lower()
        if key_l in ("count", "number"):
            vtype = "count"
            unit = "participants"
        elif key_l in {"percentage", "percent", "proportion"}:
            vtype = "percent"
            unit = "%"
        else:
            vtype = "continuous"
            unit = unit_hint
        if not unit and "age" in (measure_title or "").lower():
            unit = "Years"
        rows.append(
            {
                "nct_id": nct_id,
                "measure_title": measure_title,
                "category": category,
                "value": v,
                "unit": unit,
                "value_type": vtype,
            }
        )

    def walk(node: Any, label_path: List[str], unit_hint: Optional[str], measure_title: str):
        if isinstance(node, dict):
            local_labels = list(label_path)
            for k in _DEF_LABEL_KEYS:
                if k in node and node[k] not in (None, ""):
                    local_labels.append(str(node[k]))
            u = node.get("units") or node.get("unitOfMeasure") or unit_hint
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
        mtitle = (
            m.get("title")
            or m.get("measureTitle")
            or m.get("baselineMeasureTitle")
            or m.get("name")
            or ""
        )
        if not KEEP_RE.search(mtitle):
            continue
        munit = m.get("units") or m.get("unitOfMeasure")
        walk(m, [mtitle], munit, mtitle)

    df = pd.DataFrame(rows, columns=["nct_id", "measure_title", "category", "value", "unit", "value_type"])
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["measure_title", "category", "value", "value_type"], keep="first")
    df = df.sort_values(["measure_title", "category"], kind="stable").reset_index(drop=True)
    return df

def measure_kind(title: str) -> str:
    t = (title or "").lower()
    if re.search(r"\brace\b|race/ethnicity|racial", t):
        return "race"
    if re.search(r"\bethnic|hispanic|latino", t):
        return "ethnicity"
    if re.search(r"\bsex\b|\bgender\b|male|female", t):
        return "gender"
    if re.search(r"\bage\b", t):
        return "age"
    return "other"

def build_export_long(nct_id: str, demo_df: pd.DataFrame, selected_measures: List[str]) -> pd.DataFrame:
    """Return a long-format DataFrame limited to selected measures."""
    if demo_df.empty:
        return pd.DataFrame([{"nct_id": nct_id, "measure": None, "category": None, "value": None, "unit": None, "value_type": None}])
    demo = demo_df.copy()
    demo["measure"] = demo["measure_title"].map(measure_kind)
    demo = demo[demo["measure"].isin(selected_measures)].copy()
    return demo[["nct_id", "measure", "category", "value", "unit", "value_type"]].reset_index(drop=True)

def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s[:64]

def make_export_wide(export_long: pd.DataFrame) -> pd.Series:
    """Convert long-format rows to a single wide-format row (one study)."""
    nct_id = export_long.iloc[0].get("nct_id") if not export_long.empty else None
    out: Dict[str, Any] = {"nct_id": nct_id}

    for _, r in export_long.iterrows():
        measure = str(r.get("measure") or "")
        category = str(r.get("category") or "")
        value = r.get("value")
        vtype = r.get("value_type")
        if pd.isna(value):
            continue
        if measure == "age":
            cat_lower = category.lower()
            if "mean" in cat_lower:
                col = "age_mean"
            elif "sd" in cat_lower or "std" in cat_lower:
                col = "age_sd"
            elif "median" in cat_lower:
                col = "age_median"
            elif re.search(r"\bmin", cat_lower):
                col = "age_min"
            elif re.search(r"\bmax", cat_lower):
                col = "age_max"
            elif "iqr" in cat_lower:
                col = "age_iqr"
            else:
                col = f"age_{slugify(category) if category else 'value'}"
            out[col] = value
        else:
            suffix = "_n" if vtype == "count" else ("_pct" if vtype == "percent" else "")
            col = f"{measure}_{slugify(category)}{suffix}"
            out[col] = value

    for c in ["age_mean", "age_sd", "age_median"]:
        out.setdefault(c, None)

    return pd.Series(out)

# ========== NCT ID loading ==========

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
    return out

def load_ncts_from_upload(uploaded, column: Optional[str], allow_scan: bool) -> List[str]:
    if uploaded is None:
        return []
    name = uploaded.name or ""
    ext = os.path.splitext(name)[1].lower()
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
            return list(dict.fromkeys(re.findall(r"NCT\d{8}", text, flags=re.I)))
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
                    col_actual = c
                    break
        if col_actual is None:
            return []
        return list(dict.fromkeys(_extract_ncts_from_series(df[col_actual], allow_scan)))

    for c in df.columns:
        if re.search(r"\bnct(_?id)?\b", str(c), re.I):
            ids = _extract_ncts_from_series(df[c], allow_scan)
            if ids:
                return list(dict.fromkeys(ids))

    tokens: List[str] = []
    for c in df.columns:
        tokens.extend(_extract_ncts_from_series(df[c], allow_scan))
    return list(dict.fromkeys(tokens))

# ================= Streamlit UI =================

st.set_page_config(page_title="CT.gov Demographics Selector", page_icon="📊", layout="wide")
st.title("📊 ClinicalTrials.gov — Demographics Selector")
st.caption("Upload NCT IDs, choose which features (Age, Gender, Race, Ethnicity), then download Excel.")

with st.sidebar:
    st.header("1) Input NCT IDs")
    uploaded = st.file_uploader(
        "Upload CSV / XLSX / TXT with NCT IDs",
        type=["csv", "xlsx", "xls", "txt", "tsv"],
    )
    colname = st.text_input("Column name to read (optional)", value="")
    allow_scan = st.checkbox(
        "Scan other cells for NCT IDs if needed",
        value=True,
        help="When off, only the specified column (or NCT-like column) is read.",
    )
    pasted = st.text_area(
        "Or paste NCT IDs (one per line)",
        height=120,
        placeholder="NCT01234567\nNCT07654321",
    )

    st.divider()
    st.header("2) Demographic Features to Extract")
    age_sel = st.checkbox("Age", value=True)
    gender_sel = st.checkbox("Gender", value=True)
    race_sel = st.checkbox("Race", value=True)
    ethnic_sel = st.checkbox("Ethnicity", value=True)
    selected = []
    if age_sel:
        selected.append("age")
    if gender_sel:
        selected.append("gender")
    if race_sel:
        selected.append("race")
    if ethnic_sel:
        selected.append("ethnicity")

    st.divider()
    st.header("3) Options")
    process_all = st.checkbox("Process all NCT IDs", value=True)
    with st.expander("Advanced options", expanded=False):
        limit = 0 if process_all else st.number_input(
            "Max NCTs", min_value=1, max_value=5000, value=200, step=1
        )
        sleep_sec = st.number_input(
            "Politeness delay (sec)", min_value=0.0, max_value=5.0, value=0.25, step=0.05
        )
        timeout_sec = st.number_input(
            "Request timeout (sec)", min_value=5, max_value=120, value=60, step=5
        )

    run_btn = st.button("Run Extraction", type="primary", use_container_width=True)

# Resolve NCTs
ncts_from_file = load_ncts_from_upload(uploaded, colname or None, allow_scan)
ncts_from_paste: List[str] = []
if pasted.strip():
    ncts_from_paste = [
        ln.strip().upper() for ln in pasted.strip().splitlines() if NCT_RE.match(ln.strip().upper())
    ]
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

# ==================== SESSION STATE FOR RESUMABLE PROCESSING ====================

if "proc_state" not in st.session_state:
    st.session_state.proc_state = {
        "ncts": [],
        "processed": set(),
        "results": [],
        "errors": [],
        "running": False,
    }

submitted = run_btn or st.session_state.proc_state["running"]

if submitted and not resolved_ncts:
    st.info("To get started:\n1) Upload a file or paste NCT IDs\n2) Click **Run Extraction**")
    st.stop()

if submitted:
    proc = st.session_state.proc_state

    # New batch?
    if proc["ncts"] != resolved_ncts:
        proc["ncts"] = resolved_ncts
        proc["processed"] = set()
        proc["results"] = []
        proc["errors"] = []
        proc["running"] = True

    remaining = [n for n in proc["ncts"] if n not in proc["processed"]]
    total = len(proc["ncts"])
    done = len(proc["processed"])

    # Show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Processed", done)
    col3.metric("Remaining", len(remaining))

    if remaining:
        # Process next 50
        chunk = remaining[:50]
        prog = st.progress(done / total)
        status = st.empty()

        for i, nct in enumerate(chunk):
            curr = done + i + 1
            status.write(f"Fetching **{nct}** ({curr}/{total}) …")
            try:
                raw = fetch_study(nct, timeout=int(timeout_sec))
                study = normalize_study(raw)
                if not study:
                    raise RuntimeError("Empty study")

                demo_df = flatten_baseline_numbers_deep(study, nct)
                export_long = build_export_long(nct, demo_df, selected)
                row = make_export_wide(export_long)
                row["nct_id"] = nct
                proc["results"].append(row)
            except Exception as e:
                proc["errors"].append((nct, str(e)))

            proc["processed"].add(nct)
            prog.progress(curr / total)
            if sleep_sec > 0:
                time.sleep(float(sleep_sec))

        # More to do?
        if len(proc["processed"]) < total:
            st.info(f"✅ Processed {done + len(chunk)}/{total}. Continuing...")
            time.sleep(1)
            st.rerun()
        else:
            proc["running"] = False

    if not proc["running"] and proc["results"]:
        # All done, show results
        st.divider()
        
        try:
            st.info("Converting results to Excel... (this may take a moment for large datasets)")
            
            # Build DataFrame safely
            wide_df = pd.DataFrame(proc["results"])
            cols = wide_df.columns.tolist()
            if "nct_id" in cols:
                cols = ["nct_id"] + [c for c in cols if c != "nct_id"]
                wide_df = wide_df[cols]

            st.success(f"✅ COMPLETE! Extracted {len(wide_df)} TOTAL records from {total} NCT IDs.")
            st.write(f"**Dataframe shape: {wide_df.shape[0]} rows × {wide_df.shape[1]} columns**")
            
            # Show sample instead of all rows (prevents browser freeze)
            with st.expander(f"View data sample (showing first 100 of {len(wide_df)})"):
                st.dataframe(wide_df.head(100), use_container_width=True)

            # Download Excel with error handling
            st.write("---")
            st.subheader("📥 Download Results")
            try:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    wide_df.to_excel(writer, index=False, sheet_name="Demographics_Wide")
                buf.seek(0)
                st.success(f"✅ Excel file ready: {len(wide_df)} records")
                st.download_button(
                    "⬇️ Download Excel ({} records)".format(len(wide_df)),
                    buf.getvalue(),
                    "demographics_extracted.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as excel_err:
                st.warning(f"Excel export failed: {str(excel_err)}")
                st.info("Downloading as CSV instead...")
                csv_data = wide_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download CSV ({} records)".format(len(wide_df)),
                    csv_data,
                    "demographics_extracted.csv",
                    "text/csv",
                    use_container_width=True,
                )

            # Download errors if any
            if proc["errors"]:
                err_df = pd.DataFrame(proc["errors"], columns=["nct_id", "error"])
                st.write("---")
                st.warning(f"⚠️ {len(proc['errors'])} NCT IDs failed to extract:")
                st.download_button(
                    "⬇️ Download failures ({} records)".format(len(proc['errors'])),
                    err_df.to_csv(index=False).encode(),
                    "failures.csv",
                    use_container_width=True,
                )
                with st.expander(f"Show all {len(proc['errors'])} errors"):
                    for nct, msg in proc["errors"]:
                        st.write(f"**{nct}** — {msg}")
                        
        except Exception as e:
            st.error(f"❌ Error processing results: {str(e)}")
            st.info(f"Successfully extracted {len(proc['results'])} records before error")
            if proc["results"]:
                st.write("Attempting to save as CSV fallback...")
                try:
                    wide_df = pd.DataFrame(proc["results"])
                    csv_data = wide_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download as CSV (fallback)",
                        csv_data,
                        "demographics_fallback.csv",
                        use_container_width=True,
                    )
                except Exception as fallback_err:
                    st.error(f"Fallback also failed: {str(fallback_err)}")
    elif not proc["running"] and not proc["results"]:
        st.error("No records extracted.")
        if proc["errors"]:
            with st.expander("Errors"):
                for nct, msg in proc["errors"][:50]:
                    st.write(f"**{nct}** — {msg}")
