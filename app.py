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
        # Return one blank row so make_export_wide still works
        return pd.DataFrame([{"nct_id": nct_id, "measure": None, "category": None, "value": None, "unit": None, "value_type": None}])
    demo = demo_df.copy()
    demo["measure"] = demo["measure_title"].map(measure_kind)
    # Filter only selected measures
    demo = demo[demo["measure"].isin(selected_measures)].copy()
    return demo[["nct_id", "measure", "category", "value", "unit", "value_type"]].reset_index(drop=True)

def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s[:64]

def make_export_wide(export_long: pd.DataFrame) -> pd.Series:
    """Convert long-format rows to a single wide-format row (one study)."""
    # Grab nct_id
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
            # For gender, race, ethnicity
            suffix = "_n" if vtype == "count" else ("_pct" if vtype == "percent" else "")
            col = f"{measure}_{slugify(category)}{suffix}"
            out[col] = value

    # Guarantee basic age keys present
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
    # Let user pick:
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

submitted = bool(run_btn)

# ==================== RESUMABLE PROCESSING WITH SESSION STATE ====================

# Initialize session state for resumable processing
if "processing_state" not in st.session_state:
    st.session_state.processing_state = {
        "ncts": [],
        "processed": set(),
        "results": [],
        "errors": [],
        "is_running": False,
    }

if submitted and not resolved_ncts:
    st.info(
        "To get started:\n"
        "1) Upload a file or paste NCT IDs\n"
        "2) Click **Run Extraction**"
    )
    st.stop()

if submitted:
    state = st.session_state.processing_state
    
    # Start fresh batch if NCTs changed
    if state["ncts"] != resolved_ncts:
        state["ncts"] = resolved_ncts
        state["processed"] = set()
        state["results"] = []
        state["errors"] = []
        state["is_running"] = True

    remaining_ncts = [n for n in state["ncts"] if n not in state["processed"]]
    total_ncts = len(state["ncts"])
    processed_count = len(state["processed"])

    # Display progress info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total NCT IDs", total_ncts)
    col2.metric("Processed", processed_count)
    col3.metric("Remaining", len(remaining_ncts))

    if not remaining_ncts:
        # All done
        state["is_running"] = False
        wide_rows = state["results"]
        errors = state["errors"]
    else:
        # Process a chunk (batch of 50 at a time to avoid long reruns)
        chunk_size = 50
        chunk = remaining_ncts[:chunk_size]

        prog = st.progress(processed_count / total_ncts)
        status = st.empty()

        for idx, nct in enumerate(chunk):
            current_count = processed_count + idx + 1
            progress_pct = current_count / total_ncts
            status.write(
                f"Fetching **{nct}** ({current_count}/{total_ncts}) …"
            )
            try:
                raw = fetch_study(nct, timeout=int(timeout_sec))
                study = normalize_study(raw)
                if not study:
                    raise RuntimeError("Empty or unknown study payload")

                demo_df = flatten_baseline_numbers_deep(study, nct)
                export_long = build_export_long(nct, demo_df, selected)
                row = make_export_wide(export_long)
                row["nct_id"] = nct
                state["results"].append(row)

            except Exception as e:
                state["errors"].append((nct, str(e)))

            state["processed"].add(nct)
            prog.progress(progress_pct)

            if sleep_sec and float(sleep_sec) > 0:
                time.sleep(float(sleep_sec))

        # Rerun to process next chunk
        if len(state["processed"]) < len(state["ncts"]):
            st.info(
                f"✅ Processed {len(state['processed'])}/{total_ncts} NCT IDs. "
                f"Rerunning to fetch next batch..."
            )
            time.sleep(1)
            st.rerun()
        else:
            state["is_running"] = False
            st.success(f"✅ Done! All {total_ncts} NCT IDs have been processed.")

        wide_rows = state["results"]
        errors = state["errors"]

    # Display results once processing completes
    if not state["is_running"]:
        if not wide_rows:
            st.error("No rows produced. Check errors or try a different set of NCT IDs.")
            if errors:
                with st.expander(f"Errors ({len(errors)} total)"):
                    for nct, msg in errors[:100]:
                        st.write(f"**{nct}** — {msg}")
        else:
            wide_df = pd.DataFrame(wide_rows)
            # Reorder so nct_id is first
            cols = wide_df.columns.tolist()
            if "nct_id" in cols:
                cols = ["nct_id"] + [c for c in cols if c != "nct_id"]
                wide_df = wide_df[cols]

            st.success(f"✅ Done. Extracted {len(wide_df)} records.")
            st.dataframe(wide_df.head(50), use_container_width=True)

            # Export to Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                wide_df.to_excel(writer, index=False, sheet_name="Demographics_Wide")
            buf.seek(0)
            st.download_button(
                label="⬇️ Download Excel (Demographics_Wide)",
                data=buf.getvalue(),
                file_name="demographics_extracted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            if errors:
                err_df = pd.DataFrame(errors, columns=["nct_id", "error"])
                st.download_button(
                    "⬇️ Download failures (.csv)",
                    err_df.to_csv(index=False).encode(),
                    file_name="failures.csv",
                    use_container_width=True,
                )
                with st.expander(f"Some records failed ({len(errors)} total, showing up to 100)"):
                    for nct, msg in errors[:100]:
                        st.write(f"**{nct}** — {msg}")
