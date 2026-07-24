#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import io
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

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

# Bounded category buckets so wide-format output doesn't explode into
# thousands of near-duplicate columns across ~7,900 trials with inconsistent
# label text from CT.gov. Anything unmatched falls back to a slugified
# version of the raw text so nothing is silently dropped.
RACE_BUCKETS = {
    "white": "white",
    "black or african american": "black_or_african_american",
    "black": "black_or_african_american",
    "african american": "black_or_african_american",
    "asian": "asian",
    "american indian or alaska native": "amind_alaska_native",
    "american indian": "amind_alaska_native",
    "alaska native": "amind_alaska_native",
    "native hawaiian or other pacific islander": "nhpi",
    "native hawaiian": "nhpi",
    "pacific islander": "nhpi",
    "more than one race": "multiracial",
    "two or more races": "multiracial",
    "multiracial": "multiracial",
    "unknown or not reported": "unknown",
    "not reported": "unknown",
    "unknown": "unknown",
}

ETHNICITY_BUCKETS = {
    "hispanic or latino": "hispanic_or_latino",
    "not hispanic or latino": "not_hispanic_or_latino",
    "unknown or not reported": "unknown",
    "not reported": "unknown",
    "unknown": "unknown",
}

GENDER_BUCKETS = {
    "male": "male",
    "female": "female",
    "unknown": "unknown",
}

# Output file for incremental writes. Lives in /tmp so it survives repeated
# st.rerun() calls within the same running process (does NOT survive a full
# container restart after an OOM kill -- see app notes at bottom of page).
PROGRESS_PATH = "/tmp/demographics_progress.jsonl"
ERRORS_PATH = "/tmp/demographics_errors.jsonl"

# Free-text fields (summaries, eligibility criteria) are truncated rather
# than dropped, so a handful of unusually long trials can't blow out a
# single Excel/CSV cell or balloon memory across ~7,900 trials.
_MAX_LONG_TEXT = 32000
_MAX_DETAILED_DESC = 6000

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

# ================= Utilities =================

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

def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s[:64]

def normalize_bucket(category: str, bucket_map: Dict[str, str]) -> str:
    c = (category or "").strip().lower()
    if c in bucket_map:
        return bucket_map[c]
    for key, bucket in bucket_map.items():
        if key in c:
            return bucket
    return slugify(category) if category else "unspecified"

def _join(items: Optional[List[Any]], sep: str = "; ", limit: Optional[int] = None) -> str:
    if not items:
        return ""
    vals = [str(x).strip() for x in items if x not in (None, "")]
    if limit:
        vals = vals[:limit]
    return sep.join(vals)

def _trunc(s: Any, n: int = _MAX_LONG_TEXT) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= n else s[:n] + " …[truncated]"

# ================= API (no long-lived caching of raw payloads) =================
# NOTE: fetch_study is intentionally NOT wrapped in @st.cache_data. Caching
# every raw CT.gov JSON payload for the life of the app session was the
# primary driver of the OOM crash on large batches (~7,900 full payloads
# held in memory simultaneously, on top of everything already collected).
# Each NCT is only fetched once per run anyway, so caching bought nothing
# here except memory pressure.

def fetch_study(session: requests.Session, nct_id: str, timeout: int = 60) -> Dict[str, Any]:
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

# ============== Full protocol/trial-info extractor ==============
# Pulls the "everything else" about a trial -- identification, status,
# sponsors, design, arms/interventions, outcomes, eligibility, and
# locations -- from protocolSection. This is independent of the results
# payload used for demographics, so it works even for trials that have no
# posted results yet (most trials on CT.gov have protocol data but not
# results data).

def _pipe_join(items: Optional[List[Any]], limit: Optional[int] = None) -> Optional[str]:
    """Join list-type fields with '|', matching the delimiter CT.gov itself
    uses in its own bulk CSV/Excel exports (Interventions, Collaborators,
    Locations, outcome measures, etc.) -- so downstream cleaning scripts
    written against that export format work unchanged against this
    column's values."""
    if not items:
        return None
    vals = [str(x).strip() for x in items if x not in (None, "")]
    if limit:
        vals = vals[:limit]
    return "|".join(vals) if vals else None

def _outcome_strs(outcome_list: List[Dict[str, Any]]) -> List[str]:
    # "Measure, Description, TimeFrame" per outcome -- same shape as CT.gov's
    # own export. Blank sub-fields are dropped rather than left as empty
    # slots between commas.
    out = []
    for o in outcome_list:
        parts = [o.get("measure"), o.get("description"), o.get("timeFrame")]
        parts = [p.strip() for p in parts if p not in (None, "")]
        if parts:
            out.append(", ".join(parts))
    return out

_WS_RE = re.compile(r"\s+")

def _flatten_ws(s: Any) -> Optional[str]:
    if s in (None, ""):
        return None
    return _WS_RE.sub(" ", str(s)).strip()

def split_eligibility_criteria(raw: Any) -> Dict[str, Optional[str]]:
    """Split CT.gov's single eligibilityCriteria blob into eligibility_text
    (whitespace-flattened full text), inclusion_text, and exclusion_text --
    matching the convention already used in cleaned_dataset.xlsx: the split
    is on the literal 'Exclusion Criteria:' heading, and inclusion_text
    keeps the leading ':' left over after 'Inclusion Criteria' is stripped
    out (not a bug -- this mirrors the existing cleaning script exactly so
    both pipelines produce identical values)."""
    flat = _flatten_ws(raw)
    if not flat:
        return {"eligibility_text": None, "inclusion_text": None, "exclusion_text": None}

    if "Exclusion Criteria:" in flat:
        inc_part, exc_part = flat.split("Exclusion Criteria:", 1)
        exclusion_text = exc_part.strip() or None
    else:
        inc_part, exclusion_text = flat, None

    inclusion_text = inc_part.replace("Inclusion Criteria", "").strip() or None

    return {
        "eligibility_text": flat,
        "inclusion_text": inclusion_text,
        "exclusion_text": exclusion_text,
    }

def extract_protocol_fields(study: Dict[str, Any]) -> Dict[str, Any]:
    ps = study.get("protocolSection") or {}

    ident = ps.get("identificationModule") or {}
    status = ps.get("statusModule") or {}
    sponsor = ps.get("sponsorCollaboratorsModule") or {}
    desc = ps.get("descriptionModule") or {}
    cond = ps.get("conditionsModule") or {}
    design = ps.get("designModule") or {}
    arms_mod = ps.get("armsInterventionsModule") or {}
    outcomes = ps.get("outcomesModule") or {}
    elig = ps.get("eligibilityModule") or {}
    contacts = ps.get("contactsLocationsModule") or {}

    lead_sponsor_info = sponsor.get("leadSponsor") or {}
    lead_sponsor = lead_sponsor_info.get("name")
    funder_type = lead_sponsor_info.get("class")
    collaborators = [c.get("name") for c in (sponsor.get("collaborators") or [])]

    design_info = design.get("designInfo") or {}
    enrollment = design.get("enrollmentInfo") or {}
    masking_info = design_info.get("maskingInfo") or {}

    arm_groups = arms_mod.get("armGroups") or []
    arm_strs = [
        f"{a.get('label', '')} ({a.get('type', '')})".strip()
        for a in arm_groups
        if a.get("label") or a.get("type")
    ]

    interventions = arms_mod.get("interventions") or []
    interv_strs = [
        f"{i.get('type', '')}: {i.get('name', '')}".strip(": ").strip()
        for i in interventions
        if i.get("name")
    ]

    primary_outcomes = outcomes.get("primaryOutcomes") or []
    secondary_outcomes = outcomes.get("secondaryOutcomes") or []

    locations = contacts.get("locations") or []
    loc_strs = [
        ", ".join(
            filter(None, [l.get("facility"), l.get("city"), l.get("state"), l.get("zip"), l.get("country")])
        )
        for l in locations
    ]

    # Single combined "Study Design" string, same shape as CT.gov's own export
    masking_val = masking_info.get("masking")
    who_masked = masking_info.get("whoMasked") or []
    if masking_val and who_masked:
        masking_display = f"{masking_val} ({', '.join(who_masked)})"
    else:
        masking_display = masking_val
    study_design = "|".join(
        f"{label}: {val}"
        for label, val in [
            ("Allocation", design_info.get("allocation")),
            ("Intervention Model", design_info.get("interventionModel")),
            ("Masking", masking_display),
            ("Primary Purpose", design_info.get("primaryPurpose")),
        ]
        if val
    ) or None

    elig_split = split_eligibility_criteria(elig.get("eligibilityCriteria"))

    return {
        # --- matches cleaned_dataset.xlsx column names/format exactly ---
        "Conditions": _pipe_join(cond.get("conditions")),
        "Interventions": _pipe_join(interv_strs),
        "Primary Outcome Measures": _pipe_join(_outcome_strs(primary_outcomes)),
        "Secondary Outcome Measures": _pipe_join(_outcome_strs(secondary_outcomes)),
        "Sponsor": lead_sponsor,
        "Collaborators": _pipe_join(collaborators),
        "Phases": _pipe_join(design.get("phases")),
        "Funder Type": funder_type,
        "Study Type": design.get("studyType"),
        "Study Design": study_design,
        "Locations": _pipe_join(loc_strs, limit=25),
        "eligibility_sex": elig.get("sex"),
        "eligibility_min_age": elig.get("minimumAge"),
        "eligibility_max_age": elig.get("maximumAge"),
        "eligibility_text": _trunc(elig_split["eligibility_text"]),
        "inclusion_text": _trunc(elig_split["inclusion_text"]),
        "exclusion_text": _trunc(elig_split["exclusion_text"]),
        # --- bonus fields not in cleaned_dataset.xlsx, kept as extras ---
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "acronym": ident.get("acronym"),
        "overall_status": status.get("overallStatus"),
        "why_stopped": status.get("whyStopped"),
        "start_date": (status.get("startDateStruct") or {}).get("date"),
        "primary_completion_date": (status.get("primaryCompletionDateStruct") or {}).get("date"),
        "completion_date": (status.get("completionDateStruct") or {}).get("date"),
        "study_first_posted": (status.get("studyFirstPostDateStruct") or {}).get("date"),
        "last_update_posted": (status.get("lastUpdatePostDateStruct") or {}).get("date"),
        "keywords": _join(cond.get("keywords")),
        "enrollment_count": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "arms": _join(arm_strs),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "locations_count": len(locations),
        "brief_summary": _trunc(desc.get("briefSummary")),
        "detailed_description": _trunc(desc.get("detailedDescription"), _MAX_DETAILED_DESC),
    }

# Column order for the trial-info block, so it reads naturally left-to-right
# regardless of dict insertion order coming back from json normalization.
# First block matches cleaned_dataset.xlsx's own column names/order; the
# rest are extras this script also happens to pull.
PROTOCOL_FIELD_ORDER = [
    "Conditions", "Interventions", "Primary Outcome Measures", "Secondary Outcome Measures",
    "Sponsor", "Collaborators", "Phases", "Funder Type", "Study Type", "Study Design", "Locations",
    "eligibility_sex", "eligibility_min_age", "eligibility_max_age",
    "eligibility_text", "inclusion_text", "exclusion_text",
    "brief_title", "official_title", "acronym", "overall_status", "why_stopped",
    "start_date", "primary_completion_date", "completion_date",
    "study_first_posted", "last_update_posted", "keywords",
    "enrollment_count", "enrollment_type", "arms", "healthy_volunteers",
    "locations_count", "brief_summary", "detailed_description",
]

# ============== Baseline demographics walker =============

def flatten_baseline_numbers_deep(study: Dict[str, Any], nct_id: str) -> List[Dict[str, Any]]:
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

    return rows

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

def build_demo_columns(demo_rows: List[Dict[str, Any]], selected_measures: List[str]) -> Dict[str, Any]:
    """Collapse one trial's demographic rows into bounded, normalized
    column names (so ~7,900 trials don't produce thousands of
    near-duplicate columns from inconsistent CT.gov label text)."""
    out: Dict[str, Any] = {}

    for r in demo_rows:
        measure = measure_kind(r.get("measure_title") or "")
        if measure not in selected_measures:
            continue
        category = str(r.get("category") or "")
        value = r.get("value")
        vtype = r.get("value_type")
        if value is None:
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
            bucket_map = {
                "race": RACE_BUCKETS,
                "ethnicity": ETHNICITY_BUCKETS,
                "gender": GENDER_BUCKETS,
            }.get(measure, {})
            norm_cat = normalize_bucket(category, bucket_map) if bucket_map else slugify(category)
            suffix = "_n" if vtype == "count" else ("_pct" if vtype == "percent" else "")
            col = f"{measure}_{norm_cat}{suffix}"
            out[col] = value

    if "age" in selected_measures:
        for c in ["age_mean", "age_sd", "age_median"]:
            out.setdefault(c, None)

    return out

def build_wide_row(
    nct_id: str,
    study: Dict[str, Any],
    selected_measures: List[str],
    include_trial_info: bool,
) -> Dict[str, Any]:
    """Assemble one output row: nct_id, then (optionally) the full trial-info
    block, then the demographic columns for the selected measures."""
    out: Dict[str, Any] = {"nct_id": nct_id}

    if include_trial_info:
        protocol_fields = extract_protocol_fields(study)
        for col in PROTOCOL_FIELD_ORDER:
            out[col] = protocol_fields.get(col)

    if selected_measures:
        demo_rows = flatten_baseline_numbers_deep(study, nct_id)
        out.update(build_demo_columns(demo_rows, selected_measures))

    return out

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

# ========== Incremental disk I/O ==========

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

def read_jsonl_ids(path: str) -> set:
    """Read back only the id field from a jsonl file -- cheap way to know
    what's already been processed without holding full rows in memory."""
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("nct_id") or rec.get("_nct_id")
                if key:
                    ids.add(key)
            except Exception:
                continue
    return ids

def jsonl_to_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)

def reset_progress_files():
    for p in (PROGRESS_PATH, ERRORS_PATH):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

# ================= Streamlit UI =================

st.set_page_config(page_title="CT.gov Demographics Selector", page_icon="📊", layout="wide")
st.title("📊 ClinicalTrials.gov — Trial & Demographics Extractor")
st.caption(
    "Upload NCT IDs, choose which fields to pull (full trial info and/or baseline "
    "demographics), then download Excel."
)

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
    st.header("2) Fields to Extract")

    trial_info_sel = st.checkbox(
        "Full trial info (title, status, sponsor, design, arms/interventions, "
        "outcomes, eligibility, locations, summary)",
        value=True,
        help="Pulled from protocolSection -- available even for trials with no posted results.",
    )

    st.markdown("**Baseline demographics** (from posted results, if available)")
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

    if not trial_info_sel and not selected:
        st.warning("Select at least one field group above.")

    st.divider()
    st.header("3) Options")
    process_all = st.checkbox("Process all NCT IDs", value=True)
    with st.expander("Advanced options", expanded=False):
        limit = 0 if process_all else st.number_input(
            "Max NCTs", min_value=1, max_value=8000, value=200, step=1
        )
        chunk_size = st.number_input(
            "Trials per batch before checkpoint", min_value=5, max_value=200, value=25, step=5,
            help="Smaller batches = more frequent disk checkpoints and lower peak memory.",
        )
        sleep_sec = st.number_input(
            "Politeness delay (sec)", min_value=0.0, max_value=5.0, value=0.25, step=0.05
        )
        timeout_sec = st.number_input(
            "Request timeout (sec)", min_value=5, max_value=120, value=60, step=5
        )
        fresh_start = st.checkbox(
            "Start fresh (clear any previous in-progress results)", value=False,
            help="Leave unchecked to resume a run that was interrupted earlier in this session.",
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

# ==================== SESSION STATE (lightweight -- ids only, not full rows) ====================

if "proc_state" not in st.session_state:
    st.session_state.proc_state = {
        "ncts": [],
        "running": False,
        "completed": False,
    }

# `submitted` gates the whole processing/results block. It must stay True
# after the run finishes (proc["completed"]) so that later reruns -- e.g.
# a user clicking one of the download buttons below, which Streamlit
# handles by rerunning the whole script -- still render the results
# instead of falling back to the "click Run Extraction" placeholder.
submitted = (
    run_btn
    or st.session_state.proc_state["running"]
    or st.session_state.proc_state.get("completed", False)
)

if submitted and not resolved_ncts:
    st.info("To get started:\n1) Upload a file or paste NCT IDs\n2) Click **Run Extraction**")
    st.stop()

if submitted and not trial_info_sel and not selected:
    st.error("Select at least one field group in the sidebar (trial info and/or demographics).")
    st.stop()

if submitted:
    proc = st.session_state.proc_state

    # New batch, or explicit fresh start requested
    if proc["ncts"] != resolved_ncts or (run_btn and fresh_start):
        proc["ncts"] = resolved_ncts
        proc["running"] = True
        proc["completed"] = False
        reset_progress_files()

    # What's already done, read straight from disk -- this is what makes
    # the run resumable across reruns within the same live process, and
    # keeps session_state itself tiny regardless of batch size.
    processed_ids = read_jsonl_ids(PROGRESS_PATH) | read_jsonl_ids(ERRORS_PATH)
    remaining = [n for n in proc["ncts"] if n not in processed_ids]
    total = len(proc["ncts"])
    done = total - len(remaining)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Processed", done)
    col3.metric("Remaining", len(remaining))

    # Always-available checkpoint download so partial progress is never
    # trapped in memory if something does go wrong before the run finishes.
    # Converted to CSV here (rather than handing back the raw .jsonl
    # checkpoint file) so it's directly openable in Excel/Sheets.
    if done > 0 and os.path.exists(PROGRESS_PATH):
        progress_df = jsonl_to_dataframe(PROGRESS_PATH)
        if not progress_df.empty:
            st.download_button(
                f"⬇️ Download progress so far ({done:,} of {total:,} trials)",
                progress_df.to_csv(index=False).encode(),
                "trials_progress.csv",
                "text/csv",
                use_container_width=True,
            )

    if remaining:
        session = get_http_session()
        chunk = remaining[: int(chunk_size)]
        prog = st.progress(done / total if total else 0.0)
        status = st.empty()

        for i, nct in enumerate(chunk):
            curr = done + i + 1
            status.write(f"Fetching **{nct}** ({curr}/{total}) …")
            try:
                raw = fetch_study(session, nct, timeout=int(timeout_sec))
                study = normalize_study(raw)
                if not study:
                    raise RuntimeError("Empty study")

                row_dict = build_wide_row(nct, study, selected, trial_info_sel)
                append_jsonl(PROGRESS_PATH, row_dict)

                # Drop references immediately rather than waiting on scope exit
                del raw, study, row_dict
            except Exception as e:
                append_jsonl(ERRORS_PATH, {"nct_id": nct, "error": str(e)})

            prog.progress(curr / total if total else 0.0)
            if sleep_sec > 0:
                time.sleep(float(sleep_sec))

        gc.collect()

        new_done = done + len(chunk)
        if new_done < total:
            st.info(f"✅ Checkpointed {new_done}/{total}. Continuing...")
            time.sleep(0.5)
            st.rerun()
        else:
            # Finished. Do NOT st.rerun() here -- on the rerun, run_btn is
            # False (it's only True on an actual button click) and
            # `running` is now False too, so `submitted` would evaluate to
            # False and the results section below would never render.
            # Instead, fall through in this same script execution straight
            # into the results/download block.
            proc["running"] = False
            proc["completed"] = True

    if not proc["running"]:
        st.divider()
        n_errors = len(read_jsonl_ids(ERRORS_PATH))
        wide_df = jsonl_to_dataframe(PROGRESS_PATH)

        if not wide_df.empty:
            cols = wide_df.columns.tolist()
            if "nct_id" in cols:
                cols = ["nct_id"] + [c for c in cols if c != "nct_id"]
                wide_df = wide_df[cols]

            st.success(f"✅ COMPLETE! Extracted {len(wide_df)} records from {total} NCT IDs.")
            st.write(f"**Dataframe shape: {wide_df.shape[0]} rows × {wide_df.shape[1]} columns**")

            st.info(f"Showing first 100 of {len(wide_df)} total records")
            st.dataframe(wide_df.head(100), use_container_width=True)
            st.caption(
                "Note: the ⭳ icon in the corner of this table only exports the "
                "100 rows shown above. Use the full-dataset download buttons below."
            )

            st.write("---")
            st.subheader("📥 Download Results")

            csv_data = wide_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download CSV ({:,} records)".format(len(wide_df)),
                csv_data,
                "trials_extracted.csv",
                "text/csv",
                use_container_width=True,
            )

            try:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    if len(wide_df) > 1_000_000:
                        for i in range(0, len(wide_df), 100_000):
                            chunk_df = wide_df.iloc[i:i + 100_000]
                            sheet_name = f"Data_{i // 100_000 + 1}"
                            chunk_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    else:
                        wide_df.to_excel(writer, index=False, sheet_name="Trials_Wide")
                buf.seek(0)
                st.download_button(
                    "⬇️ Download Excel ({:,} records)".format(len(wide_df)),
                    buf.getvalue(),
                    "trials_extracted.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception:
                st.warning("Excel export unavailable for this file size")

            if n_errors:
                err_df = jsonl_to_dataframe(ERRORS_PATH)
                st.write("---")
                st.warning(f"⚠️ {n_errors} NCT IDs failed")
                st.download_button(
                    f"⬇️ Download failures ({n_errors} records)",
                    err_df.to_csv(index=False).encode(),
                    "failures.csv",
                    use_container_width=True,
                )
                with st.expander(f"Show errors (up to 100 of {n_errors})"):
                    for _, r in err_df.head(100).iterrows():
                        st.write(f"**{r['nct_id']}** — {r['error']}")
        else:
            st.error("No records extracted.")
            if n_errors:
                err_df = jsonl_to_dataframe(ERRORS_PATH)
                with st.expander("Errors"):
                    for _, r in err_df.head(50).iterrows():
                        st.write(f"**{r['nct_id']}** — {r['error']}")
