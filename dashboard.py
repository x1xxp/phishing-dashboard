"""PhishGuard — explainable phishing URL risk assessment.

Run with: python -m streamlit run dashboard.py
This module intentionally leaves feature extraction and calibration in features.py
and results.json respectively.
"""

from __future__ import annotations

import html
import ipaddress
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import streamlit as st

# Guarantee that the sibling features.py is importable even when the interpreter
# was started from a different working directory.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402  (import follows the sys.path guard above)
    BANDED_FEATURES,
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    extract_raw_subscores,
    risk_category,
    weighted_score,
)


st.set_page_config(page_title="PhishGuard | URL Risk Assessment", page_icon="🛡️", layout="wide")

# Resolve every project file relative to this module, never to the terminal's
# current working directory, so the app runs from any location.
BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results.json"
DATA_DIR = BASE_DIR / "data"
MAX_URL_LENGTH = 2_048
MAX_BATCH_ROWS = 10_000

PROJECT_INFO = {
    "author": "Abdalla",
    "programme": "BSc (Hons) Computer Science",
    "supervisor": "Dr. Loubna",
    "academic_year": "2025/2026",
}

FEATURE_LABELS = {
    "ip_address": "IP address used as hostname",
    "punycode": 'Punycode ("xn--") present',
    "at_symbol": '"@" symbol present',
    "shortener": "Recognised URL-shortening service",
    "url_length": "Long URL structure",
    "subdomain": "Deep subdomain structure",
    "keyword": "Suspicious keyword(s) present",
    "special_pattern": "Misleading URL pattern(s)",
    "https_absent": "HTTPS absent",
    "digit_hyphen_ratio": "High digit or hyphen ratio",
}

FEATURE_EXPLANATIONS = {
    "ip_address": "The hostname is written as a numeric IP address instead of a domain.",
    "punycode": "The hostname contains an internationalised-domain encoding marker.",
    "at_symbol": "An @ sign can obscure the actual destination in a URL.",
    "shortener": "A recognised shortening service can conceal a final destination.",
    "url_length": "Long URLs can be used to hide suspicious components.",
    "subdomain": "Several subdomain levels can make a hostname harder to inspect.",
    "keyword": "The URL contains terms commonly used in credential-related lures.",
    "special_pattern": "The URL contains unusual separators, encoding, or repeated patterns.",
    "https_absent": "The submitted URL does not begin with HTTPS.",
    "digit_hyphen_ratio": "The URL contains an elevated proportion of digits or hyphens.",
}

# Risk colours are tuned for the dark canvas: high chroma, AA-legible on #070c16.
RISK_STYLES = {
    "low": {
        "label": "Low Risk", "color": "#34d399", "tint": "#34d399",
        "recommendation": "Few suspicious URL indicators were detected. Continue to verify the sender and domain before sharing sensitive information.",
    },
    "medium": {
        "label": "Medium Risk", "color": "#fbbf24", "tint": "#fbbf24",
        "recommendation": "Several suspicious indicators were detected. Avoid entering credentials until the website and sender have been independently verified.",
    },
    "high": {
        "label": "High Risk", "color": "#fb7185", "tint": "#fb7185",
        "recommendation": "Multiple strong phishing indicators were detected. Do not open the link, enter credentials, download files, or provide personal information.",
    },
}


@st.cache_data(show_spinner=False)
def load_results(path: str) -> dict[str, Any]:
    """Load the immutable calibration payload and validate its required fields."""
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    required = {"calibrated_weights", "calibrated_low_high", "calibrated_med_high", "test_metrics"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"results.json is missing: {', '.join(missing)}")
    return payload


def inject_css() -> None:
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
    /* ---- Design tokens ------------------------------------------------ */
    :root {
      --page:#070c16; --panel:#0d1524; --panel-2:#111c2e; --panel-3:#16233a;
      --line:rgba(148,180,220,.14); --line-strong:rgba(148,180,220,.26);
      --cyan:#5eead4; --cyan-strong:#2dd4bf; --cyan-dim:rgba(94,234,212,.12);
      --text:#eef4fb; --muted:#8ba0ba; --muted-2:#64748b;
      --r-sm:10px; --r-md:14px; --r-lg:18px;
      --shadow:0 1px 0 rgba(255,255,255,.04) inset, 0 18px 40px -24px rgba(0,0,0,.9);
      --font-display:"Space Grotesk", ui-sans-serif, system-ui, sans-serif;
      --font-body:"Inter", ui-sans-serif, system-ui, -apple-system, sans-serif;
      --font-mono:"JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
      --ease:cubic-bezier(.22,.61,.36,1);
    }

    /* ---- Canvas & typography ------------------------------------------ */
    html, body, .stApp { font-family:var(--font-body); }
    .stApp {
      color:var(--text);
      background:
        radial-gradient(900px 520px at 78% -18%, rgba(45,212,191,.10), transparent 60%),
        radial-gradient(700px 460px at 8% -10%, rgba(79,110,247,.10), transparent 62%),
        var(--page);
      background-attachment:fixed;
    }
    .block-container { max-width:1320px; padding:1.15rem 2rem 3.5rem; }
    header[data-testid="stHeader"] { background:transparent; }
    #MainMenu, footer { visibility:hidden; }
    h1,h2,h3,h4,h5 { font-family:var(--font-display); letter-spacing:-.02em; color:var(--text); }
    .stMarkdown h4 { font-size:1.02rem; font-weight:600; margin:1.9rem 0 .55rem; color:#dbe6f3; }
    .stMarkdown h4::before { content:""; display:inline-block; width:3px; height:.85em; margin-right:.55rem;
      border-radius:2px; background:var(--cyan-strong); transform:translateY(1px); }
    div[data-testid="stCaptionContainer"] p { color:var(--muted); font-size:.82rem; }

    /* ---- Top bar ------------------------------------------------------- */
    .pg-topbar { min-height:62px; display:flex; align-items:center; justify-content:space-between; gap:1rem;
      background:linear-gradient(180deg, rgba(23,36,58,.92), rgba(13,21,36,.92));
      border:1px solid var(--line); border-radius:var(--r-lg); padding:.7rem 1.05rem;
      box-shadow:var(--shadow); backdrop-filter:blur(10px); }
    .pg-brand { display:flex; align-items:center; gap:.8rem; min-width:0; }
    .pg-logo { width:38px; height:38px; flex:0 0 38px; display:grid; place-items:center; border-radius:12px;
      color:#04201d; background:linear-gradient(150deg,#7ff2e2,#22b8a6); font-size:1.1rem; font-weight:800;
      box-shadow:0 0 0 1px rgba(94,234,212,.35), 0 10px 24px -12px rgba(45,212,191,.85); }
    .pg-brand-name { font-family:var(--font-display); color:#fff; font-size:1.06rem; font-weight:700; line-height:1.1; letter-spacing:-.01em; }
    .pg-brand-sub { color:var(--muted); font-size:.735rem; margin-top:.18rem; letter-spacing:.01em; }
    .pg-status { display:flex; align-items:center; gap:.45rem; color:#9deed5; background:rgba(45,212,191,.08);
      border:1px solid rgba(45,212,191,.28); border-radius:999px; padding:.34rem .72rem; font-size:.74rem;
      font-weight:500; white-space:nowrap; }
    .pg-status-dot { width:7px; height:7px; border-radius:50%; background:#34d399;
      box-shadow:0 0 0 3px rgba(52,211,153,.14); animation:pg-pulse 2.4s var(--ease) infinite; }
    @keyframes pg-pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

    /* ---- Hero ---------------------------------------------------------- */
    .pg-intro { padding:2.1rem .15rem 1.1rem; }
    .pg-intro h1 { font-size:clamp(1.7rem,3.1vw,2.6rem); line-height:1.08; letter-spacing:-.035em; margin:0; color:#fff; }
    .pg-intro h1 em { font-style:normal; background:linear-gradient(100deg,#7ff2e2,#5eead4 45%,#8fb8ff);
      -webkit-background-clip:text; background-clip:text; color:transparent; }
    .pg-intro p { color:var(--muted); margin:.7rem 0 0; max-width:62ch; font-size:.95rem; line-height:1.6; }

    /* ---- Surfaces ------------------------------------------------------ */
    .pg-card, .pg-result, .pg-rule, .pg-academic {
      background:linear-gradient(158deg, var(--panel-2), var(--panel) 70%);
      border:1px solid var(--line); border-radius:var(--r-lg); box-shadow:var(--shadow); }
    .pg-card { padding:1.15rem 1.25rem; margin-bottom:.8rem; }
    .pg-scanner-label { color:var(--cyan); font-family:var(--font-mono); font-size:.68rem; letter-spacing:.18em;
      font-weight:500; text-transform:uppercase; margin-bottom:.35rem; }

    /* ---- Result cards --------------------------------------------------- */
    .pg-result { position:relative; overflow:hidden; padding:1.4rem 1.45rem; min-height:250px;
      border-color:color-mix(in srgb, var(--risk) 26%, var(--line));
      background:
        radial-gradient(520px 200px at 0% 0%, color-mix(in srgb, var(--risk) 14%, transparent), transparent 70%),
        linear-gradient(158deg, var(--panel-2), var(--panel) 72%); }
    .pg-result::before { content:""; position:absolute; inset:0 0 auto 0; height:2px;
      background:linear-gradient(90deg, var(--risk), transparent 78%); }
    .pg-score { font-family:var(--font-display); color:#fff; font-size:3.6rem; line-height:1; font-weight:600;
      letter-spacing:-.055em; margin:.55rem 0 .5rem; font-variant-numeric:tabular-nums; }
    .pg-score small { color:var(--muted); font-size:.8rem; letter-spacing:0; font-weight:500; }
    .pg-badge { display:inline-flex; align-items:center; gap:.4rem; border-radius:999px; padding:.3rem .7rem;
      color:var(--risk); background:color-mix(in srgb, var(--risk) 13%, transparent);
      border:1px solid color-mix(in srgb, var(--risk) 40%, transparent);
      font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.09em; }
    .pg-kicker { text-transform:uppercase; letter-spacing:.16em; color:var(--muted-2); font-family:var(--font-mono);
      font-size:.66rem; font-weight:500; }
    .pg-result-title { margin:0; color:var(--risk); font-size:1.25rem; }
    .pg-result-copy { color:#aebccd; margin:.7rem 0 0; font-size:.86rem; line-height:1.62; }
    .pg-scale { width:100%; height:8px; overflow:hidden; border-radius:999px;
      background:rgba(148,180,220,.12); margin:1rem 0 .4rem; }
    .pg-scale-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,
      color-mix(in srgb, var(--risk) 55%, transparent), var(--risk));
      box-shadow:0 0 14px -2px var(--risk); transition:width .6s var(--ease); }
    .pg-scale-label { display:flex; justify-content:space-between; color:var(--muted-2); font-size:.68rem;
      font-variant-numeric:tabular-nums; }
    .pg-factor { display:flex; align-items:center; justify-content:space-between; gap:.8rem; padding:.62rem 0;
      border-bottom:1px solid var(--line); }
    .pg-factor:last-child { border-bottom:0; }
    .pg-factor span { color:#cdd9e7; font-size:.84rem; }
    .pg-factor b { color:var(--cyan); font-family:var(--font-mono); font-size:.8rem; white-space:nowrap; }

    /* ---- Rule rows ------------------------------------------------------ */
    .pg-rule { padding:.85rem 1rem; margin-bottom:.55rem; border-left:2px solid var(--cyan-strong);
      border-radius:var(--r-md); box-shadow:none; transition:transform .18s var(--ease), border-color .18s var(--ease); }
    .pg-rule:hover { transform:translateX(2px); border-color:var(--line-strong); border-left-color:var(--cyan); }
    .pg-rule-top { display:flex; justify-content:space-between; gap:1rem; align-items:baseline; }
    .pg-rule-title { font-family:var(--font-display); font-weight:600; color:var(--text); font-size:.94rem; }
    .pg-rule-top b { color:var(--cyan); font-family:var(--font-mono); font-size:.78rem; font-weight:500; white-space:nowrap; }
    .pg-rule-meta { color:var(--muted); font-size:.79rem; margin-top:.22rem; line-height:1.5; }
    .pg-rule-bar { height:4px; background:rgba(148,180,220,.12); border-radius:999px; overflow:hidden; margin-top:.6rem; }
    .pg-rule-bar span { display:block; height:100%; border-radius:999px;
      background:linear-gradient(90deg, rgba(45,212,191,.5), var(--cyan)); }

    /* ---- Methodology flow ------------------------------------------------ */
    .pg-flow { display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:.6rem; align-items:stretch; }
    .pg-flow-step { position:relative; background:linear-gradient(158deg,var(--panel-2),var(--panel));
      border:1px solid var(--line); border-radius:var(--r-md); padding:.9rem;
      transition:border-color .18s var(--ease), transform .18s var(--ease); }
    .pg-flow-step:hover { border-color:var(--line-strong); transform:translateY(-2px); }
    .pg-flow-step:not(:last-child)::after { content:"→"; position:absolute; right:-.52rem; top:40%; z-index:2;
      color:var(--cyan-strong); font-size:.82rem; }
    .pg-flow-num { color:var(--cyan); font-family:var(--font-mono); font-weight:500; font-size:.7rem; letter-spacing:.1em; }
    .pg-flow-step b { font-family:var(--font-display); color:#fff; display:block; margin:.3rem 0 .18rem; font-size:.85rem; }
    .pg-flow-step span { color:var(--muted); font-size:.74rem; line-height:1.45; }

    /* ---- Streamlit primitives -------------------------------------------- */
    div[data-testid="stMetric"] { background:linear-gradient(158deg,var(--panel-2),var(--panel));
      border:1px solid var(--line); border-radius:var(--r-md); padding:.85rem .95rem; box-shadow:var(--shadow); }
    div[data-testid="stMetricLabel"] p { color:var(--muted); font-size:.76rem; font-weight:500; letter-spacing:.01em; }
    div[data-testid="stMetricValue"] { color:#fff; font-family:var(--font-display); letter-spacing:-.03em;
      font-variant-numeric:tabular-nums; }

    div[data-testid="stTabs"] [role="tablist"] { gap:.25rem; background:rgba(9,15,26,.75); border:1px solid var(--line);
      border-radius:var(--r-md); padding:.3rem; backdrop-filter:blur(8px); }
    div[data-testid="stTabs"] [role="tab"] { color:var(--muted); font-weight:500; border-radius:var(--r-sm);
      padding:.48rem .9rem; border:0; transition:color .16s var(--ease), background .16s var(--ease); }
    div[data-testid="stTabs"] [role="tab"] p { font-size:.85rem; font-weight:550; }
    div[data-testid="stTabs"] [role="tab"]:hover { color:var(--text); background:rgba(148,180,220,.06); }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background:linear-gradient(160deg,#7ff2e2,#2dd4bf);
      box-shadow:0 10px 22px -14px rgba(45,212,191,.9); }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color:#04201d; font-weight:650; }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] { display:none; }

    div[data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(158deg,var(--panel-2),var(--panel) 72%);
      border-color:var(--line); border-radius:var(--r-lg); box-shadow:var(--shadow); }
    div[data-testid="stFileUploader"] section { background:rgba(148,180,220,.03); border:1px dashed var(--line-strong);
      border-radius:var(--r-md); transition:border-color .18s var(--ease), background .18s var(--ease); }
    div[data-testid="stFileUploader"] section:hover { border-color:var(--cyan-strong); background:var(--cyan-dim); }

    div[data-testid="stTextInput"] input { background:rgba(6,11,20,.85); border:1px solid var(--line-strong);
      color:#fff; border-radius:var(--r-md); min-height:50px; font-size:.92rem;
      transition:border-color .16s var(--ease), box-shadow .16s var(--ease); }
    div[data-testid="stTextInput"] input::placeholder { color:var(--muted-2); }
    div[data-testid="stTextInput"] input:focus { border-color:var(--cyan-strong);
      box-shadow:0 0 0 3px rgba(45,212,191,.16); }

    .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] button {
      width:100%; border-radius:var(--r-md); font-weight:600; font-size:.87rem; white-space:nowrap; min-height:46px;
      border:1px solid var(--line-strong); background:rgba(148,180,220,.05); color:var(--text);
      transition:transform .14s var(--ease), background .16s var(--ease), border-color .16s var(--ease); }
    .stButton > button:hover, .stDownloadButton > button:hover { background:rgba(148,180,220,.11);
      border-color:var(--cyan-strong); color:#fff; }
    .stButton > button:active, .stDownloadButton > button:active { transform:translateY(1px); }
    .stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] {
      background:linear-gradient(160deg,#7ff2e2,#2dd4bf); color:#04201d; border-color:transparent;
      box-shadow:0 14px 30px -16px rgba(45,212,191,.95); }
    .stButton > button[kind="primary"]:hover { filter:brightness(1.06); color:#04201d; }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
      outline:2px solid var(--cyan); outline-offset:2px; }
    div[data-testid="stPopover"] button { width:auto; min-height:40px; }

    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:var(--r-md); overflow:hidden; }
    div[data-testid="stExpander"] details { background:rgba(148,180,220,.03); border:1px solid var(--line);
      border-radius:var(--r-md); }
    div[data-testid="stAlert"] { border-radius:var(--r-md); border:1px solid var(--line); }
    hr { border-color:var(--line); }
    ::selection { background:rgba(45,212,191,.28); }

    /* ---- Responsive ------------------------------------------------------ */
    @media(max-width:900px){
      .block-container{padding:1rem 1rem 2.5rem}
      .pg-flow{grid-template-columns:repeat(2,1fr)}
      .pg-flow-step::after{display:none}
      .pg-status{display:none}
      .pg-intro{padding:1.5rem .1rem .8rem}
    }
    @media(max-width:640px){
      div[data-testid="stHorizontalBlock"]:has(.pg-topbar){display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:.45rem}
      div[data-testid="stHorizontalBlock"]:has(.pg-topbar)>div[data-testid="stColumn"]{width:auto!important;flex:unset!important;min-width:0}
      .block-container{padding:.7rem .7rem 2rem}
      .pg-topbar{min-height:56px;padding:.6rem .75rem}
      .pg-brand-sub{display:none}
      .pg-intro{padding:1.15rem .1rem .6rem}
      .pg-score{font-size:2.9rem}
      .pg-result{padding:1.1rem;min-height:0}
      .pg-flow{grid-template-columns:1fr}
      .pg-rule-top{align-items:flex-start;flex-direction:column;gap:.25rem}
    }
    </style>
    """.replace("\n", " "), unsafe_allow_html=True)


def show_dataframe(data: Any, **kwargs: Any) -> None:
    st.dataframe(data, width="stretch", **kwargs)


def show_chart(chart: alt.Chart | alt.LayerChart) -> None:
    st.altair_chart(chart, width="stretch")


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def validate_url(raw_url: Any) -> tuple[bool, str, str]:
    """Validate URL text without normalising or visiting the submitted address."""
    if raw_url is None or (isinstance(raw_url, float) and pd.isna(raw_url)):
        return False, "Enter a URL to assess.", ""
    url = str(raw_url).strip()
    if not url:
        return False, "Enter a URL to assess.", ""
    if len(url) > MAX_URL_LENGTH:
        return False, f"URLs must be {MAX_URL_LENGTH:,} characters or fewer.", ""
    if any(char.isspace() for char in url):
        return False, "URLs cannot contain spaces or line breaks.", ""
    has_scheme = "://" in url
    try:
        parsed = urlparse(url if has_scheme else f"//{url}")
        _ = parsed.port
    except ValueError:
        return False, "This URL has an invalid port or malformed address.", ""
    if has_scheme and parsed.scheme.lower() not in {"http", "https"}:
        return False, "Use an HTTP or HTTPS URL.", ""
    if not parsed.hostname:
        return False, "Add a valid hostname, such as example.com.", ""
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname != "localhost" and "." not in parsed.hostname:
            return False, "Add a valid hostname, such as example.com.", ""
    note = "No scheme was supplied; the URL is scored exactly as entered." if not has_scheme else ""
    return True, "", note


def score_url(url: str, weights: dict[str, float], low_high: float, med_high: float) -> tuple[dict[str, float], float, str]:
    """Run the existing, frozen scoring pipeline."""
    subscores = extract_raw_subscores(url)
    score = float(weighted_score(subscores, weights))
    category = str(risk_category(score, low_high, med_high)).lower()
    if category not in RISK_STYLES:
        raise ValueError("The scoring pipeline returned an unknown risk category.")
    return subscores, score, category


def build_triggered_rules(subscores: dict[str, float], weights: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in FEATURE_NAMES:
        raw = float(subscores.get(name, 0))
        weight = float(weights.get(name, 0))
        contribution = weight * (raw / 2) if name in BANDED_FEATURES else weight * raw
        if raw > 0:
            rows.append({"Feature": name, "Rule": feature_label(name), "Explanation": FEATURE_EXPLANATIONS[name], "Raw sub-score": raw, "Weight": weight, "Contribution": round(contribution, 3)})
    return pd.DataFrame(rows).sort_values("Contribution", ascending=False, ignore_index=True) if rows else pd.DataFrame(columns=["Feature", "Rule", "Explanation", "Raw sub-score", "Weight", "Contribution"])


def maximum_score(weights: dict[str, float]) -> float:
    return sum(max(float(value), 0) for value in weights.values()) or 1.0


def find_column(df: pd.DataFrame, wanted: str) -> Any | None:
    """Find a CSV column case-insensitively while preserving its original key."""
    return next(
        (column for column in df.columns if str(column).strip().lower() == wanted.lower()),
        None,
    )


def normalize_label(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    label = str(value).strip().lower()
    if label in {"0", "phishing", "phish", "malicious", "unsafe", "bad"}:
        return 0
    if label in {"1", "legitimate", "benign", "safe", "good"}:
        return 1
    return None


def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"precision": precision, "recall": recall, "specificity": specificity, "accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "balanced_accuracy": (recall + specificity) / 2}


def analyse_batch(source: pd.DataFrame, weights: dict[str, float], low_high: float, med_high: float) -> pd.DataFrame:
    url_column = find_column(source, "url")
    if url_column is None:
        raise ValueError("The CSV must contain a column named 'url'.")
    output = source.copy().rename(columns={url_column: "url"}) if url_column != "url" else source.copy()
    records = []
    for value in output["url"]:
        url = "" if pd.isna(value) else str(value).strip()
        valid, error, note = validate_url(url)
        record: dict[str, Any] = {"url_valid": valid, "validation_message": note if valid else error, "risk_score": None, "risk_category": "invalid", "rules_triggered": None, "top_reason": None}
        if valid:
            try:
                subscores, score, category = score_url(url, weights, low_high, med_high)
                rules = build_triggered_rules(subscores, weights)
                record.update({"risk_score": round(score, 3), "risk_category": category, "rules_triggered": int((pd.Series(subscores) > 0).sum()), "top_reason": rules.iloc[0]["Rule"] if not rules.empty else "No rule triggered"})
            except Exception as error:  # Keep one bad row from terminating a report.
                record.update({"url_valid": False, "validation_message": f"Scoring failed: {error}"})
        records.append(record)
    output = pd.concat([output.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    label_column = find_column(output, "label")
    if label_column is not None:
        output["label_normalized"] = output[label_column].map(normalize_label)
    return output


def batch_metrics(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]] | None:
    if "label_normalized" not in df:
        return None
    evaluated = df[df["url_valid"] & df["label_normalized"].isin([0, 1])].copy()
    if evaluated.empty:
        return None
    predicted = evaluated["risk_category"].isin(["medium", "high"])
    actual = evaluated["label_normalized"] == 0
    counts = {"tp": int((predicted & actual).sum()), "fp": int((predicted & ~actual).sum()), "fn": int((~predicted & actual).sum()), "tn": int((~predicted & ~actual).sum()), "n": len(evaluated)}
    return counts, calculate_metrics(counts["tp"], counts["fp"], counts["fn"], counts["tn"])


def distribution_chart(counts: pd.Series) -> alt.Chart:
    data = pd.DataFrame({"Risk": ["Low", "Medium", "High"], "Count": [int(counts.get("low", 0)), int(counts.get("medium", 0)), int(counts.get("high", 0))], "Color": ["#34d399", "#fbbf24", "#fb7185"]})
    upper = max(int(data["Count"].max()), 1)
    return alt.Chart(data).mark_bar(cornerRadiusEnd=5).encode(x=alt.X("Count:Q", title="URLs", scale=alt.Scale(domain=[0, upper])), y=alt.Y("Risk:N", sort=["Low", "Medium", "High"], title=None), color=alt.Color("Color:N", scale=None, legend=None), tooltip=["Risk", "Count"]).properties(height=180).configure_view(strokeWidth=0).configure_axis(labelColor="#8ba0ba", titleColor="#8ba0ba", gridColor="rgba(148,180,220,.10)", domainColor="rgba(148,180,220,.18)", tickColor="rgba(148,180,220,.18)", labelFont="Inter", titleFont="Inter").configure(background="transparent")


def confusion_chart(tp: int, fp: int, fn: int, tn: int) -> alt.LayerChart:
    data = pd.DataFrame([{"Actual": "Phishing", "Prediction": "Phishing", "Count": tp}, {"Actual": "Phishing", "Prediction": "Legitimate", "Count": fn}, {"Actual": "Legitimate", "Prediction": "Phishing", "Count": fp}, {"Actual": "Legitimate", "Prediction": "Legitimate", "Count": tn}])
    base = alt.Chart(data).encode(x=alt.X("Prediction:N", sort=["Phishing", "Legitimate"], title="Predicted"), y=alt.Y("Actual:N", sort=["Phishing", "Legitimate"], title="Actual"))
    upper = max(tp, fp, fn, tn, 1)
    return (base.mark_rect(cornerRadius=4).encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="teals", domain=[0, upper]), legend=None), tooltip=["Actual", "Prediction", "Count"]) + base.mark_text(fontSize=18, fontWeight=700).encode(text="Count:Q", color=alt.condition("datum.Count > 0", alt.value("#ffffff"), alt.value("#8ba0ba")))).properties(height=250).configure_view(strokeWidth=0).configure_axis(labelColor="#8ba0ba", titleColor="#8ba0ba", domainColor="rgba(148,180,220,.18)", tickColor="rgba(148,180,220,.18)", grid=False, labelFont="Inter", titleFont="Inter").configure(background="transparent")


def render_header(
    calibrated_weights: dict[str, float], low_high: float, med_high: float
) -> tuple[dict[str, float], str]:
    """Render the product bar and return the compact settings selection."""
    brand, settings = st.columns([7, 1], vertical_alignment="center")
    with brand:
        st.markdown(
            """<div class="pg-topbar"><div class="pg-brand"><div class="pg-logo">◇</div><div><div class="pg-brand-name">PhishGuard</div><div class="pg-brand-sub">Explainable URL Risk Analysis</div></div></div><div class="pg-status"><span class="pg-status-dot"></span>Analysis engine ready</div></div>""",
            unsafe_allow_html=True,
        )
    with settings:
        with st.popover("Settings"):
            profile = st.radio(
                "Weight profile",
                ["Calibrated (recommended)", "Default / uncalibrated"],
                help="Calibrated weights are the frozen values from the project evaluation.",
            )
            active_weights = DEFAULT_WEIGHTS if profile.startswith("Default") else calibrated_weights
            active_profile = "Default" if profile.startswith("Default") else "Calibrated"
            st.caption(
                f"Low ≤ {low_high:g} · Medium ≤ {med_high:g} · High > {med_high:g}"
            )
            with st.expander("Active weights"):
                show_dataframe(
                    pd.DataFrame(
                        [
                            {"Feature": feature_label(name), "Weight": active_weights.get(name, 0)}
                            for name in FEATURE_NAMES
                        ]
                    ),
                    hide_index=True,
                )
    st.markdown(
        """<div class="pg-intro"><h1>Analyse suspicious URLs <em>before</em> you trust them.</h1><p>Explainable phishing risk assessment based on URL structure. The system does not visit submitted websites.</p></div>""",
        unsafe_allow_html=True,
    )
    return active_weights, active_profile


def render_risk_result(url: str, score: float, category: str, rules: pd.DataFrame, weights: dict[str, float], low_high: float, med_high: float, profile: str) -> None:
    style = RISK_STYLES[category]
    safe_url = html.escape(url)
    percent = min(max(score / maximum_score(weights) * 100, 0), 100)
    left, right = st.columns([1.05, 1], gap="medium")
    with left:
        st.markdown(
            f"""<div class="pg-result" style="--risk:{style['color']};--risk-bg:{style['tint']}15"><div class="pg-kicker">Risk assessment</div><div class="pg-score">{score:.2f} <small>/ {maximum_score(weights):g}</small></div><span class="pg-badge">{style['label']}</span><div class="pg-scale"><div class="pg-scale-fill" style="width:{percent:.1f}%"></div></div><div class="pg-scale-label"><span>0</span><span>Low ≤ {low_high:g}</span><span>Medium ≤ {med_high:g}</span><span>{maximum_score(weights):g}</span></div><p class="pg-result-copy">{style['recommendation']}</p></div>""",
            unsafe_allow_html=True,
        )
    with right:
        factors = "".join(
            f"<div class='pg-factor'><span>{html.escape(str(row['Rule']))}</span><b>+{float(row['Contribution']):.2f}</b></div>"
            for _, row in rules.head(3).iterrows()
        ) or "<div class='pg-factor'><span>No suspicious indicators triggered</span><b>Clear</b></div>"
        st.markdown(
            f"""<div class="pg-result" style="--risk:{style['color']};--risk-bg:{style['tint']}0b"><div class="pg-kicker">Detection summary</div><h3 style="color:#fff;margin:.45rem 0 0;font-size:1.5rem;letter-spacing:-.03em">{len(rules)} triggered rule{'s' if len(rules) != 1 else ''}</h3><p class="pg-result-copy" style="margin:.25rem 0 .55rem">Top contributors · {html.escape(profile)} weights</p>{factors}</div>""",
            unsafe_allow_html=True,
        )
    st.markdown(f"<div style='margin-top:.7rem;color:#64748b;font-size:.75rem'>Analysed URL · <code style=\'color:#8ba0ba;background:rgba(148,180,220,.07);padding:.15rem .4rem;border-radius:6px;word-break:break-all\'>{safe_url}</code></div>", unsafe_allow_html=True)


def render_rules(rules: pd.DataFrame) -> None:
    if rules.empty:
        st.success("No suspicious rules were triggered for this URL.")
        return
    highest = float(rules["Contribution"].max()) or 1.0
    for _, rule in rules.iterrows():
        share = float(rule["Contribution"]) / highest * 100
        st.markdown(f"""<div class="pg-rule"><div class="pg-rule-top"><span class="pg-rule-title">{html.escape(str(rule['Rule']))}</span><b>{float(rule['Contribution']):.2f} contribution</b></div><div class="pg-rule-meta">{html.escape(str(rule['Explanation']))}</div><div class="pg-rule-meta">Raw sub-score: {float(rule['Raw sub-score']):g} · Weight: {float(rule['Weight']):g}</div><div class="pg-rule-bar"><span style="width:{share:.1f}%"></span></div></div>""", unsafe_allow_html=True)


def render_single_url_page(weights: dict[str, float], profile: str, low_high: float, med_high: float) -> None:
    st.markdown("#### URL Scanner")
    if "single_result" not in st.session_state:
        st.session_state.single_result = None
    if "scanner_url" not in st.session_state:
        st.session_state.scanner_url = ""
    with st.container(border=True):
        st.markdown("<div class='pg-scanner-label'>URL SECURITY SCANNER</div><span style='color:var(--muted);font-size:.85rem;line-height:1.55'>Paste a complete URL or domain below. Analysis runs only when you press the button.</span>", unsafe_allow_html=True)
        url = st.text_input(
            "URL to assess",
            key="scanner_url",
            label_visibility="collapsed",
            placeholder="Paste or enter a URL here",
            help="HTTP and HTTPS URLs are accepted. A URL without a scheme is scored as entered.",
        )
        analyse_col, clear_col = st.columns([5, 1])
        submitted = analyse_col.button(
            "Analyse URL", type="primary", key="analyse_url", width="stretch"
        )
        clear = clear_col.button("Clear", key="clear_url", width="stretch")
    if clear:
        st.session_state.single_result = None
        st.session_state.scanner_url = ""
        st.rerun()
    if submitted:
        valid, error, note = validate_url(url)
        if not valid:
            st.session_state.single_result = None
            st.error(error)
        else:
            subscores, score, category = score_url(url.strip(), weights, low_high, med_high)
            st.session_state.single_result = {"url": url.strip(), "subscores": subscores, "score": score, "category": category, "profile": profile, "note": note}
    result = st.session_state.single_result
    if not result:
        st.info("URL features are extracted locally, combined with frozen weights, and mapped to a risk category using stored thresholds.")
        return
    if result["profile"] != profile:
        st.info(f"This result uses the {result['profile']} profile. Analyze again to apply the {profile} profile.")
    if result["note"]:
        st.info(result["note"])
    rules = build_triggered_rules(result["subscores"], weights)
    render_risk_result(result["url"], result["score"], result["category"], rules, weights, low_high, med_high, result["profile"])
    st.markdown("#### Full explainability")
    render_rules(rules)
    with st.expander("View detailed scoring table"):
        show_dataframe(rules.drop(columns=["Feature", "Explanation"]), hide_index=True, column_config={"Contribution": st.column_config.NumberColumn(format="%.3f"), "Weight": st.column_config.NumberColumn(format="%.3f")})
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "url": result["url"], "weight_profile": result["profile"], "risk_score": round(result["score"], 6), "risk_category": result["category"], "thresholds": {"low_high": low_high, "medium_high": med_high}, "subscores": result["subscores"], "triggered_rules": rules.drop(columns=["Explanation"]).to_dict(orient="records"), "scope": "URL string analysis only; no website was opened."}
    st.download_button("Download assessment JSON", json.dumps(report, indent=2).encode("utf-8"), "url_risk_assessment.json", "application/json")


def render_batch_page(weights: dict[str, float], profile: str, low_high: float, med_high: float) -> None:
    st.subheader("Batch Analysis")
    st.caption("Upload a CSV with a required `url` column and an optional `label` column (0 = phishing, 1 = legitimate).")
    sample = pd.DataFrame({"url": ["https://example.com", "http://verify-account.example.net/login"], "label": [1, 0]})
    st.download_button("Download sample CSV", sample.to_csv(index=False).encode("utf-8"), "phishguard_sample.csv", "text/csv")
    uploaded = st.file_uploader("Drop a CSV here", type="csv", key="batch_upload")
    if "batch_result" not in st.session_state:
        st.session_state.batch_result = None
    source: pd.DataFrame | None = None
    if uploaded:
        try:
            uploaded.seek(0)
            source = pd.read_csv(uploaded)
            if source.empty:
                raise ValueError("The uploaded CSV is empty.")
            if len(source) > MAX_BATCH_ROWS:
                raise ValueError(f"The CSV exceeds the {MAX_BATCH_ROWS:,}-row limit.")
            if find_column(source, "url") is None:
                raise ValueError("The CSV must contain a column named 'url'.")
            valid_preview = source[find_column(source, "url")].map(lambda item: validate_url(item)[0])
            duplicate_count = int(source[find_column(source, "url")].duplicated().sum())
            x, y, z = st.columns(3)
            x.metric("Rows", f"{len(source):,}")
            y.metric("Valid in preview", f"{int(valid_preview.sum()):,}")
            z.metric("Duplicate URLs", f"{duplicate_count:,}")
            with st.expander("Preview uploaded data", expanded=True):
                show_dataframe(source.head(25), hide_index=True)
            if st.button("Analyze CSV", type="primary"):
                with st.status("Analyzing URL structure…", expanded=False) as status:
                    st.session_state.batch_result = analyse_batch(source, weights, low_high, med_high)
                    st.session_state.batch_profile = profile
                    status.update(label="Batch analysis complete", state="complete")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.session_state.batch_result = None
            st.error(str(error))
    result = st.session_state.batch_result
    if result is None:
        return
    if st.session_state.get("batch_profile") != profile:
        st.info(f"These results use the {st.session_state.batch_profile} profile. Analyze the CSV again to use {profile}.")
    valid = int(result["url_valid"].sum())
    invalid = len(result) - valid
    counts = result["risk_category"].value_counts()
    a, b, c, d, e = st.columns(5)
    a.metric("Total URLs", f"{len(result):,}")
    b.metric("Valid", f"{valid:,}")
    c.metric("Low risk", f"{int(counts.get('low', 0)):,}")
    d.metric("Medium risk", f"{int(counts.get('medium', 0)):,}")
    e.metric("High risk", f"{int(counts.get('high', 0)):,}")
    if invalid:
        st.caption(f"{invalid:,} row(s) could not be analysed because their URL text was invalid.")
    st.markdown("#### Risk distribution")
    show_chart(distribution_chart(counts))
    analysed = max(valid, 1)
    st.caption(" · ".join(f"{name.title()}: {int(counts.get(name, 0)) / analysed:.1%}" for name in ["low", "medium", "high"]))
    metrics_result = batch_metrics(result)
    if metrics_result:
        metric_counts, metrics = metrics_result
        st.markdown("#### Evaluation against supplied labels")
        st.caption("Medium and high risk are counted as a phishing prediction. Only valid, labelled rows are included.")
        columns = st.columns(6)
        for column, (label, key) in zip(columns, [("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1"), ("Accuracy", "accuracy"), ("Specificity", "specificity"), ("Balanced accuracy", "balanced_accuracy")]):
            column.metric(label, f"{metrics[key]:.3f}")
        show_chart(confusion_chart(metric_counts["tp"], metric_counts["fp"], metric_counts["fn"], metric_counts["tn"]))
    elif "label" in [str(c).lower() for c in result.columns]:
        st.info("A label column was found, but it contains no usable 0/1 or phishing/legitimate values.")
    st.markdown("#### Results")
    search = st.text_input("Filter results", placeholder="Search a URL, category, or rule")
    display = result.copy()
    if search:
        mask = display.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
        display = display[mask]
    columns = [column for column in ["url", "url_valid", "risk_score", "risk_category", "rules_triggered", "top_reason", "validation_message", "label_normalized"] if column in display]
    show_dataframe(display[columns], hide_index=True, column_config={"risk_score": st.column_config.NumberColumn(format="%.3f"), "url_valid": st.column_config.CheckboxColumn()})
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "weight_profile": st.session_state.get("batch_profile"), "total_urls": len(result), "valid_urls": valid, "invalid_urls": invalid, "risk_distribution": {key: int(counts.get(key, 0)) for key in ["low", "medium", "high"]}}
    if metrics_result:
        summary["labelled_evaluation"] = metrics_result[1]
    left, right = st.columns(2)
    left.download_button("Download results CSV", result.to_csv(index=False).encode("utf-8-sig"), "phishguard_results.csv", "text/csv")
    right.download_button("Download summary JSON", json.dumps(summary, indent=2).encode("utf-8"), "phishguard_summary.json", "application/json")


def render_performance_page(results: dict[str, Any]) -> None:
    st.subheader("Model Performance")
    st.caption("Frozen calibrated test-set results from results.json. These values are not changed by uploads.")
    metrics = results["test_metrics"]
    tp, fp, fn, tn = (int(metrics[key]) for key in ["tp", "fp", "fn", "tn"])
    values = calculate_metrics(tp, fp, fn, tn)
    columns = st.columns(5)
    for col, (label, value) in zip(columns, [("Precision", metrics.get("precision", values["precision"])), ("Recall", metrics.get("recall", values["recall"])), ("F1-score", metrics.get("f1", values["f1"])), ("Balanced accuracy", metrics.get("balanced_accuracy", values["balanced_accuracy"])), ("Test sample size", tp + fp + fn + tn)]):
        col.metric(label, f"{float(value):.3f}" if label != "Test sample size" else f"{int(value):,}")
    st.markdown("#### Confusion matrix")
    show_chart(confusion_chart(tp, fp, fn, tn))
    a, b, c, d = st.columns(4)
    for col, label, value in zip([a, b, c, d], ["True positives", "False positives", "False negatives", "True negatives"], [tp, fp, fn, tn]):
        col.metric(label, f"{value:,}")
    with st.expander("Metric guide"):
        st.markdown("**Precision**: predicted phishing URLs that were phishing. **Recall**: phishing URLs detected. **F1-score**: balance of precision and recall. **Specificity**: legitimate URLs correctly recognised. **Balanced accuracy**: mean of recall and specificity.")


def render_methodology_page(weights: dict[str, float], low_high: float, med_high: float) -> None:
    st.subheader("Methodology")
    flow = [("01", "URL input", "User submits URL text"), ("02", "Feature extraction", "Existing rule functions inspect structure"), ("03", "Raw sub-scores", "Each indicator is assigned a score"), ("04", "Weighted score", "Frozen weights combine contributions"), ("05", "Threshold category", f"Low ≤ {low_high:g}; medium ≤ {med_high:g}"), ("06", "Explainable output", "Rules and contributions are shown")]
    st.markdown("<div class='pg-flow'>" + "".join(f"<div class='pg-flow-step'><div class='pg-flow-num'>{number}</div><b>{title}</b><span>{copy}</span></div>" for number, title, copy in flow) + "</div>", unsafe_allow_html=True)
    st.markdown("#### Calibrated feature reference")
    table = pd.DataFrame([{"Feature": feature_label(name), "Description": FEATURE_EXPLANATIONS[name], "Type": "Banded" if name in BANDED_FEATURES else "Binary", "Calibrated weight": weights.get(name, 0)} for name in FEATURE_NAMES])
    show_dataframe(table, hide_index=True)
    st.warning("Limitations: PhishGuard evaluates the URL string only. It does not visit webpages, inspect page content, replace browser or antivirus protections, or use enterprise threat-intelligence services. Its result is decision-support information, not absolute proof.")


def render_about_page() -> None:
    st.subheader("About this project")
    st.markdown("<div class='pg-academic' style='padding:1.35rem'><div class='pg-scanner-label'>ACADEMIC PROJECT</div><h3 style='color:#fff;margin:.25rem 0 .4rem;font-size:1.45rem'>PhishGuard</h3><span style='color:var(--muted);font-size:.86rem'>Designed and developed by Abdalla as an explainable cybersecurity risk-assessment system.</span></div>", unsafe_allow_html=True)
    labels = {"author": "Author", "programme": "Programme", "supervisor": "Supervisor", "academic_year": "Academic year"}
    details = pd.DataFrame([{"Field": labels[key], "Value": value} for key, value in PROJECT_INFO.items()])
    show_dataframe(details, hide_index=True)
    st.markdown("Designed and developed by Abdalla, PhishGuard is an explainable phishing URL risk-assessment system that converts structural URL indicators into calibrated, transparent risk scores. Each assessment identifies the specific rules that influenced the result, helping users understand the evidence behind the classification. The system analyses only the submitted URL and never visits or interacts with the destination website.")


def main() -> None:
    try:
        results = load_results(str(RESULTS_PATH))
    except (OSError, json.JSONDecodeError, KeyError) as error:
        st.error(f"Unable to load results.json from the application folder: {error}")
        st.stop()
    calibrated_weights = results["calibrated_weights"]
    low_high = float(results["calibrated_low_high"])
    med_high = float(results["calibrated_med_high"])
    inject_css()
    active_weights, active_profile = render_header(calibrated_weights, low_high, med_high)
    tabs = st.tabs(["URL Scanner", "Batch Scanner", "Model Insights", "Methodology", "About"])
    with tabs[0]: render_single_url_page(active_weights, active_profile, low_high, med_high)
    with tabs[1]: render_batch_page(active_weights, active_profile, low_high, med_high)
    with tabs[2]: render_performance_page(results)
    with tabs[3]: render_methodology_page(calibrated_weights, low_high, med_high)
    with tabs[4]: render_about_page()


if __name__ == "__main__":
    main()
