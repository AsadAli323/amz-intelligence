"""Shared data loading, theme and small HTML helpers for the Amz-Intelligence app."""
from __future__ import annotations

import html as _html
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shelfsense.cleaning import (CATEGORY_RULES, DEFAULT_CATEGORY, NON_ELECTRONIC,  # noqa: E402,F401
                                 assign_category, clean)
from shelfsense.train import add_features  # noqa: E402,F401

CATEGORIES = [c for c, _ in CATEGORY_RULES] + [DEFAULT_CATEGORY]
INK, LABEL, SHELF = "#12303A", "#FFD23F", "#EEF1F3"
GOOD, BAD, MUTED = "#1F9D6B", "#D64545", "#5B7480"
TIERS = {"Low": "100 or fewer sold per month", "Medium": "200 to 999 sold per month",
         "High": "1,000 or more sold per month"}

FONTS = ("@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;"
         "12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');")

CSS = FONTS + """
.stApp, .stMarkdown, .stMarkdown p, label, input, textarea, button, [data-baseweb="select"] {
  font-family: 'Instrument Sans', system-ui, sans-serif !important; }
h1, h2, h3, .ss-display {font-family: 'Bricolage Grotesque', 'Instrument Sans', sans-serif !important;
  letter-spacing: -0.02em; color: #12303A;}
.block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem;}
header[data-testid="stHeader"] {background: transparent;}
.ss-hero h1 {font-size: clamp(2rem, 4.4vw, 3.3rem); line-height: 1.05; margin: .4rem 0 .6rem 0; font-weight: 700;}
.ss-hero p {font-size: 1.12rem; color: #3E5963; max-width: 62ch; margin: 0 0 1.1rem 0; line-height: 1.5;}
div[data-testid="stTextInput"] div[data-baseweb="input"] {border: 2px solid #12303A; border-radius: 14px;
  background: #fff; overflow: hidden;}
div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {box-shadow: 0 0 0 4px #FFD23F;}
div[data-testid="stTextInput"] input {height: 3.4rem; font-size: 1.1rem; padding: 0 1.1rem; border: none; background: #fff;}
.stButton > button {border-radius: 999px; border: 1.5px solid #12303A; background: #fff; color: #12303A;
  font-weight: 500; padding: .25rem .9rem; transition: background .15s;}
.stButton > button:hover {background: #FFD23F; border-color: #12303A; color: #12303A;}
.stDownloadButton > button {border-radius: 12px; background:#12303A; color:#fff; border:none;}
div[data-testid="stTabs"] button[role="tab"] {font-weight: 600;}
div[data-testid="stMetric"] {background:#fff; border-radius:14px; padding: 1rem 1.2rem; border:1px solid #D5DDE1;}
div[data-testid="stExpander"] {background:#fff; border-radius:14px; border:1px solid #D5DDE1;}

.ss-product {background:#fff; border-radius:16px; padding:1.2rem 1.4rem; border:1px solid #D5DDE1; margin: 1rem 0;}
.ss-product h2 {font-size:1.35rem; margin:.2rem 0 .5rem 0; line-height:1.25;}
.ss-chip {display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.85rem; font-weight:600;
  background:#EEF1F3; color:#12303A; margin-right:.4rem; border:1px solid #D5DDE1;}
.ss-chip.hl {background:#FFD23F; border-color:#12303A;}
.ss-muted {color:#5B7480; font-size:.92rem;}
.ss-note {background:#FFF6D6; border:1px solid #E5C34A; border-radius:12px; padding:.8rem 1rem; margin:.6rem 0 1rem 0;
  color:#4A3B00; font-size:.97rem; line-height:1.45;}

/* the shelf label: the one memorable element */
.ss-tag {background:#fff; border:2px solid #12303A; border-radius:6px 18px 18px 6px; padding:1.1rem 1.3rem 1.1rem 1.6rem;
  position:relative; min-height:16.5rem; box-shadow: 6px 6px 0 #12303A;}
.ss-tag:before {content:""; position:absolute; left:0; top:0; bottom:0; width:12px; background:#FFD23F;
  border-right:2px solid #12303A;}
.ss-tag .k {font-size:.95rem; color:#3E5963; margin-bottom:.1rem;}
.ss-big {font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:3.1rem; line-height:1; letter-spacing:-0.03em;}
.ss-verdict {display:inline-block; margin-top:.7rem; padding:.28rem .75rem; border-radius:8px; font-weight:600; font-size:.95rem; color:#fff;}
.ss-panel {background:#fff; border-radius:16px; border:1px solid #D5DDE1; padding:1.1rem 1.3rem; min-height:16.5rem;}
.ss-product a {color:#12303A; font-weight:600; text-decoration: underline; text-decoration-color:#FFD23F; text-decoration-thickness:3px; text-underline-offset:3px;}
.ss-panel h3 {font-size:1.05rem; margin:0 0 .7rem 0; font-weight:700;}

.ss-stack {display:flex; height:2.1rem; border-radius:8px; overflow:hidden; margin:.8rem 0 .4rem 0; border:1.5px solid #12303A;}
.ss-stack div {display:flex; align-items:center; justify-content:center; font-size:.85rem; font-weight:600; color:#12303A;}
.ss-legend {display:flex; justify-content:space-between; font-size:.85rem; color:#5B7480;}

.ss-rule {margin: .9rem 0 1.1rem 0;}
.ss-rule .row {display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:.35rem;}
.ss-rule .row b {font-weight:600;}
.ss-track {position:relative; height:10px; border-radius:6px; background: linear-gradient(90deg,#DCE4E8,#B9C8CF);}
.ss-pin {position:absolute; top:-7px; width:4px; height:24px; background:#12303A; border-radius:2px; transform:translateX(-2px);}
.ss-pin:after {content:""; position:absolute; top:-5px; left:-4px; width:12px; height:12px; border-radius:50%;
  background:#FFD23F; border:2px solid #12303A;}

.ss-peer {display:grid; grid-template-columns: 1fr auto; gap:.3rem 1rem; padding:.85rem 0; border-top:1px solid #E3E9EC;}
.ss-peer:first-child {border-top:none;}
.ss-peer .t {font-weight:600; line-height:1.3;}
.ss-peer .m {color:#5B7480; font-size:.9rem;}
.ss-match {width:90px; height:8px; background:#E3E9EC; border-radius:6px; overflow:hidden; margin-top:.4rem; justify-self:end;}
.ss-match div {height:100%; background:#12303A;}
.ss-peer a {color:#12303A; text-decoration: underline; text-decoration-color:#FFD23F; text-decoration-thickness:3px; text-underline-offset:2px;}
.ss-step {background:#fff; border-radius:16px; border:1px solid #D5DDE1; padding:1.1rem 1.3rem; min-height:9.5rem;}
.ss-step h3 {font-size:1.05rem; margin:0 0 .35rem 0;}
.ss-step p {margin:0; color:#3E5963; font-size:.96rem; line-height:1.5;}
@media (max-width: 700px) {.ss-big {font-size:2.4rem;} .ss-tag {box-shadow: 4px 4px 0 #12303A;}}
@media (prefers-reduced-motion: reduce) {* {transition: none !important;}}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def html(s: str):
    """Render an HTML snippet (indentation and blank lines stripped so Markdown never treats it as code)."""
    s = "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())
    st.markdown(s, unsafe_allow_html=True)


esc = _html.escape


def money(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"


def plot_layout(fig, height=360):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Instrument Sans, sans-serif", color=INK),
                      colorway=[INK, "#2A9D8F", LABEL, "#8FA7B2", "#E76F51", "#6C8EAD"],
                      margin=dict(l=10, r=10, t=50, b=10), title_font=dict(family="Bricolage Grotesque", size=18))
    fig.update_xaxes(gridcolor="#DCE4E8", zerolinecolor="#DCE4E8")
    fig.update_yaxes(gridcolor="#DCE4E8", zerolinecolor="#DCE4E8")
    return fig


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "sample" / "products_sample.csv", parse_dates=["collected_at"])
    df["category"] = df["title"].map(assign_category)   # always use the latest category rules
    return df


@st.cache_resource(show_spinner="Preparing models (first visit only, about 10 seconds)...")
def load_models():
    """Load saved models; if they were saved with a different library version (common between
    Colab and Streamlit Cloud) re-train quickly from the bundled sample instead."""
    try:
        return (joblib.load(ROOT / "models" / "demand_model.joblib"),
                joblib.load(ROOT / "models" / "price_model.joblib"))
    except Exception:
        from shelfsense.train import train_demand, train_price
        feat = add_features(load_sample())
        return train_demand(feat)[0], train_price(feat)[0]


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


def to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    if "Title" in df.columns and "Current/discounted_price" in df.columns:
        df = clean(df)
    return df
