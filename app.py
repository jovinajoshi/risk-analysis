"""
MCA Financial Analytics Capstone Project
Risk Analytics Dashboard — 10 Modules
Built with Python, Streamlit, Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# Page config (must be FIRST Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6e6e6; }
    .stSidebar { background-color: #0a0e1a !important; border-right: 1px solid #1e2433; }

    /* Module header */
    .module-header {
        background: linear-gradient(90deg, #1a1f3a, transparent);
        border-left: 3px solid #00d4ff;
        padding: 6px 12px;
        margin: 0 0 14px 0;
        border-radius: 0 4px 4px 0;
    }
    .module-header h3 {
        color: #00d4ff; font-size: 15px; font-weight: 700;
        margin: 0; letter-spacing: 1px;
    }
    .module-badge {
        display: inline-block; background: #00d4ff; color: #0d1117;
        font-size: 9px; font-weight: 800; padding: 2px 6px;
        border-radius: 3px; margin-right: 8px;
    }
    hr { border-color: #2d3561; margin: 16px 0; }
    .stPlotlyChart { border-radius: 8px; }

    /* Sidebar nav buttons */
    div[data-testid="stRadio"] label {
        color: #8892b0 !important;
        font-size: 13px !important;
    }
    div[data-testid="stRadio"] label:hover {
        color: #00d4ff !important;
    }

    /* Streamlit default overrides */
    .stSpinner > div { border-top-color: #00d4ff !important; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #2d3561; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Module imports
# ─────────────────────────────────────────────
from modules.data_utils import (
    TICKERS, TICKER_LIST, fetch_price_data, get_default_dates,
)
from modules.module1_executive import compute_kpis, render_executive_summary
from modules.module2_arima import render_arima_module
from modules.module3_garch import render_garch_module
from modules.module4_dcf import render_dcf_module
from modules.module5_montecarlo import render_monte_carlo_module
from modules.module6_var import render_var_module
from modules.module7_credit import render_credit_risk_module
from modules.module8_portfolio import render_portfolio_module
from modules.module9_stress import render_stress_testing_module
from modules.module10_heatmap import render_correlation_heatmap


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 16px;border-bottom:1px solid #1e2433;margin-bottom:14px;">
        <div style="font-size:18px;font-weight:900;color:#00d4ff;letter-spacing:2px;">📊 RISK ANALYTICS</div>
        <div style="font-size:9px;color:#4a5568;letter-spacing:1px;margin-top:3px;">
            MCA · FINANCIAL ANALYTICS · CAPSTONE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Ticker selector ──
    st.markdown('<p style="color:#8892b0;font-size:11px;font-weight:600;'
                'text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Select Ticker</p>',
                unsafe_allow_html=True)
    selected_ticker = st.selectbox(
        "Ticker", options=TICKER_LIST,
        format_func=lambda x: f"{x}  —  {TICKERS[x]}",
        label_visibility="collapsed",
        key="ticker_selector",
    )

    # ── Date range ──
    st.markdown('<p style="color:#8892b0;font-size:11px;font-weight:600;'
                'text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px;">Date Range</p>',
                unsafe_allow_html=True)
    default_start, default_end = get_default_dates()
    start_date = st.date_input("Start", value=datetime.strptime(default_start, "%Y-%m-%d"),
                                label_visibility="collapsed", key="start_date")
    end_date   = st.date_input("End",   value=datetime.strptime(default_end,   "%Y-%m-%d"),
                                label_visibility="collapsed", key="end_date")
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    st.markdown('<hr style="border-color:#1e2433;margin:12px 0;">', unsafe_allow_html=True)

    # ── Navigation — this is the key fix: one radio drives the whole page ──
    st.markdown('<p style="color:#8892b0;font-size:11px;font-weight:600;'
                'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Navigation</p>',
                unsafe_allow_html=True)

    NAV_OPTIONS = [
        "1 · Executive Summary",
        "2 · ARIMA Forecasting",
        "3 · GARCH Volatility",
        "4 · DCF Valuation",
        "5 · Monte Carlo",
        "6 · Value at Risk",
        "7 · Credit Risk",
        "8 · Portfolio Optimization",
        "9 · Stress Testing",
        "10 · Correlation Heatmap",
    ]
    active_module = st.radio(
        "Module", NAV_OPTIONS,
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown('<hr style="border-color:#1e2433;margin:12px 0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:9px;color:#4a5568;text-align:center;">
        Data: Yahoo Finance (NSE/BSE)<br>
        Streamlit + Plotly + Python<br>
        {datetime.now().strftime('%d %b %Y  %H:%M')}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load price data
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start, end):
    return fetch_price_data(ticker, start, end)


with st.spinner(f"Loading {selected_ticker}..."):
    prices = load_data(selected_ticker, start_str, end_str)

if prices is None or prices.empty:
    st.error(f"Could not fetch data for {selected_ticker}. Please try another ticker.")
    st.stop()

current_price = float(prices["Close"].iloc[-1]) if "Close" in prices.columns else 0.0


# ─────────────────────────────────────────────
# Page header (always visible)
# ─────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 2])
with hcol1:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1117,#1a1f3a,#0d1117);
    border-bottom:1px solid #2d3561;padding:8px 0 10px;margin-bottom:14px;">
        <div style="font-size:22px;font-weight:900;color:#00d4ff;letter-spacing:2px;">
            RISK ANALYTICS DASHBOARD
        </div>
        <div style="font-size:10px;color:#8892b0;letter-spacing:1px;margin-top:2px;">
            Comprehensive Risk & Investment Analysis Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
with hcol2:
    st.markdown(f"""
    <div style="text-align:right;padding-top:6px;">
        <div style="color:#8892b0;font-size:10px;text-transform:uppercase;letter-spacing:1px;">
            Active Ticker</div>
        <div style="color:#00d4ff;font-size:22px;font-weight:900;">{selected_ticker}</div>
        <div style="color:#e6e6e6;font-size:12px;">{TICKERS[selected_ticker]}</div>
        <div style="color:#ffd700;font-size:11px;">
            ₹{current_price:,.2f} &nbsp;|&nbsp; {start_str} → {end_str}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Module header helper
# ─────────────────────────────────────────────
def module_header(num: str, title: str, marks: str):
    st.markdown(f"""
    <div class="module-header">
        <h3>
            <span class="module-badge">{num}</span>{title}
            <span style="float:right;font-size:10px;color:#ffd700;font-weight:400;">{marks}</span>
        </h3>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Route to active module — only ONE renders
# ─────────────────────────────────────────────
mod = active_module  # shorthand

if mod == NAV_OPTIONS[0]:
    # ── Module 1: Executive Summary ──
    module_header("MODULE 1", "Executive Summary Panel", "15 marks")
    with st.spinner("Computing KPIs..."):
        kpis = compute_kpis(
            prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
        )
    render_executive_summary(selected_ticker, prices, kpis)

elif mod == NAV_OPTIONS[1]:
    # ── Module 2: ARIMA ──
    module_header("MODULE 2", "ARIMA Forecasting", "15 marks")
    render_arima_module(prices)

elif mod == NAV_OPTIONS[2]:
    # ── Module 3: GARCH ──
    module_header("MODULE 3", "GARCH Volatility Modeling", "12 marks")
    render_garch_module(prices)

elif mod == NAV_OPTIONS[3]:
    # ── Module 4: DCF ──
    module_header("MODULE 4", "DCF Valuation", "13 marks")
    render_dcf_module(selected_ticker, current_price)

elif mod == NAV_OPTIONS[4]:
    # ── Module 5: Monte Carlo ──
    module_header("MODULE 5", "Monte Carlo Simulation", "13 marks")
    render_monte_carlo_module(prices)

elif mod == NAV_OPTIONS[5]:
    # ── Module 6: VaR ──
    module_header("MODULE 6", "Value at Risk (VaR)", "15 marks")
    render_var_module(prices)

elif mod == NAV_OPTIONS[6]:
    # ── Module 7: Credit Risk ──
    module_header("MODULE 7", "Credit Risk Modeling", "13 marks")
    render_credit_risk_module(selected_ticker)

elif mod == NAV_OPTIONS[7]:
    # ── Module 8: Portfolio Optimization ──
    module_header("MODULE 8", "Portfolio Optimization", "12 marks")
    render_portfolio_module(TICKER_LIST, start_str, end_str)

elif mod == NAV_OPTIONS[8]:
    # ── Module 9: Stress Testing ──
    module_header("MODULE 9", "Stress Testing & Scenario Analysis", "10 marks")
    render_stress_testing_module(selected_ticker, prices, start_str, end_str)

elif mod == NAV_OPTIONS[9]:
    # ── Module 10: Correlation Heatmap ──
    module_header("MODULE 10", "Correlation Heatmap", "8 marks")
    render_correlation_heatmap(TICKER_LIST, start_str, end_str)

# ── Footer ──
st.markdown("""
<hr>
<div style="text-align:center;color:#4a5568;font-size:10px;padding:6px 0;">
    Risk Analytics Dashboard · Python · Streamlit · Plotly ·
    Data: Yahoo Finance (NSE/BSE) · MCA Financial Analytics Capstone
</div>
""", unsafe_allow_html=True)
