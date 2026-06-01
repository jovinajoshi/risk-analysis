"""
Module 1: Executive Summary Panel
KPI cards with sparklines, investment signal, risk level.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.data_utils import compute_log_returns


def make_sparkline(values: pd.Series, color: str = "#00d4ff") -> go.Figure:
    """Create a mini sparkline chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(values))),
        y=values.values,
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.1)",
    ))
    fig.update_layout(
        height=60, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def compute_kpis(prices: pd.Series, dcf_value: float = None, arima_forecast: float = None,
                 pd_value: float = None) -> dict:
    """Compute all KPI metrics from price series."""
    returns = compute_log_returns(prices)
    if returns.empty or len(returns) < 20:
        return {}

    current_price = float(prices.iloc[-1])
    expected_return_1y = float(returns.mean() * 252)
    volatility = float(returns.std() * np.sqrt(252))

    var_95 = float(np.percentile(returns, 5))
    rf = 0.06 / 252
    sharpe = float((returns.mean() - rf) / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    portfolio_return = expected_return_1y

    if arima_forecast is None:
        arima_forecast = current_price * (1 + expected_return_1y)
    if dcf_value is None:
        dcf_value = current_price * 1.15
    if pd_value is None:
        pd_value = 2.5

    margin_of_safety = (dcf_value - current_price) / dcf_value * 100 if dcf_value > 0 else 0

    if margin_of_safety > 20 and var_95 > -0.05:
        signal, signal_color = "BUY", "#00ff88"
    elif 5 <= margin_of_safety <= 20:
        signal, signal_color = "HOLD", "#ffd700"
    else:
        signal, signal_color = "SELL", "#ff4444"

    if var_95 > -0.01:
        risk_level, risk_color = "LOW", "#00ff88"
    elif var_95 > -0.025:
        risk_level, risk_color = "MEDIUM", "#ffd700"
    else:
        risk_level, risk_color = "HIGH", "#ff4444"

    return {
        "current_price": current_price,
        "arima_forecast": arima_forecast,
        "dcf_value": dcf_value,
        "expected_return": expected_return_1y,
        "portfolio_return": portfolio_return,
        "volatility": volatility,
        "var_95": var_95,
        "pd_value": pd_value,
        "sharpe": sharpe,
        "margin_of_safety": margin_of_safety,
        "signal": signal,
        "signal_color": signal_color,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "returns": returns,
        "prices": prices,
    }


def render_kpi_card(label: str, value: str, delta: str, sparkline_data: pd.Series,
                    delta_color: str = "#00d4ff", chart_key: str = ""):
    """Render a single KPI card with sparkline. chart_key must be unique."""
    delta_html = f'<span style="color:{delta_color};font-size:11px;">{delta}</span>'
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1f3a,#0d1117);border:1px solid #2d3561;
    border-radius:10px;padding:14px 16px;min-height:110px;">
        <div style="color:#8892b0;font-size:11px;font-weight:600;text-transform:uppercase;
        letter-spacing:1px;margin-bottom:4px;">{label}</div>
        <div style="color:#e6e6e6;font-size:22px;font-weight:700;line-height:1.2;">{value}</div>
        <div style="margin-top:2px;">{delta_html}</div>
    </div>
    """, unsafe_allow_html=True)
    if sparkline_data is not None and len(sparkline_data) > 2:
        fig = make_sparkline(sparkline_data[-30:])
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False},
                        key=f"spark_{chart_key}")


def render_executive_summary(ticker: str, prices: pd.DataFrame, kpis: dict):
    """Render the full Executive Summary panel."""
    if not kpis:
        st.warning("Insufficient data to compute KPIs.")
        return

    price_series = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    # Use ticker as part of key prefix so switching tickers resets keys
    t = ticker.replace(".", "_")

    # ── Row 1: Price KPI cards ──
    c1, c2, c3 = st.columns(3)
    with c1:
        delta_pct = (kpis["current_price"] / float(price_series.iloc[-2]) - 1) * 100 if len(price_series) > 1 else 0
        render_kpi_card(
            "Current Price",
            f"₹{kpis['current_price']:,.2f}",
            f"+{delta_pct:.2f}%" if delta_pct >= 0 else f"{delta_pct:.2f}%",
            price_series,
            "#00ff88" if delta_pct >= 0 else "#ff4444",
            chart_key=f"{t}_price",
        )
    with c2:
        diff_pct = (kpis["arima_forecast"] / kpis["current_price"] - 1) * 100
        render_kpi_card(
            "Forecasted Price (ARIMA)",
            f"₹{kpis['arima_forecast']:,.2f}",
            f"+{diff_pct:.2f}%" if diff_pct >= 0 else f"{diff_pct:.2f}%",
            price_series,
            "#00d4ff",
            chart_key=f"{t}_forecast",
        )
    with c3:
        mos = kpis["margin_of_safety"]
        render_kpi_card(
            "DCF Intrinsic Value",
            f"₹{kpis['dcf_value']:,.2f}",
            f"MoS: +{mos:.2f}%" if mos >= 0 else f"MoS: {mos:.2f}%",
            price_series,
            "#ffd700",
            chart_key=f"{t}_dcf",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Row 2: Risk & return metrics ──
    c4, c5, c6, c7, c8, c9 = st.columns(6)
    metrics = [
        (c4, "Expected Return (1Y)", f"{kpis['expected_return']*100:.2f}%"),
        (c5, "Portfolio Return (1Y)", f"{kpis['portfolio_return']*100:.2f}%"),
        (c6, "Portfolio Risk", f"{kpis['volatility']*100:.2f}%"),
        (c7, "VaR 95% (1-Day)", f"{kpis['var_95']*100:.2f}%"),
        (c8, "Prob. of Default", f"{kpis['pd_value']:.2f}%"),
        (c9, "Sharpe Ratio (1Y)", f"{kpis['sharpe']:.2f}"),
    ]
    for col, label, val in metrics:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1f3a,#0d1117);border:1px solid #2d3561;
            border-radius:8px;padding:10px 12px;text-align:center;">
                <div style="color:#8892b0;font-size:9px;font-weight:600;text-transform:uppercase;
                letter-spacing:.5px;margin-bottom:3px;">{label}</div>
                <div style="color:#e6e6e6;font-size:16px;font-weight:700;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Row 3: Investment Signal ──
    cs, cr = st.columns([1, 1])
    with cs:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1f3a,#0d1117);border:2px solid {kpis['signal_color']};
        border-radius:12px;padding:20px;text-align:center;">
            <div style="color:#8892b0;font-size:11px;font-weight:600;text-transform:uppercase;
            letter-spacing:1px;">Investment Signal</div>
            <div style="color:{kpis['signal_color']};font-size:36px;font-weight:900;
            letter-spacing:4px;margin:8px 0;">{kpis['signal']}</div>
            <div style="color:#8892b0;font-size:10px;">
                MoS: {kpis['margin_of_safety']:.1f}% | VaR: {kpis['var_95']*100:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with cr:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1f3a,#0d1117);border:2px solid {kpis['risk_color']};
        border-radius:12px;padding:20px;text-align:center;">
            <div style="color:#8892b0;font-size:11px;font-weight:600;text-transform:uppercase;
            letter-spacing:1px;">Risk Level</div>
            <div style="color:{kpis['risk_color']};font-size:36px;font-weight:900;
            letter-spacing:4px;margin:8px 0;">{kpis['risk_level']}</div>
            <div style="color:#8892b0;font-size:10px;">
                Sharpe: {kpis['sharpe']:.2f} | Vol: {kpis['volatility']*100:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
