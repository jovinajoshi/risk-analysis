"""
Module 9: Stress Testing & Scenario Analysis
Five predefined + one custom scenario with factor betas.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.data_utils import fetch_price_data, compute_log_returns


def compute_factor_betas(ticker_returns: pd.Series, start: str, end: str) -> dict:
    """Compute factor betas via historical regression against NIFTY 50."""
    try:
        import yfinance as yf
        nifty = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
        if nifty is not None and not nifty.empty:
            nifty.columns = [c[0] if isinstance(c, tuple) else c for c in nifty.columns]
            nifty_ret = compute_log_returns(nifty["Close"].dropna())
            common = ticker_returns.index.intersection(nifty_ret.index)
            if len(common) >= 30:
                y = ticker_returns.loc[common].values
                x = nifty_ret.loc[common].values
                beta = np.cov(y, x)[0, 1] / np.var(x) if np.var(x) > 0 else 1.0
                return {
                    "market_beta": float(beta),
                    "rate_beta": float(-0.5 * beta),
                    "oil_beta": float(0.3 * beta),
                }
    except Exception:
        pass

    # Fallback: estimate beta from return volatility ratio (proxy method)
    vol = float(ticker_returns.std() * np.sqrt(252))
    nifty_vol = 0.16  # approximate NIFTY annual vol
    beta_est = min(max(vol / nifty_vol, 0.4), 2.5)
    return {
        "market_beta": round(beta_est, 2),
        "rate_beta": round(-0.5 * beta_est, 2),
        "oil_beta": round(0.3 * beta_est, 2),
    }


def compute_scenario_impacts(betas: dict, scenarios: list) -> list:
    """Compute portfolio impact for each scenario."""
    results = []
    for name, market_shock, rate_shock, oil_shock, desc in scenarios:
        impact = (
            betas["market_beta"] * market_shock
            + betas["rate_beta"] * rate_shock
            + betas["oil_beta"] * oil_shock
        )
        results.append({
            "Scenario": name,
            "Portfolio Impact (%)": round(impact * 100, 2),
            "Description": desc,
        })
    return results


def render_stress_testing_module(ticker: str, prices: pd.DataFrame,
                                 start: str, end: str):
    """Render Stress Testing & Scenario Analysis module."""
    price_series = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    returns = compute_log_returns(price_series)

    with st.spinner("Computing factor betas..."):
        betas = compute_factor_betas(returns, start, end)

    # Predefined scenarios: (name, market_shock, rate_shock, oil_shock, description)
    base_scenarios = [
        ("Market Crash (-20%)", -0.20, 0.0, 0.0, "Broad market falls 20%"),
        ("Interest Rate Hike (+2%)", 0.0, 0.02, 0.0, "RBI hikes rates by 200bps"),
        ("Recession Scenario", -0.15, 0.01, -0.10, "GDP contraction -3%"),
        ("Oil Price Shock (+25%)", 0.0, 0.0, 0.25, "Crude surges +25%"),
        ("Best Case (+15%)", 0.15, 0.0, 0.0, "Bull market rally +15%"),
    ]

    # Custom scenario
    st.markdown("**Custom Scenario**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        custom_name = st.text_input("Name", "Custom Scenario", key="sc_name")
    with c2:
        market_s = st.slider("Market Shock (%)", -30, 30, 0, key="sc_market") / 100
    with c3:
        rate_s = st.slider("Rate Change (%)", -5, 5, 0, key="sc_rate") / 100
    with c4:
        oil_s = st.slider("Oil Shock (%)", -30, 30, 0, key="sc_oil") / 100

    all_scenarios = base_scenarios + [(custom_name, market_s, rate_s, oil_s, "User-defined scenario")]
    impacts = compute_scenario_impacts(betas, all_scenarios)
    df_impacts = pd.DataFrame(impacts)

    col_table, col_chart = st.columns([1, 2])

    with col_table:
        st.markdown("**Scenario Impact Table**")
        # Color-code display
        for _, row in df_impacts.iterrows():
            impact = row["Portfolio Impact (%)"]
            color = "#00ff88" if impact > 0 else "#ff4444"
            st.markdown(f"""
            <div style="background:#1a1f3a;border:1px solid #2d3561;border-radius:6px;
            padding:8px 10px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#e6e6e6;font-size:11px;">{row['Scenario']}</span>
                <span style="color:{color};font-weight:700;font-size:13px;">{impact:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Factor betas info
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #2d3561;border-radius:6px;
        padding:8px;margin-top:8px;font-size:10px;color:#8892b0;">
            Market β: {betas['market_beta']:.2f} |
            Rate β: {betas['rate_beta']:.2f} |
            Oil β: {betas['oil_beta']:.2f}
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        scenarios_names = [r["Scenario"] for r in impacts]
        impact_vals = [r["Portfolio Impact (%)"] for r in impacts]
        bar_colors = ["#00ff88" if v > 0 else "#ff4444" for v in impact_vals]

        fig = go.Figure(go.Bar(
            y=scenarios_names,
            x=impact_vals,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.2f}%" for v in impact_vals],
            textposition="outside",
            textfont=dict(color="#ffffff", size=10),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=360,
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(
                title="Portfolio Impact (%)",
                range=[-30, 25],
                showgrid=True, gridcolor="#1e2433",
                zeroline=True, zerolinecolor="#4a5568",
            ),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="stress_bar")

    # ── Risk Narrative ──
    worst_scenario = min(impacts, key=lambda x: x["Portfolio Impact (%)"])
    best_scenario = max(impacts, key=lambda x: x["Portfolio Impact (%)"])
    abs_market = abs(betas["market_beta"])
    abs_rate = abs(betas["rate_beta"])
    abs_oil = abs(betas["oil_beta"])
    dominant = "market" if abs_market >= abs_rate and abs_market >= abs_oil else \
               "interest rate" if abs_rate >= abs_oil else "oil price"

    hedge_action = {
        "market": "consider protective puts or index hedging to reduce market exposure",
        "interest rate": "consider duration management through bond laddering or interest rate swaps",
        "oil price": "consider commodity futures or exposure reduction to oil-sensitive sectors",
    }.get(dominant, "diversify across uncorrelated assets")

    st.markdown(f"""
    <div style="background:#1a1f3a;border-left:3px solid #ffd700;border-radius:0 8px 8px 0;
    padding:12px 16px;margin-top:8px;">
        <div style="color:#ffd700;font-size:11px;font-weight:600;margin-bottom:4px;">📊 Risk Interpretation</div>
        <div style="color:#c8d0e0;font-size:12px;line-height:1.6;">
            The highest-risk scenario is <strong style="color:#ff4444">{worst_scenario['Scenario']}</strong>
            with a projected portfolio impact of
            <strong style="color:#ff4444">{worst_scenario['Portfolio Impact (%)']:+.2f}%</strong>.
            The portfolio's most significant factor exposure is
            <strong style="color:#00d4ff">{dominant} risk</strong>
            (β = {betas[dominant.split()[0]+'_beta']:.2f}).
            To mitigate downside risk, it is recommended to
            <strong style="color:#00ff88">{hedge_action}</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
