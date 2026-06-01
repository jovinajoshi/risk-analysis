"""
Module 6: Value at Risk (VaR)
Three methods: Historical, Parametric, Monte Carlo. CVaR. Kupiec test.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from modules.data_utils import compute_log_returns
from modules.module5_montecarlo import run_gbm_simulation


def compute_var_historical(returns: pd.Series, confidence: float) -> float:
    """Historical Simulation VaR."""
    return float(np.percentile(returns, (1 - confidence) * 100))


def compute_var_parametric(returns: pd.Series, confidence: float) -> float:
    """Parametric Normal VaR."""
    mu = returns.mean()
    sigma = returns.std()
    return float(mu + sigma * stats.norm.ppf(1 - confidence))


def compute_var_montecarlo(current_price: float, mu: float, sigma: float,
                           confidence: float, n_paths: int = 10000) -> float:
    """Monte Carlo VaR using GBM 1-day simulation."""
    paths = run_gbm_simulation(current_price, mu, sigma, n_paths, 1, seed=101)
    daily_pnl = (paths[:, -1] - current_price) / current_price
    return float(np.percentile(daily_pnl, (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, var_95: float) -> float:
    """Expected Shortfall (CVaR) at 95%."""
    tail_losses = returns[returns <= var_95]
    return float(tail_losses.mean()) if len(tail_losses) > 0 else var_95


def kupiec_test(returns: pd.Series, var_level: float, confidence: float = 0.95,
                window: int = 252) -> dict:
    """Kupiec Proportion of Failures (POF) test."""
    recent = returns.tail(window)
    exceptions = int((recent < var_level).sum())
    n = len(recent)
    expected = n * (1 - confidence)
    alpha = 1 - confidence

    # LR statistic
    if exceptions == 0:
        lr_stat = 0.0
    else:
        p_hat = exceptions / n
        try:
            lr_stat = -2 * (
                np.log((1 - alpha) ** (n - exceptions) * alpha ** exceptions)
                - np.log((1 - p_hat) ** (n - exceptions) * p_hat ** exceptions)
            )
        except Exception:
            lr_stat = 0.0

    p_value = float(1 - stats.chi2.cdf(lr_stat, df=1))
    valid = p_value > 0.05

    return {
        "exceptions": exceptions,
        "expected": round(expected, 1),
        "p_value": round(p_value, 4),
        "verdict": "Valid ✅" if valid else "Invalid ❌",
        "verdict_color": "#00ff88" if valid else "#ff4444",
    }


def render_var_module(prices: pd.DataFrame):
    """Render the Value at Risk module."""
    price_series = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    returns = compute_log_returns(price_series)
    current_price = float(price_series.iloc[-1])
    mu = float(returns.mean() * 252)
    sigma = float(returns.std() * np.sqrt(252))

    # Compute all VaR values
    with st.spinner("Computing VaR (10,000 MC paths)..."):
        var95_hist = compute_var_historical(returns, 0.95)
        var99_hist = compute_var_historical(returns, 0.99)
        var95_param = compute_var_parametric(returns, 0.95)
        var99_param = compute_var_parametric(returns, 0.99)
        var95_mc = compute_var_montecarlo(current_price, mu, sigma, 0.95, 10000)
        var99_mc = compute_var_montecarlo(current_price, mu, sigma, 0.99, 10000)

    cvar_95 = compute_cvar(returns, var95_hist)
    kupiec = kupiec_test(returns, var95_hist, 0.95)

    # ── VaR Comparison Table ──
    var_table = pd.DataFrame({
        "Method": ["Historical Simulation", "Parametric Normal", "Monte Carlo"],
        "95% VaR": [f"{var95_hist*100:.2f}%", f"{var95_param*100:.2f}%", f"{var95_mc*100:.2f}%"],
        "99% VaR": [f"{var99_hist*100:.2f}%", f"{var99_param*100:.2f}%", f"{var99_mc*100:.2f}%"],
    })

    col_table, col_chart = st.columns([1, 2])

    with col_table:
        st.markdown("**VaR (1-Day Holding Period)**")
        st.dataframe(var_table, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:#1a1f3a;border:1px solid #2d3561;border-radius:8px;padding:10px;margin-top:8px;">
            <div style="color:#8892b0;font-size:10px;">Expected Shortfall (CVaR 95%)</div>
            <div style="color:#ff4444;font-size:22px;font-weight:700;">{cvar_95*100:.2f}%</div>
            <div style="color:#8892b0;font-size:9px;margin-top:4px;">
                CVaR captures the average loss in the worst 5% of scenarios,
                exceeding VaR by measuring tail severity, not just the threshold.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Kupiec Test
        st.markdown("**Kupiec Backtesting (252 days)**")
        st.markdown(f"""
        <div style="background:#1a1f3a;border:1px solid #2d3561;border-radius:8px;padding:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#8892b0;font-size:10px;">Exceptions</span>
                <span style="color:#ffffff;font-size:12px;font-weight:600;">{kupiec['exceptions']}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#8892b0;font-size:10px;">Expected</span>
                <span style="color:#ffffff;font-size:12px;font-weight:600;">{kupiec['expected']}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#8892b0;font-size:10px;">p-value</span>
                <span style="color:#ffd700;font-size:12px;font-weight:600;">{kupiec['p_value']}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8892b0;font-size:10px;">Model</span>
                <span style="color:{kupiec['verdict_color']};font-size:12px;font-weight:700;">{kupiec['verdict']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        # Loss Distribution Chart (Monte Carlo 1-day P&L)
        mc_paths = run_gbm_simulation(current_price, mu, sigma, 10000, 1, seed=200)
        pnl = (mc_paths[:, -1] - current_price) / current_price * 100

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=pnl, nbinsx=80,
            marker_color="#4da6ff",
            opacity=0.7,
            name="P&L Distribution",
        ))

        # Shade tail
        tail_mask = pnl <= var95_mc * 100
        fig.add_trace(go.Histogram(
            x=pnl[tail_mask], nbinsx=40,
            marker_color="rgba(255,68,68,0.6)",
            name="VaR 95% Tail",
        ))

        # VaR line
        fig.add_vline(
            x=var95_mc * 100,
            line_dash="dash", line_color="#ff4444", line_width=2,
            annotation_text=f"VaR 95%: {var95_mc*100:.2f}%",
            annotation_position="top right",
            annotation_font_color="#ff4444",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=360,
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="1-Day P&L (%)", showgrid=True, gridcolor="#1e2433"),
            yaxis=dict(title="Frequency", showgrid=True, gridcolor="#1e2433"),
        )
        st.plotly_chart(fig, use_container_width=True, key="var_loss_dist")
