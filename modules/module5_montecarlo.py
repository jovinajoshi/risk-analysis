"""
Module 5: Monte Carlo Simulation
GBM price paths with probability summary and interactive controls.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from modules.data_utils import compute_log_returns


def run_gbm_simulation(current_price: float, mu: float, sigma: float,
                       n_sims: int, n_days: int, seed: int = 42) -> np.ndarray:
    """
    Simulate price paths using Geometric Brownian Motion.
    S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """
    np.random.seed(seed)
    dt = 1 / 252
    paths = np.zeros((n_sims, n_days + 1))
    paths[:, 0] = current_price

    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = np.random.standard_normal((n_sims, n_days))
    log_returns = drift + diffusion * Z
    paths[:, 1:] = current_price * np.exp(np.cumsum(log_returns, axis=1))

    return paths


def render_monte_carlo_module(prices: pd.DataFrame):
    """Render the Monte Carlo Simulation module."""
    price_series = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    returns = compute_log_returns(price_series)
    current_price = float(price_series.iloc[-1])
    mu = float(returns.mean() * 252)
    sigma = float(returns.std() * np.sqrt(252))

    # Interactive Controls
    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.select_slider(
            "Number of Simulations",
            options=list(range(500, 10001, 500)),
            value=1000,
            key="mc_sims"
        )
    with col2:
        horizon = st.select_slider(
            "Time Horizon (Trading Days)",
            options=[63, 84, 105, 126, 147, 168, 189, 210, 231, 252],
            value=252,
            key="mc_horizon"
        )

    start_time = time.time()
    paths = run_gbm_simulation(current_price, mu, sigma, n_sims, horizon)
    elapsed = time.time() - start_time

    final_prices = paths[:, -1]
    p5 = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)
    median_p = np.median(final_prices)
    expected_p = np.mean(final_prices)

    prob_up_10 = np.mean(final_prices > current_price * 1.10) * 100
    prob_down_10 = np.mean(final_prices < current_price * 0.90) * 100

    # Color-code paths
    top5_mask = final_prices >= np.percentile(final_prices, 95)
    bot5_mask = final_prices <= np.percentile(final_prices, 5)
    mid_mask = ~top5_mask & ~bot5_mask

    days_axis = list(range(horizon + 1))

    fig = go.Figure()

    # Middle paths (sample for performance)
    mid_indices = np.where(mid_mask)[0]
    sample_size = min(200, len(mid_indices))
    sampled = np.random.choice(mid_indices, sample_size, replace=False)
    for i in sampled:
        fig.add_trace(go.Scatter(
            x=days_axis, y=paths[i],
            mode="lines", line=dict(color="rgba(77,166,255,0.08)", width=0.5),
            showlegend=False,
        ))

    # Top 5% paths
    top_indices = np.where(top5_mask)[0]
    for i in top_indices[:30]:
        fig.add_trace(go.Scatter(
            x=days_axis, y=paths[i],
            mode="lines", line=dict(color="rgba(0,255,136,0.25)", width=0.8),
            showlegend=False,
        ))

    # Bottom 5% paths
    bot_indices = np.where(bot5_mask)[0]
    for i in bot_indices[:30]:
        fig.add_trace(go.Scatter(
            x=days_axis, y=paths[i],
            mode="lines", line=dict(color="rgba(255,68,68,0.25)", width=0.8),
            showlegend=False,
        ))

    # Legend dummy traces
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="#00ff88", width=2), name="Top 5%"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="#ff4444", width=2), name="Bottom 5%"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="rgba(77,166,255,0.6)", width=1), name="Other Paths"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Trading Days", showgrid=True, gridcolor="#1e2433"),
        yaxis=dict(title="Price (INR)", showgrid=True, gridcolor="#1e2433"),
    )

    col_chart, col_summary = st.columns([3, 2])

    with col_chart:
        st.plotly_chart(fig, use_container_width=True, key="mc_paths")
        st.markdown(f'<div style="color:#8892b0;font-size:10px;text-align:right;">'
                    f'Computation time: {elapsed:.2f}s | {n_sims} paths</div>',
                    unsafe_allow_html=True)

    with col_summary:
        st.markdown("**Summary (1 Year)**")
        summary_data = {
            "Metric": [
                "Expected Price", "Median Price",
                "Best Case (95th %ile)", "Worst Case (5th %ile)",
                "Prob(Price > +10%)", "Prob(Price < -10%)"
            ],
            "Value": [
                f"₹{expected_p:,.2f}",
                f"₹{median_p:,.2f}",
                f"₹{p95:,.2f}",
                f"₹{p5:,.2f}",
                f"{prob_up_10:.1f}%",
                f"{prob_down_10:.1f}%",
            ]
        }
        df_s = pd.DataFrame(summary_data)
        st.dataframe(df_s, use_container_width=True, hide_index=True)
