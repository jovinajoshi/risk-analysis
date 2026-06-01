"""
Module 10: Correlation Heatmap
Portfolio asset correlation with annotations and diversification insights.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.data_utils import (fetch_multi_ticker_data, simulate_bond_returns,
                                 simulate_gold_prices)


def render_correlation_heatmap(tickers: list, start: str, end: str):
    """Render the Correlation Heatmap module."""
    with st.spinner("Computing correlations..."):
        price_data = fetch_multi_ticker_data(tickers, start, end)

    if price_data.empty:
        st.warning("No data available for correlation heatmap.")
        return

    # Add Bond and Gold
    bond_prices = simulate_bond_returns(price_data.index)
    gold_prices = simulate_gold_prices(price_data.index)
    price_data["Bond"] = bond_prices.values
    price_data["Gold"] = gold_prices.values

    # Compute log returns
    log_returns = np.log(price_data / price_data.shift(1)).dropna()

    # Clean column names for display
    clean_names = {col: col.replace(".NS", "") for col in log_returns.columns}
    log_returns_clean = log_returns.rename(columns=clean_names)

    corr_matrix = log_returns_clean.corr()
    labels = list(corr_matrix.columns)
    n = len(labels)

    # Build annotation text
    annotations = []
    for i in range(n):
        for j in range(n):
            val = corr_matrix.iloc[i, j]
            text = f"{val:.2f}"
            if i != j and abs(val) > 0.70:
                text += " ⚠"
            annotations.append(
                dict(
                    x=labels[j], y=labels[i],
                    text=text,
                    font=dict(
                        color="#ffffff" if abs(val) > 0.3 else "#a0a0a0",
                        size=11,
                    ),
                    showarrow=False,
                )
            )

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=labels, y=labels,
        colorscale=[
            [0.0, "#2979ff"],
            [0.25, "#7fb3ff"],
            [0.5, "#ffffff"],
            [0.75, "#ff8585"],
            [1.0, "#ff4444"],
        ],
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(
            title="Correlation",
            titlefont=dict(color="#8892b0"),
            tickfont=dict(color="#8892b0"),
            thickness=12,
        ),
    ))

    fig.update_layout(
        annotations=annotations,
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickfont=dict(color="#8892b0")),
        yaxis=dict(tickfont=dict(color="#8892b0")),
    )

    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")

    # ── Diversification Insights ──
    # Find most diversifying pair (most negative correlation)
    best_pair = None
    best_corr = 1.0
    worst_pair = None
    worst_corr = -1.0

    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix.iloc[i, j]
            if val < best_corr:
                best_corr = val
                best_pair = (labels[i], labels[j])
            if val > worst_corr:
                worst_corr = val
                worst_pair = (labels[i], labels[j])

    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-top:8px;">
        <div style="flex:1;background:#1a1f3a;border-left:3px solid #00ff88;
        border-radius:0 8px 8px 0;padding:10px 14px;">
            <div style="color:#00ff88;font-size:10px;font-weight:600;">
                🌱 Most Diversifying Pair
            </div>
            <div style="color:#e6e6e6;font-size:13px;font-weight:700;margin-top:2px;">
                {best_pair[0]} & {best_pair[1]}
            </div>
            <div style="color:#8892b0;font-size:10px;">Correlation: {best_corr:.2f}</div>
        </div>
        <div style="flex:1;background:#1a1f3a;border-left:3px solid #ff4444;
        border-radius:0 8px 8px 0;padding:10px 14px;">
            <div style="color:#ff4444;font-size:10px;font-weight:600;">
                ⚠️ Most Redundant Pair
            </div>
            <div style="color:#e6e6e6;font-size:13px;font-weight:700;margin-top:2px;">
                {worst_pair[0]} & {worst_pair[1]}
            </div>
            <div style="color:#8892b0;font-size:10px;">Correlation: {worst_corr:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
