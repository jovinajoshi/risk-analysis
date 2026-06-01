"""
Module 8: Portfolio Optimization
Efficient frontier, max-Sharpe weights, optimal allocation pie.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.data_utils import (fetch_multi_ticker_data, simulate_bond_returns,
                                 simulate_gold_prices, compute_log_returns)


def compute_efficient_frontier(returns_df: pd.DataFrame, n_portfolios: int = 5000,
                                rf: float = 0.06) -> pd.DataFrame:
    """Generate random portfolio weights and compute metrics."""
    np.random.seed(42)
    n_assets = returns_df.shape[1]
    mu = returns_df.mean() * 252
    cov = returns_df.cov() * 252

    results = {"return": [], "volatility": [], "sharpe": [], "weights": []}

    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        port_return = float(np.dot(w, mu))
        port_vol = float(np.sqrt(w @ cov.values @ w))
        sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0
        results["return"].append(port_return * 100)
        results["volatility"].append(port_vol * 100)
        results["sharpe"].append(sharpe)
        results["weights"].append(w)

    return pd.DataFrame(results)


def get_max_sharpe_portfolio(frontier_df: pd.DataFrame, asset_names: list) -> dict:
    """Get the max Sharpe ratio portfolio weights."""
    idx = frontier_df["sharpe"].idxmax()
    best = frontier_df.iloc[idx]
    return {
        "return": best["return"],
        "volatility": best["volatility"],
        "sharpe": best["sharpe"],
        "weights": dict(zip(asset_names, best["weights"])),
    }


def render_portfolio_module(tickers: list, start: str, end: str):
    """Render Portfolio Optimization module."""
    with st.spinner("Loading portfolio data..."):
        price_data = fetch_multi_ticker_data(tickers, start, end)

    if price_data.empty or len(price_data.columns) < 2:
        st.warning("Need at least 2 tickers with data for portfolio optimization.")
        return

    # Add Bond and Gold
    bond_prices = simulate_bond_returns(price_data.index)
    gold_prices = simulate_gold_prices(price_data.index)
    price_data["Bond"] = bond_prices.values
    price_data["Gold"] = gold_prices.values

    all_assets = list(price_data.columns)

    # Asset toggle
    st.markdown("**Toggle Assets**")
    toggle_cols = st.columns(len(all_assets))
    selected_assets = []
    for i, asset in enumerate(all_assets):
        with toggle_cols[i]:
            checked = st.checkbox(asset.replace(".NS", ""), value=True, key=f"port_{asset}")
            if checked:
                selected_assets.append(asset)

    if len(selected_assets) < 2:
        st.warning("Select at least 2 assets.")
        return

    # Compute log returns
    returns_df = np.log(price_data[selected_assets] / price_data[selected_assets].shift(1)).dropna()

    with st.spinner("Computing Efficient Frontier (5,000 portfolios)..."):
        frontier = compute_efficient_frontier(returns_df)

    optimal = get_max_sharpe_portfolio(frontier, selected_assets)

    # ── Efficient Frontier Plot ──
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier["volatility"],
        y=frontier["return"],
        mode="markers",
        marker=dict(
            color=frontier["sharpe"],
            colorscale="Viridis",
            size=3,
            opacity=0.6,
            colorbar=dict(title="Sharpe", thickness=12,
                         tickfont=dict(color="#8892b0")),
        ),
        name="Portfolios",
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Max Sharpe marker
    fig.add_trace(go.Scatter(
        x=[optimal["volatility"]],
        y=[optimal["return"]],
        mode="markers+text",
        marker=dict(color="#ffd700", size=14, symbol="star", line=dict(color="#ffffff", width=1)),
        text=["Max Sharpe"],
        textposition="top right",
        textfont=dict(color="#ffd700", size=11),
        name="Max Sharpe",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Volatility (%)", showgrid=True, gridcolor="#1e2433"),
        yaxis=dict(title="Expected Return (%)", showgrid=True, gridcolor="#1e2433"),
    )

    col_frontier, col_pie = st.columns([3, 2])

    with col_frontier:
        st.plotly_chart(fig, use_container_width=True, key="frontier_chart")

        # Metrics
        m1, m2, m3 = st.columns(3)
        metric_style = """background:linear-gradient(135deg,#1a1f3a,#0d1117);
        border:1px solid #2d3561;border-radius:8px;padding:10px;text-align:center;"""
        with m1:
            st.markdown(f'<div style="{metric_style}"><div style="color:#8892b0;font-size:10px;">'
                        f'Expected Return</div><div style="color:#00ff88;font-size:18px;font-weight:700;">'
                        f'{optimal["return"]:.2f}%</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div style="{metric_style}"><div style="color:#8892b0;font-size:10px;">'
                        f'Portfolio Vol</div><div style="color:#ffd700;font-size:18px;font-weight:700;">'
                        f'{optimal["volatility"]:.2f}%</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div style="{metric_style}"><div style="color:#8892b0;font-size:10px;">'
                        f'Sharpe Ratio</div><div style="color:#00d4ff;font-size:18px;font-weight:700;">'
                        f'{optimal["sharpe"]:.2f}</div></div>', unsafe_allow_html=True)

    with col_pie:
        weights = optimal["weights"]
        labels = [a.replace(".NS", "") for a in weights.keys()]
        values = list(weights.values())

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            textinfo="label+percent",
            textfont=dict(color="#ffffff", size=11),
            hole=0.35,
            marker=dict(colors=[
                "#00d4ff", "#00ff88", "#ffd700", "#ff8c00",
                "#ff4444", "#9c27b0", "#2979ff"
            ][:len(labels)]),
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            height=340,
            showlegend=True,
            legend=dict(font=dict(color="#8892b0", size=10)),
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Optimal Portfolio Allocation", font=dict(color="#8892b0", size=12)),
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_pie")
