"""
Module 3: GARCH Volatility Modeling
GARCH(1,1) conditional volatility with regime detection.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.data_utils import compute_log_returns, compute_rolling_volatility


def fit_garch_model(returns: pd.Series):
    """Fit GARCH(1,1) model on log returns."""
    try:
        from arch import arch_model
        # Scale returns to percentage for numerical stability
        scaled = returns * 100
        model = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        result = model.fit(disp="off", show_warning=False)
        return result
    except Exception as e:
        st.error(f"GARCH fitting error: {e}")
        return None


def render_garch_module(prices: pd.DataFrame):
    """Render the GARCH Volatility module."""
    price_series = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    returns = compute_log_returns(price_series)

    if len(returns) < 60:
        st.warning("Need at least 60 observations for GARCH.")
        return

    with st.spinner("Fitting GARCH(1,1)..."):
        result = fit_garch_model(returns)

    if result is None:
        st.error("GARCH model fitting failed.")
        return

    # Conditional volatility — annualised
    cond_vol = result.conditional_volatility / 100  # back to decimal
    cond_vol_annualised = cond_vol * np.sqrt(252)
    cond_vol_series = pd.Series(
        cond_vol_annualised.values * 100,  # as percentage
        index=returns.index[-len(cond_vol_annualised):]
    )

    # 20-day Rolling Volatility (annualised)
    rolling_vol = compute_rolling_volatility(returns, 20) * 100
    rolling_vol = rolling_vol.dropna()

    # Find highest spike
    max_vol_date = cond_vol_series.idxmax()
    max_vol_val = cond_vol_series.max()

    # Summary statistics
    current_vol = float(cond_vol_series.iloc[-1])
    lt_avg = float(cond_vol_series.mean())
    p25 = np.percentile(cond_vol_series, 25)
    p75 = np.percentile(cond_vol_series, 75)

    if current_vol > p75:
        regime = "HIGH"
        regime_color = "#ff4444"
    elif current_vol < p25:
        regime = "LOW"
        regime_color = "#00ff88"
    else:
        regime = "MODERATE"
        regime_color = "#ffd700"

    # ── Chart ──
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cond_vol_series.index, y=cond_vol_series.values,
        mode="lines", name="Conditional Volatility (GARCH)",
        line=dict(color="#ff8c00", width=2),
    ))

    common_idx = rolling_vol.index.intersection(cond_vol_series.index)
    fig.add_trace(go.Scatter(
        x=common_idx, y=rolling_vol.loc[common_idx].values,
        mode="lines", name="20-Day Rolling Volatility",
        line=dict(color="#ffffff", width=1.5),
    ))

    # Spike annotation — use scatter trace to avoid Timestamp arithmetic bug in add_vline
    spike_x = max_vol_date.strftime("%Y-%m-%d")
    y_max = float(cond_vol_series.max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[spike_x, spike_x], y=[0, y_max],
        mode="lines+text",
        line=dict(color="#ff4444", width=1.5, dash="dash"),
        text=["", f"Peak: {max_vol_val:.1f}%"],
        textposition="top right",
        textfont=dict(color="#ff4444", size=10),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#1e2433"),
        yaxis=dict(showgrid=True, gridcolor="#1e2433", title="Volatility (%)"),
    )

    st.plotly_chart(fig, use_container_width=True, key="garch_chart")

    # ── Summary Stats ──
    s1, s2, s3, s4 = st.columns(4)
    stat_style = """background:linear-gradient(135deg,#1a1f3a,#0d1117);
    border:1px solid #2d3561;border-radius:8px;padding:12px;text-align:center;"""

    with s1:
        st.markdown(f'<div style="{stat_style}"><div style="color:#8892b0;font-size:10px;">Current Vol</div>'
                    f'<div style="color:#ff8c00;font-size:18px;font-weight:700;">{current_vol:.2f}%</div></div>',
                    unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div style="{stat_style}"><div style="color:#8892b0;font-size:10px;">Long-Term Avg</div>'
                    f'<div style="color:#ffffff;font-size:18px;font-weight:700;">{lt_avg:.2f}%</div></div>',
                    unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div style="{stat_style}"><div style="color:#8892b0;font-size:10px;">Volatility Regime</div>'
                    f'<div style="color:{regime_color};font-size:16px;font-weight:700;">{regime}</div></div>',
                    unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div style="{stat_style}"><div style="color:#8892b0;font-size:10px;">Last Spike Date</div>'
                    f'<div style="color:#ff4444;font-size:13px;font-weight:700;">'
                    f'{max_vol_date.strftime("%d %b %Y")}</div></div>',
                    unsafe_allow_html=True)
