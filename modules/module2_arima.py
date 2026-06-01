"""
Module 2: ARIMA Forecasting
Auto ARIMA with walk-forward validation, CI bands, metrics.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def fit_arima_model(prices: pd.Series):
    """Fit auto ARIMA model and return fitted model."""
    try:
        from pmdarima import auto_arima
        model = auto_arima(
            prices,
            stepwise=True,
            seasonal=False,
            information_criterion="aic",
            suppress_warnings=True,
            error_action="ignore",
            max_p=5, max_q=5, max_d=2,
        )
        return model
    except Exception as e:
        st.error(f"ARIMA fitting error: {e}")
        return None


def walk_forward_validation(prices: pd.Series, split: float = 0.8):
    """80/20 walk-forward validation, returns in-sample & out-of-sample RMSE."""
    try:
        from pmdarima import auto_arima
        n = len(prices)
        split_idx = int(n * split)
        train = prices.iloc[:split_idx]
        test = prices.iloc[split_idx:]

        model = auto_arima(train, stepwise=True, seasonal=False,
                           suppress_warnings=True, error_action="ignore")

        in_sample_pred = model.predict_in_sample()
        in_rmse = np.sqrt(mean_squared_error(train.iloc[len(train)-len(in_sample_pred):], in_sample_pred))

        # Rolling 1-step-ahead predictions
        preds = []
        history = list(train)
        for t in range(len(test)):
            fc, _ = model.predict(n_periods=1, return_conf_int=True)
            preds.append(fc[0])
            model.update(test.iloc[t:t+1])

        out_rmse = np.sqrt(mean_squared_error(test, preds))
        return in_rmse, out_rmse, test.index, test.values, np.array(preds)
    except Exception:
        return None, None, None, None, None


def compute_mape(actual, predicted):
    """Mean Absolute Percentage Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def render_arima_module(prices: pd.DataFrame):
    """Render the ARIMA Forecasting module."""
    price_series = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    price_series = price_series.dropna()

    if len(price_series) < 60:
        st.warning("Need at least 60 data points for ARIMA.")
        return None

    with st.spinner("Fitting ARIMA model..."):
        model = fit_arima_model(price_series)

    if model is None:
        st.error("Could not fit ARIMA model.")
        return None

    # Forecast 90 trading days
    horizon = 90
    forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)

    last_date = price_series.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

    # In-sample fitted values
    fitted = model.predict_in_sample()
    fitted_index = price_series.index[len(price_series) - len(fitted):]

    # Metrics
    actual_vals = price_series.iloc[len(price_series) - len(fitted):]
    rmse = np.sqrt(mean_squared_error(actual_vals, fitted))
    mae = mean_absolute_error(actual_vals, fitted)
    mape = compute_mape(actual_vals, fitted)
    direction = "Uptrend 📈" if forecast[-1] > float(price_series.iloc[-1]) else "Downtrend 📉"
    dir_color = "#00ff88" if "Up" in direction else "#ff4444"

    # Walk-forward validation
    with st.spinner("Running walk-forward validation..."):
        in_rmse, out_rmse, _, _, _ = walk_forward_validation(price_series)

    # ── Chart ──
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        mode="lines", name="Actual Price",
        line=dict(color="#ffffff", width=1.8),
    ))

    fig.add_trace(go.Scatter(
        x=list(future_dates), y=forecast,
        mode="lines", name="ARIMA Forecast",
        line=dict(color="#4da6ff", width=2, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(conf_int[:, 1]) + list(conf_int[::-1, 0]),
        fill="toself", fillcolor="rgba(77,166,255,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#1e2433"),
        yaxis=dict(showgrid=True, gridcolor="#1e2433", title="Price (INR)"),
    )

    st.plotly_chart(fig, use_container_width=True, key="arima_chart")

    # ── Metrics Row ──
    m1, m2, m3, m4 = st.columns(4)
    metric_style = """background:linear-gradient(135deg,#1a1f3a,#0d1117);
    border:1px solid #2d3561;border-radius:8px;padding:12px;text-align:center;"""

    with m1:
        st.markdown(f'<div style="{metric_style}"><div style="color:#8892b0;font-size:10px;">RMSE</div>'
                    f'<div style="color:#00d4ff;font-size:18px;font-weight:700;">₹{rmse:.2f}</div></div>',
                    unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div style="{metric_style}"><div style="color:#8892b0;font-size:10px;">MAE</div>'
                    f'<div style="color:#00d4ff;font-size:18px;font-weight:700;">₹{mae:.2f}</div></div>',
                    unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div style="{metric_style}"><div style="color:#8892b0;font-size:10px;">MAPE</div>'
                    f'<div style="color:#00d4ff;font-size:18px;font-weight:700;">{mape:.2f}%</div></div>',
                    unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div style="{metric_style}"><div style="color:#8892b0;font-size:10px;">Direction</div>'
                    f'<div style="color:{dir_color};font-size:14px;font-weight:700;">{direction}</div></div>',
                    unsafe_allow_html=True)

    # ── Model Summary ──
    order = model.order
    aic = model.aic()
    bic = model.bic()
    st.markdown(f"""
    <div style="background:#1a1f3a;border:1px solid #2d3561;border-radius:8px;
    padding:10px 14px;margin-top:8px;font-size:12px;">
        <span style="color:#8892b0;">Model: </span><span style="color:#00d4ff;font-weight:700;">
        ARIMA{order}</span>&nbsp;&nbsp;
        <span style="color:#8892b0;">AIC: </span><span style="color:#ffd700;">{aic:.2f}</span>&nbsp;&nbsp;
        <span style="color:#8892b0;">BIC: </span><span style="color:#ffd700;">{bic:.2f}</span>&nbsp;&nbsp;
        <span style="color:#8892b0;">In-Sample RMSE: </span>
        <span style="color:#00ff88;">₹{f"{in_rmse:.2f}" if in_rmse is not None else "N/A"}</span>&nbsp;&nbsp;
        <span style="color:#8892b0;">Out-of-Sample RMSE: </span>
        <span style="color:#ff4444;">₹{f"{out_rmse:.2f}" if out_rmse is not None else "N/A"}</span>
    </div>
    """, unsafe_allow_html=True)

    return float(forecast[-1])
