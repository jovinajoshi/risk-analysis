"""
Module 7: Credit Risk Modeling
Logistic regression PD model, credit score gauge, confusion matrix.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from modules.data_utils import fetch_stock_info


def generate_synthetic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic training data for PD model."""
    np.random.seed(seed)
    data = {
        "debt_to_equity": np.abs(np.random.normal(1.5, 1.0, n)),
        "interest_coverage": np.abs(np.random.normal(5.0, 3.0, n)),
        "current_ratio": np.abs(np.random.normal(1.8, 0.6, n)),
        "roe": np.random.normal(0.15, 0.10, n),
        "net_profit_margin": np.random.normal(0.12, 0.08, n),
    }
    df = pd.DataFrame(data)

    # Default label logic: high D/E, low coverage → default
    default_score = (
        (df["debt_to_equity"] > 3.0).astype(int) * 2
        + (df["interest_coverage"] < 2.0).astype(int) * 2
        + (df["current_ratio"] < 1.0).astype(int)
        + (df["roe"] < 0).astype(int)
        + (df["net_profit_margin"] < 0).astype(int)
    )
    noise = np.random.binomial(1, 0.05, n)
    df["default"] = ((default_score >= 3) | noise.astype(bool)).astype(int)
    return df


def train_pd_model(df: pd.DataFrame):
    """Train logistic regression PD model."""
    features = ["debt_to_equity", "interest_coverage", "current_ratio", "roe", "net_profit_margin"]
    X = df[features]
    y = df["default"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, auc, acc, cm


def fetch_company_features(ticker: str) -> dict:
    """Get financial ratios for the selected stock."""
    info = fetch_stock_info(ticker)
    return {
        "debt_to_equity": info.get("debtToEquity", 80) / 100 if info.get("debtToEquity") else 1.2,
        "interest_coverage": info.get("ebitda", 1) / max(info.get("totalDebt", 1) * 0.05, 1)
                             if info.get("ebitda") else 5.0,
        "current_ratio": info.get("currentRatio", 1.5),
        "roe": info.get("returnOnEquity", 0.15),
        "net_profit_margin": info.get("profitMargins", 0.12),
    }


def pd_to_credit_score(pd_pct: float) -> tuple:
    """Map PD (%) to credit score (300-850) and risk grade."""
    pd_clamped = min(max(pd_pct, 0), 100)
    score = int(850 - (pd_clamped / 100) * 550)
    score = max(300, min(850, score))

    if score >= 800:
        grade = "AAA"
    elif score >= 750:
        grade = "AA"
    elif score >= 700:
        grade = "A"
    elif score >= 650:
        grade = "BBB"
    elif score >= 600:
        grade = "BB"
    elif score >= 550:
        grade = "B"
    elif score >= 500:
        grade = "CCC"
    else:
        grade = "D"

    if score >= 700:
        risk_level = "LOW"
        risk_color = "#00ff88"
    elif score >= 600:
        risk_level = "MEDIUM"
        risk_color = "#ffd700"
    else:
        risk_level = "HIGH"
        risk_color = "#ff4444"

    return score, grade, risk_level, risk_color


def render_credit_risk_module(ticker: str):
    """Render Credit Risk Modeling module."""
    with st.spinner("Training PD model..."):
        df = generate_synthetic_dataset(500)
        model, scaler, auc, acc, cm = train_pd_model(df)

    features = fetch_company_features(ticker)
    X_stock = np.array([[
        features["debt_to_equity"],
        features["interest_coverage"],
        features["current_ratio"],
        features["roe"],
        features["net_profit_margin"],
    ]])
    X_scaled = scaler.transform(X_stock)
    pd_prob = float(model.predict_proba(X_scaled)[0, 1]) * 100

    score, grade, risk_level, risk_color = pd_to_credit_score(pd_prob)

    col1, col2 = st.columns([2, 3])

    with col1:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 36, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [300, 850], "tickcolor": "#8892b0",
                         "tickfont": {"color": "#8892b0", "size": 9}},
                "bar": {"color": risk_color},
                "bgcolor": "#1a1f3a",
                "bordercolor": "#2d3561",
                "steps": [
                    {"range": [300, 500], "color": "rgba(255,68,68,0.15)"},
                    {"range": [500, 650], "color": "rgba(255,215,0,0.1)"},
                    {"range": [650, 850], "color": "rgba(0,255,136,0.1)"},
                ],
                "threshold": {
                    "line": {"color": risk_color, "width": 3},
                    "thickness": 0.8,
                    "value": score
                },
            },
            title={"text": "Credit Score", "font": {"color": "#8892b0", "size": 12}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d1117",
            height=240,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="credit_gauge")

        st.markdown(f"""
        <div style="background:#1a1f3a;border:1px solid #2d3561;border-radius:8px;padding:12px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#8892b0;font-size:11px;">PD (%)</span>
                <span style="color:#ff4444;font-size:16px;font-weight:700;">{pd_prob:.2f}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#8892b0;font-size:11px;">Risk Grade</span>
                <span style="color:{risk_color};font-size:16px;font-weight:700;">{grade}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8892b0;font-size:11px;">Risk Level</span>
                <span style="color:{risk_color};font-size:14px;font-weight:700;">{risk_level}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Model performance
        st.markdown("**Model Performance**")
        perf_col1, perf_col2 = st.columns(2)
        stat_style = """background:linear-gradient(135deg,#1a1f3a,#0d1117);
        border:1px solid #2d3561;border-radius:8px;padding:10px;text-align:center;"""
        with perf_col1:
            st.markdown(f'<div style="{stat_style}"><div style="color:#8892b0;font-size:10px;">AUC-ROC</div>'
                        f'<div style="color:#00d4ff;font-size:20px;font-weight:700;">{auc:.2f}</div></div>',
                        unsafe_allow_html=True)
        with perf_col2:
            st.markdown(f'<div style="{stat_style}"><div style="color:#8892b0;font-size:10px;">Accuracy</div>'
                        f'<div style="color:#00ff88;font-size:20px;font-weight:700;">{acc*100:.1f}%</div></div>',
                        unsafe_allow_html=True)

        # Confusion Matrix
        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(cm, columns=["Pred: Non-Default", "Pred: Default"],
                             index=["Actual: Non-Default", "Actual: Default"])
        st.dataframe(cm_df, use_container_width=True)

        # PD Trend (15-month simulation)
        base_features = X_stock[0].copy()
        trend_pds = []
        for m in range(15):
            varied = base_features + np.random.normal(0, 0.05, len(base_features))
            x_v = scaler.transform(varied.reshape(1, -1))
            pd_m = float(model.predict_proba(x_v)[0, 1]) * 100
            trend_pds.append(pd_m)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=list(range(1, 16)), y=trend_pds,
            mode="lines+markers",
            line=dict(color="#ff4444", width=2),
            marker=dict(size=5),
            name="PD Trend",
        ))
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=180,
            margin=dict(l=10, r=10, t=10, b=10),
            title=dict(text="15-Month PD Trend", font=dict(color="#8892b0", size=11)),
            xaxis=dict(title="Month", showgrid=True, gridcolor="#1e2433"),
            yaxis=dict(title="PD (%)", showgrid=True, gridcolor="#1e2433"),
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="credit_trend")

    return pd_prob
