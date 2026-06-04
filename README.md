# Risk Analytics Dashboard
### MCA · Financial Analytics · Capstone Project

> **Student:** Jovina Mariya Joshi · **Roll No:** LC25MCA035 · **Batch:** 2025  
> **College:** LEAD College (Autonomous) · **Submitted:** June 2026

A fully functional, 10-module professional Risk Analytics Dashboard built with **Python, Streamlit, and Plotly**.  
Supports live ticker switching across 5 NSE/BSE-listed stocks with auto-refresh of all modules on every user interaction.

---

## 🚀 Quick Start

```bash
# 1. Unzip and enter the project folder
cd risk_dashboard

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Modules

| # | Module | Marks | Key Deliverables |
|---|--------|:-----:|-----------------|
| 1 | Executive Summary Panel | 15 | KPI cards, sparklines, BUY/HOLD/SELL signal, Risk Level |
| 2 | ARIMA Forecasting | 15 | Auto ARIMA, 90-day forecast, 95% CI, walk-forward validation |
| 3 | GARCH Volatility Modeling | 12 | GARCH(1,1), regime detection (Low/Moderate/High), spike annotation |
| 4 | DCF Valuation | 13 | Gordon Growth Model, waterfall chart, Margin of Safety |
| 5 | Monte Carlo Simulation | 13 | GBM paths (1k–10k), probability summary, interactive sliders |
| 6 | Value at Risk (VaR) | 15 | Historical + Parametric + Monte Carlo VaR, CVaR, Kupiec test |
| 7 | Credit Risk Modeling | 13 | Logistic regression PD (AUC=0.74), credit score gauge, confusion matrix |
| 8 | Portfolio Optimization | 12 | Efficient frontier (5,000 portfolios), max-Sharpe allocation, asset toggle |
| 9 | Stress Testing | 10 | 5 predefined + 1 custom scenario, factor betas, risk narrative |
| 10 | Correlation Heatmap | 8 | Pearson correlation, ⚠ annotations (|ρ|>0.70), diversification insights |
| | **Base Total** | **130** | |
| | Bonus — Deployed App | +20 | Streamlit Cloud public URL |

---

## 📁 Project Structure

```
risk_dashboard/
├── app.py                         ← Main Streamlit app (routing, sidebar, layout)
├── requirements.txt               ← All 13 dependencies pinned to specific versions
├── packages.txt                   ← System dependencies for Streamlit Cloud
├── README.md                      ← This file
├── .streamlit/
│   └── config.toml                ← Dark theme, server config
└── modules/
    ├── __init__.py
    ├── data_utils.py              ← yfinance fetch, caching, preprocessing, synthetic fallback
    ├── module1_executive.py       ← KPI computation, sparklines, investment signal logic
    ├── module2_arima.py           ← auto_arima, 90-day forecast, CI bands, walk-forward
    ├── module3_garch.py           ← GARCH(1,1), conditional volatility, regime detection
    ├── module4_dcf.py             ← DCF model, Gordon Growth TV, waterfall chart, MoS
    ├── module5_montecarlo.py      ← GBM simulation, path visualisation, probability table
    ├── module6_var.py             ← 3×VaR methods, CVaR, loss distribution, Kupiec POF
    ├── module7_credit.py          ← Logistic PD model, credit score gauge, confusion matrix
    ├── module8_portfolio.py       ← Efficient frontier, max-Sharpe, optimal allocation pie
    ├── module9_stress.py          ← Stress scenarios, factor betas, impact bar chart
    └── module10_heatmap.py        ← Correlation matrix, heatmap, diversification insights
```

---

## 📈 Supported Tickers

| Ticker | Company | Approx. Price | Annual Vol |
|--------|---------|:-------------:|:----------:|
| RELIANCE.NS | Reliance Industries | ₹2,980 | 22.45% |
| TCS.NS | Tata Consultancy Services | ₹3,900 | 20.18% |
| INFY.NS | Infosys Ltd. | ₹1,750 | 22.87% |
| HDFCBANK.NS | HDFC Bank Ltd. | ₹1,620 | 18.39% |
| WIPRO.NS | Wipro Ltd. | ₹480 | 25.55% |

---

## 🔧 Technology Stack

| Category | Libraries |
|----------|-----------|
| Dashboard | `streamlit==1.41.1`, `streamlit-option-menu==0.3.13` |
| Visualisation | `plotly==5.24.1` |
| Data & APIs | `pandas==2.2.3`, `numpy==1.26.4`, `yfinance==0.2.51` |
| Time-Series | `pmdarima==2.0.4`, `statsmodels==0.14.4` |
| Volatility | `arch==7.2.0` |
| ML / Stats | `scikit-learn==1.5.2`, `scipy==1.14.1` |
| Portfolio | `PyPortfolioOpt==1.5.6`, `cvxpy==1.6.0` |

---

## 📐 Model Details

### Module 2 — ARIMA Forecasting
- `pmdarima.auto_arima(stepwise=True, seasonal=False)` — selects optimal (p,d,q) by AIC
- 90 trading-day forecast with 95% confidence intervals
- Walk-forward validation: 80% train / 20% test, rolling 1-step-ahead predictions

### Module 3 — GARCH(1,1)
- `arch_model(scaled_returns, vol='Garch', p=1, q=1)` — returns scaled ×100 for stability
- Annualised conditional volatility = σ_t × √252
- Regime: **High** (>P75), **Moderate** (P25–P75), **Low** (<P25)

### Module 4 — DCF Valuation
- Operating cash flows from `yfinance.Ticker.cashflow`
- Terminal Value (Gordon Growth): `TV = FCF_n × (1+g) / (WACC−g)`
- Margin of Safety = `(Intrinsic − Market) / Intrinsic × 100`

### Module 5 — Monte Carlo GBM
- `S(t+1) = S(t) × exp((μ − 0.5σ²)dt + σ√dt × Z)`, Z∼N(0,1)
- Configurable: 500–10,000 paths, 63–252 trading days
- Paths colour-coded: Top 5% green, Bottom 5% red, Others blue

### Module 6 — Value at Risk
- **Historical**: `VaR = percentile(returns, 5%)`
- **Parametric**: `VaR = μ + σ × Φ⁻¹(0.05)`
- **Monte Carlo**: percentile of 10,000 simulated 1-day P&L paths
- **CVaR**: mean loss in worst 5% of scenarios
- **Kupiec POF Test**: χ²(1) likelihood ratio test, p > 0.05 = Valid

### Module 7 — Credit Risk
- Logistic regression on 500 synthetic companies; **AUC-ROC = 0.7412** ✅
- Features: D/E, Interest Coverage, Current Ratio, ROE, Net Profit Margin
- Credit score: `850 − (PD% / 100) × 550`, mapped to AAA–D grades

### Module 8 — Portfolio Optimization
- 5,000 random Dirichlet weight combinations
- Max-Sharpe portfolio identified from frontier (risk-free rate = 6% p.a.)
- Assets: 5 NSE stocks + simulated Bond + Gold proxy

### Module 9 — Stress Testing
- Factor betas from OLS regression against NIFTY 50 (^NSEI)
- Impact = β_market × Δmarket + β_rate × Δrate + β_oil × Δoil
- Fallback beta estimation via volatility ratio when NIFTY data unavailable

### Module 10 — Correlation Heatmap
- `pandas.DataFrame.corr()` on daily log returns across 7 assets
- Cells with |ρ| > 0.70 annotated with ⚠ (high co-movement warning)
- Dynamic insights: most diversifying pair and most redundant pair

---

## ⚠️ Note on Data

The dashboard uses **Yahoo Finance (yfinance)** as its primary data source for NSE/BSE tickers.  
If Yahoo Finance is unreachable (network restrictions, sandbox environments), the dashboard automatically falls back to **realistic GBM-simulated price data** calibrated to each stock's known approximate price level and volatility. A small `ℹ️` caption appears when simulated data is in use.

---

## 🤖 AI Tool Usage Declaration

Portions of the boilerplate code structure and documentation were generated with assistance from **Claude (Anthropic)**. All financial models, mathematical formulas, and statistical implementations were verified against course materials and standard academic references. The student is responsible for understanding and being able to explain every line of code in this repository.

---

## 📋 Submission Checklist

- [x] GitHub repository with full commit history
- [x] `requirements.txt` with all dependencies pinned
- [x] `README.md` with setup instructions and model documentation
- [x] Project report (PDF) — 10 pages
- [x] Source code (`risk_analytics_dashboard.zip`)
- [ ] Live dashboard URL *(Streamlit Cloud deployment — optional, +20 bonus)*

---

*Built with ❤️ · Python + Streamlit + Plotly · MCA Financial Analytics Capstone*
