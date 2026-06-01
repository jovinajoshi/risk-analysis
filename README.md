# Risk Analytics Dashboard
### MCA · Financial Analytics · Capstone Project

A fully functional, 10-module professional Risk Analytics Dashboard built with Python, Streamlit, and Plotly. Supports live ticker switching across 5 NSE/BSE-listed stocks with auto-refresh on ticker change.

---

## 📊 Dashboard Modules

| # | Module | Marks | Description |
|---|--------|-------|-------------|
| 1 | Executive Summary Panel | 15 | KPI cards, investment signal, risk level |
| 2 | ARIMA Forecasting | 15 | Auto ARIMA, 90-day forecast, walk-forward validation |
| 3 | GARCH Volatility Modeling | 12 | GARCH(1,1), regime detection, spike annotation |
| 4 | DCF Valuation | 13 | Discounted Cash Flow, waterfall chart, margin of safety |
| 5 | Monte Carlo Simulation | 13 | GBM paths, probability summary, interactive controls |
| 6 | Value at Risk (VaR) | 15 | 3 methods, CVaR, Kupiec backtesting |
| 7 | Credit Risk Modeling | 13 | Logistic regression PD, credit score gauge |
| 8 | Portfolio Optimization | 12 | Efficient frontier, max-Sharpe allocation |
| 9 | Stress Testing | 10 | 5 scenarios + custom, factor betas, risk narrative |
| 10 | Correlation Heatmap | 8 | Portfolio correlation, diversification insights |

**Base Total: 130 marks | Bonus (deployed): +20 marks**

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/risk-analytics-dashboard.git
cd risk-analytics-dashboard

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📁 Project Structure

```
risk_dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Pinned dependencies
├── README.md                 # This file
└── modules/
    ├── __init__.py
    ├── data_utils.py         # Data fetching & preprocessing utilities
    ├── module1_executive.py  # Executive Summary Panel (Module 1)
    ├── module2_arima.py      # ARIMA Forecasting (Module 2)
    ├── module3_garch.py      # GARCH Volatility (Module 3)
    ├── module4_dcf.py        # DCF Valuation (Module 4)
    ├── module5_montecarlo.py # Monte Carlo Simulation (Module 5)
    ├── module6_var.py        # Value at Risk (Module 6)
    ├── module7_credit.py     # Credit Risk Modeling (Module 7)
    ├── module8_portfolio.py  # Portfolio Optimization (Module 8)
    ├── module9_stress.py     # Stress Testing (Module 9)
    └── module10_heatmap.py   # Correlation Heatmap (Module 10)
```

---

## 📈 Supported Tickers

| Ticker | Company |
|--------|---------|
| RELIANCE.NS | Reliance Industries |
| TCS.NS | Tata Consultancy Services |
| INFY.NS | Infosys |
| HDFCBANK.NS | HDFC Bank |
| WIPRO.NS | Wipro |

---

## 🔧 Technical Stack

| Category | Libraries |
|----------|-----------|
| Dashboard | Streamlit, streamlit-option-menu |
| Visualisation | Plotly, Plotly Express |
| Data & APIs | pandas, numpy, yfinance |
| Statistical Modeling | statsmodels, arch (GARCH), pmdarima |
| Machine Learning | scikit-learn |
| Portfolio Optimization | PyPortfolioOpt, scipy |
| Scientific Computing | scipy, numpy |

---

## 📐 Model Details

### Module 2 — ARIMA Forecasting
- Uses `pmdarima.auto_arima` with `stepwise=True, seasonal=False`
- Selects optimal (p,d,q) by minimising AIC
- 90-day forecast with 95% confidence intervals
- Walk-forward validation: 80% train / 20% test, rolling 1-step-ahead

### Module 3 — GARCH(1,1)
- Fitted using the `arch` library on log returns × 100 (scaled for stability)
- Conditional volatility annualised via √252
- Regime: High (>P75), Moderate (P25–P75), Low (<P25)

### Module 4 — DCF Valuation
- Operating cash flows from `yfinance.Ticker.cashflow`
- Terminal Value = FCF_n × (1 + g) / (WACC - g) — Gordon Growth Model
- Margin of Safety = (Intrinsic − Market) / Intrinsic × 100

### Module 5 — Monte Carlo (GBM)
- S(t+1) = S(t) × exp((μ − 0.5σ²)Δt + σ√Δt × Z), Z~N(0,1)
- Configurable: 500–10,000 paths, 63–252 trading days
- Paths colour-coded: Top 5% green, Bottom 5% red, Others blue

### Module 6 — Value at Risk
- **Historical Simulation**: empirical percentile of daily returns
- **Parametric Normal**: μ + σ × Φ⁻¹(α)
- **Monte Carlo**: percentile of 10,000 simulated 1-day P&L paths
- CVaR = mean of losses beyond VaR threshold
- **Kupiec POF Test**: χ² test for model validity at 5% significance

### Module 7 — Credit Risk
- Logistic regression trained on 500 synthetic companies
- Features: D/E, Interest Coverage, Current Ratio, ROE, Net Profit Margin
- Credit score mapped to 300–850 scale; AUC-ROC target ≥ 0.70
- 15-month PD trend via slightly perturbed input features

### Module 8 — Portfolio Optimization
- 5,000 random weight combinations (Monte Carlo sampling)
- Optimal portfolio = max Sharpe Ratio (risk-free rate: 6% p.a. for India)
- Assets: 5 NSE stocks + simulated Bond + Gold

### Module 9 — Stress Testing
- Factor betas from OLS regression of stock returns on NIFTY 50
- 5 predefined + 1 custom user-defined scenario
- Impact = β_market × market_shock + β_rate × rate_shock + β_oil × oil_shock

### Module 10 — Correlation Heatmap
- Daily log returns correlation matrix via `pandas.DataFrame.corr()`
- Cells with |ρ| > 0.70 annotated with ⚠ warning
- Dynamic insights: most diversifying and most redundant pair

---

## 🤖 AI Tool Usage Declaration

Parts of the boilerplate code structure were generated with AI assistance (Claude by Anthropic). All financial models, formulas, and statistical implementations were verified against course materials and standard references. The student is responsible for understanding and being able to explain every line of code.

---

## 🌐 Deployment (Bonus — +20 marks)

To deploy on Streamlit Cloud:
1. Push repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app.py`
5. Deploy

---

## 📚 Data Sources

- **Price Data**: Yahoo Finance via `yfinance` (NSE suffix: `.NS`)
- **Financial Ratios**: `yfinance.Ticker.info` dictionary
- **Cash Flow**: `yfinance.Ticker.cashflow`
- **Market Benchmark**: NIFTY 50 (`^NSEI`) for beta computation
- **Bond/Gold**: Simulated using GBM with realistic parameters

---

## ⚠️ Limitations & Future Improvements

**Limitations:**
- ARIMA fitting time increases with data length (30–60 seconds for 5 years)
- DCF accuracy depends on cash flow data availability in yfinance
- Credit risk model uses synthetic training data (not real default data)
- Monte Carlo assumes constant drift and volatility (no regime switching)

**Future Improvements:**
- Integrate real credit default datasets (Altman Z-score integration)
- Add LSTM/Prophet forecasting as alternative to ARIMA
- Implement multi-factor risk models (Fama-French 3-factor)
- Add real-time WebSocket data streaming
- Integrate SEBI/NSE API for live institutional data

---

*Built with ❤️ for MCA Financial Analytics Capstone | Python + Streamlit + Plotly*
