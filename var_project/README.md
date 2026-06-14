# VaR Risk Modeling Dashboard

Comprehensive Value at Risk (VaR) modeling project covering portfolio construction,
three VaR calculation methods, two backtesting approaches, and a Streamlit dashboard.

## Project Structure

```
var_project/
├── portfolio_construction/
│   └── portfolio.py          ← Markowitz Mean-Variance Optimization
├── var_calculation/
│   └── var_methods.py        ← Historical, Variance-Covariance, Monte Carlo
├── var_backtesting/
│   └── backtesting.py        ← Traffic Light & Kupiec POF Test
├── streamlit_app/
│   └── app.py                ← Interactive Streamlit Dashboard
├── data/
│   └── stock_prices.csv      ← Pre-generated stock price data
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Run individual modules
python portfolio_construction/portfolio.py
python var_calculation/var_methods.py        # set PYTHONPATH=. first
python var_backtesting/backtesting.py

# Launch the dashboard
cd streamlit_app
streamlit run app.py
```

## Stocks
JPMorgan Chase (JPM) · Morgan Stanley (MS) · Bank of America (BAC)
Data: 2019–2024 (synthetic correlated GBM, calibrated to historical parameters)

## Methodology

### 1. Portfolio Construction
- **Markowitz Mean-Variance Optimization**
- Maximize Sharpe Ratio: max (μₚ − Rf) / σₚ
- Weights constrained: Σwᵢ = 1, wᵢ ≥ 0
- Efficient frontier plotted from 5,000 Monte Carlo portfolios

### 2. VaR Methods

| Method | Type | Key Assumption |
|--------|------|----------------|
| Historical Simulation | Non-parametric | Past returns repeat |
| Variance-Covariance | Parametric | Normal distribution |
| Monte Carlo | Simulation-based | GBM + Cholesky correlation |

### 3. Backtesting

| Test | Framework | Decision Rule |
|------|-----------|---------------|
| Traffic Light | Basel III | Green/Yellow/Red by breach count in 250 days |
| Kupiec POF | Statistical | LR ~ χ²(1); reject if LR > 3.841 |
