# Value at Risk (VaR) Analysis

> Quantitative risk estimation for financial portfolios using Historical Simulation, Variance-Covariance, and Monte Carlo methods — with backtesting and performance validation.

---

## Overview

This project implements three widely-used Value at Risk (VaR) methodologies to measure downside risk in financial assets and portfolios. It covers the full pipeline from data collection and return computation to VaR estimation, Expected Shortfall (CVaR), and rigorous backtesting.

The analysis is documented in `VaR_Report.docx`, which details the theoretical foundations, methodology, results, and interpretation for each approach.

---

## Methods Implemented

| Method | Assumption | Key Feature |
|---|---|---|
| **Historical Simulation** | Non-parametric | Uses actual empirical return distribution |
| **Variance-Covariance (Parametric)** | Normal returns | Analytical; fast for large portfolios |
| **Monte Carlo Simulation** | Flexible | Handles non-linear instruments and fat tails |

---

## Key Features

- **VaR Estimation** at 95% and 99% confidence levels (1-day and multi-day horizons)
- **Expected Shortfall (CVaR)** — average loss beyond the VaR threshold
- **Backtesting** — Kupiec POF test and exception rate analysis to validate model accuracy
- **Return Distribution Analysis** — normality tests, skewness, kurtosis, Q-Q plots
- **Visualizations** — return histograms, VaR breach plots, loss distribution curves

---

## Project Structure

```
Value-at-Risk-Var-/
│
├── VaR_Report.docx          # Full project report (methodology, results, interpretation)
├── var_analysis.py          # Main VaR computation script
├── data/                    # Historical price/return data
├── plots/                   # Output charts and figures
└── README.md
```

---

## Tech Stack

- **Python 3.x**
- `numpy`, `pandas` — data manipulation and return computation
- `scipy` — statistical distributions and hypothesis tests
- `matplotlib`, `seaborn` — visualization
- `yfinance` — market data retrieval

---

## Methodology Summary

### 1. Historical Simulation
Returns are sorted empirically and VaR is read off at the desired quantile — no distributional assumption required. Captures fat tails and skewness present in actual market data.

### 2. Variance-Covariance
Assumes log returns are normally distributed. VaR is computed analytically:

```
VaR = μ - z_α · σ
```

where `z_α` is the standard normal quantile at confidence level α. Fast and interpretable, but underestimates tail risk in non-normal markets.

### 3. Monte Carlo Simulation
Simulates a large number of return paths (typically 10,000+) via Geometric Brownian Motion or empirically calibrated distributions. VaR is extracted from the simulated loss distribution at the target quantile.

---

## Backtesting

Model performance is validated using:
- **Kupiec's POF (Proportion of Failures) Test** — checks if observed exception rate matches the theoretical confidence level
- **Exception Rate Analysis** — counts VaR breaches over the backtesting window and flags over/under-estimation

---

## Results Summary

| Model | 95% VaR (1-day) | 99% VaR (1-day) | Exceptions (Kupiec) |
|---|---|---|---|
| Historical Simulation | — | — | — |
| Variance-Covariance | — | — | — |
| Monte Carlo | — | — | — |

> Fill in results from `VaR_Report.docx` after running the analysis.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Balkrishna5/Value-at-Risk-Var-.git
cd Value-at-Risk-Var-

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn yfinance

# Run the analysis
python var_analysis.py
```

---

## Academic Context

This project was completed as part of the Data Science & AI curriculum at **IIT Guwahati**, with a focus on quantitative finance applications. It demonstrates core risk management concepts aligned with Basel III market risk standards.

---

## References

- Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk*.
- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.). McGraw-Hill.

---

## Author

**Balkrishna**  
B.Tech, Data Science & AI — IIT Guwahati  
[GitHub](https://github.com/Balkrishna5)

---

## License

This project is intended for academic and educational purposes.
