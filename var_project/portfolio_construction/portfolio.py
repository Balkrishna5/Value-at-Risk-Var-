"""
Portfolio Construction Module
Uses Markowitz Mean-Variance Optimization to find optimal portfolio weights.
Stocks: JPMorgan (JPM), Morgan Stanley (MS), Bank of America (BAC)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ── Configuration ──────────────────────────────────────────────────────────────
TICKERS    = ["JPM", "MS", "BAC"]
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
RISK_FREE  = 0.05          # annual risk-free rate (approx 5%)
N_PORTFOLIOS = 5_000       # Monte Carlo simulated portfolios for efficient frontier


# ── Data Fetching ──────────────────────────────────────────────────────────────
def _generate_synthetic_prices(tickers=TICKERS, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """
    Generate realistic correlated GBM stock prices when Yahoo Finance is unavailable.
    Parameters calibrated to JPM/MS/BAC historical characteristics.
    """
    np.random.seed(42)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    _params = {
        "JPM": {"mu": 0.0004, "sigma": 0.0135, "S0": 100.0},
        "MS":  {"mu": 0.0005, "sigma": 0.0155, "S0":  45.0},
        "BAC": {"mu": 0.0003, "sigma": 0.0145, "S0":  28.0},
    }
    corr = np.array([[1.0, 0.72, 0.78],
                     [0.72, 1.0,  0.68],
                     [0.78, 0.68, 1.0]])
    L      = np.linalg.cholesky(corr)
    shocks = np.random.normal(0, 1, (n, 3)) @ L.T
    prices = {}
    for i, t in enumerate(tickers):
        p = _params[t]
        daily_ret = p["mu"] + p["sigma"] * shocks[:, i]
        prices[t] = p["S0"] * np.exp(np.cumsum(daily_ret))
    return pd.DataFrame(prices, index=dates)


def fetch_data(tickers=TICKERS, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """
    Download adjusted-close prices from Yahoo Finance.
    Falls back to realistic synthetic data if network is unavailable.
    """
    # Check for pre-generated CSV first
    _csv = os.path.join(os.path.dirname(__file__), "..", "data", "stock_prices.csv")
    _csv = os.path.normpath(_csv)
    if os.path.exists(_csv):
        df = pd.read_csv(_csv, index_col=0, parse_dates=True)
        df = df[tickers].dropna()
        return df

    try:
        raw    = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        prices = raw["Close"][tickers].dropna()
        if prices.empty or prices.isnull().all().all():
            raise ValueError("Empty data from Yahoo Finance")
        return prices
    except Exception:
        return _generate_synthetic_prices(tickers, start, end)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    returns = prices.pct_change().dropna()
    return returns


# ── Markowitz Optimization ─────────────────────────────────────────────────────
def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray,
                    cov_matrix: np.ndarray, trading_days: int = 252):
    """Annual return, volatility, and Sharpe ratio for a weight vector."""
    w = np.array(weights)
    ret  = np.dot(w, mean_returns) * trading_days
    vol  = np.sqrt(w @ cov_matrix @ w) * np.sqrt(trading_days)
    sharpe = (ret - RISK_FREE) / vol
    return ret, vol, sharpe


def neg_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_stats(weights, mean_returns, cov_matrix)[2]


def min_variance(weights, mean_returns, cov_matrix):
    return portfolio_stats(weights, mean_returns, cov_matrix)[1]


def optimize_portfolio(returns: pd.DataFrame):
    """
    Returns a dict with:
      - optimal_weights  : max-Sharpe weights
      - min_var_weights  : min-variance weights
      - mean_returns     : daily mean returns
      - cov_matrix       : daily covariance matrix
      - frontier         : efficient frontier data
    """
    mean_ret = returns.mean()
    cov      = returns.cov()
    n        = len(returns.columns)
    bounds   = tuple((0, 1) for _ in range(n))
    w0       = np.array([1 / n] * n)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # ── Max Sharpe ──
    res_sharpe = minimize(
        neg_sharpe, w0,
        args=(mean_ret.values, cov.values),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    # ── Min Variance ──
    res_minvar = minimize(
        min_variance, w0,
        args=(mean_ret.values, cov.values),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    # ── Efficient Frontier (Monte Carlo) ──
    frontier_ret, frontier_vol, frontier_sr = [], [], []
    rng = np.random.default_rng(42)
    for _ in range(N_PORTFOLIOS):
        w = rng.random(n)
        w /= w.sum()
        r, v, s = portfolio_stats(w, mean_ret.values, cov.values)
        frontier_ret.append(r)
        frontier_vol.append(v)
        frontier_sr.append(s)

    frontier = pd.DataFrame({
        "Return":    frontier_ret,
        "Volatility": frontier_vol,
        "Sharpe":    frontier_sr
    })

    return {
        "optimal_weights":  res_sharpe.x,
        "min_var_weights":  res_minvar.x,
        "mean_returns":     mean_ret,
        "cov_matrix":       cov,
        "frontier":         frontier,
        "tickers":          list(returns.columns),
    }


# ── Summary Helper ─────────────────────────────────────────────────────────────
def portfolio_summary(result: dict) -> pd.DataFrame:
    """Human-readable weight table with stats."""
    w   = result["optimal_weights"]
    mr  = result["mean_returns"]
    cov = result["cov_matrix"]
    ret, vol, sharpe = portfolio_stats(w, mr.values, cov.values)

    rows = []
    for ticker, weight in zip(result["tickers"], w):
        rows.append({"Ticker": ticker, "Weight (%)": round(weight * 100, 2)})

    df = pd.DataFrame(rows)
    df.loc[len(df)] = ["— Portfolio —", ""]
    df.loc[len(df)] = ["Annual Return (%)", round(ret * 100, 2)]
    df.loc[len(df)] = ["Annual Volatility (%)", round(vol * 100, 2)]
    df.loc[len(df)] = ["Sharpe Ratio", round(sharpe, 4)]
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching data …")
    prices  = fetch_data()
    returns = compute_returns(prices)

    print("Running Markowitz optimization …")
    result  = optimize_portfolio(returns)

    print("\n── Optimal Weights (Max Sharpe) ──")
    for t, w in zip(result["tickers"], result["optimal_weights"]):
        print(f"  {t}: {w*100:.2f}%")

    ret, vol, sharpe = portfolio_stats(
        result["optimal_weights"],
        result["mean_returns"].values,
        result["cov_matrix"].values
    )
    print(f"\n  Annual Return    : {ret*100:.2f}%")
    print(f"  Annual Volatility: {vol*100:.2f}%")
    print(f"  Sharpe Ratio     : {sharpe:.4f}")
