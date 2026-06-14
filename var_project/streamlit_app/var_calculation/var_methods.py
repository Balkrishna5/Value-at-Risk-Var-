"""
VaR Calculation Module
Three methods:
  1. Historical Simulation
  2. Variance-Covariance (Parametric)
  3. Monte Carlo Simulation
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ── Shared Helper ──────────────────────────────────────────────────────────────
def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted daily portfolio returns."""
    return returns.values @ weights


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1 – Historical Simulation
# ═══════════════════════════════════════════════════════════════════════════════
def var_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    portfolio_value: float = 1_000_000,
    holding_days: int = 1,
) -> dict:
    """
    Historical Simulation VaR.
    Steps:
      1. Compute weighted portfolio daily returns (1-year look-back or full history)
      2. Sort returns ascending
      3. Pick the return at the (1-confidence) percentile → VaR return
      4. VaR ($) = portfolio_value × |VaR return| × sqrt(holding_days)
    """
    port_ret = portfolio_returns(returns, weights)
    sorted_ret = np.sort(port_ret)

    # Number of observations
    n = len(sorted_ret)
    # Rank corresponding to the tail
    rank = int(np.ceil((1 - confidence) * n))
    rank = max(rank, 1)

    var_return = sorted_ret[rank - 1]           # most negative tail return
    var_dollar = abs(var_return) * portfolio_value * np.sqrt(holding_days)
    es_returns = sorted_ret[:rank]              # tail returns beyond VaR
    es_return  = np.mean(es_returns)
    es_dollar  = abs(es_return) * portfolio_value * np.sqrt(holding_days)

    return {
        "method":          "Historical Simulation",
        "confidence":      confidence,
        "holding_days":    holding_days,
        "portfolio_value": portfolio_value,
        "var_return":      var_return,
        "var_dollar":      var_dollar,
        "es_dollar":       es_dollar,
        "tail_returns":    es_returns,
        "all_returns":     port_ret,
        "var_rank":        rank,
        "n_obs":           n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Method 2 – Variance-Covariance (Parametric)
# ═══════════════════════════════════════════════════════════════════════════════
def var_variance_covariance(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    portfolio_value: float = 1_000_000,
    holding_days: int = 1,
) -> dict:
    """
    Variance-Covariance (Parametric) VaR.
    Assumes normally distributed returns.
    VaR = portfolio_value × (μ_p + z × σ_p) × sqrt(holding_days)
    where z is the z-score at (1 - confidence).
    """
    port_ret = portfolio_returns(returns, weights)
    mu_p  = port_ret.mean()
    sig_p = port_ret.std(ddof=1)

    # Portfolio std via covariance matrix (more precise for multi-asset)
    w   = np.array(weights)
    cov = returns.cov().values
    sig_p_cov = np.sqrt(w @ cov @ w)      # daily std from cov matrix

    z = stats.norm.ppf(1 - confidence)    # negative (left tail)

    var_return = mu_p + z * sig_p_cov
    var_dollar = abs(var_return) * portfolio_value * np.sqrt(holding_days)

    # Expected Shortfall under normality
    phi_z = stats.norm.pdf(z)
    es_return = -(mu_p - sig_p_cov * phi_z / (1 - confidence))
    es_dollar = abs(es_return) * portfolio_value * np.sqrt(holding_days)

    return {
        "method":          "Variance-Covariance",
        "confidence":      confidence,
        "holding_days":    holding_days,
        "portfolio_value": portfolio_value,
        "var_return":      var_return,
        "var_dollar":      var_dollar,
        "es_dollar":       es_dollar,
        "mu_p":            mu_p,
        "sigma_p":         sig_p_cov,
        "z_score":         z,
        "all_returns":     port_ret,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Method 3 – Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════════
def var_monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    portfolio_value: float = 1_000_000,
    holding_days: int = 1,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo VaR using Geometric Brownian Motion (GBM) + Cholesky decomposition.

    Steps:
      1. Estimate mean (μ) and std (σ) of each stock from history
      2. Build correlation matrix; decompose via Cholesky → L
      3. Generate n_simulations × n_stocks uncorrelated N(0,1) shocks Z
      4. Correlated shocks W = Z @ L.T
      5. Simulate 1-day returns per stock: r = μ*dt + σ*W*sqrt(dt)
      6. Portfolio return = weights @ r  for each simulation
      7. Sort → percentile approach for VaR
    """
    rng = np.random.default_rng(seed)
    n_assets = len(returns.columns)
    dt = 1  # 1-day; holding_days applied via sqrt scaling

    mu  = returns.mean().values       # daily mean
    sig = returns.std(ddof=1).values  # daily std

    # Correlation matrix → Cholesky
    corr = returns.corr().values
    # Ensure positive-definite
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Fallback: nearst PD
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-8)
        corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(corr_pd)

    # Simulate uncorrelated standard normal shocks
    Z = rng.standard_normal((n_simulations, n_assets))
    # Correlated shocks via Cholesky
    W = Z @ L.T

    # GBM-style 1-day return for each asset
    sim_asset_returns = mu + sig * W

    # Portfolio return for each simulation
    w = np.array(weights)
    sim_port_returns = sim_asset_returns @ w

    sorted_sim = np.sort(sim_port_returns)
    var_idx    = int(np.ceil((1 - confidence) * n_simulations)) - 1
    var_idx    = max(var_idx, 0)

    var_return = sorted_sim[var_idx]
    var_dollar = abs(var_return) * portfolio_value

    es_returns = sorted_sim[: var_idx + 1]
    es_return  = np.mean(es_returns)
    es_dollar  = abs(es_return) * portfolio_value

    return {
        "method":           "Monte Carlo Simulation",
        "confidence":       confidence,
        "holding_days":     holding_days,
        "portfolio_value":  portfolio_value,
        "var_return":       var_return,
        "var_dollar":       var_dollar,
        "es_dollar":        es_dollar,
        "sim_returns":      sim_port_returns,
        "n_simulations":    n_simulations,
        "var_idx":          var_idx,
    }


# ── Summary Table ──────────────────────────────────────────────────────────────
def var_summary(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Method":           r["method"],
            "Confidence Level": f"{r['confidence']*100:.0f}%",
            "Holding Days":     r["holding_days"],
            "VaR Return (%)":   f"{r['var_return']*100:.4f}%",
            "VaR ($)":          f"${r['var_dollar']:,.0f}",
            "ES ($)":           f"${r['es_dollar']:,.0f}",
        })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from portfolio_construction.portfolio import fetch_data, compute_returns, optimize_portfolio

    print("Fetching data …")
    prices  = fetch_data()
    returns = compute_returns(prices)
    result  = optimize_portfolio(returns)
    weights = result["optimal_weights"]

    print("Computing VaR …\n")
    r1 = var_historical(returns, weights)
    r2 = var_variance_covariance(returns, weights)
    r3 = var_monte_carlo(returns, weights)

    df = var_summary([r1, r2, r3])
    print(df.to_string(index=False))
