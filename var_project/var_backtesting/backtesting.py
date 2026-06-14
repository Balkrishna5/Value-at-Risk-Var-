"""
VaR Backtesting Module
Two methods:
  1. Traffic Light Approach (Basel framework)
  2. Kupiec Proportion of Failures (POF) Test
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# Shared: Count Breaches
# ═══════════════════════════════════════════════════════════════════════════════
def count_breaches(port_returns: np.ndarray, var_return: float) -> dict:
    """
    Given actual daily portfolio returns and a daily VaR return threshold,
    find days where the actual loss exceeded VaR (breaches / violations).

    port_returns: array of actual daily returns (negative = loss)
    var_return  : VaR return threshold (e.g. -0.015 for -1.5%)
    """
    # A breach occurs when the actual return is worse (more negative) than VaR
    breaches   = port_returns < var_return
    n_breaches = int(breaches.sum())
    n_obs      = len(port_returns)
    breach_rate = n_breaches / n_obs
    return {
        "breaches":    breaches,
        "n_breaches":  n_breaches,
        "n_obs":       n_obs,
        "breach_rate": breach_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1 – Traffic Light Approach (Basel)
# ═══════════════════════════════════════════════════════════════════════════════
TRAFFIC_THRESHOLDS = {
    "green":  (0, 4),    # 0–4 breaches in 250 days → Low Risk
    "yellow": (5, 9),    # 5–9 breaches             → Moderate Risk
    "red":    (10, None) # ≥10 breaches              → High Risk
}

ZONE_DESCRIPTIONS = {
    "green":  {
        "label":       "Green Zone (Low Risk)",
        "range":       "0–4 breaches",
        "implication": "Risk levels are low. Frequency of breaches is within expected range. "
                       "Model is performing well and is within acceptable limits.",
        "action":      "Normal monitoring and operations can continue without immediate concern.",
        "color":       "#22c55e",
    },
    "yellow": {
        "label":       "Yellow Zone (Moderate Risk)",
        "range":       "5–9 breaches",
        "implication": "Risk levels are moderate. The frequency of breaches is higher and nearing "
                       "the upper tolerance, indicating the model might not capture some risks effectively.",
        "action":      "Increase scrutiny, review portfolio strategies, perform daily monitoring.",
        "color":       "#f59e0b",
    },
    "red":    {
        "label":       "Red Zone (High Risk)",
        "range":       "≥10 breaches",
        "implication": "Risk levels are high with significant frequency of breaches, suggesting "
                       "model is inadequate or there are severe risk-management issues.",
        "action":      "Immediate recalibration of the model and reassessment of the portfolio required.",
        "color":       "#ef4444",
    },
}


def traffic_light_backtest(
    port_returns: pd.Series | np.ndarray,
    var_return: float,
    confidence: float = 0.99,
    window: int = 250,
) -> dict:
    """
    Basel Traffic Light Approach.
    Uses a rolling window of `window` trading days (default 250 = ~1 year).
    Classifies model into Green / Yellow / Red zone based on breach count.
    """
    ret_arr = np.asarray(port_returns)

    # Use the last `window` days for backtesting
    if len(ret_arr) > window:
        bt_returns = ret_arr[-window:]
    else:
        bt_returns = ret_arr

    breach_info = count_breaches(bt_returns, var_return)
    n_b = breach_info["n_breaches"]

    # Determine zone
    if n_b <= 4:
        zone = "green"
    elif n_b <= 9:
        zone = "yellow"
    else:
        zone = "red"

    # Build breach time series for plotting
    breach_dates = np.where(breach_info["breaches"])[0]

    # Expected breaches under H0
    expected_breaches = (1 - confidence) * len(bt_returns)

    return {
        "method":             "Traffic Light Approach",
        "window":             len(bt_returns),
        "confidence":         confidence,
        "var_return":         var_return,
        "n_breaches":         n_b,
        "expected_breaches":  expected_breaches,
        "breach_rate":        breach_info["breach_rate"],
        "zone":               zone,
        "zone_info":          ZONE_DESCRIPTIONS[zone],
        "bt_returns":         bt_returns,
        "breach_indices":     breach_dates,
        "all_breaches":       breach_info["breaches"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Method 2 – Kupiec POF (Proportion of Failures) Test
# ═══════════════════════════════════════════════════════════════════════════════
def kupiec_pof_test(
    port_returns: pd.Series | np.ndarray,
    var_return: float,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> dict:
    """
    Kupiec (1995) Proportion of Failures test.

    H0: p̂ = p  (observed failure rate equals expected)
    H1: p̂ ≠ p

    Test statistic:
      LR_pof = -2 * ln[ p^x * (1-p)^(n-x) / (p̂^x * (1-p̂)^(n-x)) ]

    LR_pof ~ χ²(1) under H0.
    Reject H0 (model inadequate) if LR_pof > χ²_{1-α}(1).
    """
    ret_arr = np.asarray(port_returns)
    breach_info = count_breaches(ret_arr, var_return)

    x = breach_info["n_breaches"]   # observed failures
    n = breach_info["n_obs"]        # total observations
    p = 1 - confidence              # expected failure probability (e.g. 0.01 for 99% CI)
    p_hat = x / n if n > 0 else 0  # observed failure rate

    # Compute LR statistic
    # Handle edge cases
    eps = 1e-12
    p_hat_safe = np.clip(p_hat, eps, 1 - eps)

    try:
        log_h0 = x * np.log(p)       + (n - x) * np.log(1 - p)
        log_h1 = x * np.log(p_hat_safe) + (n - x) * np.log(1 - p_hat_safe)
        lr_stat = -2 * (log_h0 - log_h1)
    except (ValueError, ZeroDivisionError):
        lr_stat = np.nan

    # Chi-squared critical value with 1 dof
    chi2_crit = stats.chi2.ppf(1 - significance, df=1)

    # p-value
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1) if not np.isnan(lr_stat) else np.nan

    reject_h0 = bool(lr_stat > chi2_crit) if not np.isnan(lr_stat) else False
    model_adequate = not reject_h0

    return {
        "method":            "Kupiec POF Test",
        "confidence":        confidence,
        "significance":      significance,
        "n_obs":             n,
        "n_breaches":        x,
        "expected_failures": round(p * n, 2),
        "p_expected":        p,
        "p_hat":             p_hat,
        "lr_statistic":      lr_stat,
        "chi2_critical":     chi2_crit,
        "p_value":           p_value,
        "reject_h0":         reject_h0,
        "model_adequate":    model_adequate,
        "verdict":           "Model Adequate ✓" if model_adequate else "Model Inadequate ✗",
        "all_returns":       ret_arr,
        "breaches":          breach_info["breaches"],
        "var_return":        var_return,
    }


# ── Summary Helper ─────────────────────────────────────────────────────────────
def backtest_summary(tl_result: dict, kupiec_result: dict) -> pd.DataFrame:
    rows = [
        {
            "Test":            tl_result["method"],
            "N Obs":           tl_result["window"],
            "Breaches":        tl_result["n_breaches"],
            "Expected":        f"{tl_result['expected_breaches']:.1f}",
            "Outcome":         tl_result["zone_info"]["label"],
            "Statistic":       "—",
            "p-value":         "—",
            "Verdict":         tl_result["zone"].upper(),
        },
        {
            "Test":            kupiec_result["method"],
            "N Obs":           kupiec_result["n_obs"],
            "Breaches":        kupiec_result["n_breaches"],
            "Expected":        kupiec_result["expected_failures"],
            "Outcome":         kupiec_result["verdict"],
            "Statistic":       f"{kupiec_result['lr_statistic']:.4f}",
            "p-value":         f"{kupiec_result['p_value']:.4f}",
            "Verdict":         "PASS" if kupiec_result["model_adequate"] else "FAIL",
        },
    ]
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from portfolio_construction.portfolio import fetch_data, compute_returns, optimize_portfolio
    from var_calculation.var_methods import var_historical

    print("Fetching data …")
    prices  = fetch_data()
    returns = compute_returns(prices)
    result  = optimize_portfolio(returns)
    weights = result["optimal_weights"]

    var_res = var_historical(returns, weights, confidence=0.99)
    port_ret = var_res["all_returns"]
    var_ret  = var_res["var_return"]

    tl  = traffic_light_backtest(port_ret, var_ret)
    kup = kupiec_pof_test(port_ret, var_ret)

    df = backtest_summary(tl, kup)
    print("\n── Backtest Results ──")
    print(df.to_string(index=False))

    print(f"\nTraffic Light Zone : {tl['zone_info']['label']}")
    print(f"Kupiec Verdict     : {kup['verdict']}")
    print(f"  LR Statistic     : {kup['lr_statistic']:.4f}")
    print(f"  χ² Critical (1df): {kup['chi2_critical']:.4f}")
    print(f"  p-value          : {kup['p_value']:.4f}")
