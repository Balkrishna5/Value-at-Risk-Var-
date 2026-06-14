"""
VaR Risk Dashboard — Streamlit Application
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from portfolio_construction.portfolio import (
    fetch_data, compute_returns, optimize_portfolio, portfolio_stats, TICKERS
)
from var_calculation.var_methods import (
    var_historical, var_variance_covariance, var_monte_carlo, var_summary
)
from var_backtesting.backtesting import (
    traffic_light_backtest, kupiec_pof_test, backtest_summary
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VaR Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,h2,h3 { color: #60a5fa !important; }

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0f1629 0%, #1a2744 50%, #0f1629 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.dash-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 0 0 6px 0;
    letter-spacing: -1px;
}
.dash-header p { color: #64748b; margin: 0; font-size: 0.9rem; }

/* Metric cards */
.metric-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3b82f6; }
.metric-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #475569; }
.metric-card .value { font-family: 'Space Mono', monospace; font-size: 1.5rem; color: #60a5fa; font-weight: 700; margin: 6px 0 0; }
.metric-card .sub   { font-size: 0.75rem; color: #64748b; }

/* Section titles */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #3b82f6;
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
    margin: 28px 0 16px;
}

/* Traffic light */
.zone-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    font-family: 'Space Mono', monospace;
}
.zone-green  { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid #22c55e; }
.zone-yellow { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b; }
.zone-red    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; }

/* Tables */
.stDataFrame { background: #0f1629 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0f1629; border-bottom: 1px solid #1e2d4a; gap: 4px; }
.stTabs [data-baseweb="tab"]      { color: #64748b; padding: 8px 20px; border-radius: 6px 6px 0 0; }
.stTabs [aria-selected="true"]    { color: #60a5fa !important; background: #1e2d4a !important; }

/* Slider / select */
.stSlider > div > div { color: #60a5fa !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>📊 VaR Risk Dashboard</h1>
  <p>Value at Risk · Portfolio Optimization · Backtesting · JPM · MS · BAC</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar Parameters
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")

    confidence = st.selectbox(
        "Confidence Level",
        options=[0.95, 0.99, 0.999],
        index=1,
        format_func=lambda x: f"{x*100:.1f}%"
    )

    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value=10_000,
        max_value=100_000_000,
        value=1_000_000,
        step=100_000,
        format="%d"
    )

    holding_days = st.selectbox("Holding Period", [1, 5, 10], index=0,
                                format_func=lambda x: f"{x} day(s)")

    mc_sims = st.select_slider(
        "Monte Carlo Simulations",
        options=[1_000, 5_000, 10_000, 50_000],
        value=10_000
    )

    bt_window = st.selectbox("Backtesting Window (days)", [250, 500, 750, 1000], index=0)

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#334155;'>
    <b>Stocks:</b> JPM · MS · BAC<br>
    <b>Period:</b> 2019–2024<br>
    <b>Method:</b> Markowitz Max-Sharpe
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Data & Computation (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_optimize():
    prices  = fetch_data()
    returns = compute_returns(prices)
    result  = optimize_portfolio(returns)
    return prices, returns, result

@st.cache_data(show_spinner=False)
def run_var(confidence, portfolio_value, holding_days, mc_sims, _returns, _weights):
    r1 = var_historical(          _returns, _weights, confidence, portfolio_value, holding_days)
    r2 = var_variance_covariance( _returns, _weights, confidence, portfolio_value, holding_days)
    r3 = var_monte_carlo(         _returns, _weights, confidence, portfolio_value, holding_days, mc_sims)
    return r1, r2, r3

@st.cache_data(show_spinner=False)
def run_backtests(var_return, _port_returns, confidence, bt_window):
    tl  = traffic_light_backtest(_port_returns, var_return, confidence, bt_window)
    kup = kupiec_pof_test(       _port_returns, var_return, confidence)
    return tl, kup

# ── Load data on first run or when button pressed ──────────────────────────────
if "loaded" not in st.session_state or run_btn:
    with st.spinner("Fetching data from Yahoo Finance …"):
        prices, returns, opt_result = load_and_optimize()
    st.session_state.loaded      = True
    st.session_state.prices      = prices
    st.session_state.returns     = returns
    st.session_state.opt_result  = opt_result

prices     = st.session_state.prices
returns    = st.session_state.returns
opt_result = st.session_state.opt_result
weights    = opt_result["optimal_weights"]

# Compute VaR
with st.spinner("Computing VaR …"):
    r1, r2, r3 = run_var(confidence, portfolio_value, holding_days, mc_sims,
                         returns, weights)

# Backtesting uses historical VaR return
with st.spinner("Running backtests …"):
    port_ret = r1["all_returns"]
    tl, kup  = run_backtests(r1["var_return"], port_ret, confidence, bt_window)

# ═══════════════════════════════════════════════════════════════════════════════
# Top KPI Row
# ═══════════════════════════════════════════════════════════════════════════════
ret_a, vol_a, sharpe = portfolio_stats(weights, opt_result["mean_returns"].values,
                                       opt_result["cov_matrix"].values)

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "Portfolio Value",   f"${portfolio_value:,.0f}", ""),
    (c2, "Annual Return",     f"{ret_a*100:.2f}%",        "Markowitz Max-Sharpe"),
    (c3, "Annual Volatility", f"{vol_a*100:.2f}%",        ""),
    (c4, "Sharpe Ratio",      f"{sharpe:.3f}",            f"RF={5}%"),
    (c5, "Backtest Zone",
     tl["zone"].upper(),
     tl["zone_info"]["range"]),
]
for col, label, value, sub in kpis:
    with col:
        zone_style = f"color:{tl['zone_info']['color']}" if label == "Backtest Zone" else ""
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value" style="{zone_style}">{value}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Portfolio", "🎯 VaR Calculation", "🔍 Backtesting", "📋 Summary"
])

CHART_THEME = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0f1629",
    font_color="#94a3b8",
    font_family="DM Sans",
    xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Portfolio Construction
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Markowitz Mean-Variance Optimization</div>',
                unsafe_allow_html=True)

    col_w, col_ef = st.columns([1, 2])

    with col_w:
        # Weights pie
        fig_pie = go.Figure(go.Pie(
            labels=opt_result["tickers"],
            values=weights * 100,
            hole=0.55,
            marker=dict(colors=["#3b82f6","#06b6d4","#8b5cf6"]),
            textfont=dict(color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>"
        ))
        fig_pie.update_layout(
            title=dict(text="Optimal Weights", font=dict(color="#60a5fa", size=14)),
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8")),
            **{k:v for k,v in CHART_THEME.items() if k not in ("xaxis","yaxis")}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**Weight Summary**")
        w_df = pd.DataFrame({
            "Ticker": opt_result["tickers"],
            "Weight": [f"{w*100:.2f}%" for w in weights]
        })
        st.dataframe(w_df, hide_index=True, use_container_width=True)

    with col_ef:
        # Efficient frontier
        frontier = opt_result["frontier"]
        opt_r, opt_v, _ = portfolio_stats(weights, opt_result["mean_returns"].values,
                                          opt_result["cov_matrix"].values)
        mv_r,  mv_v,  _ = portfolio_stats(opt_result["min_var_weights"],
                                          opt_result["mean_returns"].values,
                                          opt_result["cov_matrix"].values)

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=frontier["Volatility"]*100,
            y=frontier["Return"]*100,
            mode="markers",
            marker=dict(size=3, color=frontier["Sharpe"],
                        colorscale="Blues", opacity=0.6,
                        colorbar=dict(title="Sharpe", tickfont=dict(color="#94a3b8"))),
            name="Simulated Portfolios",
            hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"
        ))
        # Max Sharpe
        fig_ef.add_trace(go.Scatter(
            x=[opt_v*100], y=[opt_r*100],
            mode="markers", marker=dict(symbol="star", size=18, color="#f59e0b",
                                         line=dict(color="white", width=1)),
            name="Max Sharpe ⭐"
        ))
        # Min Variance
        fig_ef.add_trace(go.Scatter(
            x=[mv_v*100], y=[mv_r*100],
            mode="markers", marker=dict(symbol="diamond", size=12, color="#22c55e",
                                         line=dict(color="white", width=1)),
            name="Min Variance 💎"
        ))
        fig_ef.update_layout(
            title=dict(text="Efficient Frontier (5,000 Portfolios)", font=dict(color="#60a5fa", size=14)),
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Annual Return (%)",
            legend=dict(font=dict(color="#94a3b8")),
            **CHART_THEME
        )
        st.plotly_chart(fig_ef, use_container_width=True)

    # Cumulative returns
    st.markdown('<div class="section-title">Cumulative Returns</div>', unsafe_allow_html=True)
    cum = (1 + returns).cumprod()
    cum_port = (1 + returns.values @ weights).cumprod()

    fig_cum = go.Figure()
    colors = ["#3b82f6","#06b6d4","#8b5cf6"]
    for i, t in enumerate(TICKERS):
        fig_cum.add_trace(go.Scatter(
            x=cum.index, y=cum[t],
            mode="lines", name=t,
            line=dict(color=colors[i], width=1.5)
        ))
    fig_cum.add_trace(go.Scatter(
        x=cum.index, y=cum_port,
        mode="lines", name="Portfolio",
        line=dict(color="#f59e0b", width=2.5, dash="dash")
    ))
    fig_cum.update_layout(
        xaxis_title="Date", yaxis_title="Cumulative Return",
        legend=dict(font=dict(color="#94a3b8")),
        **CHART_THEME
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – VaR Calculation
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Value at Risk — Three Methods</div>',
                unsafe_allow_html=True)

    # Summary table
    var_df = var_summary([r1, r2, r3])
    st.dataframe(var_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    mc1, mc2, mc3 = st.columns(3)
    for col, res, clr in zip([mc1, mc2, mc3], [r1, r2, r3],
                              ["#3b82f6", "#06b6d4", "#8b5cf6"]):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{clr}">
              <div class="label">{res['method']}</div>
              <div class="value" style="color:{clr}">${res['var_dollar']:,.0f}</div>
              <div class="sub">VaR @ {res['confidence']*100:.0f}% CI · {holding_days}d</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Historical return distribution
    v1col, v2col = st.columns(2)
    with v1col:
        st.markdown("**Historical Simulation — Return Distribution**")
        all_ret = r1["all_returns"]
        var_ret = r1["var_return"]

        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=all_ret * 100,
            nbinsx=80,
            marker_color="#3b82f6",
            opacity=0.7,
            name="Portfolio Returns"
        ))
        # Tail fill
        tail = r1["tail_returns"] * 100
        if len(tail) > 0:
            fig_h.add_trace(go.Histogram(
                x=tail, nbinsx=20,
                marker_color="#ef4444", opacity=0.9,
                name=f"Tail (< VaR)"
            ))
        fig_h.add_vline(
            x=var_ret * 100,
            line=dict(color="#f59e0b", width=2, dash="dash"),
            annotation=dict(text=f"VaR={var_ret*100:.2f}%",
                            font=dict(color="#f59e0b"), showarrow=False)
        )
        fig_h.update_layout(
            xaxis_title="Daily Return (%)", yaxis_title="Frequency",
            barmode="overlay", **CHART_THEME
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with v2col:
        st.markdown("**Monte Carlo — Simulated Return Distribution**")
        sim = r3["sim_returns"]
        var_mc = r3["var_return"]

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=sim * 100, nbinsx=100,
            marker_color="#8b5cf6", opacity=0.7,
            name="MC Simulations"
        ))
        tail_mc = sim[sim < var_mc] * 100
        fig_mc.add_trace(go.Histogram(
            x=tail_mc, nbinsx=20,
            marker_color="#ef4444", opacity=0.9,
            name="Tail"
        ))
        fig_mc.add_vline(
            x=var_mc * 100,
            line=dict(color="#f59e0b", width=2, dash="dash"),
            annotation=dict(text=f"VaR={var_mc*100:.2f}%",
                            font=dict(color="#f59e0b"), showarrow=False)
        )
        fig_mc.update_layout(
            xaxis_title="Simulated Return (%)", yaxis_title="Frequency",
            barmode="overlay", **CHART_THEME
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # ── Variance-Covariance normal curve
    st.markdown("**Variance-Covariance — Normal Distribution Fit**")
    from scipy.stats import norm
    mu_p  = r2["mu_p"]
    sig_p = r2["sigma_p"]
    x_lin = np.linspace(mu_p - 4*sig_p, mu_p + 4*sig_p, 400)
    y_lin = norm.pdf(x_lin, mu_p, sig_p)

    fig_vc = go.Figure()
    fig_vc.add_trace(go.Scatter(
        x=x_lin*100, y=y_lin,
        mode="lines", fill="tozeroy",
        line=dict(color="#06b6d4", width=2),
        fillcolor="rgba(6,182,212,0.15)",
        name="Normal PDF"
    ))
    # Tail area
    x_tail = x_lin[x_lin < r2["var_return"]]
    y_tail = norm.pdf(x_tail, mu_p, sig_p)
    fig_vc.add_trace(go.Scatter(
        x=np.concatenate([x_tail, [x_tail[-1]]])*100 if len(x_tail) else [],
        y=np.concatenate([y_tail, [0]]) if len(y_tail) else [],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(239,68,68,0.3)",
        line=dict(color="#ef4444", width=1),
        name=f"{(1-confidence)*100:.1f}% Tail"
    ))
    fig_vc.add_vline(
        x=r2["var_return"]*100,
        line=dict(color="#f59e0b", width=2, dash="dash"),
        annotation=dict(text=f"VaR={r2['var_return']*100:.2f}%",
                        font=dict(color="#f59e0b"))
    )
    fig_vc.update_layout(
        xaxis_title="Return (%)", yaxis_title="Probability Density",
        **CHART_THEME
    )
    st.plotly_chart(fig_vc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 – Backtesting
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">VaR Backtesting</div>', unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    # ── Traffic Light card
    with bc1:
        zone = tl["zone"]
        zi   = tl["zone_info"]
        st.markdown(f"""
        <div class="metric-card" style="border-color:{zi['color']};text-align:left;padding:24px;">
          <div style="font-family:'Space Mono',monospace;font-size:0.8rem;text-transform:uppercase;
                      letter-spacing:2px;color:{zi['color']};">🚦 Traffic Light Approach</div>
          <div style="font-family:'Space Mono',monospace;font-size:2rem;color:{zi['color']};
                      margin:12px 0;">
            {zi['label']}
          </div>
          <div style="color:#64748b;font-size:0.85rem;margin-bottom:12px;">{zi['range']}</div>
          <hr style="border-color:#1e2d4a;">
          <p style="color:#94a3b8;font-size:0.85rem;">{zi['implication']}</p>
          <p style="color:#60a5fa;font-size:0.82rem;"><b>Action:</b> {zi['action']}</p>
          <table style="color:#94a3b8;font-size:0.82rem;width:100%;">
            <tr><td>Observed Breaches</td><td style="text-align:right;color:{zi['color']};font-weight:700;">{tl['n_breaches']}</td></tr>
            <tr><td>Expected Breaches</td><td style="text-align:right;">{tl['expected_breaches']:.1f}</td></tr>
            <tr><td>Breach Rate</td><td style="text-align:right;">{tl['breach_rate']*100:.2f}%</td></tr>
            <tr><td>Window</td><td style="text-align:right;">{tl['window']} days</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    # ── Kupiec card
    with bc2:
        verdict_color = "#22c55e" if kup["model_adequate"] else "#ef4444"
        st.markdown(f"""
        <div class="metric-card" style="border-color:{verdict_color};text-align:left;padding:24px;">
          <div style="font-family:'Space Mono',monospace;font-size:0.8rem;text-transform:uppercase;
                      letter-spacing:2px;color:{verdict_color};">📐 Kupiec POF Test</div>
          <div style="font-family:'Space Mono',monospace;font-size:2rem;color:{verdict_color};
                      margin:12px 0;">
            {kup['verdict']}
          </div>
          <div style="color:#64748b;font-size:0.85rem;margin-bottom:12px;">
            χ²(1) likelihood ratio test
          </div>
          <hr style="border-color:#1e2d4a;">
          <table style="color:#94a3b8;font-size:0.82rem;width:100%;">
            <tr><td>LR Statistic</td><td style="text-align:right;color:{verdict_color};font-weight:700;">{kup['lr_statistic']:.4f}</td></tr>
            <tr><td>χ² Critical (1 dof)</td><td style="text-align:right;">{kup['chi2_critical']:.4f}</td></tr>
            <tr><td>p-value</td><td style="text-align:right;">{kup['p_value']:.4f}</td></tr>
            <tr><td>Observed Failures</td><td style="text-align:right;">{kup['n_breaches']}</td></tr>
            <tr><td>Expected Failures</td><td style="text-align:right;">{kup['expected_failures']}</td></tr>
            <tr><td>Significance Level</td><td style="text-align:right;">{kup['significance']*100:.0f}%</td></tr>
            <tr><td>p̂ (observed rate)</td><td style="text-align:right;">{kup['p_hat']*100:.3f}%</td></tr>
            <tr><td>p (expected rate)</td><td style="text-align:right;">{kup['p_expected']*100:.1f}%</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    # ── Breach timeline chart
    st.markdown('<div class="section-title">Breach Timeline</div>', unsafe_allow_html=True)

    bt_ret = tl["bt_returns"]
    breach_mask = tl["all_breaches"]
    n_days = len(bt_ret)
    x_days = list(range(n_days))

    fig_bt = go.Figure()
    # Returns line
    fig_bt.add_trace(go.Scatter(
        x=x_days, y=bt_ret * 100,
        mode="lines", name="Daily Return",
        line=dict(color="#3b82f6", width=1), opacity=0.8
    ))
    # VaR line
    fig_bt.add_hline(
        y=tl["var_return"] * 100,
        line=dict(color="#f59e0b", width=2, dash="dash"),
        annotation=dict(text=f"VaR={tl['var_return']*100:.2f}%",
                        font=dict(color="#f59e0b"), showarrow=False)
    )
    # Breach markers
    breach_idx = np.where(breach_mask)[0]
    if len(breach_idx) > 0:
        fig_bt.add_trace(go.Scatter(
            x=breach_idx.tolist(),
            y=(bt_ret[breach_idx] * 100).tolist(),
            mode="markers",
            marker=dict(size=8, color="#ef4444", symbol="x",
                        line=dict(color="white", width=1)),
            name=f"Breaches ({len(breach_idx)})"
        ))
    fig_bt.update_layout(
        xaxis_title="Trading Day",
        yaxis_title="Return (%)",
        legend=dict(font=dict(color="#94a3b8")),
        **CHART_THEME
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── Traffic light zone bar
    st.markdown('<div class="section-title">Basel Zone Classification</div>',
                unsafe_allow_html=True)
    zones_data = {
        "Zone":       ["Green (Low Risk)", "Yellow (Moderate Risk)", "Red (High Risk)"],
        "Min Breach": [0, 5, 10],
        "Max Breach": [4, 9, 999],
        "Color":      ["#22c55e", "#f59e0b", "#ef4444"],
    }
    fig_zones = go.Figure()
    for z, mn, mx, clr in zip(zones_data["Zone"], zones_data["Min Breach"],
                               zones_data["Max Breach"], zones_data["Color"]):
        w = min(mx, 15) - mn + 1
        fig_zones.add_trace(go.Bar(
            x=[w], y=[z], orientation="h",
            marker_color=clr, opacity=0.7,
            name=z,
            text=f"  {mn}–{'' if mx==999 else mx} breaches",
            textposition="inside",
            textfont=dict(color="white"),
        ))
    # Observed breaches marker
    fig_zones.add_vline(
        x=tl["n_breaches"] - zones_data["Min Breach"][[0,5,10].index(
            min([0,5,10], key=lambda b: abs(b - (0 if tl["n_breaches"]<=4 else 5 if tl["n_breaches"]<=9 else 10)))
        )] + 0.5,
        line=dict(color="white", width=2, dash="dot"),
    )
    fig_zones.update_layout(
        barmode="stack",
        xaxis_title="Breach Count",
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8")),
        height=200,
        **CHART_THEME
    )
    st.plotly_chart(fig_zones, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 – Summary
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Comprehensive Summary</div>',
                unsafe_allow_html=True)

    # ── Portfolio weights
    st.markdown("#### 📌 Portfolio Weights (Markowitz Max-Sharpe)")
    w_summary = pd.DataFrame({
        "Ticker": opt_result["tickers"],
        "Weight (%)": [f"{w*100:.2f}%" for w in weights],
        "Invested ($)": [f"${w*portfolio_value:,.0f}" for w in weights],
    })
    st.dataframe(w_summary, hide_index=True, use_container_width=True)

    # ── VaR comparison
    st.markdown("#### 🎯 VaR Results — All Methods")
    st.dataframe(var_df, hide_index=True, use_container_width=True)

    # Method comparison bar
    fig_bar = go.Figure()
    methods = [r["method"] for r in [r1, r2, r3]]
    var_vals = [r["var_dollar"] for r in [r1, r2, r3]]
    es_vals  = [r["es_dollar"] for r in [r1, r2, r3]]

    fig_bar.add_trace(go.Bar(
        x=methods, y=var_vals,
        name="VaR ($)",
        marker_color="#3b82f6", opacity=0.85,
        text=[f"${v:,.0f}" for v in var_vals],
        textposition="outside", textfont=dict(color="#94a3b8")
    ))
    fig_bar.add_trace(go.Bar(
        x=methods, y=es_vals,
        name="ES ($)",
        marker_color="#ef4444", opacity=0.7,
        text=[f"${v:,.0f}" for v in es_vals],
        textposition="outside", textfont=dict(color="#94a3b8")
    ))
    fig_bar.update_layout(
        barmode="group",
        yaxis_title="Dollar Amount",
        yaxis_tickformat="$,.0f",
        legend=dict(font=dict(color="#94a3b8")),
        **CHART_THEME
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Backtesting summary
    st.markdown("#### 🔍 Backtesting Summary")
    bt_df = backtest_summary(tl, kup)
    st.dataframe(bt_df, hide_index=True, use_container_width=True)

    # ── Methodology notes
    st.markdown("#### 📚 Methodology Notes")
    with st.expander("View detailed methodology"):
        st.markdown("""
**Portfolio Construction — Markowitz Optimization**
- Stocks: JPMorgan Chase (JPM), Morgan Stanley (MS), Bank of America (BAC)
- Data: 2019–2024 daily adjusted closing prices (Yahoo Finance)
- Returns: Simple daily returns (Pₜ/Pₜ₋₁ − 1)
- Optimization: Maximize Sharpe Ratio subject to Σwᵢ = 1, wᵢ ≥ 0
- Efficient Frontier: 5,000 Monte Carlo portfolios

**VaR Method 1 — Historical Simulation**
- Non-parametric; no distributional assumptions
- Rank portfolio returns ascending; VaR = return at (1-CI) percentile
- Rank = ceil((1 − confidence) × n)

**VaR Method 2 — Variance-Covariance (Parametric)**
- Assumes normally distributed returns
- VaR = μₚ + z × σₚ where z = Φ⁻¹(1 − CI)
- σₚ computed via covariance matrix: σₚ = √(wᵀΣw)

**VaR Method 3 — Monte Carlo Simulation**
- Models stock prices via Geometric Brownian Motion (GBM)
- Sₜ = S₀ exp((μ − σ²/2)t + σWₜ)
- Correlated random shocks via Cholesky decomposition: W = ZLᵀ
- 10,000+ simulated paths; VaR from percentile of simulated returns

**Backtesting 1 — Traffic Light Approach (Basel)**
- Count breaches (actual loss > VaR) over rolling 250-day window
- Green: 0–4, Yellow: 5–9, Red: ≥10 breaches
- Basel III regulatory standard

**Backtesting 2 — Kupiec POF Test**
- H₀: p̂ = p (observed failure rate = expected 1−CI)
- LR = −2 ln[pˣ(1−p)ⁿ⁻ˣ / p̂ˣ(1−p̂)ⁿ⁻ˣ] ~ χ²(1)
- Reject H₀ if LR > χ²₀.₉₅(1) = 3.841 → model inadequate
        """)
