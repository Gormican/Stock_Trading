"""
dashboard.py — Streamlit UI: sliders, toggles, live GPA view, performance
Run: streamlit run dashboard.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taylor Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_PATH   = Path(__file__).parent / "config.json"
TRADE_LOG     = Path.home() / "taylor_trade_log.json"
BACKTEST_FILE = Path(__file__).parent / "backtest_results.json"
AGENT_LOG     = Path.home() / "taylor_agent.log"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

def save_config(cfg: dict):
    existing = load_config()
    existing.update(cfg)
    with open(CONFIG_PATH, "w") as f:
        json.dump(existing, f, indent=2)

def load_trades() -> list:
    if TRADE_LOG.exists():
        try:
            return json.loads(TRADE_LOG.read_text())
        except Exception:
            return []
    return []

def load_backtest() -> dict:
    if BACKTEST_FILE.exists():
        try:
            return json.loads(BACKTEST_FILE.read_text())
        except Exception:
            return {}
    return {}

def gpa_color(gpa: float) -> str:
    if gpa >= 3.5:  return "#16a34a"
    if gpa >= 3.0:  return "#65a30d"
    if gpa >= 2.5:  return "#ca8a04"
    if gpa >= 2.0:  return "#ea580c"
    return "#dc2626"

def gpa_badge(gpa: float, grade: str) -> str:
    c = gpa_color(gpa)
    return f'<span style="color:{c};font-size:1.4em;font-weight:bold">{gpa:.2f} ({grade})</span>'

# ==============================================================================
# SIDEBAR — CONTROLS
# ==============================================================================

cfg = load_config()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stocks.png", width=60)
    st.title("Agent Controls")
    st.caption("Changes save instantly to config.json")

    st.divider()
    st.subheader("Risk Settings")

    risk_slider = st.slider(
        "Risk Tolerance", min_value=1.0, max_value=5.0,
        value=float(cfg.get("risk", {}).get("position_size_pct", 3.0)),
        step=0.5,
        help="1=Conservative (1% per trade) → 5=Aggressive (5% per trade)",
        format="%.1f%%"
    )

    stop_loss = st.slider(
        "Stop Loss %", min_value=3.0, max_value=15.0,
        value=float(cfg.get("risk", {}).get("stop_loss_pct", 7.0)),
        step=0.5, format="%.1f%%"
    )

    take_profit = st.slider(
        "Take Profit %", min_value=5.0, max_value=30.0,
        value=float(cfg.get("risk", {}).get("take_profit_pct", 14.0)),
        step=1.0, format="%.1f%%"
    )

    daily_limit = st.slider(
        "Daily Loss Kill-Switch %", min_value=2.0, max_value=10.0,
        value=float(cfg.get("risk", {}).get("daily_loss_limit_pct", 5.0)),
        step=0.5, format="%.1f%%"
    )

    st.divider()
    st.subheader("Stock Filters")

    beta_range = st.slider(
        "Beta Range", min_value=0.2, max_value=3.0,
        value=(
            float(cfg.get("filters", {}).get("beta_min", 0.8)),
            float(cfg.get("filters", {}).get("beta_max", 1.8)),
        ),
        step=0.1
    )

    CAPS = {"Micro (<$300M)": 0, "Small ($300M-2B)": 0.3,
            "Mid ($2B-10B)": 2, "Large ($10B-200B)": 10, "Mega (>$200B)": 200}
    cap_labels = list(CAPS.keys())
    cap_values = list(CAPS.values())

    cap_min_idx = st.select_slider(
        "Min Market Cap",
        options=cap_labels,
        value=cap_labels[2]  # Mid default
    )
    cap_max_idx = st.select_slider(
        "Max Market Cap",
        options=cap_labels,
        value=cap_labels[4]  # Mega default
    )

    st.divider()
    st.subheader("Trading Mode")

    frequency = st.radio(
        "Scan Frequency",
        options=["low", "medium", "high"],
        index=["low","medium","high"].index(
            cfg.get("trading", {}).get("frequency", "medium")
        ),
        horizontal=True,
        help="Low=daily, Medium=30min, High=5min"
    )

    direction = st.radio(
        "Trade Direction",
        options=["long_only", "long_short"],
        index=0 if cfg.get("trading", {}).get("direction") == "long_only" else 1,
        horizontal=True,
        format_func=lambda x: "Long Only" if x == "long_only" else "Long/Short"
    )

    paper_mode = st.toggle("Paper Trading Mode", value=True)

    st.divider()
    st.subheader("GPA Thresholds")
    min_gpa = st.slider("Min GPA to BUY", 2.5, 4.0,
                        float(cfg.get("trading", {}).get("min_gpa_to_buy", 3.5)), 0.1)
    max_gpa_sell = st.slider("Max GPA to SELL", 1.5, 3.5,
                             float(cfg.get("trading", {}).get("max_gpa_to_sell", 2.5)), 0.1)

    if st.button("💾 Save Settings", use_container_width=True, type="primary"):
        updates = {
            "risk": {
                **cfg.get("risk", {}),
                "position_size_pct":   risk_slider,
                "stop_loss_pct":       stop_loss,
                "take_profit_pct":     take_profit,
                "daily_loss_limit_pct": daily_limit,
            },
            "filters": {
                **cfg.get("filters", {}),
                "beta_min": beta_range[0],
                "beta_max": beta_range[1],
                "market_cap_min_billions": CAPS[cap_min_idx],
                "market_cap_max_billions": CAPS[cap_max_idx] if CAPS[cap_max_idx] > 0 else 99999,
            },
            "trading": {
                **cfg.get("trading", {}),
                "frequency":        frequency,
                "direction":        direction,
                "mode":             "paper" if paper_mode else "live",
                "min_gpa_to_buy":   min_gpa,
                "max_gpa_to_sell":  max_gpa_sell,
                "min_gpa_to_alert": min_gpa,
            },
        }
        save_config(updates)
        st.success("Settings saved!")


# ==============================================================================
# MAIN CONTENT
# ==============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🏆 GPA Scores", "📈 Backtest", "📋 Trade Log", "⚙️ GPA Weights"
])

# ── Tab 1: Portfolio Dashboard ────────────────────────────────────────────────
with tab1:
    st.title("📈 Taylor's Trading Agent")
    mode_badge = "🟡 PAPER" if cfg.get("trading", {}).get("mode") == "paper" else "🔴 LIVE"
    st.caption(f"{mode_badge} | Alpaca API | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    trades    = load_trades()
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t.get("timestamp","")[:10] == today_str]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Net Account Value", "$14,516.50", delta="+$5,001.06",
                  help="From TaylorApr2026.csv — updated by agent")
    with c2:
        buys  = [t for t in today_trades if t["action"] == "BUY"]
        sells = [t for t in today_trades if t["action"] == "SELL"]
        st.metric("Today's Trades", f"{len(today_trades)}", delta=f"+{len(buys)} buys")
    with c3:
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        st.metric("Total P&L", f"${total_pnl:+,.2f}")
    with c4:
        risk_pct = cfg.get("risk", {}).get("position_size_pct", 3.0)
        st.metric("Risk / Trade", f"{risk_pct:.1f}%",
                  help="Percent of portfolio per position")

    st.divider()

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Current Holdings (from Apr 2026 CSV)")
        holdings = {
            "ADC": {"qty": 2,  "avg_cost": 70.125,  "last": 78.18,  "sector": "REIT"},
            "AMZN":{"qty": 4,  "avg_cost": 100.763, "last": 238.38, "sector": "Tech"},
            "COST":{"qty": 3,  "avg_cost": 627.47,  "last": 998.47, "sector": "Retail"},
            "DNP": {"qty": 15, "avg_cost": 10.72,   "last": 10.515, "sector": "Utility"},
            "FVRR":{"qty": 10, "avg_cost": 28.249,  "last": 10.32,  "sector": "Tech"},
            "NVDA":{"qty": 14, "avg_cost": 89.443,  "last": 188.745,"sector": "Tech"},
            "QQQ": {"qty": 3,  "avg_cost": 617.517, "last": 611.07, "sector": "ETF"},
            "TSLA":{"qty": 4,  "avg_cost": 104.913, "last": 348.996,"sector": "Auto"},
            "WMT": {"qty": 15, "avg_cost": 49.679,  "last": 126.787,"sector": "Retail"},
        }
        rows = []
        for sym, h in holdings.items():
            gain_pct = (h["last"] / h["avg_cost"] - 1) * 100
            rows.append({
                "Symbol": sym, "Qty": h["qty"],
                "Avg Cost": f"${h['avg_cost']:.2f}",
                "Last": f"${h['last']:.2f}",
                "Gain %": f"{gain_pct:+.1f}%",
                "Value": f"${h['qty']*h['last']:,.2f}",
                "Sector": h["sector"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Risk Parameters")
        st.info(f"""
**Risk / Trade:** {risk_slider:.1f}%
**Stop Loss:** {stop_loss:.1f}%
**Take Profit:** {take_profit:.1f}%
**Kill Switch:** -{daily_limit:.1f}%/day
**Beta Range:** {beta_range[0]:.1f} – {beta_range[1]:.1f}
**Direction:** {direction.replace('_',' ').title()}
**Frequency:** {frequency.title()}
""")

    # Recent log tail
    if AGENT_LOG.exists():
        with st.expander("📜 Recent Agent Log"):
            lines = AGENT_LOG.read_text().splitlines()[-40:]
            st.code("\n".join(lines), language="text")

# ── Tab 2: GPA Scores ─────────────────────────────────────────────────────────
with tab2:
    st.title("🏆 GPA Scoring Model")
    st.caption("Based on Stock Eval.xls — 5 pillars, 0.0–4.0 GPA scale")

    st.markdown("""
    | Grade | GPA Range | Action |
    |-------|-----------|--------|
    | A+    | 3.7–4.0   | Strong Buy |
    | A     | 3.5–3.7   | Buy |
    | B+    | 3.0–3.5   | Watch / Hold |
    | B     | 2.7–3.0   | Hold |
    | C+    | 2.0–2.7   | Reduce |
    | D     | < 2.0     | Sell |
    """)

    st.subheader("Pillar Weights")
    weights = cfg.get("gpa_weights", {})
    w_data  = {
        "Pillar": ["Technical Momentum", "Fundamentals", "Sentiment",
                   "Relative Strength", "Volatility Profile"],
        "Weight": [
            weights.get("technical_momentum", 0.25),
            weights.get("fundamentals", 0.25),
            weights.get("sentiment", 0.25),
            weights.get("relative_strength", 0.15),
            weights.get("volatility_profile", 0.10),
        ],
        "Sub-factors": [
            "RSI(14), MACD, SMA20/50, Volume, Stochastics",
            "ROE, EPS Growth, D/E, P/E, PEG, Revenue Growth",
            "News NLP (VADER), Reddit mentions, Velocity",
            "1-month & 3-month return vs SPY",
            "Beta fit to user range, ATR%",
        ],
        "Max Score": ["4.0"] * 5,
    }
    st.dataframe(pd.DataFrame(w_data), use_container_width=True, hide_index=True)

    st.subheader("From Stock Eval.xls — Rating Scale")
    scale = {
        "Rating": [4, 3, 2, 1],
        "Technical": [">30° Uptrend + Vol Surge", "Up <30°, Above SMA", "Flat / Mixed", "Downtrend"],
        "Fundamentals": ["ROE>18%, EPS>25%, D/E=0", "Strong most factors", "Mixed", "Weak/Negative"],
        "Sentiment": ["Headline + Great Trend", "Positive trend", "Mixed", "Negative"],
        "Relative Strength": ["Leading Sector", "Above Average", "Below Average", "Worst in Sector"],
    }
    st.dataframe(pd.DataFrame(scale), use_container_width=True, hide_index=True)

# ── Tab 3: Backtest ───────────────────────────────────────────────────────────
with tab3:
    st.title("📈 Backtest Results")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("Run backtest from terminal:\n```\npython trading_agent.py --backtest\n```")

    bt = load_backtest()
    if bt:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            delta_color = "normal" if bt.get("total_return_pct",0) >= 0 else "inverse"
            st.metric("Strategy Return",
                      f"{bt.get('total_return_pct',0):+.1f}%",
                      delta=f"vs SPY {bt.get('spy_return_pct',0):+.1f}%")
        with m2:
            st.metric("Alpha", f"{bt.get('alpha_pct',0):+.1f}%")
        with m3:
            st.metric("Sharpe Ratio", f"{bt.get('sharpe_ratio',0):.2f}")
        with m4:
            st.metric("Win Rate", f"{bt.get('win_rate_pct',0):.1f}%",
                      delta=f"{bt.get('total_trades',0)} trades")

        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("Max Drawdown", f"{bt.get('max_drawdown_pct',0):.1f}%")
        with c_b:
            st.metric("Profit Factor", f"{bt.get('profit_factor',0):.2f}")

        # Equity curve
        eq = bt.get("equity_curve", [])
        if eq:
            eq_df = pd.DataFrame(eq)
            eq_df["date"] = pd.to_datetime(eq_df["date"])
            eq_df.set_index("date", inplace=True)
            st.subheader("Equity Curve")
            st.line_chart(eq_df["value"])

        # Trade log
        bt_trades = bt.get("trades", [])
        if bt_trades:
            st.subheader("Recent Backtest Trades")
            st.dataframe(
                pd.DataFrame(bt_trades)[["date","symbol","action","shares","price","pnl","reason"]],
                use_container_width=True, hide_index=True
            )
    else:
        st.warning("No backtest results yet. Run: `python trading_agent.py --backtest`")

# ── Tab 4: Trade Log ──────────────────────────────────────────────────────────
with tab4:
    st.title("📋 Trade Log")
    trades = load_trades()
    if trades:
        df = pd.DataFrame(trades)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values("timestamp", ascending=False, inplace=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            buys  = len([t for t in trades if t["action"] == "BUY"])
            sells = len([t for t in trades if t["action"] == "SELL"])
            st.metric("Total Buys", buys)
        with col_b:
            st.metric("Total Sells", sells)
        with col_c:
            total_vol = sum(t.get("value", 0) for t in trades)
            st.metric("Total Volume", f"${total_vol:,.0f}")

        show_cols = [c for c in ["timestamp","action","symbol","shares","price",
                                  "value","gpa","reason","mode"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button("Download Trade Log CSV", csv,
                           file_name="taylor_trades.csv", mime="text/csv")
    else:
        st.info("No trades recorded yet. Start the agent to begin trading.")

# ── Tab 5: GPA Weight Editor ──────────────────────────────────────────────────
with tab5:
    st.title("⚙️ GPA Weight Configuration")
    st.caption("Adjust how much each pillar contributes to the 0–4.0 GPA score")

    weights = cfg.get("gpa_weights", {
        "technical_momentum": 0.25, "fundamentals": 0.25, "sentiment": 0.25,
        "relative_strength": 0.15, "volatility_profile": 0.10,
    })

    st.info("Weights must sum to 1.0 (100%). The agent uses these to compute the final GPA.")

    col1, col2 = st.columns(2)
    with col1:
        w_tech = st.slider("Technical Momentum", 0.05, 0.60,
                           float(weights.get("technical_momentum", 0.25)), 0.05)
        w_fund = st.slider("Fundamentals", 0.05, 0.60,
                           float(weights.get("fundamentals", 0.25)), 0.05)
        w_sent = st.slider("Sentiment (News + Social)", 0.05, 0.60,
                           float(weights.get("sentiment", 0.25)), 0.05)
    with col2:
        w_rs   = st.slider("Relative Strength (vs SPY)", 0.05, 0.40,
                           float(weights.get("relative_strength", 0.15)), 0.05)
        w_vol  = st.slider("Volatility Profile", 0.05, 0.30,
                           float(weights.get("volatility_profile", 0.10)), 0.05)

        total = w_tech + w_fund + w_sent + w_rs + w_vol
        color = "🟢" if abs(total - 1.0) < 0.01 else "🔴"
        st.metric("Weight Total", f"{color} {total:.2f}", delta="should be 1.00")

    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f} — they must equal 1.00. Adjust sliders.")
    else:
        if st.button("💾 Save GPA Weights", type="primary"):
            save_config({
                "gpa_weights": {
                    "technical_momentum": w_tech,
                    "fundamentals": w_fund,
                    "sentiment": w_sent,
                    "relative_strength": w_rs,
                    "volatility_profile": w_vol,
                }
            })
            st.success("GPA weights saved!")

    st.divider()
    st.subheader("Preset Weight Profiles")
    presets = {
        "Default (Balanced)":     [0.25, 0.25, 0.25, 0.15, 0.10],
        "Momentum-First":         [0.35, 0.15, 0.30, 0.15, 0.05],
        "Fundamentals-First":     [0.20, 0.40, 0.15, 0.15, 0.10],
        "Sentiment-Driven (WSB)": [0.20, 0.10, 0.45, 0.15, 0.10],
        "Low-Risk / Quality":     [0.20, 0.35, 0.20, 0.15, 0.10],
    }
    for name, vals in presets.items():
        cols = st.columns([3, 1])
        with cols[0]:
            st.write(f"**{name}**: Tech={vals[0]:.0%} Fund={vals[1]:.0%} "
                     f"Sent={vals[2]:.0%} RS={vals[3]:.0%} Vol={vals[4]:.0%}")
        with cols[1]:
            if st.button(f"Apply", key=name):
                save_config({"gpa_weights": {
                    "technical_momentum": vals[0], "fundamentals": vals[1],
                    "sentiment": vals[2], "relative_strength": vals[3],
                    "volatility_profile": vals[4],
                }})
                st.rerun()
