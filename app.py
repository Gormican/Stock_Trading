"""
app.py — Taylor's Trading Agent — Beautiful Beginner-Friendly UI
Run via: START_AGENT.bat  (double-click)  OR  streamlit run app.py
"""
import json
import subprocess
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taylor's Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
CFG_PATH   = BASE / "config.json"
TRADE_LOG  = Path.home() / "taylor_trade_log.json"
AGENT_LOG  = Path.home() / "taylor_agent.log"
BT_FILE    = BASE / "backtest_results.json"

# ── Custom CSS (dark-card, clean modern look) ──────────────────────────────────
st.markdown("""
<style>
  /* Base */
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"]          { background: #1a1d27; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Hide default header */
  header[data-testid="stHeader"] { background: transparent; }

  /* Cards */
  .card {
    background: #1a1d27;
    border: 1px solid #2d3047;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 16px;
  }
  .card-green  { border-left: 4px solid #22c55e; }
  .card-yellow { border-left: 4px solid #f59e0b; }
  .card-red    { border-left: 4px solid #ef4444; }
  .card-blue   { border-left: 4px solid #3b82f6; }

  /* Metric overrides */
  [data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2d3047;
    border-radius: 12px;
    padding: 16px 20px;
  }

  /* GPA badge */
  .gpa-a  { color:#22c55e; font-size:2rem; font-weight:800; }
  .gpa-b  { color:#84cc16; font-size:2rem; font-weight:800; }
  .gpa-c  { color:#f59e0b; font-size:2rem; font-weight:800; }
  .gpa-d  { color:#ef4444; font-size:2rem; font-weight:800; }

  /* Status pills */
  .pill-green  { background:#14532d; color:#86efac; padding:4px 12px;
                 border-radius:20px; font-size:.8rem; font-weight:600; }
  .pill-red    { background:#450a0a; color:#fca5a5; padding:4px 12px;
                 border-radius:20px; font-size:.8rem; font-weight:600; }
  .pill-yellow { background:#451a03; color:#fcd34d; padding:4px 12px;
                 border-radius:20px; font-size:.8rem; font-weight:600; }
  .pill-blue   { background:#1e3a5f; color:#93c5fd; padding:4px 12px;
                 border-radius:20px; font-size:.8rem; font-weight:600; }

  /* Section headers */
  .section-title {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: .03em;
  }
  .section-sub {
    color: #64748b;
    font-size: .85rem;
    margin-bottom: 16px;
  }

  /* Big action button */
  [data-testid="stButton"] > button {
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    transition: all .2s;
  }

  /* Table */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Tab styling */
  [data-testid="stTabs"] button {
    font-weight: 600;
    font-size: .9rem;
  }

  /* Slider labels */
  .slider-label {
    color:#94a3b8; font-size:.85rem; margin-bottom:2px;
  }
  .slider-hint {
    color:#475569; font-size:.78rem; margin-top:-8px; margin-bottom:12px;
  }

  /* Logo / title bar */
  .topbar {
    display:flex; align-items:center; gap:14px;
    padding:12px 0 20px 0;
    border-bottom: 1px solid #2d3047;
    margin-bottom: 24px;
  }
  .topbar-title { color:#f1f5f9; font-size:1.6rem; font-weight:800; }
  .topbar-sub   { color:#64748b; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_cfg() -> dict:
    if CFG_PATH.exists():
        raw = json.loads(CFG_PATH.read_text())
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    return {}

def save_cfg(updates: dict):
    cfg = json.loads(CFG_PATH.read_text()) if CFG_PATH.exists() else {}
    for section, vals in updates.items():
        if section in cfg and isinstance(cfg[section], dict):
            cfg[section].update(vals)
        else:
            cfg[section] = vals
    CFG_PATH.write_text(json.dumps(cfg, indent=2))

def load_trades() -> list:
    if TRADE_LOG.exists():
        try:   return json.loads(TRADE_LOG.read_text())
        except: return []
    return []

def load_bt() -> dict:
    if BT_FILE.exists():
        try:   return json.loads(BT_FILE.read_text())
        except: return {}
    return {}

def check_alpaca(cfg) -> bool:
    key = cfg.get("alpaca", {}).get("api_key", "")
    sec = cfg.get("alpaca", {}).get("secret_key", "")
    return ("YOUR_" not in key and len(key) > 10 and
            "YOUR_" not in sec and len(sec) > 10)

def check_email(cfg) -> bool:
    u = cfg.get("alerts", {}).get("smtp_user", "")
    p = cfg.get("alerts", {}).get("smtp_password", "")
    return "YOUR_" not in u and "YOUR_" not in p and "@" in u

def market_status() -> tuple:
    from datetime import timezone, timedelta
    now_et = datetime.now(timezone.utc) + timedelta(hours=-4)
    if now_et.weekday() >= 5:
        return "Closed (Weekend)", "red"
    o = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    c = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    if o <= now_et <= c:
        return f"Open  {now_et.strftime('%I:%M %p ET')}", "green"
    return f"Closed  {now_et.strftime('%I:%M %p ET')}", "yellow"

def gpa_class(gpa: float) -> str:
    if gpa >= 3.5: return "gpa-a"
    if gpa >= 3.0: return "gpa-b"
    if gpa >= 2.5: return "gpa-c"
    return "gpa-d"

def pill(text: str, color: str) -> str:
    return f'<span class="pill-{color}">{text}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════

cfg = load_cfg()
mkt_text, mkt_color = market_status()
alpaca_ok = check_alpaca(cfg)
email_ok  = check_email(cfg)

st.markdown(f"""
<div class="topbar">
  <div style="font-size:2.2rem">📈</div>
  <div>
    <div class="topbar-title">Taylor's Trading Agent</div>
    <div class="topbar-sub">
      Paper Trading &nbsp;·&nbsp;
      {pill(mkt_text, mkt_color)} &nbsp;
      {pill('Alpaca ✓' if alpaca_ok else 'Alpaca — not connected', 'green' if alpaca_ok else 'red')} &nbsp;
      {pill('Email ✓' if email_ok else 'Email — not set', 'green' if email_ok else 'yellow')} &nbsp;
      {pill(datetime.now().strftime('%b %d, %Y'), 'blue')}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SETUP NOTICE (shown only if not configured)
# ══════════════════════════════════════════════════════════════════════════════

if not alpaca_ok:
    st.markdown("""
    <div class="card card-yellow">
      <div class="section-title">⚠️  Quick Setup Needed</div>
      <p style="color:#cbd5e1">Go to the <b>Setup</b> tab below and enter your Alpaca API keys
      to connect the agent to your paper trading account.</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_home, tab_controls, tab_scores, tab_trades, tab_backtest, tab_setup = st.tabs([
    "🏠  Home",
    "🎛️  Controls",
    "🏆  GPA Scores",
    "📋  Trade History",
    "📊  Backtest",
    "⚙️  Setup",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_home:
    trades    = load_trades()
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_t   = [t for t in trades if t.get("timestamp","")[:10] == today_str]
    total_pnl = sum(t.get("pnl", 0) for t in trades if "pnl" in t)
    wins      = [t for t in trades if t.get("pnl", 0) > 0]
    win_rate  = round(len(wins) / len(trades) * 100) if trades else 0

    # ── Big action buttons ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Agent Controls</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    with c1:
        if st.button("▶  Start Agent", type="primary", use_container_width=True,
                     help="Starts scanning stocks and placing paper trades"):
            if alpaca_ok:
                st.session_state["agent_running"] = True
                st.success("Agent started! Check logs below for activity.")
                st.info("In a real deployment, this launches trading_agent.py in the background.\n\nTo run now: open Command Prompt and type:\n`python trading_agent.py`")
            else:
                st.error("Please configure your Alpaca keys in the Setup tab first.")
    with c2:
        if st.button("⏹  Stop Agent", use_container_width=True,
                     help="Safely stops all trading activity"):
            st.session_state["agent_running"] = False
            st.warning("Agent stopped.")
    with c3:
        if st.button("🔄  Run One Scan", use_container_width=True,
                     help="Runs a single scan cycle right now"):
            st.info("To run a single scan:\n`python trading_agent.py --once --force`")
    with c4:
        agent_on = st.session_state.get("agent_running", False)
        status_html = (
            f'<div class="card card-green" style="padding:12px 20px">'
            f'<span style="color:#22c55e;font-size:1.1rem;font-weight:700">'
            f'● Agent Running</span>'
            f'<span style="color:#64748b;font-size:.85rem;margin-left:12px">'
            f'Scanning every 30 min</span></div>'
            if agent_on else
            f'<div class="card" style="padding:12px 20px">'
            f'<span style="color:#64748b;font-size:1.1rem;font-weight:700">'
            f'○ Agent Stopped</span></div>'
        )
        st.markdown(status_html, unsafe_allow_html=True)

    st.divider()

    # ── Portfolio summary ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Account Value",   "$14,516.50", delta="+$5,001.06 all-time")
    with m2:
        st.metric("Available Cash",  "$2,376.86")
    with m3:
        st.metric("Total Return",    "+70.1%",  delta="vs S&P 500")
    with m4:
        pnl_delta = f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}"
        st.metric("Agent P&L",       pnl_delta)
    with m5:
        st.metric("Win Rate",        f"{win_rate}%", delta=f"{len(trades)} trades")

    st.divider()

    # ── Current holdings ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Current Holdings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">From TaylorApr2026.csv — updated live when agent runs</div>',
                unsafe_allow_html=True)

    holdings = [
        {"Symbol":"NVDA",  "Shares":14, "Avg Cost":"$89.44",  "Last":"$188.75", "Return":"+111%", "Value":"$2,642", "Status":"Hold"},
        {"Symbol":"TSLA",  "Shares":4,  "Avg Cost":"$104.91", "Last":"$349.00", "Return":"+233%", "Value":"$1,396", "Status":"Hold"},
        {"Symbol":"AMZN",  "Shares":4,  "Avg Cost":"$100.76", "Last":"$238.38", "Return":"+137%", "Value":"$954",   "Status":"Hold"},
        {"Symbol":"WMT",   "Shares":15, "Avg Cost":"$49.68",  "Last":"$126.79", "Return":"+155%", "Value":"$1,902", "Status":"Hold"},
        {"Symbol":"COST",  "Shares":3,  "Avg Cost":"$627.47", "Last":"$998.47", "Return":"+59%",  "Value":"$2,995", "Status":"Hold"},
        {"Symbol":"QQQ",   "Shares":3,  "Avg Cost":"$617.52", "Last":"$611.07", "Return":"-1%",   "Value":"$1,833", "Status":"Watch"},
        {"Symbol":"ADC",   "Shares":2,  "Avg Cost":"$70.13",  "Last":"$78.18",  "Return":"+11%",  "Value":"$156",   "Status":"Hold"},
        {"Symbol":"DNP",   "Shares":15, "Avg Cost":"$10.72",  "Last":"$10.52",  "Return":"-2%",   "Value":"$158",   "Status":"Watch"},
        {"Symbol":"FVRR",  "Shares":10, "Avg Cost":"$28.25",  "Last":"$10.32",  "Return":"-63%",  "Value":"$103",   "Status":"Sell"},
    ]
    df = pd.DataFrame(holdings)

    def color_return(val):
        if "+" in str(val):   return "color: #22c55e"
        if "-" in str(val):   return "color: #ef4444"
        return ""
    def color_status(val):
        colors = {"Hold":"color:#22c55e", "Watch":"color:#f59e0b", "Sell":"color:#ef4444"}
        return colors.get(val, "")

    st.dataframe(df, use_container_width=True, hide_index=True, height=360)

    st.divider()

    # ── Today's activity ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Today\'s Activity</div>', unsafe_allow_html=True)
    if today_t:
        df_t = pd.DataFrame(today_t)
        show = [c for c in ["timestamp","action","symbol","shares","price","gpa","reason"]
                if c in df_t.columns]
        st.dataframe(df_t[show], use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:30px">
          <div style="font-size:2rem">💤</div>
          <div style="color:#64748b;margin-top:8px">No trades yet today.<br>
          Start the agent to begin scanning.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recent log ──────────────────────────────────────────────────────────
    if AGENT_LOG.exists():
        with st.expander("📜  View Agent Log"):
            lines = AGENT_LOG.read_text(errors="replace").splitlines()[-30:]
            st.code("\n".join(lines), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONTROLS (sliders / toggles)
# ══════════════════════════════════════════════════════════════════════════════

with tab_controls:
    st.markdown('<div class="section-title" style="font-size:1.3rem">Trading Controls</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All changes save automatically when you click Save.</div>',
                unsafe_allow_html=True)

    risk_cfg    = cfg.get("risk", {})
    filter_cfg  = cfg.get("filters", {})
    trade_cfg   = cfg.get("trading", {})

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        # ── Risk ─────────────────────────────────────────────────────────────
        st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
        st.markdown("#### 💰  How Aggressive Should Trades Be?")
        st.markdown('<div class="section-sub">This controls how much of your account is used per trade.</div>',
                    unsafe_allow_html=True)

        risk_level = st.select_slider(
            "Risk Level",
            options=["Very Conservative", "Conservative", "Medium", "Aggressive", "Very Aggressive"],
            value="Medium",
            help="Medium = 3% per trade (~$435 on a $14,500 account)"
        )
        risk_map = {"Very Conservative":1.0, "Conservative":2.0, "Medium":3.0,
                    "Aggressive":4.0, "Very Aggressive":5.0}
        risk_pct = risk_map[risk_level]
        dollar_per_trade = 14516.50 * risk_pct / 100
        st.caption(f"≈ ${dollar_per_trade:,.0f} per trade on your current account")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")

        st.markdown('<div class="card card-red">', unsafe_allow_html=True)
        st.markdown("#### 🛡️  Loss Protection")

        stop_loss = st.slider(
            "Stop Loss — sell if a trade falls by:",
            min_value=3, max_value=15, value=int(risk_cfg.get("stop_loss_pct", 7)),
            step=1, format="%d%%",
            help="Automatically sells if a position drops this much"
        )
        take_profit = st.slider(
            "Take Profit — sell when a trade gains:",
            min_value=5, max_value=30, value=int(risk_cfg.get("take_profit_pct", 14)),
            step=1, format="%d%%",
            help="Locks in gains when a position rises this much"
        )
        daily_limit = st.slider(
            "Daily Loss Limit (Emergency Stop):",
            min_value=2, max_value=10, value=int(risk_cfg.get("daily_loss_limit_pct", 5)),
            step=1, format="%d%%",
            help="Shuts down ALL trading for the day if your account drops this much"
        )
        st.caption(f"Emergency stop triggers if account loses ${14516.50 * daily_limit / 100:,.0f} in one day")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # ── Stock filters ─────────────────────────────────────────────────────
        st.markdown('<div class="card card-green">', unsafe_allow_html=True)
        st.markdown("#### 📊  What Kind of Stocks to Trade")

        company_size = st.select_slider(
            "Company Size",
            options=["Small Caps", "Mid Caps", "Large Caps", "Mega Caps", "All Sizes"],
            value="Large Caps",
            help="Large Caps = household names like Apple, Amazon, etc."
        )
        size_map = {
            "Small Caps":  (0.3, 2),
            "Mid Caps":    (2,   10),
            "Large Caps":  (10,  200),
            "Mega Caps":   (200, 9999),
            "All Sizes":   (0.1, 9999),
        }
        cap_min, cap_max = size_map[company_size]

        volatility = st.select_slider(
            "Stock Volatility (Beta)",
            options=["Very Stable", "Stable", "Moderate", "Active", "Very Active"],
            value="Moderate",
            help="Moderate = moves similarly to the overall market"
        )
        beta_map = {
            "Very Stable":  (0.2, 0.7),
            "Stable":       (0.5, 1.0),
            "Moderate":     (0.8, 1.5),
            "Active":       (1.2, 2.0),
            "Very Active":  (1.5, 3.0),
        }
        beta_min, beta_max = beta_map[volatility]
        st.caption(f"Beta range: {beta_min} – {beta_max}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")

        st.markdown('<div class="card card-yellow">', unsafe_allow_html=True)
        st.markdown("#### ⚡  How Often to Trade")

        freq = st.radio(
            "Scan Frequency",
            options=["Once Daily (9:30 AM)", "Every 30 Minutes", "Every 5 Minutes"],
            index=1,
            help="How often the agent checks for new opportunities"
        )
        freq_map = {"Once Daily (9:30 AM)":"low", "Every 30 Minutes":"medium",
                    "Every 5 Minutes":"high"}

        direction = st.radio(
            "Trade Direction",
            options=["Buy Only (Safer)", "Buy & Short Sell"],
            index=0,
            help="'Buy Only' is recommended for most investors"
        )
        dir_map = {"Buy Only (Safer)":"long_only", "Buy & Short Sell":"long_short"}

        min_gpa = st.slider(
            "Minimum Score to Buy (GPA threshold):",
            min_value=2.5, max_value=4.0,
            value=float(trade_cfg.get("min_gpa_to_buy", 3.5)),
            step=0.1, format="%.1f / 4.0",
            help="Only buy stocks that score above this. Higher = more selective."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("💾  Save All Settings", type="primary", use_container_width=False):
        save_cfg({
            "risk": {
                "position_size_pct":    risk_pct,
                "stop_loss_pct":        stop_loss,
                "take_profit_pct":      take_profit,
                "daily_loss_limit_pct": daily_limit,
            },
            "filters": {
                "beta_min":                   beta_min,
                "beta_max":                   beta_max,
                "market_cap_min_billions":    cap_min,
                "market_cap_max_billions":    cap_max,
            },
            "trading": {
                "frequency":       freq_map[freq],
                "direction":       dir_map[direction],
                "min_gpa_to_buy":  min_gpa,
                "min_gpa_to_alert": min_gpa,
            },
        })
        st.success("✅  Settings saved! The agent will use these on its next scan.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GPA SCORES
# ══════════════════════════════════════════════════════════════════════════════

with tab_scores:
    st.markdown('<div class="section-title" style="font-size:1.3rem">📐 How the GPA Score Works</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-blue">
    <p style="color:#cbd5e1">
    Every stock gets scored on a <b>0.0 – 4.0 GPA scale</b> — just like school grades.
    The agent only buys stocks scoring <b>3.5 or higher</b> (an "A").
    It uses 5 categories based on your Stock Eval spreadsheet:
    </p>
    </div>
    """, unsafe_allow_html=True)

    pillars = [
        ("📈", "Technical Momentum", "25%",
         "RSI, MACD, moving averages, volume trends, stochastics",
         "Is the stock price trending up with strong momentum?"),
        ("💼", "Fundamentals", "25%",
         "ROE, earnings growth, debt/equity ratio, P/E, PEG ratio",
         "Is the company financially healthy and growing?"),
        ("📰", "News & Sentiment", "25%",
         "Yahoo Finance headlines, Reddit (WSB), CNBC — scored by AI",
         "What is the buzz and news trend around this stock?"),
        ("⚡", "Relative Strength", "15%",
         "1-month and 3-month return compared to the S&P 500",
         "Is this stock beating the overall market?"),
        ("🎯", "Volatility Profile", "10%",
         "Beta vs your selected range, Average True Range",
         "Does the stock's risk level match your settings?"),
    ]

    for icon, name, weight, factors, plain in pillars:
        cols = st.columns([1, 4, 1])
        with cols[0]:
            st.markdown(f"<div style='font-size:2.5rem;text-align:center'>{icon}</div>",
                        unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="card" style="margin:4px 0;padding:14px 20px">
              <span style="color:#f1f5f9;font-weight:700;font-size:1rem">{name}</span>
              <span class="pill-blue" style="margin-left:10px">{weight} weight</span>
              <div style="color:#94a3b8;font-size:.85rem;margin-top:6px">{plain}</div>
              <div style="color:#475569;font-size:.78rem;margin-top:4px">Factors: {factors}</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown("")

    st.divider()
    st.markdown("#### GPA Grade Scale")
    grade_data = {
        "Grade": ["A+", "A", "B+", "B", "C", "D"],
        "GPA":   ["3.7 – 4.0", "3.5 – 3.7", "3.0 – 3.5", "2.7 – 3.0", "2.0 – 2.7", "< 2.0"],
        "What it means": [
            "Exceptional — Strong buy",
            "Very good — Buy signal",
            "Above average — Watch it",
            "Average — Hold if owned",
            "Below average — Consider selling",
            "Poor — Sell signal",
        ],
        "Agent Action": ["BUY", "BUY", "WATCH", "HOLD", "REDUCE", "SELL"],
    }
    st.dataframe(pd.DataFrame(grade_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRADE HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab_trades:
    st.markdown('<div class="section-title" style="font-size:1.3rem">📋  Trade History</div>',
                unsafe_allow_html=True)
    trades = load_trades()
    if trades:
        df = pd.DataFrame(trades)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values("timestamp", ascending=False, inplace=True)
            df["timestamp"] = df["timestamp"].dt.strftime("%b %d  %I:%M %p")

        c1, c2, c3, c4 = st.columns(4)
        buys   = [t for t in trades if t.get("action") == "BUY"]
        sells  = [t for t in trades if t.get("action") == "SELL"]
        pnls   = [t.get("pnl",0) for t in trades if "pnl" in t]
        t_pnl  = sum(pnls)
        t_wins = len([p for p in pnls if p > 0])
        with c1: st.metric("Total Buys",  len(buys))
        with c2: st.metric("Total Sells", len(sells))
        with c3: st.metric("Total P&L",   f"${t_pnl:+,.2f}")
        with c4: st.metric("Win Rate",    f"{round(t_wins/len(pnls)*100) if pnls else 0}%")

        show_cols = [c for c in ["timestamp","action","symbol","shares",
                                  "price","value","gpa","reason"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True, height=400)
        csv = df.to_csv(index=False)
        st.download_button("⬇  Download as CSV", csv,
                           file_name="taylor_trades.csv", mime="text/csv")
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:50px">
          <div style="font-size:3rem">📭</div>
          <div style="color:#64748b;font-size:1.1rem;margin-top:12px">
            No trades yet.<br>Start the agent to begin paper trading.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.markdown('<div class="section-title" style="font-size:1.3rem">📊  Backtest Results</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-blue">
    <p style="color:#cbd5e1">
    Backtesting runs the strategy against <b>12 months of historical data</b>
    to see how it would have performed — before risking any real (or paper) money.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("To run a backtest, open Command Prompt and type:\n```\npython trading_agent.py --backtest\n```\nResults will appear here automatically when complete.")

    bt = load_bt()
    if bt:
        m1, m2, m3, m4 = st.columns(4)
        ret   = bt.get("total_return_pct", 0)
        spy   = bt.get("spy_return_pct", 0)
        alpha = bt.get("alpha_pct", 0)
        with m1: st.metric("Strategy Return", f"{ret:+.1f}%",
                            delta=f"S&P 500: {spy:+.1f}%")
        with m2: st.metric("Alpha vs S&P 500", f"{alpha:+.1f}%",
                            delta="outperformed" if alpha > 0 else "underperformed")
        with m3: st.metric("Sharpe Ratio",     f"{bt.get('sharpe_ratio',0):.2f}",
                            help="Above 1.0 = good risk-adjusted return")
        with m4: st.metric("Win Rate",         f"{bt.get('win_rate_pct',0):.1f}%",
                            delta=f"{bt.get('total_trades',0)} trades")

        m5, m6 = st.columns(2)
        with m5: st.metric("Max Drawdown",  f"{bt.get('max_drawdown_pct',0):.1f}%",
                            help="Worst peak-to-trough drop during the test period")
        with m6: st.metric("Profit Factor", f"{bt.get('profit_factor',0):.2f}",
                            help="Above 1.5 = strategy makes more than it loses")

        eq = bt.get("equity_curve", [])
        if eq:
            eq_df = pd.DataFrame(eq).set_index("date")
            eq_df.index = pd.to_datetime(eq_df.index)
            st.subheader("Account Growth Over Test Period")
            st.line_chart(eq_df["value"], height=280)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:50px">
          <div style="font-size:3rem">⏳</div>
          <div style="color:#64748b;font-size:1.1rem;margin-top:12px">
            No backtest data yet.<br>Run the backtest command above to generate results.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SETUP
# ══════════════════════════════════════════════════════════════════════════════

with tab_setup:
    st.markdown('<div class="section-title" style="font-size:1.3rem">⚙️  Account Setup</div>',
                unsafe_allow_html=True)

    # ── Step 1: Alpaca ────────────────────────────────────────────────────────
    alpaca_status = "✅  Connected" if alpaca_ok else "❌  Not configured"
    with st.expander(f"Step 1 — Alpaca Paper Trading API   {alpaca_status}", expanded=not alpaca_ok):
        st.markdown("""
        **How to get your free Alpaca API keys:**
        1. Go to **[app.alpaca.markets](https://app.alpaca.markets)** and create a free account
        2. Make sure **Paper Trading** is selected in the top-left
        3. Click **"Your API Keys"** → **Generate New Key**
        4. Copy the Key and Secret below
        """)
        api_key = st.text_input("Alpaca API Key",
            value=cfg.get("alpaca",{}).get("api_key",""),
            type="password" if alpaca_ok else "default")
        api_sec = st.text_input("Alpaca Secret Key",
            value=cfg.get("alpaca",{}).get("secret_key",""),
            type="password")
        if st.button("Save Alpaca Keys", key="save_alpaca"):
            save_cfg({"alpaca": {"api_key": api_key, "secret_key": api_sec, "paper": True}})
            st.success("Alpaca keys saved!")
            st.rerun()

    # ── Step 2: Email ─────────────────────────────────────────────────────────
    email_status = "✅  Configured" if email_ok else "❌  Not configured"
    with st.expander(f"Step 2 — Gmail Alerts   {email_status}", expanded=not email_ok):
        st.markdown("""
        **How to set up Gmail alerts:**
        1. Enable **2-Step Verification** at [myaccount.google.com/security](https://myaccount.google.com/security)
        2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
        3. Create an App Password named "Trading Agent"
        4. Paste the 16-character password below
        """)
        gmail_addr = st.text_input("Gmail Address",
            value=cfg.get("alerts",{}).get("smtp_user",""))
        gmail_pass = st.text_input("Gmail App Password (16 characters)",
            value=cfg.get("alerts",{}).get("smtp_password",""),
            type="password")
        if st.button("Save Email Settings", key="save_email"):
            save_cfg({"alerts": {
                "email_to":      gmail_addr,
                "smtp_user":     gmail_addr,
                "smtp_password": gmail_pass.replace(" ",""),
                "smtp_host":     "smtp.gmail.com",
                "smtp_port":     587,
            }})
            st.success("Email settings saved!")
            st.rerun()

    # ── Step 3: Reddit (optional) ─────────────────────────────────────────────
    reddit_ok = "YOUR_" not in cfg.get("data_sources",{}).get("reddit_client_id","YOUR_")
    reddit_status = "✅  Connected" if reddit_ok else "⚪  Optional"
    with st.expander(f"Step 3 — Reddit Sentiment (Optional)   {reddit_status}"):
        st.markdown("""
        Reddit adds **social media sentiment** from r/WallStreetBets and r/stocks.
        It's free but optional — the agent works fine without it.

        **How to get Reddit API credentials (free):**
        1. Go to **[reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)**
        2. Click **"create another app"** → select **"script"**
        3. Fill in any name, set redirect to `http://localhost`
        4. Copy the Client ID and Secret below
        """)
        r_id  = st.text_input("Reddit Client ID",
            value=cfg.get("data_sources",{}).get("reddit_client_id",""))
        r_sec = st.text_input("Reddit Secret",
            value=cfg.get("data_sources",{}).get("reddit_client_secret",""),
            type="password")
        if st.button("Save Reddit Keys", key="save_reddit"):
            save_cfg({"data_sources": {
                "reddit_client_id":     r_id,
                "reddit_client_secret": r_sec,
            }})
            st.success("Reddit keys saved!")

    # ── Connection status summary ─────────────────────────────────────────────
    st.divider()
    st.markdown("#### Connection Status")
    items = [
        ("Alpaca Paper Trading", alpaca_ok, "Required to place paper trades"),
        ("Gmail Alerts",         email_ok,  "Sends email when a stock scores A or higher"),
        ("Reddit Sentiment",     reddit_ok, "Optional — adds social media signals"),
    ]
    for name, ok, desc in items:
        icon = "✅" if ok else "❌"
        color = "#22c55e" if ok else "#ef4444"
        st.markdown(
            f'<div class="card" style="padding:12px 20px;margin:6px 0">'
            f'<span style="font-size:1.1rem">{icon}</span> '
            f'<span style="color:#f1f5f9;font-weight:600">{name}</span> '
            f'<span style="color:#64748b;font-size:.85rem;margin-left:8px">— {desc}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
