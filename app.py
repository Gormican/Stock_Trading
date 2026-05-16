"""
Taylor's Trading Agent — Streamlit App
3-category GPA model · Multi-account Alpaca · Named strategies
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

# ── Module imports ─────────────────────────────────────────────────────────────
from modules.account_manager  import AccountManager
from modules.strategy_manager import StrategyManager
from modules.data_engine      import DataEngine
from modules.gpa_scorer       import GPAEngine, _recommendation as gpa_recommendation

log = logging.getLogger("TradingApp")
logging.basicConfig(level=logging.INFO)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
CFG_FILE  = BASE_DIR / "config.json"
ACCT_MGR  = AccountManager(CFG_FILE)
STRAT_MGR = StrategyManager()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taylor's Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Metric cards — dark background, force white text ── */
[data-testid="stMetric"] {
    background: #1e2130;
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #2d3250;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] p { color: #c8cfe8 !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"]   { color: #ffffff   !important; }
[data-testid="stMetric"] [data-testid="stMetricDelta"]   { color: #a0aec0   !important; }

/* ── Widget labels on light background (sliders, number inputs, radio) ── */
[data-testid="stWidgetLabel"] p,
.stSlider label,
div[data-testid="stNumberInput"] label { color: #1f2937 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { color: #555e7a; font-size: 14px; }
.stTabs [aria-selected="true"] { color: #111827; font-weight: 600; }

/* ── Sidebar text ── */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] label { color: #e8eaf6 !important; }

/* ── GPA badges ── */
.gpa-a  { color: #16a34a; font-weight: 700; }
.gpa-b  { color: #ca8a04; font-weight: 700; }
.gpa-c  { color: #ea580c; font-weight: 700; }
.gpa-d  { color: #dc2626; font-weight: 700; }

/* ── Signal chips ── */
.chip-buy  { background:#166534; color:#bbf7d0; padding:2px 8px; border-radius:4px; font-size:12px; }
.chip-sell { background:#7f1d1d; color:#fecaca; padding:2px 8px; border-radius:4px; font-size:12px; }
.chip-hold { background:#374151; color:#d1d5db; padding:2px 8px; border-radius:4px; font-size:12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# GPA CACHE  (persists across restarts)
# ─────────────────────────────────────────────────────────────────────────────
GPA_CACHE_FILE = BASE_DIR / "gpa_cache.json"

def load_gpa_cache() -> dict:
    """Load saved GPA scores from disk so they survive restarts."""
    try:
        if GPA_CACHE_FILE.exists():
            return json.loads(GPA_CACHE_FILE.read_text())
    except Exception:
        pass
    return {}

def _to_native(obj):
    """Recursively convert numpy/pandas types to plain Python for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_gpa_cache(gpa_dict: dict):
    """Save current GPA scores to disk, surviving restarts."""
    try:
        slim = {}
        for sym, r in gpa_dict.items():
            slim[sym] = _to_native({
                "gpa":         r.get("gpa"),
                "grade":       r.get("grade"),
                "buy_signal":  r.get("buy_signal"),
                "sell_signal": r.get("sell_signal"),
                "top_drivers": r.get("top_drivers", []),
                "categories":  r.get("categories", {}),
                "timestamp":   r.get("timestamp", ""),
            })
        GPA_CACHE_FILE.write_text(json.dumps(slim, indent=2))
    except Exception as e:
        log.warning(f"GPA cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL ALERTS
# ─────────────────────────────────────────────────────────────────────────────
def _load_alert_cfg() -> dict:
    """Read the alerts section from config.json."""
    try:
        cfg = json.loads(CFG_FILE.read_text())
        return cfg.get("alerts", {})
    except Exception:
        return {}


def send_email(subject: str, body: str) -> bool:
    """Send a plain-text email using the SMTP settings in config.json.
    Returns True on success, False on failure (never raises)."""
    import smtplib
    from email.mime.text import MIMEText
    alert = _load_alert_cfg()
    to_addr   = alert.get("email_to", "")
    smtp_host = alert.get("smtp_host", "smtp.gmail.com")
    smtp_port = int(alert.get("smtp_port", 587))
    smtp_user = alert.get("smtp_user", "")
    smtp_pass = alert.get("smtp_password", "")

    if not (to_addr and smtp_user and smtp_pass):
        log.warning("Email not sent — SMTP credentials missing in config.json")
        return False

    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = f"[Taylor's Trading Agent] {subject}"
        msg["From"]    = smtp_user
        msg["To"]      = to_addr
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_addr], msg.as_string())
        log.info(f"Email sent: {subject}")
        return True
    except Exception as e:
        log.warning(f"Email failed: {e}")
        return False


def send_trade_email(symbol: str, side: str, qty: int,
                     account_name: str, order_id: str = "", error: str = ""):
    """Send a trade confirmation or failure alert."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    if error:
        subject = f"❌ Trade FAILED — {side.upper()} {qty}× {symbol}"
        body = (
            f"Trade failed at {ts}\n\n"
            f"  Account : {account_name}\n"
            f"  Action  : {side.upper()}\n"
            f"  Symbol  : {symbol}\n"
            f"  Shares  : {qty}\n"
            f"  Error   : {error}\n"
        )
    else:
        emoji = "🟢" if side == "buy" else "🔴"
        subject = f"{emoji} Trade Executed — {side.upper()} {qty}× {symbol}"
        body = (
            f"Trade executed at {ts}\n\n"
            f"  Account  : {account_name}\n"
            f"  Action   : {side.upper()}\n"
            f"  Symbol   : {symbol}\n"
            f"  Shares   : {qty}\n"
            f"  Order ID : {order_id}\n"
        )
    send_email(subject, body)


def send_gpa_alert_email(gpa_dict: dict):
    """After a GPA refresh, email a summary of any BUY or SELL signals."""
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    buys    = [(s, r) for s, r in gpa_dict.items() if r.get("buy_signal")]
    sells   = [(s, r) for s, r in gpa_dict.items() if r.get("sell_signal")]

    if not buys and not sells:
        return  # nothing to alert on

    lines = [f"GPA Refresh completed at {ts}\n"]

    if buys:
        lines.append("── BUY Signals ──────────────────────")
        for sym, r in sorted(buys, key=lambda x: -x[1].get("gpa", 0)):
            drivers = ", ".join(r.get("top_drivers", []))
            lines.append(f"  {sym:6s}  GPA {r['gpa']:.2f} {r.get('grade','?')}  "
                         f"Drivers: {drivers or '—'}")

    if sells:
        lines.append("\n── SELL Signals ─────────────────────")
        for sym, r in sorted(sells, key=lambda x: x[1].get("gpa", 4)):
            drivers = ", ".join(r.get("top_drivers", []))
            lines.append(f"  {sym:6s}  GPA {r['gpa']:.2f} {r.get('grade','?')}  "
                         f"Drivers: {drivers or '—'}")

    count = len(buys) + len(sells)
    subject = f"GPA Alert — {count} signal{'s' if count != 1 else ''} detected"
    send_email(subject, "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    # If a Strategies.xlsx with an ACTIVE flag is present, prefer it as the
    # startup default. Falls back to the account's saved strategy otherwise.
    xlsx_active = STRAT_MGR.active_strategy_from_xlsx()
    acct_default = ACCT_MGR.get_strategy(ACCT_MGR.get_last())
    defaults = {
        "account_name":   ACCT_MGR.get_last(),
        "strategy_name":  xlsx_active or acct_default,
        "trade_queue":    [],
        "scan_results":        [],
        "scan_strategy_name":  "Default",
        "lookup_result":       None,
        "positions_gpa":  load_gpa_cache(),   # ← loads saved scores on startup
        "last_scan_time": None,
        "spy_df":         None,
        "spy_fetch_date": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_alpaca_portfolio(account_name: str, api_key: str, secret_key: str,
                            base_url: str) -> dict:
    try:
        from alpaca.trading.client import TradingClient
        paper = "paper-api" in base_url
        client = TradingClient(api_key, secret_key, paper=paper)
        account = client.get_account()
        positions = client.get_all_positions()
        pos_list = []
        for p in positions:
            pos_list.append({
                "symbol":    p.symbol,
                "qty":       float(p.qty),
                "avg_cost":  float(p.avg_entry_price),
                "mkt_value": float(p.market_value),
                "cur_price": float(p.current_price),
                "gain_loss": float(p.unrealized_pl),
                "gain_pct":  float(p.unrealized_plpc) * 100,
                "side":      p.side.value,
            })
        return {
            "ok":           True,
            "equity":       float(account.equity),
            "cash":         float(account.cash),
            "buying_power": float(account.buying_power),
            "day_pl":       float(account.equity) - float(account.last_equity),
            "positions":    pos_list,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "positions": []}


def get_alpaca_client(account_name: str):
    acct = ACCT_MGR.get(account_name)
    k = acct.get("api_key", "")
    s = acct.get("secret_key", "")
    if not k or not s or len(k) < 10:
        return None
    try:
        from alpaca.trading.client import TradingClient
        paper = acct.get("paper", True)
        return TradingClient(k, s, paper=paper)
    except Exception:
        return None


def submit_order(account_name: str, symbol: str, side: str, qty: int) -> dict:
    client = get_alpaca_client(account_name)
    if client is None:
        return {"ok": False, "error": "Account not configured"}
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums    import OrderSide, TimeInForce
        req = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_data=req)
        return {"ok": True, "order_id": str(order.id), "status": str(order.status)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# GPA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_spy_df():
    today = datetime.now().date()
    if st.session_state.spy_df is None or st.session_state.spy_fetch_date != today:
        try:
            spy = yf.download("SPY", period="1y", progress=False, auto_adjust=True)
            st.session_state.spy_df = spy
            st.session_state.spy_fetch_date = today
        except Exception:
            pass
    return st.session_state.spy_df


def _load_av_key() -> str:
    """
    Read the Alpha Vantage API key. Checks ALPHA_VANTAGE_API_KEY env var first,
    then falls back to config.json using case-insensitive section matching.
    """
    import os
    env_val = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if env_val:
        return env_val
    try:
        cfg = json.loads(CFG_FILE.read_text())
        # Case-insensitive search: any section whose name contains these tokens
        for section_key, sec in cfg.items():
            low = section_key.lower().replace(" ", "").replace("_", "")
            if any(tok in low for tok in ("alpha", "vantage", "advantage", "alphav")):
                if isinstance(sec, dict):
                    for key_name in ("api_key", "key", "apikey", "API_KEY", "Key",
                                     "api key", "APIKey", "apiKey"):
                        val = sec.get(key_name, "")
                        if val:
                            log.info(f"AV key loaded from config section '{section_key}'")
                            return str(val).strip()
        # Flat key fallback
        for flat in ("alpha_vantage_key", "av_api_key", "alphavantage_key", "av_key"):
            val = cfg.get(flat, "")
            if val:
                return str(val).strip()
        # Key not found — log all section names so user can diagnose
        log.warning(f"AV key not found. Config sections present: {list(cfg.keys())}")
    except Exception as e:
        log.warning(f"Could not load AV key: {e}")
    return ""


def build_engine() -> GPAEngine:
    strategy = STRAT_MGR.get(st.session_state.strategy_name)
    engine   = GPAEngine(strategy=strategy)
    spy_df   = get_spy_df()
    if spy_df is not None:
        engine.set_spy_df(spy_df)
    av_key = _load_av_key()
    engine.set_api_keys(av_key=av_key)
    return engine


def score_symbol(symbol: str, engine: GPAEngine = None):
    if engine is None:
        engine = build_engine()
    try:
        de     = DataEngine()
        ticker = yf.Ticker(symbol.upper())
        ohlcv  = ticker.history(period="1y", auto_adjust=True)
        if ohlcv is None or len(ohlcv) < 20:
            return None

        # Fix yfinance multi-level columns (newer versions return ("Close","AAPL") etc.)
        if isinstance(ohlcv.columns, pd.MultiIndex):
            ohlcv.columns = ohlcv.columns.get_level_values(0)

        fundamentals = de.get_fundamentals(symbol)
        sentiment    = de.get_sentiment(symbol)
        return engine.score(symbol.upper(), ohlcv, fundamentals, sentiment)
    except Exception as e:
        log.warning(f"score_symbol({symbol}): {e}")
        return None


def gpa_color_class(gpa: float) -> str:
    if gpa >= 3.0: return "gpa-a"
    if gpa >= 2.5: return "gpa-b"
    if gpa >= 2.0: return "gpa-c"
    return "gpa-d"


def signal_chip(result: dict) -> str:
    if result.get("buy_signal"):  return '<span class="chip-buy">BUY</span>'
    if result.get("sell_signal"): return '<span class="chip-sell">SELL</span>'
    return '<span class="chip-hold">HOLD</span>'


def reweight_result(r: dict, strategy: dict) -> dict:
    """
    Re-apply strategy weights to a cached GPA result.
    Uses the raw sub-scores stored in r["categories"] — no API calls.
    Returns a new dict; the original is not mutated.
    """
    import copy
    from modules.strategy_manager import DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS

    cats = r.get("categories", {})
    if not cats:
        return r  # no raw sub-scores available

    w          = strategy.get("weights",    DEFAULT_WEIGHTS)
    fund_sub   = w.get("fund_sub", {"valuation": 0.33, "financial": 0.33, "estimates": 0.34})
    tech_sub   = w.get("tech_sub", {"trend": 0.60, "oscillators": 0.40})
    thresholds = strategy.get("thresholds", DEFAULT_THRESHOLDS)

    sent_score = cats.get("sentiment",    {}).get("score", 2.0)
    fund_subs  = cats.get("fundamentals", {}).get("sub_scores", {})
    tech_subs  = cats.get("technical",    {}).get("sub_scores", {})

    # Re-apply fundamentals sub-weights
    fund_score = round(
        fund_subs.get("valuation",  {}).get("score", 2.0) * fund_sub.get("valuation", 0.33) +
        fund_subs.get("financial",  {}).get("score", 2.0) * fund_sub.get("financial",  0.33) +
        fund_subs.get("estimates",  {}).get("score", 2.0) * fund_sub.get("estimates",  0.34),
        3
    )

    # Re-apply technical sub-weights
    tech_score = round(
        tech_subs.get("trend",       {}).get("score", 2.0) * tech_sub.get("trend",       0.60) +
        tech_subs.get("oscillators", {}).get("score", 2.0) * tech_sub.get("oscillators", 0.40),
        3
    )

    # Re-apply main weights
    gpa = round(
        sent_score * w.get("sentiment",    0.20) +
        fund_score * w.get("fundamentals", 0.45) +
        tech_score * w.get("technical",    0.35),
        3
    )

    # Grade & recommendation (9-tier scale, see _grade/_recommendation in gpa_scorer.py)
    if   gpa >= 3.7: grade, rec = "A+", "Strong Buy"
    elif gpa >= 3.5: grade, rec = "A",  "Buy"
    elif gpa >= 3.3: grade, rec = "A-", "Weak Buy"
    elif gpa >= 3.0: grade, rec = "B+", "Hold/Buy"
    elif gpa >= 2.7: grade, rec = "B",  "Hold"
    elif gpa >= 2.3: grade, rec = "B-", "Hold/Sell"
    elif gpa >= 2.0: grade, rec = "C+", "Weak Sell"
    elif gpa >= 1.7: grade, rec = "C",  "Sell"
    else:            grade, rec = "C-", "Strong Sell"

    result = copy.deepcopy(r)
    result.update({
        "gpa":            gpa,
        "grade":          grade,
        "recommendation": rec,
        "buy_signal":     gpa >= thresholds.get("min_gpa_to_buy",  3.5),
        "sell_signal":    gpa <= thresholds.get("max_gpa_to_sell", 2.5),
        "thresholds":     thresholds,
        "weights_used":   w,
    })
    # Update per-category display weights so render_gpa_detail shows correct numbers
    if "sentiment" in result["categories"]:
        result["categories"]["sentiment"]["weight"] = w.get("sentiment", 0.20)
    if "fundamentals" in result["categories"]:
        result["categories"]["fundamentals"]["score"]       = fund_score
        result["categories"]["fundamentals"]["weight"]      = w.get("fundamentals", 0.45)
        result["categories"]["fundamentals"]["sub_weights"] = fund_sub
    if "technical" in result["categories"]:
        result["categories"]["technical"]["score"]       = tech_score
        result["categories"]["technical"]["weight"]      = w.get("technical", 0.35)
        result["categories"]["technical"]["sub_weights"] = tech_sub
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHART DATA
# ─────────────────────────────────────────────────────────────────────────────
def get_comparison_df(symbol: str, period_label: str):
    period_map = {
        "1D":    ("1d",  "5m"),
        "1M":    ("1mo", "1d"),
        "1Y":    ("1y",  "1d"),
        "Start": ("max", "1wk"),
    }
    period, interval = period_map.get(period_label, ("1y", "1d"))
    try:
        sym_df = yf.download(symbol.upper(), period=period,
                              interval=interval, progress=False, auto_adjust=True)
        spy_df = yf.download("SPY", period=period,
                              interval=interval, progress=False, auto_adjust=True)
        if sym_df is None or spy_df is None or len(sym_df) < 2:
            return None
        # Fix yfinance multi-level columns
        if isinstance(sym_df.columns, pd.MultiIndex):
            sym_df.columns = sym_df.columns.get_level_values(0)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        sym_norm = (sym_df["Close"] / sym_df["Close"].iloc[0] * 100).rename(symbol.upper())
        spy_norm = (spy_df["Close"] / spy_df["Close"].iloc[0] * 100).rename("S&P 500")
        return pd.concat([sym_norm, spy_norm], axis=1).dropna()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NEWS SCAN
# ─────────────────────────────────────────────────────────────────────────────
def get_news_symbols() -> list:
    try:
        import feedparser, re
        symbols = set()
        trending = [
            "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
            "JPM","V","UNH","JNJ","WMT","PG","HD","CVX",
            "NFLX","AMD","INTC","BAC","DIS",
        ]
        symbols.update(trending)
        feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI&region=US&lang=en-US",
        ]
        for url in feeds:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title","") + " " + entry.get("summary","")
                found = re.findall(r'\b([A-Z]{2,5})\b', title)
                skip = {"US","CEO","IPO","ETF","GDP","FDA","AI","EPS","FED","SEC"}
                for sym in found:
                    if 2 <= len(sym) <= 5 and sym not in skip:
                        symbols.add(sym)
        return list(symbols)[:30]
    except Exception:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
                "JPM","V","UNH","JNJ","WMT","BAC","NFLX","AMD"]


def run_news_scan(min_gpa: float = 3.0) -> list:
    """Score all news symbols and return ALL results with raw sub-scores.
    Filtering by min_gpa happens at display time so strategy-switching works
    without re-scanning."""
    symbols = get_news_symbols()
    engine  = build_engine()
    results = []
    prog = st.progress(0, text="Scanning news stocks…")
    for i, sym in enumerate(symbols):
        prog.progress((i + 1) / len(symbols), text=f"Scoring {sym}…")
        r = score_symbol(sym, engine)
        if r and r.get("categories"):   # store everything that has raw sub-scores
            results.append(r)
        time.sleep(0.2)
    prog.empty()
    # Record which strategy was active at scan time
    st.session_state.scan_strategy_name = st.session_state.strategy_name
    results.sort(key=lambda x: x["gpa"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GPA DETAIL RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_gpa_detail(result: dict):
    gpa  = result["gpa"]
    cats = result.get("categories", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("GPA", f"{gpa:.2f}", delta=result.get("grade",""))
    col2.metric("Signal", "🟢 BUY" if result.get("buy_signal") else
                           ("🔴 SELL" if result.get("sell_signal") else "🟡 HOLD"))
    col3.metric("Top Drivers", ", ".join(result.get("top_drivers", [])) or "—")

    st.divider()

    # Sentiment
    sent = cats.get("sentiment", {})
    with st.expander(f"😶 Sentiment  {sent.get('score',0):.2f} / 4.0  "
                     f"(weight {sent.get('weight',0.20)*100:.0f}%)", expanded=False):
        d      = sent.get("detail", {})
        source = d.get("source", "none")
        av     = d.get("alpha_vantage", {})
        st_tw  = d.get("stocktwits", {})
        legacy = d.get("legacy_vader", {})

        from modules.gpa_scorer import _av_calls_today, AV_DAILY_LIMIT
        calls_used = _av_calls_today()
        st.caption(f"Source: **{source}**  ·  AV quota: {calls_used}/{AV_DAILY_LIMIT} calls used today")
        c1, c2, c3 = st.columns(3)

        if av:
            c1.metric("AV Score",    f"{av.get('score', 0):.2f}")
            c2.metric("AV Label",    av.get("label", "—"))
            c3.metric("AV Articles", av.get("article_count", 0))
        elif legacy:
            c1.metric("Combined Score", f"{legacy.get('combined', 0):.3f}")
            c2.metric("Velocity",       f"{legacy.get('velocity', 0):.3f}")
            c3.metric("Headlines",      d.get("headline_count", 0))
        else:
            c1.metric("AV Score", "—")
            c2.metric("AV Label", "No data")
            c3.metric("AV Articles", 0)

        if st_tw:
            c4, c5, c6 = st.columns(3)
            bull_pct = f"{st_tw.get('bullish_ratio', 0)*100:.0f}%"
            c4.metric("StockTwits Score",   f"{st_tw.get('score', 0):.2f}")
            c5.metric("Bullish Ratio",      bull_pct)
            c6.metric("Messages / Tagged",  f"{st_tw.get('messages',0)} / {st_tw.get('tagged',0)}")

    # Fundamentals
    fund = cats.get("fundamentals", {})
    with st.expander(f"📊 Fundamentals  {fund.get('score',0):.2f} / 4.0  "
                     f"(weight {fund.get('weight',0.45)*100:.0f}%)", expanded=False):
        sub_s = fund.get("sub_scores", {})
        sub_w = fund.get("sub_weights", {})
        for label, key in [("Valuation","valuation"),("Financial","financial"),("Estimates","estimates")]:
            sub = sub_s.get(key, {})
            w_pct = sub_w.get(key, 0.33) * 100
            st.markdown(f"**{label}**  ·  score {sub.get('score',0):.2f}  ·  weight {w_pct:.0f}%")
            rows = [{"Factor": k.replace("_"," ").title(),
                     "Value":  str(v.get("value","n/a")),
                     "Score":  f"{v.get('score',0):.1f}",
                     "Note":   v.get("label","")}
                    for k, v in sub.get("detail", {}).items()]
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Technical
    tech = cats.get("technical", {})
    with st.expander(f"📉 Technical  {tech.get('score',0):.2f} / 4.0  "
                     f"(weight {tech.get('weight',0.35)*100:.0f}%)", expanded=False):
        sub_s = tech.get("sub_scores", {})
        sub_w = tech.get("sub_weights", {})
        for label, key in [("Trend","trend"),("Oscillators","oscillators")]:
            sub = sub_s.get(key, {})
            w_pct = sub_w.get(key, 0.5) * 100
            st.markdown(f"**{label}**  ·  score {sub.get('score',0):.2f}  ·  weight {w_pct:.0f}%")
            rows = []
            for k, v in sub.get("detail", {}).items():
                if isinstance(v, dict):
                    rows.append({
                        "Indicator": k.replace("_"," ").title(),
                        "Value":     str(v.get("value", v.get("k_pct", v.get("ratio","n/a")))),
                        "Score":     f"{v.get('score',0):.1f}",
                        "Note":      v.get("label",""),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Taylor's Trading Agent")
        st.divider()

        st.markdown("### Account")
        acct_names = ACCT_MGR.account_names()
        acct_all   = ACCT_MGR.get_all()

        def acct_label(n):
            a    = acct_all[n]
            icon = a.get("icon","📘")
            lbl  = a.get("label", n)
            cfg  = " ✓" if ACCT_MGR.is_configured(n) else " ✗"
            return f"{icon} {lbl}{cfg}"

        current_idx = acct_names.index(st.session_state.account_name) \
                      if st.session_state.account_name in acct_names else 0
        chosen = st.radio("Select account", acct_names, index=current_idx,
                           format_func=acct_label, key="acct_radio")
        if chosen != st.session_state.account_name:
            st.session_state.account_name  = chosen
            st.session_state.strategy_name = ACCT_MGR.get_strategy(chosen)
            ACCT_MGR.set_last(chosen)
            st.cache_data.clear()
            st.rerun()

        acct = acct_all[st.session_state.account_name]
        if acct.get("paper") is False:
            st.error("⚠️ LIVE ACCOUNT — Real money!", icon="🔴")
        else:
            st.info("Paper trading account", icon="📘")

        st.divider()

        st.markdown("### Strategy")
        strat_names = STRAT_MGR.names()
        strat_idx   = strat_names.index(st.session_state.strategy_name) \
                      if st.session_state.strategy_name in strat_names else 0
        chosen_strat = st.selectbox("Active strategy", strat_names,
                                     index=strat_idx, key="strat_select")
        if chosen_strat != st.session_state.strategy_name:
            st.session_state.strategy_name = chosen_strat
            ACCT_MGR.save_strategy(st.session_state.account_name, chosen_strat)

        strat = STRAT_MGR.get(st.session_state.strategy_name)
        t     = strat.get("thresholds", {})
        auto  = strat.get("auto_trade", False)
        st.caption(
            f"Buy ≥ {t.get('min_gpa_to_buy',3.5):.1f}  "
            f"· Sell ≤ {t.get('max_gpa_to_sell',2.5):.1f}  "
            f"· {'🤖 Auto' if auto else '✋ Confirm'}"
        )
        if strat.get("description"):
            st.caption(strat["description"])

        st.divider()

        n_pending = len([tr for tr in st.session_state.trade_queue
                         if tr.get("status") in ("pending","auto")])
        if n_pending:
            st.warning(f"🔔 {n_pending} trade{'s' if n_pending > 1 else ''} pending")

        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: HOME
# ─────────────────────────────────────────────────────────────────────────────
def render_home():
    st.header("Portfolio Overview")
    acct = ACCT_MGR.get(st.session_state.account_name)

    if not ACCT_MGR.is_configured(st.session_state.account_name):
        st.warning("⚙️ API keys not set. Go to **Configure → API Keys**.")
        return

    portfolio = fetch_alpaca_portfolio(
        st.session_state.account_name,
        acct.get("api_key",""), acct.get("secret_key",""), acct.get("base_url","")
    )
    if not portfolio.get("ok"):
        st.error(f"Alpaca error: {portfolio.get('error','Unknown')}")
        return

    positions = portfolio["positions"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Value",  f"${portfolio['equity']:,.2f}")
    c2.metric("Cash",             f"${portfolio['cash']:,.2f}")
    c3.metric("Buying Power",     f"${portfolio['buying_power']:,.2f}")
    day_pl = portfolio["day_pl"]
    c4.metric("Day P&L", f"${day_pl:+,.2f}",
              delta=f"{day_pl/portfolio['equity']*100:+.2f}%"
              if portfolio["equity"] > 0 else None)
    c5.metric("Positions", len(positions))

    st.divider()

    if not positions:
        st.info("No open positions. Check the **Opportunities** tab for buy candidates.")
        return

    active_strat = STRAT_MGR.get(st.session_state.strategy_name)
    st.subheader(f"Current Holdings  ·  Strategy: {st.session_state.strategy_name}")
    rows = []
    for p in positions:
        sym    = p["symbol"]
        raw_r  = st.session_state.positions_gpa.get(sym)
        gpa_r  = reweight_result(raw_r, active_strat) if raw_r else None
        if gpa_r:
            rec = gpa_r.get("recommendation") or gpa_recommendation(gpa_r["gpa"])
        else:
            rec = "—"
        rows.append({
            "Symbol":         sym,
            "Grade":          gpa_r.get("grade","—") if gpa_r else "—",
            "GPA":            f"{gpa_r['gpa']:.2f}" if gpa_r else "—",
            "Recommendation": rec,
            "Shares":         p["qty"],
            "Avg Cost":       f"${p['avg_cost']:.2f}",
            "Price":          f"${p['cur_price']:.2f}",
            "Value":          f"${p['mkt_value']:,.0f}",
            "Gain $":         f"${p['gain_loss']:+,.2f}",
            "Gain %":         f"{p['gain_pct']:+.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── Strategy comparison table ──────────────────────────────────────────────
    scored_syms = [p["symbol"] for p in positions
                   if st.session_state.positions_gpa.get(p["symbol"], {}).get("categories")]
    if scored_syms:
        with st.expander("📊 Strategy GPA Comparison", expanded=False):
            all_strats = STRAT_MGR.load_all()
            comp_rows  = []
            for p in positions:
                sym   = p["symbol"]
                raw_r = st.session_state.positions_gpa.get(sym)
                if not raw_r or not raw_r.get("categories"):
                    continue
                row = {"Symbol": sym}
                for sname, sdef in all_strats.items():
                    rw = reweight_result(raw_r, sdef)
                    arrow = ""
                    if rw.get("buy_signal"):  arrow = " 🟢"
                    elif rw.get("sell_signal"): arrow = " 🔴"
                    row[sname] = f"{rw['gpa']:.2f} {rw['grade']}{arrow}"
                comp_rows.append(row)
            if comp_rows:
                st.caption("GPA under each strategy  ·  🟢 = BUY signal  ·  🔴 = SELL signal")
                st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)
    elif positions:
        st.caption("Run **Refresh GPA** to enable Strategy Comparison.")

    if st.button("🔄 Refresh GPA for all positions", key="home_refresh_gpa"):
        with st.spinner("Scoring…"):
            engine = build_engine()
            for p in positions:
                r = score_symbol(p["symbol"], engine)
                if r:
                    st.session_state.positions_gpa[p["symbol"]] = r
        save_gpa_cache(st.session_state.positions_gpa)
        send_gpa_alert_email(st.session_state.positions_gpa)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB: POSITIONS
# ─────────────────────────────────────────────────────────────────────────────
def render_positions():
    st.header("Positions — Potential Sells")
    acct = ACCT_MGR.get(st.session_state.account_name)

    if not ACCT_MGR.is_configured(st.session_state.account_name):
        st.warning("API keys not configured.")
        return

    portfolio = fetch_alpaca_portfolio(
        st.session_state.account_name,
        acct.get("api_key",""), acct.get("secret_key",""), acct.get("base_url","")
    )
    if not portfolio.get("ok"):
        st.error(f"Alpaca error: {portfolio.get('error')}")
        return

    positions = portfolio["positions"]
    if not positions:
        st.info("No open positions.")
        return

    ca, cb = st.columns([2,1])
    with ca:
        if st.button("🔄 Score All Positions", key="pos_score_all"):
            with st.spinner("Scoring…"):
                engine = build_engine()
                for p in positions:
                    r = score_symbol(p["symbol"], engine)
                    if r:
                        st.session_state.positions_gpa[p["symbol"]] = r
            save_gpa_cache(st.session_state.positions_gpa)
            send_gpa_alert_email(st.session_state.positions_gpa)
            st.rerun()
    with cb:
        show_all = st.checkbox("Show all positions", value=True, key="pos_show_all")

    strat          = STRAT_MGR.get(st.session_state.strategy_name)
    sell_threshold = strat.get("thresholds", {}).get("max_gpa_to_sell", 2.5)
    auto_trade     = strat.get("auto_trade", False)

    st.divider()

    for p in positions:
        sym   = p["symbol"]
        gpa_r = st.session_state.positions_gpa.get(sym)
        sell_flag = gpa_r and gpa_r.get("sell_signal", False)

        if not show_all and not sell_flag:
            continue

        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.5, 1, 1, 1, 1.2, 1.2, 1.5])
        c1.markdown(f"**{sym}**")
        c2.markdown(f"{p['qty']:.0f} sh")
        c3.markdown(f"${p['cur_price']:.2f}")
        g_icon = "🟢" if p["gain_loss"] >= 0 else "🔴"
        c4.markdown(f"{g_icon} {p['gain_pct']:+.1f}%")

        if gpa_r:
            cls = gpa_color_class(gpa_r["gpa"])
            c5.markdown(f'<span class="{cls}">GPA {gpa_r["gpa"]:.2f}</span>',
                        unsafe_allow_html=True)
            c6.markdown(signal_chip(gpa_r), unsafe_allow_html=True)
        else:
            c5.markdown("GPA —")
            c6.markdown("—")

        if sell_flag:
            already = any(tr["symbol"] == sym and tr["action"] == "sell"
                          for tr in st.session_state.trade_queue)
            if not already:
                label = "🤖 Auto-queue SELL" if auto_trade else f"Queue SELL {sym}"
                if c7.button(label, key=f"sell_{sym}"):
                    st.session_state.trade_queue.append({
                        "symbol": sym, "action": "sell",
                        "qty": int(p["qty"]), "gpa": gpa_r["gpa"],
                        "reason": f"GPA {gpa_r['gpa']:.2f} ≤ {sell_threshold:.1f}",
                        "status": "auto" if auto_trade else "pending",
                    })
                    st.success(f"Added SELL {sym} to Trade Queue")

        if gpa_r:
            with st.expander(f"📊 GPA breakdown — {sym}"):
                render_gpa_detail(gpa_r)

        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TAB: OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────
def render_opportunities():
    st.header("Opportunities — Potential Buys")

    strat      = STRAT_MGR.get(st.session_state.strategy_name)
    thresholds = strat.get("thresholds", {})
    min_show   = thresholds.get("min_gpa_to_show", 3.0)
    auto_trade = strat.get("auto_trade", False)

    # Manual lookup
    st.subheader("Manual Symbol Lookup")
    sym_col, btn_col = st.columns([2,1])
    manual_sym = sym_col.text_input("Ticker symbol",
                                     placeholder="e.g. NVDA",
                                     key="manual_sym").upper().strip()
    btn_col.markdown("<br>", unsafe_allow_html=True)
    if btn_col.button("Get GPA", key="manual_gpa_btn") and manual_sym:
        with st.spinner(f"Scoring {manual_sym}…"):
            r = score_symbol(manual_sym)
        st.session_state.lookup_result = r
        if not r:
            st.error(f"Could not score {manual_sym}")

    if st.session_state.lookup_result:
        r   = st.session_state.lookup_result
        sym = r["symbol"]
        gpa = r["gpa"]
        cls = gpa_color_class(gpa)
        st.markdown(
            f'**{sym}** — GPA: <span class="{cls}">{gpa:.2f} ({r["grade"]})</span>  '
            f'{signal_chip(r)}',
            unsafe_allow_html=True
        )
        with st.expander("📊 Full GPA Breakdown", expanded=True):
            render_gpa_detail(r)

        if r.get("buy_signal") and ACCT_MGR.is_configured(st.session_state.account_name):
            q, b = st.columns([1,2])
            qty = q.number_input("Shares", 1, value=1, key="manual_qty")
            if b.button(f"Queue BUY {sym}", key="manual_buy"):
                st.session_state.trade_queue.append({
                    "symbol": sym, "action": "buy", "qty": int(qty),
                    "gpa": gpa, "reason": f"Manual lookup GPA {gpa:.2f}",
                    "status": "auto" if auto_trade else "pending",
                })
                st.success(f"Added BUY {qty}× {sym} to Trade Queue")

    st.divider()

    # News Scan
    st.subheader(f"News Scan  (showing GPA ≥ {min_show:.1f})")
    sc, si = st.columns([1,3])
    with sc:
        scan_btn = st.button("🔍 Scan Now", key="scan_btn")
    with si:
        if st.session_state.last_scan_time:
            st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

    if scan_btn:
        results = run_news_scan(min_gpa=min_show)
        st.session_state.scan_results   = results
        st.session_state.last_scan_time = datetime.now()
        st.rerun()

    all_results = st.session_state.scan_results
    if not all_results:
        st.info("Click **Scan Now** to discover buy candidates from today's market news.")
        return

    # Re-weight all stored results with current strategy (no re-scan needed)
    cur_strat    = STRAT_MGR.get(st.session_state.strategy_name)
    scan_strat   = st.session_state.get("scan_strategy_name", st.session_state.strategy_name)
    reweighted   = [reweight_result(r, cur_strat) for r in all_results]
    results      = [r for r in reweighted if r["gpa"] >= min_show]
    results.sort(key=lambda x: x["gpa"], reverse=True)

    # Show context when strategy differs from scan time
    if scan_strat != st.session_state.strategy_name:
        st.info(
            f"📌 Scan was run with **{scan_strat}**.  "
            f"GPAs below are re-weighted for **{st.session_state.strategy_name}** — no re-scan needed.",
            icon="🔄"
        )
        hidden = len(all_results) - len(results)
        if hidden:
            st.caption(f"{hidden} stock{'s' if hidden != 1 else ''} below GPA {min_show:.1f} threshold hidden.")
    st.success(f"{'Re-weighted: ' if scan_strat != st.session_state.strategy_name else ''}"
               f"{len(results)} stocks with GPA ≥ {min_show:.1f}")

    for r in results:
        sym = r["symbol"]
        gpa = r["gpa"]
        cls = gpa_color_class(gpa)
        top = ", ".join(r.get("top_drivers", []))

        c1, c2, c3, c4 = st.columns([1.5, 1.5, 2, 2])
        c1.markdown(f"**{sym}**")
        c2.markdown(f'<span class="{cls}">{gpa:.2f} ({r["grade"]})</span>',
                    unsafe_allow_html=True)
        c3.markdown(signal_chip(r), unsafe_allow_html=True)
        c4.markdown(f"Drivers: {top}")

        with st.expander(f"📊 {sym} GPA Details"):
            render_gpa_detail(r)

        if r.get("buy_signal") and ACCT_MGR.is_configured(st.session_state.account_name):
            q2, b2 = st.columns([1,2])
            qty2 = q2.number_input("Shares", 1, value=1, key=f"qty_{sym}")
            if b2.button(f"Queue BUY {sym}", key=f"buy_{sym}"):
                st.session_state.trade_queue.append({
                    "symbol": sym, "action": "buy", "qty": int(qty2),
                    "gpa": gpa, "reason": f"News scan GPA {gpa:.2f}",
                    "status": "auto" if auto_trade else "pending",
                })
                st.success(f"Queued BUY {qty2}× {sym}")

        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TAB: TRADE QUEUE
# ─────────────────────────────────────────────────────────────────────────────
def render_trade_queue():
    st.header("Trade Queue")
    queue      = st.session_state.trade_queue
    strat      = STRAT_MGR.get(st.session_state.strategy_name)
    auto_trade = strat.get("auto_trade", False)

    if not queue:
        st.info("No trades queued. Add them from Positions or Opportunities.")
        return

    ce, cc = st.columns(2)
    with ce:
        lbl = "🤖 Execute All Auto Trades" if auto_trade else "✅ Execute All Pending"
        if st.button(lbl, key="exec_all"):
            statuses = ("auto",) if auto_trade else ("pending",)
            for tr in [t for t in queue if t["status"] in statuses]:
                res = submit_order(st.session_state.account_name,
                                   tr["symbol"], tr["action"], tr["qty"])
                if res["ok"]:
                    tr["status"] = "executed"
                    send_trade_email(tr["symbol"], tr["action"], tr["qty"],
                                     st.session_state.account_name,
                                     order_id=res.get("order_id",""))
                else:
                    err = res.get("error","")
                    tr["status"] = f"failed: {err}"
                    send_trade_email(tr["symbol"], tr["action"], tr["qty"],
                                     st.session_state.account_name, error=err)
            st.rerun()
    with cc:
        if st.button("🗑 Clear Completed/Failed", key="clear_queue"):
            st.session_state.trade_queue = [
                t for t in queue if t["status"] in ("pending","auto")
            ]
            st.rerun()

    st.divider()

    pending = [t for t in queue if t["status"] in ("pending","auto")]
    done    = [t for t in queue if t["status"] not in ("pending","auto")]

    def render_trade_rows(trades, allow_exec):
        for i, tr in enumerate(trades):
            c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1, 0.8, 0.8, 2, 1.5])
            c1.markdown(f"**{tr['symbol']}**")
            c2.markdown("🟢 BUY" if tr["action"] == "buy" else "🔴 SELL")
            c3.markdown(f"{tr['qty']} sh")
            c4.markdown(f"GPA {tr['gpa']:.2f}")
            c5.markdown(tr.get("reason","—"))
            if allow_exec:
                if c6.button("Execute", key=f"exec_{i}_{tr['symbol']}"):
                    res = submit_order(st.session_state.account_name,
                                       tr["symbol"], tr["action"], tr["qty"])
                    if res["ok"]:
                        tr["status"] = "executed"
                        send_trade_email(tr["symbol"], tr["action"], tr["qty"],
                                         st.session_state.account_name,
                                         order_id=res.get("order_id",""))
                    else:
                        err = res.get("error","")
                        tr["status"] = f"failed: {err}"
                        send_trade_email(tr["symbol"], tr["action"], tr["qty"],
                                         st.session_state.account_name, error=err)
                    st.rerun()
            else:
                c6.markdown(f"`{tr['status']}`")

    if pending:
        st.subheader(f"Pending ({len(pending)})")
        render_trade_rows(pending, allow_exec=True)
        st.divider()

    if done:
        st.subheader("History")
        render_trade_rows(done, allow_exec=False)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: CHART
# ─────────────────────────────────────────────────────────────────────────────
def render_chart():
    st.header("Performance vs S&P 500")

    acct      = ACCT_MGR.get(st.session_state.account_name)
    portfolio = {}
    if ACCT_MGR.is_configured(st.session_state.account_name):
        portfolio = fetch_alpaca_portfolio(
            st.session_state.account_name,
            acct.get("api_key",""), acct.get("secret_key",""), acct.get("base_url","")
        )
    positions = portfolio.get("positions", [])

    # Period picker
    period = st.radio("Period", ["1D","1M","1Y","Start"],
                       index=2, horizontal=True, key="chart_period")

    # ── Portfolio chart (all positions combined) ───────────────────────────────
    st.subheader("📂 Full Portfolio vs S&P 500")

    if not positions:
        st.info("No positions found. Connect your Alpaca account to see portfolio performance.")
    else:
        with st.spinner("Loading portfolio chart…"):
            period_map = {"1D":("1d","5m"), "1M":("1mo","1d"), "1Y":("1y","1d"), "Start":("max","1wk")}
            yf_period, yf_interval = period_map.get(period, ("1y","1d"))

            # Build equal-weighted portfolio return by averaging all position returns
            all_series = []
            for p in positions:
                try:
                    sym_df = yf.download(p["symbol"], period=yf_period,
                                          interval=yf_interval, progress=False, auto_adjust=True)
                    if sym_df is not None and len(sym_df) >= 2:
                        if isinstance(sym_df.columns, pd.MultiIndex):
                            sym_df.columns = sym_df.columns.get_level_values(0)
                        # Weight by current market value
                        norm = sym_df["Close"] / sym_df["Close"].iloc[0]
                        all_series.append((norm, p["mkt_value"]))
                except Exception:
                    pass

            if all_series:
                total_value = sum(w for _, w in all_series)
                # Weighted average return across all positions
                portfolio_norm = sum(s * (w / total_value) for s, w in all_series)
                portfolio_norm = (portfolio_norm * 100).rename("My Portfolio")

                spy_df = yf.download("SPY", period=yf_period,
                                      interval=yf_interval, progress=False, auto_adjust=True)
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df.columns = spy_df.columns.get_level_values(0)
                spy_norm = (spy_df["Close"] / spy_df["Close"].iloc[0] * 100).rename("S&P 500")

                chart_df = pd.concat([portfolio_norm, spy_norm], axis=1).dropna()
                st.line_chart(chart_df, width="stretch")

                port_ret = chart_df["My Portfolio"].iloc[-1] - 100
                spy_ret  = chart_df["S&P 500"].iloc[-1] - 100
                alpha    = port_ret - spy_ret
                m1, m2, m3 = st.columns(3)
                m1.metric("Portfolio Return", f"{port_ret:+.2f}%")
                m2.metric("S&P 500 Return",   f"{spy_ret:+.2f}%")
                m3.metric("Alpha vs S&P",     f"{alpha:+.2f}%",
                           delta_color="normal" if alpha >= 0 else "inverse")

    st.divider()

    # ── Individual stock chart ─────────────────────────────────────────────────
    st.subheader("🔍 Individual Stock vs S&P 500")
    pos_symbols = [p["symbol"] for p in positions]
    cs, _ = st.columns([2,1])
    chart_sym = cs.text_input("Symbol", value=pos_symbols[0] if pos_symbols else "AAPL",
                               key="chart_sym").upper().strip()

    if chart_sym:
        with st.spinner(f"Loading {chart_sym}…"):
            df = get_comparison_df(chart_sym, period)

        if df is not None and not df.empty:
            st.line_chart(df, width="stretch")
            sym_ret = df[chart_sym].iloc[-1] - 100
            spy_ret = df["S&P 500"].iloc[-1]  - 100
            alpha   = sym_ret - spy_ret
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{chart_sym} Return", f"{sym_ret:+.2f}%")
            m2.metric("S&P 500 Return",      f"{spy_ret:+.2f}%")
            m3.metric("Alpha vs S&P",        f"{alpha:+.2f}%")

            if st.button(f"Get GPA for {chart_sym}", key="chart_gpa_btn"):
                with st.spinner("Scoring…"):
                    r = score_symbol(chart_sym)
                if r:
                    with st.expander("📊 GPA Breakdown", expanded=True):
                        render_gpa_detail(r)
        else:
            st.error(f"Could not load data for {chart_sym}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: CONFIGURE
# ─────────────────────────────────────────────────────────────────────────────
def render_configure():
    st.header("Configure")
    cfg_tab1, cfg_tab4, cfg_tab2, cfg_tab3 = st.tabs(
        ["🎯 Strategy & Weights", "📊 Strategy Tree (Excel)", "🔑 API Keys", "⚙️ Advanced"]
    )

    # ── Strategy & Weights ─────────────────────────────────────────────────────
    with cfg_tab1:
        st.subheader("Strategy Manager")
        all_strats  = STRAT_MGR.load_all()
        strat_names = list(all_strats.keys())
        active_name = st.session_state.strategy_name

        edit_name = st.selectbox(
            "Edit strategy", strat_names,
            index=strat_names.index(active_name) if active_name in strat_names else 0,
            key="cfg_strat_edit"
        )
        edit_strat = all_strats[edit_name]
        is_builtin = STRAT_MGR.is_builtin(edit_name)
        if is_builtin:
            st.caption("ℹ️ Built-in strategy — save under a new name to customize.")

        w  = edit_strat.get("weights", {})
        fs = w.get("fund_sub", {"valuation":0.33,"financial":0.33,"estimates":0.34})
        ts = w.get("tech_sub", {"trend":0.60,"oscillators":0.40})
        t  = edit_strat.get("thresholds", {})

        st.divider()
        st.markdown("**Main Category Weights**")
        mc1, mc2, mc3 = st.columns(3)
        w_sent = mc1.slider("Sentiment %",    0, 100, int(w.get("sentiment",0.20)*100),    key="w_sent")
        w_fund = mc2.slider("Fundamentals %", 0, 100, int(w.get("fundamentals",0.45)*100), key="w_fund")
        w_tech = mc3.slider("Technical %",    0, 100, int(w.get("technical",0.35)*100),    key="w_tech")
        total_main = w_sent + w_fund + w_tech
        st.caption(f"Total: {total_main}% (will auto-normalize to 100%)")

        st.markdown("**Fundamentals Sub-Weights**")
        fc1, fc2, fc3 = st.columns(3)
        fs_val = fc1.slider("Valuation %", 0, 100, int(fs.get("valuation",0.33)*100), key="fs_val")
        fs_fin = fc2.slider("Financial %", 0, 100, int(fs.get("financial",0.33)*100), key="fs_fin")
        fs_est = fc3.slider("Estimates %", 0, 100, int(fs.get("estimates",0.34)*100), key="fs_est")

        st.markdown("**Technical Sub-Weights**")
        tc1, tc2 = st.columns(2)
        ts_trnd = tc1.slider("Trend %",       0, 100, int(ts.get("trend",0.60)*100),       key="ts_trnd")
        ts_osc  = tc2.slider("Oscillators %", 0, 100, int(ts.get("oscillators",0.40)*100), key="ts_osc")

        st.divider()
        st.markdown("**Buy / Sell Thresholds**")
        th1, th2, th3, th4 = st.columns(4)
        t_buy   = th1.number_input("Min GPA → Buy",   0.0, 4.0, float(t.get("min_gpa_to_buy",3.5)),   0.1, key="t_buy")
        t_show  = th2.number_input("Min GPA → Show",  0.0, 4.0, float(t.get("min_gpa_to_show",3.0)),  0.1, key="t_show")
        t_sell  = th3.number_input("Max GPA → Sell",  0.0, 4.0, float(t.get("max_gpa_to_sell",2.5)),  0.1, key="t_sell")
        t_alert = th4.number_input("Min GPA → Alert", 0.0, 4.0, float(t.get("min_gpa_to_alert",3.5)), 0.1, key="t_alert")

        st.divider()
        auto_new = st.toggle(
            "🤖 Auto-Trade — execute without confirmation",
            value=edit_strat.get("auto_trade", False), key="cfg_auto"
        )
        if auto_new:
            st.warning("Orders will be submitted automatically when signals trigger.", icon="⚠️")

        desc_new = st.text_input("Description",
                                  value=edit_strat.get("description",""), key="cfg_desc")

        st.divider()
        nc, sc2, dc = st.columns([2,1,1])
        save_as = nc.text_input(
            "Save as name",
            value="" if is_builtin else edit_name,
            placeholder="New strategy name",
            key="cfg_save_name"
        )
        sc2.markdown("<br>", unsafe_allow_html=True)
        if sc2.button("💾 Save", key="cfg_save"):
            sname = save_as.strip()
            if not sname:
                st.error("Enter a strategy name.")
            elif STRAT_MGR.is_builtin(sname):
                st.error("Cannot overwrite built-in strategies.")
            else:
                tm = total_main or 100
                fm = (fs_val + fs_fin + fs_est) or 100
                om = (ts_trnd + ts_osc) or 100
                new_w = {
                    "sentiment":    w_sent / tm,
                    "fundamentals": w_fund / tm,
                    "technical":    w_tech / tm,
                    "fund_sub": {"valuation": fs_val/fm, "financial": fs_fin/fm, "estimates": fs_est/fm},
                    "tech_sub": {"trend": ts_trnd/om, "oscillators": ts_osc/om},
                }
                new_t = {
                    "min_gpa_to_buy":   t_buy,
                    "min_gpa_to_show":  t_show,
                    "max_gpa_to_sell":  t_sell,
                    "min_gpa_to_alert": t_alert,
                }
                STRAT_MGR.save(sname, new_w, new_t, auto_new, desc_new)
                st.session_state.strategy_name = sname
                st.success(f"Strategy '{sname}' saved!")
                st.rerun()

        dc.markdown("<br>", unsafe_allow_html=True)
        if not is_builtin and dc.button("🗑 Delete", key="cfg_del"):
            STRAT_MGR.delete(edit_name)
            st.session_state.strategy_name = "Default"
            st.success(f"Deleted '{edit_name}'")
            st.rerun()

    # ── Strategy Tree (Excel) ─────────────────────────────────────────────────
    with cfg_tab4:
        st.subheader("Strategy Tree — Edit in Excel")
        st.caption(
            "Edit the full 3-level strategy tree (categories → subcategories → "
            "components) in `Strategies.xlsx`. Upload the file here when you "
            "want the app to pick up your changes."
        )

        xlsx_path = BASE_DIR / "Strategies.xlsx"

        # File state
        if xlsx_path.exists():
            mtime = datetime.fromtimestamp(xlsx_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.success(f"📄 Strategies.xlsx present · last modified {mtime}")
        else:
            st.info("No Strategies.xlsx in your Stock folder yet — upload one below to get started.")

        col_up, col_dl = st.columns([3, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Replace Strategies.xlsx",
                type=["xlsx"],
                key="cfg_xlsx_upload",
                help="Saves to Stock\\Strategies.xlsx and re-imports all strategies."
            )
            if uploaded is not None:
                xlsx_path.write_bytes(uploaded.getvalue())
                st.success(f"Saved {uploaded.name} → Strategies.xlsx")
        with col_dl:
            st.markdown("<br>", unsafe_allow_html=True)
            if xlsx_path.exists():
                st.download_button(
                    "⬇️ Download current",
                    data=xlsx_path.read_bytes(),
                    file_name="Strategies.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="cfg_xlsx_dl",
                )

        if not xlsx_path.exists():
            st.stop()

        # Import button
        st.divider()
        ic1, ic2 = st.columns([1, 4])
        if ic1.button("📥 Import strategies", key="cfg_xlsx_import"):
            result = STRAT_MGR.import_from_xlsx(xlsx_path)
            if result["errors"]:
                for e in result["errors"]:
                    st.error(e)
            else:
                names = ", ".join(result["imported"]) or "(none)"
                st.success(
                    f"Imported {len(result['imported'])} strategies: {names}. "
                    + (f"Active: **{result['active']}**." if result['active'] else "No ACTIVE flag set.")
                )
                if result["active"]:
                    st.session_state.strategy_name = result["active"]
        ic2.caption(
            "Reads every strategy registered in the xlsx's `Strategies` sheet and "
            "saves them to `strategies.json`. The strategy marked ACTIVE becomes "
            "the current one."
        )

        # Preview of what's in the file
        st.divider()
        try:
            from modules import strategy_xlsx as _sx
            xlsx_data = _sx.load_xlsx(xlsx_path)
        except Exception as e:
            st.error(f"Couldn't parse Strategies.xlsx: {e}")
            st.stop()

        index = xlsx_data.get("_index", [])
        if not index:
            st.warning("The `Strategies` index sheet is empty. Add a row there for each strategy you want to use.")
            st.stop()

        # Show the strategies index
        st.markdown("**Strategies in this file**")
        import pandas as _pd
        idx_df = _pd.DataFrame([{
            "Active":       "✅" if e["active"] else "",
            "Name":         e["name"],
            "Sheet":        e["sheet"],
            "Description":  e["description"],
            "Last Modified": e["last_modified"],
        } for e in index])
        st.dataframe(idx_df, hide_index=True, use_container_width=True)

        # Pick one to preview
        names = [e["name"] for e in index if e["name"] in xlsx_data]
        if not names:
            st.warning("None of the strategies in the index could be parsed.")
            st.stop()
        active_idx = next((i for i, n in enumerate(names)
                           if xlsx_data[n].get("active")), 0)
        preview_name = st.selectbox("Preview tree for",
                                    names, index=active_idx, key="cfg_xlsx_preview")
        rich = xlsx_data[preview_name]
        tree = rich["tree"]
        known_keys = _sx.get_known_data_keys()

        # Tree rendering
        st.markdown(f"**{preview_name} — component tree**")
        rows = []
        for c in tree["categories"]:
            rows.append({
                "Level":         "Category",
                "Item":          c["name"],
                "Weight (norm)": f"{c['weight']*100:.1f}%",
                "% of GPA":      f"{c['weight']*100:.1f}%",
                "Higher=Better": "",
                "Thresholds (x/y/z)": "",
                "Data Key":      "",
                "Available?":    "",
            })
            for s in c["subcategories"]:
                rows.append({
                    "Level":         "  Subcategory",
                    "Item":          s["name"],
                    "Weight (norm)": f"{s['weight']*100:.1f}%",
                    "% of GPA":      f"{c['weight']*s['weight']*100:.1f}%",
                    "Higher=Better": "",
                    "Thresholds (x/y/z)": "",
                    "Data Key":      "",
                    "Available?":    "",
                })
                for p in s["components"]:
                    avail = "✅" if p.get("data_key") in known_keys else "⚠️ needs wiring"
                    thr_x, thr_y, thr_z = p.get("x"), p.get("y"), p.get("z")
                    thr = ("—" if None in (thr_x, thr_y, thr_z)
                           else f"{thr_x:g} / {thr_y:g} / {thr_z:g}")
                    rows.append({
                        "Level":         "    Component",
                        "Item":          p["name"],
                        "Weight (norm)": f"{p['weight']*100:.1f}%",
                        "% of GPA":      f"{p['pct_of_gpa']*100:.2f}%",
                        "Higher=Better": "Yes" if p.get("higher_better") else "No",
                        "Thresholds (x/y/z)": thr,
                        "Data Key":      p.get("data_key", ""),
                        "Available?":    avail,
                    })
        st.dataframe(_pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Data availability summary
        comps_all = [p for c in tree["categories"]
                     for s in c["subcategories"] for p in s["components"]]
        comps_ok  = [p for p in comps_all if p.get("data_key") in known_keys]
        comps_no  = [p for p in comps_all if p.get("data_key") not in known_keys]
        weight_no = sum(p["pct_of_gpa"] for p in comps_no)
        st.caption(
            f"✅ {len(comps_ok)} of {len(comps_all)} components have data wired up · "
            f"⚠️ {len(comps_no)} need wiring "
            f"(carrying {weight_no*100:.1f}% of GPA — these are skipped at scoring time)."
        )

        st.divider()
        st.markdown(
            "**Note:** the scorer currently honors the category and subcategory "
            "weights from this tree, but not yet the per-component weights, "
            "x/y/z thresholds, or Higher=Better flag. Those are stored and shown "
            "above so you can preview / edit in Excel; wiring them into the "
            "scorer is a follow-up step."
        )

    # ── API Keys ───────────────────────────────────────────────────────────────
    with cfg_tab2:
        st.subheader("Alpaca API Keys")
        st.info(
            "🔒  API keys are not accessible through this interface.\n\n"
            "To add or update keys, edit **config.json** directly in the Stock folder "
            "under the `accounts` section.",
            icon="🔐"
        )

        # Status-only display — no key values are rendered to the page
        for aname in ACCT_MGR.account_names():
            acct2   = ACCT_MGR.get(aname)
            icon2   = acct2.get("icon", "📘")
            label2  = acct2.get("label", aname)
            configured = ACCT_MGR.is_configured(aname)
            status  = "✅  Configured" if configured else "⚠️  Not configured — add keys to config.json"
            st.markdown(f"{icon2} **{label2}** — {status}")

        # ── Editable key fields preserved below for future use ──────────────
        # To re-enable: uncomment this block and deploy via COPY_NEW_APP.bat
        #
        # for aname in ACCT_MGR.account_names():
        #     acct2   = ACCT_MGR.get(aname)
        #     icon2   = acct2.get("icon","📘")
        #     label2  = acct2.get("label", aname)
        #     is_live = not acct2.get("paper", True)
        #     with st.expander(f"{icon2} {label2}",
        #                       expanded=not ACCT_MGR.is_configured(aname)):
        #         if is_live:
        #             st.error("⚠️ LIVE credentials — real money at risk!")
        #         new_k = st.text_input("API Key", value=acct2.get("api_key",""),
        #                                type="password", key=f"k_{aname}")
        #         new_s = st.text_input("Secret",  value=acct2.get("secret_key",""),
        #                                type="password", key=f"s_{aname}")
        #         if st.button(f"Save {aname} Keys", key=f"save_{aname}"):
        #             ACCT_MGR.save_keys(aname, new_k.strip(), new_s.strip())
        #             st.cache_data.clear()
        #             st.success(f"Keys saved for {aname}")
        #         st.caption("✅ Configured" if ACCT_MGR.is_configured(aname) else "⚠️ Not configured")

    # ── Advanced ───────────────────────────────────────────────────────────────
    with cfg_tab3:
        st.subheader("Advanced Settings")
        cfg_data = {}
        if CFG_FILE.exists():
            try:
                cfg_data = json.loads(CFG_FILE.read_text())
            except Exception:
                pass
        trading = cfg_data.get("trading", {})

        freq_opts = ["daily_noon","daily_open","every_2h","every_scan"]
        freq_lbls = {
            "daily_noon":  "Once Daily at Noon",
            "daily_open":  "Once Daily at Market Open",
            "every_2h":    "Every 2 Hours",
            "every_scan":  "Every Scan",
        }
        cur_freq = trading.get("news_check_frequency","daily_noon")
        new_freq = st.radio("News check frequency", freq_opts,
                             index=freq_opts.index(cur_freq) if cur_freq in freq_opts else 0,
                             format_func=lambda x: freq_lbls[x],
                             key="adv_freq")

        scan_int = st.number_input("Scan interval (minutes)", 1, 1440,
                                    int(trading.get("scan_interval_minutes", 60)),
                                    key="adv_scan_int")

        if st.button("Save Advanced Settings", key="adv_save"):
            cfg_data.setdefault("trading", {})
            cfg_data["trading"]["news_check_frequency"]  = new_freq
            cfg_data["trading"]["scan_interval_minutes"] = scan_int
            CFG_FILE.write_text(json.dumps(cfg_data, indent=2))
            st.success("Settings saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    tab_home, tab_pos, tab_opp, tab_queue, tab_chart, tab_cfg = st.tabs([
        "🏠 Home",
        "📋 Positions",
        "🔍 Opportunities",
        "🔁 Trade Queue",
        "📈 Chart",
        "⚙️ Configure",
    ])

    with tab_home:   render_home()
    with tab_pos:    render_positions()
    with tab_opp:    render_opportunities()
    with tab_queue:  render_trade_queue()
    with tab_chart:  render_chart()
    with tab_cfg:    render_configure()


if __name__ == "__main__":
    main()
