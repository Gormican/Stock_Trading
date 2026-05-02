"""
trading_agent.py — Main orchestrator for Taylor's Autonomous Trading Agent
Run: python trading_agent.py [--dry-run] [--force] [--backtest] [--once]
"""
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from modules.data_engine    import MarketDataEngine, SentimentEngine, TrendingScanner
from modules.gpa_scorer     import GPAEngine
from modules.risk_manager   import RiskManager
from modules.trade_executor import TradeExecutor
from modules.alert_engine   import AlertEngine
from modules.backtester     import Backtester

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = Path.home() / "taylor_agent.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("TradingAgent")


# ==============================================================================
# CONFIG LOADER
# ==============================================================================

def load_config(path: str = None) -> dict:
    cfg_path = Path(path or Path(__file__).parent / "config.json")
    if not cfg_path.exists():
        log.error(f"Config not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)
    # Remove comment keys
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


# ==============================================================================
# MARKET HOURS
# ==============================================================================

def is_market_open() -> bool:
    now_et  = datetime.now(timezone.utc) + timedelta(hours=-4)
    if now_et.weekday() >= 5:
        return False
    o = now_et.replace(hour=9, minute=25, second=0, microsecond=0)
    c = now_et.replace(hour=16, minute=5,  second=0, microsecond=0)
    return o <= now_et <= c


def sleep_until_market_open():
    now_et  = datetime.now(timezone.utc) + timedelta(hours=-4)
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et >= open_et:
        open_et += timedelta(days=1)
    while open_et.weekday() >= 5:
        open_et += timedelta(days=1)
    wait = (open_et - now_et).total_seconds()
    log.info(f"Market closed — sleeping {wait/3600:.1f}h until next open")
    time.sleep(wait)


# ==============================================================================
# SCAN INTERVAL  (based on frequency setting)
# ==============================================================================

FREQUENCY_INTERVALS = {
    "low":    86400,  # once daily
    "medium":  1800,  # every 30 min
    "high":     300,  # every 5 min
}


# ==============================================================================
# NEWS SCHEDULE HELPERS
# ==============================================================================

def should_refresh_news(cfg: dict, last_news_check: datetime | None) -> bool:
    """
    Returns True if it's time to pull fresh news/sentiment data.
    news_check_frequency options:
        daily_noon  — once per day at 12:00 PM ET  (default)
        daily_open  — once per day at 9:30 AM ET
        every_2h    — every 2 hours
        every_scan  — always (current behaviour)
    """
    freq = cfg.get("trading", {}).get("news_check_frequency", "daily_noon")
    now_et = datetime.now(timezone.utc) + timedelta(hours=-4)

    if freq == "every_scan" or last_news_check is None:
        return True

    if freq == "every_2h":
        return (now_et - last_news_check).total_seconds() >= 7200

    # For daily schedules, only refresh once per day at the target hour
    if last_news_check.date() == now_et.date():
        return False   # already refreshed today

    if freq == "daily_noon":
        return now_et.hour >= 12
    if freq == "daily_open":
        return now_et.hour >= 9 and now_et.minute >= 30

    return True  # fallback


# ==============================================================================
# CORE EVALUATION LOOP
# ==============================================================================

def run_evaluation(cfg: dict, executor: TradeExecutor, risk: RiskManager,
                   data: MarketDataEngine, sentiment: SentimentEngine,
                   gpa_engine: GPAEngine, alerts: AlertEngine,
                   scanner: TrendingScanner,
                   sentiment_cache: dict | None = None) -> list:
    """
    One full scan cycle:
    1. Get trending stocks
    2. Filter by beta / market cap
    3. Score each with GPA model
    4. Execute buys/sells
    5. Send alerts
    Returns list of GPA results.
    sentiment_cache: if provided, use cached sentiment data instead of fetching fresh.
    """
    if sentiment_cache is None:
        sentiment_cache = {}
    log.info("=" * 60)
    log.info(f"SCAN CYCLE — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    log.info("=" * 60)

    # Update portfolio state
    portfolio_value = executor.get_portfolio_value()
    risk.update_portfolio_value(portfolio_value)

    if risk.check_kill_switch():
        alerts._send(
            subject="KILL SWITCH TRIGGERED — Trading Halted",
            html_body="<h2>Daily loss limit reached. All trading halted for today.</h2>"
        )
        return []

    # Get SPY for relative strength baseline
    spy_df = data.get_bars("SPY", days=120)
    gpa_engine.set_spy_df(spy_df)

    # Get candidate universe (trending + seeds)
    log.info("Identifying trending stocks...")
    trending = scanner.get_top_trending(data, n=cfg["trading"]["scan_universe_size"])
    log.info(f"Candidates: {trending}")

    gpa_results   = []
    current_pos   = executor.get_positions()
    alerted_today = set()

    for symbol in trending:
        try:
            log.info(f"\n  Evaluating {symbol}...")
            ohlcv        = data.get_bars(symbol, days=90)
            fundamentals = data.get_fundamentals(symbol)
            # Use cached sentiment if available, otherwise fetch fresh
            if symbol in sentiment_cache:
                sent = sentiment_cache[symbol]
                log.info(f"    Using cached sentiment for {symbol}")
            else:
                sent = sentiment.get_combined_sentiment(symbol)
                sentiment_cache[symbol] = sent
            snapshot     = data.get_snapshot(symbol)

            # ── Beta and market cap filter ───────────────────────────────────
            beta   = fundamentals.get("beta")
            mcap   = fundamentals.get("market_cap", 0) or 0
            mcap_b = mcap / 1e9

            beta_ok = (beta is None or
                       cfg["filters"]["beta_min"] <= beta <= cfg["filters"]["beta_max"])
            cap_ok  = (cfg["filters"]["market_cap_min_billions"] <= mcap_b <=
                       cfg["filters"]["market_cap_max_billions"])

            if not beta_ok:
                log.info(f"    Skipped {symbol}: beta={beta:.2f} outside range")
                continue
            if not cap_ok:
                log.info(f"    Skipped {symbol}: mktcap=${mcap_b:.1f}B outside range")
                continue

            # ── GPA Score ────────────────────────────────────────────────────
            result = gpa_engine.score(symbol, ohlcv, fundamentals, sent)
            result["snapshot"]      = snapshot
            result["top_headlines"] = sent.get("top_headlines", [])
            gpa = result["gpa"]

            log.info(f"    GPA={gpa:.2f} ({result['grade']}) | "
                     f"price=${snapshot.get('price',0):.2f} | "
                     f"beta={beta}")
            gpa_results.append(result)

            price  = snapshot.get("price", 0)
            sector = fundamentals.get("sector", "Unknown")

            # ── SELL existing position? ──────────────────────────────────────
            if symbol in current_pos:
                pos = current_pos[symbol]
                entry_price = pos.get("avg_cost", price)
                pct_chg     = (price / entry_price - 1) * 100

                stop_hit    = pct_chg <= -cfg["risk"]["stop_loss_pct"]
                target_hit  = pct_chg >= cfg["risk"]["take_profit_pct"]
                gpa_sell    = gpa < cfg["trading"]["max_gpa_to_sell"]
                sent_neg    = sent.get("combined_score", 0) < -0.3

                if stop_hit or target_hit or (gpa_sell and sent_neg):
                    reason = ("stop_loss"   if stop_hit else
                              "take_profit" if target_hit else "gpa_sell")
                    shares = int(pos.get("qty", 0))
                    if shares > 0:
                        executor.sell_limit(symbol, shares, price,
                                            gpa=gpa, reason=reason)
                        risk.record_sell(symbol, price)
                        log.info(f"    SELL {symbol} — {reason} | "
                                 f"pnl=${(price-entry_price)*shares:+,.2f}")
                continue  # don't try to buy something we just sold

            # ── BUY new position? ────────────────────────────────────────────
            if result.get("buy_signal") and gpa >= cfg["trading"]["min_gpa_to_buy"]:
                ok, reason = risk.pre_trade_ok(symbol, price, sector)
                if ok and price > 0:
                    shares = risk.position_size_shares(price)
                    oid    = executor.buy_limit(symbol, shares, price,
                                               gpa=gpa, reason="gpa_buy")
                    if oid:
                        risk.record_buy(symbol, shares, price, sector)
                        executor.place_stop_loss(
                            symbol, shares, risk.stop_price(price)
                        )
                        log.info(f"    BUY {symbol} — GPA={gpa:.2f} | "
                                 f"{shares}x@${price:.2f}")

            # ── GPA Alert email ──────────────────────────────────────────────
            if (gpa >= cfg["trading"]["min_gpa_to_alert"] and
                    symbol not in alerted_today):
                alerts.send_gpa_alert(result, snapshot)
                alerted_today.add(symbol)

        except Exception as e:
            log.error(f"  Error evaluating {symbol}: {e}", exc_info=True)

    log.info(f"\nCycle complete — {len(gpa_results)} stocks scored")
    return gpa_results


# ==============================================================================
# BACKTEST MODE
# ==============================================================================

def run_backtest(cfg: dict):
    log.info("Starting backtest mode...")
    data       = MarketDataEngine(cfg)
    gpa_engine = GPAEngine(
        weights  = cfg["gpa_weights"],
        beta_min = cfg["filters"]["beta_min"],
        beta_max = cfg["filters"]["beta_max"],
    )
    bt = Backtester(cfg, gpa_engine, data)
    symbols = cfg["scan_seeds"]["tickers"][:15]  # use first 15 seeds
    metrics = bt.run(symbols, lookback_days=252)

    if metrics:
        result_file = Path(__file__).parent / "backtest_results.json"
        result_file.write_text(json.dumps(metrics, indent=2, default=str))
        log.info(f"Backtest results saved to {result_file}")

        log.info("\n" + "=" * 50)
        log.info("BACKTEST RESULTS")
        log.info("=" * 50)
        log.info(f"  Return:       {metrics['total_return_pct']:+.2f}%")
        log.info(f"  S&P 500:      {metrics['spy_return_pct']:+.2f}%")
        log.info(f"  Alpha:        {metrics['alpha_pct']:+.2f}%")
        log.info(f"  Sharpe:       {metrics['sharpe_ratio']:.3f}")
        log.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        log.info(f"  Win Rate:     {metrics['win_rate_pct']:.1f}%")
        log.info(f"  Trades:       {metrics['total_trades']}")
        log.info("=" * 50)
    return metrics


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Taylor's Autonomous Trading Agent")
    ap.add_argument("--dry-run",  action="store_true", help="Paper mode: no real orders")
    ap.add_argument("--force",    action="store_true", help="Run outside market hours")
    ap.add_argument("--backtest", action="store_true", help="Run historical backtest")
    ap.add_argument("--once",     action="store_true", help="Run one cycle then exit")
    ap.add_argument("--config",   type=str,            help="Path to config.json")
    args = ap.parse_args()

    log.info("=" * 60)
    log.info("  TAYLOR'S AUTONOMOUS TRADING AGENT")
    log.info(f"  Mode: {'DRY-RUN' if args.dry_run else 'PAPER TRADE'}")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    log.info("=" * 60)

    cfg = load_config(args.config)

    if args.backtest:
        run_backtest(cfg)
        return

    # Initialise all components
    data      = MarketDataEngine(cfg)
    sentiment = SentimentEngine(cfg)
    scanner   = TrendingScanner(cfg["scan_seeds"]["tickers"])
    gpa_engine = GPAEngine(
        weights  = cfg["gpa_weights"],
        beta_min = cfg["filters"]["beta_min"],
        beta_max = cfg["filters"]["beta_max"],
    )
    risk     = RiskManager(cfg)
    executor = TradeExecutor(cfg, dry_run=args.dry_run)
    alerts   = AlertEngine(cfg)

    risk.update_portfolio_value(executor.get_portfolio_value())

    interval = FREQUENCY_INTERVALS.get(cfg["trading"]["frequency"], 1800)

    # News caching — tracks last fetch time and cached results
    sentiment_cache: dict = {}
    last_news_check: datetime | None = None
    news_freq = cfg.get("trading", {}).get("news_check_frequency", "daily_noon")
    log.info(f"News check schedule: {news_freq}")

    # Main loop
    session_scores = []
    try:
        while True:
            if not is_market_open() and not args.force:
                if args.once:
                    log.info("Market closed. Use --force to override.")
                    break
                sleep_until_market_open()

            # Decide whether to refresh news or use cache
            if should_refresh_news(cfg, last_news_check):
                log.info("Refreshing news & sentiment data...")
                sentiment_cache = {}   # clear cache — fresh fetch in run_evaluation
                last_news_check = datetime.now(timezone.utc) + timedelta(hours=-4)
            else:
                log.info(f"Using cached news sentiment (next refresh: {news_freq})")

            scores = run_evaluation(
                cfg, executor, risk, data, sentiment,
                gpa_engine, alerts, scanner,
                sentiment_cache=sentiment_cache,
            )
            session_scores.extend(scores)

            if args.once:
                break

            log.info(f"Sleeping {interval//60} min until next scan...")
            time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Interrupted by user")

    # End-of-session: send daily summary
    log.info("Sending daily summary email...")
    trades = executor.get_trade_history()
    today_trades = [
        t for t in trades
        if t.get("timestamp", "")[:10] == datetime.now().strftime("%Y-%m-%d")
    ]
    alerts.send_daily_summary(
        all_scores      = session_scores,
        trades          = today_trades,
        portfolio_value = executor.get_portfolio_value(),
        daily_pnl       = risk.daily_pnl,
    )
    log.info("Agent session complete.")


if __name__ == "__main__":
    main()
