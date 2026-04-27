# 📈 Taylor's Autonomous Stock Trading Agent

A rules-based, data-driven stock trading agent that identifies, scores, and executes paper trades using a **GPA-style scoring model** — built on the Alpaca API with a beginner-friendly Streamlit dashboard.

---

## Features

- **GPA Scoring Engine (0.0 – 4.0)** — Every stock is scored across 5 pillars: Technical Momentum, Fundamentals, News Sentiment, Relative Strength, and Volatility Profile
- **Autonomous Paper Trading** — Connects to Alpaca's paper trading API to place simulated trades with no real money at risk
- **Email Alerts** — Instant Gmail alerts when any stock scores A (3.5+), with full GPA breakdown and trade rationale
- **Beautiful Dashboard** — Streamlit UI with sliders for risk tolerance, company size, beta range, trade frequency, and GPA thresholds
- **Risk Management** — Position sizing, stop-loss, take-profit, sector exposure limits, and a daily kill switch
- **Backtesting** — Run the strategy against 12 months of historical data and compare performance vs S&P 500
- **News & Social Sentiment** — Scans Yahoo Finance RSS, CNBC, MarketWatch, and Reddit (WSB, r/stocks)

---

## GPA Scoring Model

Based on a custom factor scoring model across 5 weighted pillars:

| Pillar | Weight | Factors |
|--------|--------|---------|
| Technical Momentum | 25% | RSI(14), MACD, SMA 20/50, Volume, Stochastics |
| Fundamentals | 25% | ROE, Earnings Growth, Debt/Equity, P/E, PEG |
| News & Sentiment | 25% | Yahoo Finance, CNBC, Reddit — scored by VADER NLP |
| Relative Strength | 15% | 1-month & 3-month return vs S&P 500 |
| Volatility Profile | 10% | Beta range fit, Average True Range |

| Grade | GPA | Action |
|-------|-----|--------|
| A+ | 3.7 – 4.0 | Strong Buy |
| A | 3.5 – 3.7 | Buy |
| B+ | 3.0 – 3.5 | Watch |
| B | 2.7 – 3.0 | Hold |
| C | 2.0 – 2.7 | Reduce |
| D | < 2.0 | Sell |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
Copy `config.example.json` to `config.json` and fill in:
- **Alpaca API key & secret** — free at [app.alpaca.markets](https://app.alpaca.markets) (use Paper Trading)
- **Gmail App Password** — for email alerts ([myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords))
- **Reddit credentials** — optional, free at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Or on Windows, double-click **START_AGENT.bat**

### 4. Run the trading agent
```bash
# Test mode (no real orders)
python trading_agent.py --dry-run --force

# Live paper trading
python trading_agent.py

# Run backtest
python trading_agent.py --backtest
```

---

## Project Structure

```
Stock_Trading/
├── app.py                  # Streamlit dashboard (UI)
├── trading_agent.py        # Main agent orchestrator
├── config.json             # API keys & settings (not committed)
├── requirements.txt        # Python dependencies
├── START_AGENT.bat         # Windows one-click launcher
├── setup_guide.md          # Detailed setup instructions
└── modules/
    ├── data_engine.py      # Market data + sentiment fetching
    ├── gpa_scorer.py       # GPA scoring engine
    ├── risk_manager.py     # Position sizing & risk controls
    ├── trade_executor.py   # Alpaca order placement
    ├── alert_engine.py     # Gmail email alerts
    └── backtester.py       # Historical backtesting
```

---

## Risk Management (Default Settings)

| Parameter | Default |
|-----------|---------|
| Position size | 3% of portfolio per trade |
| Stop loss | 7% below entry |
| Take profit | 14% above entry |
| Daily kill switch | -5% portfolio loss halts all trading |
| Beta range | 0.8 – 1.8 |
| Max open positions | 10 |
| Scan frequency | Every 30 minutes |

All parameters are adjustable via the dashboard sliders — no code editing required.

---

## Tech Stack

- **Python 3.11+**
- **Alpaca API** — paper trading execution + market data
- **yfinance** — historical price data & fundamentals
- **Streamlit** — web dashboard
- **VADER Sentiment** — NLP news scoring
- **PRAW** — Reddit API
- **feedparser** — RSS news feeds (Yahoo Finance, CNBC, MarketWatch)
- **pandas / numpy** — data processing

---

## Disclaimer

This project is for **educational and paper trading purposes only**. It does not constitute financial advice. Never trade with money you cannot afford to lose. Past performance in backtests does not guarantee future results.
