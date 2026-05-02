# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bat
# Normal launch (use this — it kills any existing port 8501 process first)
START_AGENT.bat

# Direct command if bat file is unavailable
cd C:\Users\sgorm\Stock
python -m streamlit run app.py --server.port 8501
```

The app runs at `http://localhost:8501`. Phone access is via Tailscale at `http://100.67.249.60:8501`.

## Architecture

This is a single-process Streamlit app. There is no backend server, job queue, or database — everything runs in the Streamlit session. The entry point is `app.py`; all business logic lives in `modules/`.

### GPA Scoring Model (3-category, 0.0–4.0 scale)

The scoring model in `modules/gpa_scorer.py` is a **3-category weighted average**:

| Category | Default Weight | Sub-categories |
|---|---|---|
| Sentiment | 20% | Combined VADER/news score + velocity + Reddit |
| Fundamentals | 45% | Valuation (33%), Financial (33%), Estimates (34%) |
| Technical | 35% | Trend (60%), Oscillators (40%) |

The `GPAEngine` class is the entry point. Call `engine.score(symbol, ohlcv_df, fundamentals, sentiment)` — it returns a full report dict with `gpa`, `grade`, `buy_signal`, `sell_signal`, `categories`, and `top_drivers`.

**Critical:** this is a 3-category model. Do not refactor it into a 5-pillar or any other architecture. The weights described above are defaults — strategies can override them.

### Strategy System

`modules/strategy_manager.py` manages named weight presets (Default, Growth, Income, Momentum, plus any user-created). Built-in strategies cannot be deleted or overwritten. User strategies are saved to `strategies.json`. All weights are auto-normalized to sum to 1.0 via `StrategyManager.normalize()`.

The active strategy is stored in `st.session_state.strategy_name` and passed to `GPAEngine` at scoring time.

### Account System

`modules/account_manager.py` manages three Alpaca accounts: Taylor (paper), $100K (paper), Live (real money). API keys are stored in `config.json` under the `accounts` key. The last-selected account persists to `~/taylor_last_account.json`.

### Data Flow

```
app.py → score_symbol()
    → DataEngine (modules/data_engine.py)
        → MarketDataEngine   — yfinance fundamentals, OHLCV bars
        → SentimentEngine    — RSS/news VADER sentiment
    → GPAEngine.score()
        → score_sentiment(), score_fundamentals_full(), score_technical_full()
```

`DataEngine` is a wrapper class at the bottom of `data_engine.py` that unifies `MarketDataEngine` and `SentimentEngine`. Import `DataEngine`, not the underlying classes.

yfinance sometimes returns MultiIndex columns on newer versions. Always apply this fix after downloading OHLCV data:
```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
```

### Persistent State

| File | Purpose | Notes |
|---|---|---|
| `config.json` | API keys, SMTP, risk/filter settings | **Never overwrite** — user's live keys are here |
| `gpa_cache.json` | Last GPA scores for held positions | Written by `save_gpa_cache()`, read on startup |
| `strategies.json` | User-created strategies | Only user strategies; built-ins are hardcoded |
| `~/taylor_last_account.json` | Last selected account | Written to home dir, not Stock folder |

`gpa_cache.json` is written with a `_to_native()` helper that recursively converts numpy types to plain Python before JSON serialization. Any code path that saves GPA results must use this helper — numpy floats/ints silently cause `json.dumps()` to fail.

### Strategy Re-weighting (no re-scan)

`reweight_result(r, strategy)` in `app.py` takes any cached GPA result dict and re-applies new strategy weights using the raw sub-scores stored in `r["categories"]`. This is pure arithmetic — no API calls, no yfinance downloads.

This powers two UX features:
- **Opportunities tab**: Scan results are stored with all raw sub-scores regardless of GPA threshold. Switching strategy re-weights the existing stocks instantly. A banner shows when the displayed strategy differs from the scan-time strategy.
- **Home tab**: The main positions table always reflects the currently selected strategy. A "Strategy GPA Comparison" expander shows GPA under every strategy side-by-side.

`st.session_state.scan_strategy_name` tracks which strategy was active when the last scan ran. `run_news_scan()` stores all scored stocks that have `categories` data (not filtered by min_gpa at scan time — filtering happens at display time).

### Email Alerts

`send_email(subject, body)` in `app.py` reads SMTP config from `config.json → alerts`. It uses Gmail with an App Password (not the account password). Two alert functions sit on top of it:
- `send_trade_email()` — called after every trade execution (success or failure)
- `send_gpa_alert_email()` — called after GPA refresh, only fires if BUY or SELL signals exist

### Deployment

`COPY_NEW_APP.bat` copies updated files from the Claude Cowork outputs folder to `C:\Users\sgorm\Stock`. It intentionally **skips `config.json`** to preserve API keys. Run this bat file after any Cowork session that modifies app files, then relaunch via `START_AGENT.bat`.

`SETUP_AUTOSTART.bat` registers `START_AGENT.bat` as a Windows login task via `schtasks`.
