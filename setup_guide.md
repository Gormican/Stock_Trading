# Taylor's Autonomous Trading Agent — Setup Guide

## 1. Install Dependencies (5 minutes)
```bash
cd trading_system
pip install -r requirements.txt
```

## 2. Configure API Keys (edit config.json)

### Alpaca (Paper Trading — Required)
1. Go to https://app.alpaca.markets → sign up free
2. Switch to **Paper Trading** (toggle at top-left)
3. Click **Your API Keys** → Generate New Key
4. Paste into config.json:
   - `alpaca.api_key`
   - `alpaca.secret_key`

### Gmail Alerts (Optional but Recommended)
1. Enable 2-Factor Authentication on your Gmail account
2. Visit https://myaccount.google.com/apppasswords
3. Generate an App Password for "Mail"
4. Paste into config.json:
   - `alerts.smtp_user` = your Gmail address
   - `alerts.smtp_password` = the 16-char App Password

### Reddit Sentiment (Optional — free)
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" → select "script"
3. Paste into config.json:
   - `data_sources.reddit_client_id`
   - `data_sources.reddit_client_secret`

### NewsAPI (Optional — free 100 req/day)
1. Register at https://newsapi.org
2. Copy API key to `data_sources.news_api_key`

## 3. Launch the Dashboard
```bash
streamlit run dashboard.py
```
Opens at http://localhost:8501 — adjust sliders, save settings.

## 4. Test with Dry Run
```bash
python trading_agent.py --dry-run --force
```
Runs full scan cycle with no real orders. Check output and logs.

## 5. Run Backtest
```bash
python trading_agent.py --backtest
```
Results appear in dashboard under the Backtest tab.

## 6. Start Live Paper Trading
```bash
python trading_agent.py
```
Runs continuously during market hours (9:30 AM – 4:00 PM ET).
Sends email alerts to sgormican@gmail.com when GPA ≥ 3.5.

## Risk Summary (Medium Settings)
| Parameter          | Default Value |
|--------------------|---------------|
| Position size      | 3% of portfolio |
| Stop loss          | 7% below entry |
| Take profit        | 14% above entry |
| Kill switch        | -5% daily loss |
| Beta range         | 0.8 – 1.8 |
| Min GPA to buy     | 3.5 / 4.0 |
| Max concurrent     | 10 positions |
| Scan frequency     | Every 30 min |

## Files
| File | Purpose |
|------|---------|
| `config.json` | All settings (edited by dashboard or directly) |
| `trading_agent.py` | Main agent — run this |
| `dashboard.py` | Streamlit UI — run separately |
| `modules/data_engine.py` | Market data + sentiment |
| `modules/gpa_scorer.py` | GPA scoring engine |
| `modules/risk_manager.py` | Risk controls |
| `modules/trade_executor.py` | Alpaca order placement |
| `modules/alert_engine.py` | Email alerts |
| `modules/backtester.py` | Historical backtest |
| `~/taylor_trade_log.json` | All trade history |
| `~/taylor_agent.log` | Full agent log |
