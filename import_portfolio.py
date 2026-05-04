# import_portfolio.py
# Reads Transfers.csv, scales all positions to fit within $95K,
# and submits market orders to the Alpaca $100K paper account.
# Run from C:/Users/sgorm/Stock:  python import_portfolio.py

import json, csv, time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV_FILE   = Path(__file__).parent / "Transfers.csv"
CFG_FILE   = Path(__file__).parent / "config.json"
ACCOUNT    = "$100K"
BUDGET     = 95_000   # leave $5K buffer for fees / rounding

# Tickers Alpaca cannot trade as equities
SKIP = {
    "ARKVX",   # mutual fund
    "MINGX",   # mutual fund
    "PDGIX",   # mutual fund
    "BTC",     # crypto — needs separate crypto API
}

# ── Load API keys ─────────────────────────────────────────────────────────────
cfg      = json.loads(CFG_FILE.read_text())
api_key  = cfg["accounts"][ACCOUNT]["api_key"]
secret   = cfg["accounts"][ACCOUNT]["secret_key"]

if not api_key or not secret:
    print(f"ERROR: No API keys found for {ACCOUNT} in config.json")
    input("Press Enter to exit.")
    raise SystemExit(1)

# ── Parse CSV ─────────────────────────────────────────────────────────────────
raw = []
with open(CSV_FILE, newline="") as f:
    for row in csv.DictReader(f):
        ticker = row["Ticker"].strip().upper()
        if ticker in SKIP:
            print(f"  SKIP  {ticker:8s}  (not tradeable on Alpaca)")
            continue
        shares = float(row["Shares"].replace(",", "").replace("*", "").strip())
        raw.append({"ticker": ticker, "shares": shares})

print(f"\nLoaded {len(raw)} tradeable positions from CSV\n")

# ── Fetch current prices via yfinance ─────────────────────────────────────────
import yfinance as yf

print("Fetching current prices…")
prices = {}
for p in raw:
    t = p["ticker"]
    try:
        px = yf.Ticker(t).fast_info.last_price
        if px and px > 0:
            prices[t] = px
            print(f"  {t:8s}  ${px:.2f}")
        else:
            print(f"  {t:8s}  no price — will skip")
    except Exception as e:
        print(f"  {t:8s}  error ({e}) — will skip")
    time.sleep(0.1)

# ── Calculate scale factor ────────────────────────────────────────────────────
total_cost = sum(p["shares"] * prices[p["ticker"]]
                 for p in raw if p["ticker"] in prices)

scale = min(1.0, BUDGET / total_cost)

print(f"\nTotal at current prices : ${total_cost:>12,.0f}")
print(f"Budget                  : ${BUDGET:>12,.0f}")
print(f"Scale factor            :  {scale:.6f}  ({scale*100:.2f}%)")

# ── Connect to Alpaca ─────────────────────────────────────────────────────────
from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums    import OrderSide, TimeInForce

client  = TradingClient(api_key, secret, paper=True)
account = client.get_account()
buying_power = float(account.buying_power)
print(f"Account buying power    : ${buying_power:>12,.2f}")

if buying_power < 1000:
    print("\nWARNING: Very low buying power — orders will likely fail.")

print("\n" + "="*60)
confirm = input(f"Submit {len([p for p in raw if p['ticker'] in prices])} scaled orders? (yes/no): ").strip().lower()
if confirm != "yes":
    print("Cancelled.")
    raise SystemExit(0)

# ── Submit orders ─────────────────────────────────────────────────────────────
print()
ok, failed, skipped = [], [], []

for p in raw:
    t = p["ticker"]
    if t not in prices:
        skipped.append(t)
        continue

    scaled_qty = round(p["shares"] * scale, 4)
    if scaled_qty < 0.001:
        print(f"  SKIP  {t:8s}  qty {scaled_qty:.4f} too small")
        skipped.append(t)
        continue

    try:
        order = client.submit_order(MarketOrderRequest(
            symbol        = t,
            qty           = scaled_qty,
            side          = OrderSide.BUY,
            time_in_force = TimeInForce.DAY,
        ))
        print(f"  ✓  {t:8s}  {scaled_qty:>10.4f} shares  (order {str(order.id)[:8]}…)")
        ok.append(t)
    except Exception as e:
        print(f"  ✗  {t:8s}  FAILED — {e}")
        failed.append({"ticker": t, "error": str(e)})

    time.sleep(0.3)   # stay inside rate limits

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  Submitted : {len(ok)}")
print(f"  Failed    : {len(failed)}")
print(f"  Skipped   : {len(skipped)}")
if failed:
    print("\nFailed orders:")
    for f in failed:
        print(f"  {f['ticker']:8s}  {f['error']}")

input("\nDone. Press Enter to close.")
