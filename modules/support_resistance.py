# support_resistance.py
# Auto-computes Support & Resistance GPA scores from price/volume data.
# Drop this in your Stock/modules/ folder and import from gpa_scorer.py.
#
# Scores every parameter on the 1-4 GPA scale matching your Codes sheet:
#   Value of Support   4=Solid/New  3=Mixed/Old  2=Thin Lower Low  1=New Low
#   Lack of Resistance 4=New High   3=Thin/High  2=Thick/Old       1=Thick/New
#   3 Mo Trend         4=>30%       3=Up<30%     2=Flat            1=Down
#   1 Yr Trend         4=>30%       3=Up<30%     2=Flat            1=Down
#   > 50d MA           4=Up+Above   3=Above/Near 2=Below/Up        1=Down+Below
#   > 200d MA          4=Up+Above   3=Above/Near 2=Below/Up        1=Down+Below
#   Volume Trend       4=HighVol+Up 3=NormVol+Up 2=NormVol+Dn     1=HighVol+Dn
#
# NEW metrics suggested (better than manual angle estimation):
#   Pivot Point Score  -- weekly S1/R1 relative to current price
#   ATR Support Buffer -- how many ATRs above nearest support level
#   Price Cluster      -- how many historical closes cluster near current price
#   Fib Retracement    -- where is current price in the fib ladder?

import numpy as np
import pandas as pd
import yfinance as yf
import time

# ── Constants ────────────────────────────────────────────────────────────────
FLAT_THRESHOLD   = 0.03   # <3% move = "flat" for trend scoring
CLUSTER_WINDOW   = 0.02   # price levels within 2% count as the same cluster
VOLUME_HIGH_MULT = 1.5    # volume > 1.5x 50-day avg = "High Volume"
NEAR_HIGH_PCT    = 0.05   # within 5% of 52W high = "thin" resistance
NEW_HIGH_PCT     = 0.02   # within 2% of 52W high = "no resistance" (score 4)
NEW_LOW_PCT      = 0.03   # within 3% of 52W low  = "new low" (score 1)

# ── Data fetch ────────────────────────────────────────────────────────────────
def _get_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Return daily OHLCV as a clean DataFrame. Returns empty DF on error."""
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()

# ── Individual scorers ────────────────────────────────────────────────────────

def score_value_of_support(df: pd.DataFrame) -> float:
    """
    Combines three signals:
      1. Distance above 52-week low  (new low = 1, solid cushion = 4)
      2. Price cluster density below (many historical closes near = solid support)
      3. Weekly pivot S1/S2 relative to current price
    """
    if df.empty or len(df) < 20:
        return 2.0

    close   = float(df["Close"].iloc[-1])
    low_52w = float(df["Low"].min())
    high_52w= float(df["High"].max())
    price_range = high_52w - low_52w if high_52w > low_52w else 1.0

    # ── Signal 1: proximity to 52W low ──────────────────────────────────────
    pct_above_low = (close - low_52w) / price_range  # 0 = at low, 1 = at high
    if pct_above_low < NEW_LOW_PCT:
        sig1 = 1.0    # new low — no support
    elif pct_above_low < 0.15:
        sig1 = 1.5
    elif pct_above_low < 0.35:
        sig1 = 2.5
    elif pct_above_low < 0.60:
        sig1 = 3.0
    else:
        sig1 = 4.0    # solid cushion above 52W low

    # ── Signal 2: historical price cluster below current price ───────────────
    closes = df["Close"].values
    support_zone_low  = close * (1 - CLUSTER_WINDOW * 3)
    support_zone_high = close * (1 - CLUSTER_WINDOW * 0.1)
    cluster_count = np.sum((closes >= support_zone_low) & (closes <= support_zone_high))
    # More prior closes near this price = stronger support floor
    if cluster_count >= 15:
        sig2 = 4.0
    elif cluster_count >= 8:
        sig2 = 3.0
    elif cluster_count >= 3:
        sig2 = 2.5
    elif cluster_count >= 1:
        sig2 = 2.0
    else:
        sig2 = 1.5

    # ── Signal 3: weekly pivot S1 ────────────────────────────────────────────
    if len(df) >= 5:
        week = df.iloc[-5:]
        pivot = (float(week["High"].max()) + float(week["Low"].min()) +
                 float(week["Close"].iloc[-1])) / 3
        s1 = 2 * pivot - float(week["High"].max())
        s2 = pivot - (float(week["High"].max()) - float(week["Low"].min()))
        if close > pivot:
            sig3 = 4.0    # above pivot = strong support
        elif close > s1:
            sig3 = 3.0
        elif close > s2:
            sig3 = 2.0
        else:
            sig3 = 1.0    # below S2 = support broken
    else:
        sig3 = 2.5

    # Weighted average (proximity to 52W low most important)
    score = sig1 * 0.40 + sig2 * 0.35 + sig3 * 0.25
    return round(min(4.0, max(1.0, score)), 2)


def score_lack_of_resistance(df: pd.DataFrame) -> float:
    """
    Combines three signals:
      1. Proximity to 52-week high   (new high = no overhead resistance = 4)
      2. Price cluster density above (many historical closes above = thick resistance)
      3. Weekly pivot R1 relative to current price
    """
    if df.empty or len(df) < 20:
        return 2.0

    close    = float(df["Close"].iloc[-1])
    high_52w = float(df["High"].max())
    low_52w  = float(df["Low"].min())
    price_range = high_52w - low_52w if high_52w > low_52w else 1.0

    # ── Signal 1: how close to 52W high ──────────────────────────────────────
    pct_below_high = (high_52w - close) / high_52w
    if pct_below_high <= NEW_HIGH_PCT:
        sig1 = 4.0   # at/near new high — no overhead resistance
    elif pct_below_high <= NEAR_HIGH_PCT:
        sig1 = 3.5
    elif pct_below_high <= 0.15:
        sig1 = 2.5
    elif pct_below_high <= 0.30:
        sig1 = 1.5
    else:
        sig1 = 1.0   # deep hole below 52W high — thick overhead resistance

    # ── Signal 2: price cluster density ABOVE current price ──────────────────
    closes = df["Close"].values
    resist_zone_low  = close * (1 + CLUSTER_WINDOW * 0.1)
    resist_zone_high = close * (1 + CLUSTER_WINDOW * 4)
    cluster_count = np.sum((closes >= resist_zone_low) & (closes <= resist_zone_high))
    # More prior closes just above = thicker resistance overhead
    if cluster_count >= 15:
        sig2 = 1.0   # thick resistance
    elif cluster_count >= 8:
        sig2 = 2.0
    elif cluster_count >= 3:
        sig2 = 3.0
    else:
        sig2 = 4.0   # thin/no resistance overhead

    # ── Signal 3: weekly pivot R1 ────────────────────────────────────────────
    if len(df) >= 5:
        week = df.iloc[-5:]
        pivot = (float(week["High"].max()) + float(week["Low"].min()) +
                 float(week["Close"].iloc[-1])) / 3
        r1 = 2 * pivot - float(week["Low"].min())
        r2 = pivot + (float(week["High"].max()) - float(week["Low"].min()))
        pct_to_r1 = (r1 - close) / close if r1 > close else 0
        if close >= r1:
            sig3 = 4.0   # already broken through R1
        elif pct_to_r1 < 0.02:
            sig3 = 3.5
        elif pct_to_r1 < 0.05:
            sig3 = 3.0
        elif pct_to_r1 < 0.10:
            sig3 = 2.0
        else:
            sig3 = 1.5
    else:
        sig3 = 2.5

    score = sig1 * 0.45 + sig2 * 0.35 + sig3 * 0.20
    return round(min(4.0, max(1.0, score)), 2)


def score_trend(df: pd.DataFrame, days: int) -> float:
    """
    Computes percentage price change over N days and slope of linear regression.
    Better than estimating a chart angle visually.
      4 = >30% gain over period  (steep uptrend)
      3 = 5-30% gain             (moderate uptrend)
      2 = -5% to +5%             (flat)
      1 = negative               (downtrend)
    days=63 → 3 Month,  days=252 → 1 Year
    """
    if df.empty or len(df) < days:
        days = len(df)
    if days < 5:
        return 2.0

    subset = df["Close"].iloc[-days:].values.astype(float)
    pct_change = (subset[-1] - subset[0]) / subset[0] * 100

    # Also check linear regression slope for consistency
    x = np.arange(len(subset))
    slope, _ = np.polyfit(x, subset, 1)
    slope_pct_per_day = slope / subset[0] * 100  # daily % move

    if pct_change > 30:
        base = 4.0
    elif pct_change > 5:
        base = 3.0
    elif pct_change > -FLAT_THRESHOLD * 100:
        base = 2.0
    else:
        base = 1.0

    # Bonus 0.25 if regression slope confirms the direction
    if slope_pct_per_day > 0.05 and base >= 3.0:
        base = min(4.0, base + 0.25)
    elif slope_pct_per_day < -0.05 and base <= 2.0:
        base = max(1.0, base - 0.25)

    return round(base, 2)


def score_moving_average(df: pd.DataFrame, ma_period: int) -> float:
    """
    Scores position relative to a moving average.
    Matches the Codes sheet: U+A=4, D+A or U(near)=3, U+B=2, D+B=1
    Also considers the slope of the MA itself (uptrend vs downtrend).
    """
    if df.empty or len(df) < ma_period + 5:
        return 2.0

    closes = df["Close"].values.astype(float)
    ma = np.mean(closes[-(ma_period):])
    ma_prev = np.mean(closes[-(ma_period + 5):-5])
    current = closes[-1]
    ma_trend = "U" if ma > ma_prev else "D"   # MA itself trending up or down

    pct_vs_ma = (current - ma) / ma * 100

    if ma_trend == "U" and pct_vs_ma > 0:
        score = 4.0   # Uptrend AND above MA — best case
    elif pct_vs_ma > 0:
        score = 3.0   # Above MA but MA in downtrend (or neutral)
    elif ma_trend == "U" and pct_vs_ma > -3:
        score = 3.0   # Near MA (within 3%) and MA uptrending
    elif ma_trend == "U":
        score = 2.0   # Below MA but MA still uptrending
    else:
        score = 1.0   # Below AND downtrending MA

    # Fine-tune by % distance
    if score == 4.0 and pct_vs_ma > 10:
        score = 3.5   # Too far extended above MA = potential reversion risk

    return round(score, 2)


def score_volume_trend(df: pd.DataFrame, days: int = 20) -> float:
    """
    High Volume + Uptrend = accumulation (bullish) = 4
    Normal Volume + Uptrend = 3
    Normal Volume + Downtrend = 2
    High Volume + Downtrend = distribution (very bearish) = 1
    """
    if df.empty or len(df) < days + 10:
        return 2.0

    close_now  = float(df["Close"].iloc[-1])
    close_prev = float(df["Close"].iloc[-days])
    avg_vol    = float(df["Volume"].iloc[-50:].mean()) if len(df) >= 50 else float(df["Volume"].mean())
    recent_vol = float(df["Volume"].iloc[-days:].mean())

    price_up  = close_now > close_prev * (1 + FLAT_THRESHOLD)
    price_down= close_now < close_prev * (1 - FLAT_THRESHOLD)
    high_vol  = recent_vol > avg_vol * VOLUME_HIGH_MULT

    if high_vol and price_up:
        return 4.0   # Institutional accumulation
    elif not high_vol and price_up:
        return 3.0
    elif not high_vol and price_down:
        return 2.0
    elif high_vol and price_down:
        return 1.0   # Institutional distribution — very bearish
    else:
        return 2.0   # Flat


# ── NEW METRICS — better than manual chart reading ────────────────────────────

def score_atr_support_buffer(df: pd.DataFrame) -> float:
    """
    NEW: Average True Range (ATR) buffer above nearest support.
    Measures how "thick" the support cushion is in terms of volatility units.
    If price is 3+ ATRs above the 52W low → very solid, Score 4.
    If price is < 0.5 ATR above 52W low → dangerously thin, Score 1.
    """
    if df.empty or len(df) < 20:
        return 2.0

    high = df["High"].values.astype(float)
    low  = df["Low"].values.astype(float)
    close= df["Close"].values.astype(float)

    # Compute ATR (14-day)
    n = min(14, len(df) - 1)
    tr = np.maximum(high[-n:] - low[-n:],
         np.maximum(np.abs(high[-n:] - close[-n-1:-1]),
                    np.abs(low[-n:] - close[-n-1:-1])))
    atr = np.mean(tr) if len(tr) > 0 else 1.0

    current   = close[-1]
    low_52w   = float(df["Low"].min())
    buffer_atrs = (current - low_52w) / atr if atr > 0 else 0

    if buffer_atrs >= 5.0:
        return 4.0
    elif buffer_atrs >= 3.0:
        return 3.0
    elif buffer_atrs >= 1.5:
        return 2.0
    else:
        return 1.0


def score_fibonacci_position(df: pd.DataFrame) -> float:
    """
    NEW: Fibonacci retracement level.
    Uses the most recent significant swing high and low over 52 weeks.
    Key fib levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    Price ABOVE 61.8% fib = strong (held major retracement) = Score 4
    Price between 38.2-61.8% = neutral/mixed = Score 3
    Price between 23.6-38.2% = weak = Score 2
    Price below 23.6% = Score 1
    """
    if df.empty or len(df) < 50:
        return 2.0

    swing_high = float(df["High"].max())
    swing_low  = float(df["Low"].min())
    current    = float(df["Close"].iloc[-1])
    span       = swing_high - swing_low

    if span == 0:
        return 2.0

    fib_position = (current - swing_low) / span  # 0 = at low, 1 = at high

    if fib_position >= 0.618:
        return 4.0   # Above golden ratio — strong uptrend holding
    elif fib_position >= 0.382:
        return 3.0   # In the middle fib zone
    elif fib_position >= 0.236:
        return 2.0   # Deep retracement
    else:
        return 1.0   # Below 23.6% — breakdown territory


def score_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> float:
    """
    NEW: Chaikin Money Flow (CMF) — measures buying vs selling pressure.
    CMF > +0.20 = strong buying (accumulation) = Score 4
    CMF > 0     = mild buying = Score 3
    CMF < 0     = mild selling = Score 2
    CMF < -0.20 = strong selling (distribution) = Score 1
    Better than just looking at volume alone.
    """
    if df.empty or len(df) < period:
        return 2.0

    sub = df.iloc[-period:].copy()
    high  = sub["High"].values.astype(float)
    low   = sub["Low"].values.astype(float)
    close = sub["Close"].values.astype(float)
    vol   = sub["Volume"].values.astype(float)

    hl_range = high - low
    hl_range[hl_range == 0] = 0.001  # avoid div/0
    mfm = ((close - low) - (high - close)) / hl_range  # Money Flow Multiplier
    mfv = mfm * vol                                      # Money Flow Volume
    cmf = np.sum(mfv) / np.sum(vol) if np.sum(vol) > 0 else 0

    if cmf >= 0.20:
        return 4.0
    elif cmf >= 0.05:
        return 3.5
    elif cmf >= 0:
        return 3.0
    elif cmf >= -0.10:
        return 2.0
    elif cmf >= -0.20:
        return 1.5
    else:
        return 1.0


# ── Main entry point ─────────────────────────────────────────────────────────

def compute_support_resistance_scores(ticker: str) -> dict:
    """
    Master function. Fetches 1 year of data and returns all S/R scores
    as a dict ready to plug into the Technical category of gpa_scorer.py.

    Returns:
    {
        "value_of_support":    float,   # replaces manual Codes sheet scoring
        "lack_of_resistance":  float,
        "trend_3mo":           float,
        "trend_1yr":           float,
        "ma_50":               float,
        "ma_200":              float,
        "volume_trend":        float,
        "atr_support_buffer":  float,   # NEW — volatility-adjusted cushion
        "fibonacci_position":  float,   # NEW — fib ladder position
        "chaikin_money_flow":  float,   # NEW — buying vs selling pressure
        "trend_gpa":           float,   # average of core 7
        "enhanced_trend_gpa":  float,   # average including 3 new metrics
        "data_quality":        str,     # "good" | "partial" | "no_data"
    }
    """
    time.sleep(0.1)  # rate limit courtesy

    # Fetch data (need 200+ days for MA200)
    df_1y = _get_data(ticker, "1y")
    df_2y = _get_data(ticker, "2y")   # for MA200 we need >200 trading days

    if df_1y.empty:
        empty = {k: 2.0 for k in [
            "value_of_support", "lack_of_resistance",
            "trend_3mo", "trend_1yr", "ma_50", "ma_200", "volume_trend",
            "atr_support_buffer", "fibonacci_position", "chaikin_money_flow"
        ]}
        empty["trend_gpa"] = 2.0
        empty["enhanced_trend_gpa"] = 2.0
        empty["data_quality"] = "no_data"
        return empty

    df = df_2y if not df_2y.empty else df_1y
    quality = "good" if len(df) >= 200 else "partial"

    scores = {
        "value_of_support":   score_value_of_support(df),
        "lack_of_resistance": score_lack_of_resistance(df),
        "trend_3mo":          score_trend(df, 63),
        "trend_1yr":          score_trend(df, 252),
        "ma_50":              score_moving_average(df, 50),
        "ma_200":             score_moving_average(df, 200),
        "volume_trend":       score_volume_trend(df),
        "atr_support_buffer": score_atr_support_buffer(df),
        "fibonacci_position": score_fibonacci_position(df),
        "chaikin_money_flow": score_chaikin_money_flow(df),
    }

    core_keys = ["value_of_support", "lack_of_resistance",
                 "trend_3mo", "trend_1yr", "ma_50", "ma_200", "volume_trend"]
    scores["trend_gpa"] = round(
        sum(scores[k] for k in core_keys) / len(core_keys), 3)

    all_keys = core_keys + ["atr_support_buffer", "fibonacci_position", "chaikin_money_flow"]
    scores["enhanced_trend_gpa"] = round(
        sum(scores[k] for k in all_keys) / len(all_keys), 3)

    scores["data_quality"] = quality
    return scores


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tickers = ["AAPL", "NVDA", "F", "SPY"]
    print(f"\n{'Ticker':<8} {'Support':>8} {'Resist':>8} {'3Mo':>6} {'1Yr':>6} "
          f"{'MA50':>6} {'MA200':>6} {'Vol':>6} {'ATR':>6} {'Fib':>6} {'CMF':>6} "
          f"{'TrendGPA':>9} {'EnhGPA':>8}")
    print("-" * 110)
    for t in test_tickers:
        s = compute_support_resistance_scores(t)
        print(f"{t:<8} {s['value_of_support']:>8.2f} {s['lack_of_resistance']:>8.2f} "
              f"{s['trend_3mo']:>6.2f} {s['trend_1yr']:>6.2f} "
              f"{s['ma_50']:>6.2f} {s['ma_200']:>6.2f} {s['volume_trend']:>6.2f} "
              f"{s['atr_support_buffer']:>6.2f} {s['fibonacci_position']:>6.2f} "
              f"{s['chaikin_money_flow']:>6.2f} "
              f"{s['trend_gpa']:>9.3f} {s['enhanced_trend_gpa']:>8.3f}")
