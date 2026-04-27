"""
gpa_scorer.py — GPA Scoring Engine (0.0 – 4.0 scale)
Based on Stock Eval.xls factor model (S. Gormican)

Pillars:
  1. Technical Momentum  (RSI, MACD, MAs, Volume, Stochastics)
  2. Fundamentals        (ROE, Earnings Growth, D/E, P/E, PEG, Estimates)
  3. Sentiment           (News + Social combined score)
  4. Relative Strength   (vs sector and market)
  5. Volatility Profile  (Beta range, ATR)
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger("GPAScorer")

DEFAULT_WEIGHTS = {
    "technical_momentum": 0.25,
    "fundamentals":       0.25,
    "sentiment":          0.25,
    "relative_strength":  0.15,
    "volatility_profile": 0.10,
}


# ==============================================================================
# TECHNICAL PILLAR  (0–4 points)
# ==============================================================================

def score_technical(df: pd.DataFrame) -> dict:
    """
    Scores technical momentum 0–4 using:
      RSI(14), MACD, SMA20/50/200, Volume, Stochastics
    Mirrors the 4-point rating in Stock Eval.xls.
    """
    if df is None or len(df) < 20:
        return {"score": 2.0, "detail": {"error": "insufficient data"}}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    price  = close.iloc[-1]

    components = {}

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    # From xls: 4=30-50 (momentum building), 3=50-65, 2=65-75, 1=>75 or <30
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]
    if 30 <= rsi <= 50:
        rsi_score = 4.0
    elif 50 < rsi <= 65:
        rsi_score = 3.0
    elif (20 <= rsi < 30) or (65 < rsi <= 75):
        rsi_score = 2.0
    else:
        rsi_score = 1.0
    components["rsi"] = {"value": round(rsi, 1), "score": rsi_score}

    # ── MACD ─────────────────────────────────────────────────────────────────
    # 4=bullish crossover (MACD just crossed above signal)
    # 3=MACD positive and above signal  2=MACD positive but below  1=negative
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val    = macd.iloc[-1]
        signal_val  = signal.iloc[-1]
        macd_prev   = macd.iloc[-2]
        signal_prev = signal.iloc[-2]
        crossover = (macd_prev <= signal_prev) and (macd_val > signal_val)
        if crossover and macd_val > 0:
            macd_score = 4.0
        elif macd_val > 0 and macd_val > signal_val:
            macd_score = 3.0
        elif macd_val > 0:
            macd_score = 2.0
        else:
            macd_score = 1.0
        components["macd"] = {"value": round(macd_val, 3), "crossover": crossover, "score": macd_score}
    else:
        macd_score = 2.0
        components["macd"] = {"score": macd_score}

    # ── Moving Averages ───────────────────────────────────────────────────────
    # From xls: 4=>30 degree uptrend, 3=up<30, 2=flat, 1=downtrend
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    # Measure trend angle: slope of SMA20 over last 5 days
    sma20_series = close.rolling(20).mean().dropna()
    if len(sma20_series) >= 5:
        slope_pct = (sma20_series.iloc[-1] / sma20_series.iloc[-5] - 1) * 100
        above_50  = price > sma50
        above_20  = price > sma20
        if slope_pct > 2.0 and above_20 and above_50:
            ma_score = 4.0
        elif slope_pct > 0.5 and above_20:
            ma_score = 3.0
        elif slope_pct > -0.5:
            ma_score = 2.0
        else:
            ma_score = 1.0
    else:
        slope_pct, ma_score = 0, 2.0
    components["moving_averages"] = {
        "sma20": round(sma20, 2), "sma50": round(sma50, 2),
        "slope_5d_pct": round(slope_pct, 2), "score": ma_score
    }

    # ── Volume Trend ─────────────────────────────────────────────────────────
    # From xls: 4=High Vol Up, 3=Normal Vol Up, 2=Normal Vol Down, 1=High Vol Down
    avg_vol   = volume.rolling(20).mean().iloc[-1]
    today_vol = volume.iloc[-1]
    vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1.0
    price_up  = close.iloc[-1] > close.iloc[-2]
    high_vol  = vol_ratio >= 1.5
    if high_vol and price_up:
        vol_score = 4.0
    elif (not high_vol) and price_up:
        vol_score = 3.0
    elif (not high_vol) and (not price_up):
        vol_score = 2.0
    else:  # high volume down
        vol_score = 1.0
    components["volume"] = {"ratio": round(vol_ratio, 2), "price_up": price_up, "score": vol_score}

    # ── Stochastics (14,3) ────────────────────────────────────────────────────
    # 4=oversold (<20) turning up, 3=20-50 rising, 2=50-80, 1=overbought (>80)
    if len(high) >= 14:
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k_pct  = ((close - low14) / (high14 - low14).replace(0, np.nan) * 100).iloc[-1]
        k_prev = ((close - low14) / (high14 - low14).replace(0, np.nan) * 100).iloc[-2]
        rising = k_pct > k_prev
        if k_pct < 20 and rising:
            stoch_score = 4.0
        elif k_pct < 50 and rising:
            stoch_score = 3.0
        elif k_pct < 80:
            stoch_score = 2.0
        else:
            stoch_score = 1.0
        components["stochastics"] = {"k_pct": round(k_pct, 1), "rising": rising, "score": stoch_score}
    else:
        stoch_score = 2.0
        components["stochastics"] = {"score": stoch_score}

    # ── Aggregate (equal weight across 5 sub-components → 0–4) ──────────────
    sub_scores = [rsi_score, macd_score, ma_score, vol_score, stoch_score]
    final      = round(np.mean(sub_scores), 3)
    return {"score": final, "detail": components}


# ==============================================================================
# FUNDAMENTALS PILLAR  (0–4 points)
# ==============================================================================

def score_fundamentals(fundamentals: dict) -> dict:
    """
    Scores financials 0–4 using Stock Eval.xls criteria:
      ROE, Earnings Growth, Debt/Equity, P/E, PEG, Estimates, Dividend
    """
    components = {}
    scores     = []

    def safe(val, multiplier=1.0):
        try:
            return float(val) * multiplier if val is not None else None
        except Exception:
            return None

    # ── ROE (>18% = 4pts) ─────────────────────────────────────────────────────
    roe = safe(fundamentals.get("roe"), 100)  # convert to %
    if roe is not None:
        if roe >= 18:   roe_s = 4.0
        elif roe >= 12: roe_s = 3.0
        elif roe >= 6:  roe_s = 2.0
        else:           roe_s = 1.0
        scores.append(roe_s)
        components["roe"] = {"value": round(roe, 1), "score": roe_s}

    # ── Earnings Growth YoY (>25% = 4pts) ────────────────────────────────────
    eg = safe(fundamentals.get("earnings_growth_yoy"), 100)
    if eg is not None:
        if eg >= 25:    eg_s = 4.0
        elif eg >= 15:  eg_s = 3.0
        elif eg >= 5:   eg_s = 2.0
        else:           eg_s = 1.0
        scores.append(eg_s)
        components["earnings_growth"] = {"value": round(eg, 1), "score": eg_s}

    # ── Debt/Equity (0 = 4pts) ────────────────────────────────────────────────
    de = safe(fundamentals.get("debt_equity"))
    if de is not None:
        de_adj = de / 100 if de > 10 else de  # normalize
        if de_adj <= 0:     de_s = 4.0
        elif de_adj <= 0.5: de_s = 3.0
        elif de_adj <= 1.0: de_s = 2.0
        else:               de_s = 1.0
        scores.append(de_s)
        components["debt_equity"] = {"value": round(de_adj, 2), "score": de_s}

    # ── P/E TTM (<10=4, <20=3, <30=2, >30=1) ─────────────────────────────────
    pe = safe(fundamentals.get("pe_ttm"))
    if pe is not None and pe > 0:
        if pe < 10:    pe_s = 4.0
        elif pe < 20:  pe_s = 3.0
        elif pe < 30:  pe_s = 2.0
        else:          pe_s = 1.0
        scores.append(pe_s)
        components["pe_ttm"] = {"value": round(pe, 1), "score": pe_s}

    # ── PEG (<1.5=4) ─────────────────────────────────────────────────────────
    peg = safe(fundamentals.get("peg"))
    if peg is not None and peg > 0:
        if peg < 1.5:  peg_s = 4.0
        elif peg < 2.0: peg_s = 3.0
        elif peg < 2.5: peg_s = 2.0
        else:           peg_s = 1.0
        scores.append(peg_s)
        components["peg"] = {"value": round(peg, 2), "score": peg_s}

    # ── Revenue Growth ─────────────────────────────────────────────────────────
    rg = safe(fundamentals.get("revenue_growth"), 100)
    if rg is not None:
        if rg >= 20:   rg_s = 4.0
        elif rg >= 10: rg_s = 3.0
        elif rg >= 0:  rg_s = 2.0
        else:          rg_s = 1.0
        scores.append(rg_s)
        components["revenue_growth"] = {"value": round(rg, 1), "score": rg_s}

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


# ==============================================================================
# SENTIMENT PILLAR  (0–4 points)
# ==============================================================================

def score_sentiment(sentiment: dict) -> dict:
    """
    Converts raw sentiment dict [-1,+1] to a 0–4 score.
    Includes velocity bonus (recent positive trend).
    From xls: 4=Headline & Great Trend, 3=Positive, 2=Mixed, 1=Negative
    """
    s = sentiment.get("combined_score", 0.0)
    v = sentiment.get("velocity", 0.0)

    # Base score from compound sentiment
    if s >= 0.35:   base = 4.0
    elif s >= 0.10: base = 3.0
    elif s >= -0.1: base = 2.0
    else:           base = 1.0

    # Velocity modifier (+/- 0.5)
    velocity_bonus = 0.3 if v > 0.05 else (-0.3 if v < -0.05 else 0.0)
    reddit_boost   = 0.2 if sentiment.get("reddit_mentions", 0) >= 5 else 0.0
    final = max(1.0, min(4.0, base + velocity_bonus + reddit_boost))

    return {
        "score": round(final, 3),
        "detail": {
            "base_score": base,
            "velocity_bonus": velocity_bonus,
            "reddit_boost": reddit_boost,
            "combined_raw": round(s, 3),
            "velocity": round(v, 3),
            "mentions": sentiment.get("reddit_mentions", 0),
        },
    }


# ==============================================================================
# RELATIVE STRENGTH PILLAR  (0–4 points)
# ==============================================================================

def score_relative_strength(symbol_df: pd.DataFrame, spy_df: pd.DataFrame,
                             sector: str = None) -> dict:
    """
    Measures performance vs SPY over 1-month and 3-month windows.
    From xls: 4=Leading Sector/Industry, 3=Above Avg, 2=Below Avg, 1=Worst
    """
    if symbol_df is None or spy_df is None or len(symbol_df) < 20:
        return {"score": 2.0, "detail": {"error": "insufficient data"}}

    try:
        sym_ret_1m  = (symbol_df["Close"].iloc[-1] / symbol_df["Close"].iloc[-21] - 1) * 100
        spy_ret_1m  = (spy_df["Close"].iloc[-1]    / spy_df["Close"].iloc[-21]    - 1) * 100
        rel_1m      = sym_ret_1m - spy_ret_1m

        sym_ret_3m  = (symbol_df["Close"].iloc[-1] / symbol_df["Close"].iloc[-63] - 1) * 100 \
                      if len(symbol_df) >= 63 else sym_ret_1m
        spy_ret_3m  = (spy_df["Close"].iloc[-1]    / spy_df["Close"].iloc[-63]    - 1) * 100 \
                      if len(spy_df) >= 63 else spy_ret_1m
        rel_3m      = sym_ret_3m - spy_ret_3m

        combined_rs = (rel_1m * 0.6) + (rel_3m * 0.4)

        if combined_rs >= 5.0:   rs_score = 4.0
        elif combined_rs >= 0.0: rs_score = 3.0
        elif combined_rs >= -5:  rs_score = 2.0
        else:                    rs_score = 1.0

        return {
            "score": round(rs_score, 3),
            "detail": {
                "ret_1m_vs_spy": round(rel_1m, 2),
                "ret_3m_vs_spy": round(rel_3m, 2),
                "combined_rs":   round(combined_rs, 2),
            },
        }
    except Exception as e:
        log.debug(f"RS score failed: {e}")
        return {"score": 2.0, "detail": {"error": str(e)}}


# ==============================================================================
# VOLATILITY PROFILE PILLAR  (0–4 points)
# ==============================================================================

def score_volatility(df: pd.DataFrame, fundamentals: dict,
                     beta_min: float, beta_max: float) -> dict:
    """
    Rewards volatility that's in the user's desired range.
    High volatility in correct direction = opportunity.
    From xls: measures beta fit and ATR profile.
    """
    if df is None or len(df) < 20:
        return {"score": 2.0, "detail": {}}

    try:
        beta = fundamentals.get("beta")
        # ATR (14-day average true range as % of price)
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_pct = (tr.rolling(14).mean().iloc[-1] / close.iloc[-1]) * 100

        detail = {"atr_pct": round(atr_pct, 2), "beta": beta}

        # Beta fit score
        beta_score = 2.0
        if beta is not None:
            if beta_min <= beta <= beta_max:
                beta_score = 4.0  # perfect fit
            elif abs(beta - beta_min) <= 0.3 or abs(beta - beta_max) <= 0.3:
                beta_score = 3.0
            elif abs(beta - beta_min) <= 0.7 or abs(beta - beta_max) <= 0.7:
                beta_score = 2.0
            else:
                beta_score = 1.0  # way outside range
            detail["beta_in_range"] = beta_min <= beta <= beta_max

        # ATR score: reward moderate volatility (1–4%), penalize extremes
        if 1.0 <= atr_pct <= 4.0:
            atr_score = 4.0
        elif 0.5 <= atr_pct < 1.0 or 4.0 < atr_pct <= 6.0:
            atr_score = 3.0
        elif atr_pct <= 0.5 or atr_pct <= 8.0:
            atr_score = 2.0
        else:
            atr_score = 1.0  # extremely volatile

        final = round((beta_score * 0.6) + (atr_score * 0.4), 3)
        detail.update({"beta_score": beta_score, "atr_score": atr_score})
        return {"score": final, "detail": detail}

    except Exception as e:
        log.debug(f"Volatility score failed: {e}")
        return {"score": 2.0, "detail": {"error": str(e)}}


# ==============================================================================
# MASTER GPA ENGINE
# ==============================================================================

class GPAEngine:
    """
    Orchestrates all 5 pillars into a composite 0.0–4.0 GPA score.
    Weights are user-configurable (see config.json → gpa_weights).
    """

    def __init__(self, weights: dict = None, beta_min: float = 0.8, beta_max: float = 1.8):
        self.weights  = weights or DEFAULT_WEIGHTS
        self.beta_min = beta_min
        self.beta_max = beta_max
        self._spy_df  = None  # cached SPY data

    def set_spy_df(self, spy_df: pd.DataFrame):
        self._spy_df = spy_df

    def score(self, symbol: str, ohlcv_df: pd.DataFrame,
              fundamentals: dict, sentiment: dict) -> dict:
        """
        Returns full GPA report for one stock.
        """
        # Score each pillar
        tech_result = score_technical(ohlcv_df)
        fund_result = score_fundamentals(fundamentals)
        sent_result = score_sentiment(sentiment)
        rs_result   = score_relative_strength(ohlcv_df, self._spy_df,
                                              fundamentals.get("sector"))
        vol_result  = score_volatility(ohlcv_df, fundamentals,
                                       self.beta_min, self.beta_max)

        pillars = {
            "technical_momentum": tech_result,
            "fundamentals":       fund_result,
            "sentiment":          sent_result,
            "relative_strength":  rs_result,
            "volatility_profile": vol_result,
        }

        # Weighted composite GPA
        gpa = 0.0
        for pillar, weight in self.weights.items():
            gpa += pillars[pillar]["score"] * weight

        gpa = round(gpa, 3)

        # Letter grade mapping
        if gpa >= 3.7:   grade = "A+"
        elif gpa >= 3.5: grade = "A"
        elif gpa >= 3.0: grade = "B+"
        elif gpa >= 2.7: grade = "B"
        elif gpa >= 2.3: grade = "B-"
        elif gpa >= 2.0: grade = "C+"
        elif gpa >= 1.7: grade = "C"
        else:            grade = "D"

        # Key drivers (top 2 contributors and detractors)
        scored_pillars = [(k, pillars[k]["score"] * self.weights[k]) for k in pillars]
        scored_pillars.sort(key=lambda x: x[1], reverse=True)
        top_drivers = [p[0] for p in scored_pillars[:2]]
        detractors  = [p[0] for p in scored_pillars[-2:] if p[1] < 0.3]

        # Buy/sell signals
        snapshot   = {}  # populated by caller
        buy_signal  = (
            gpa >= 3.5 and
            sentiment.get("combined_score", 0) > 0 and
            tech_result["score"] >= 3.0 and
            vol_result["detail"].get("beta_in_range", True)
        )
        sell_signal = gpa < 2.5

        return {
            "symbol":      symbol,
            "gpa":         gpa,
            "grade":       grade,
            "buy_signal":  buy_signal,
            "sell_signal": sell_signal,
            "pillars":     {k: {"score": pillars[k]["score"], "detail": pillars[k]["detail"]}
                            for k in pillars},
            "top_drivers": top_drivers,
            "detractors":  detractors,
            "weights_used": self.weights,
            "timestamp":   pd.Timestamp.now().isoformat(),
        }
