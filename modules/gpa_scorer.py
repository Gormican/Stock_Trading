"""
gpa_scorer.py — GPA Scoring Engine  (0.0 – 4.0 scale)
3-Category model: Sentiment / Fundamentals / Technical
Each category has configurable sub-categories.

Sentiment    (default 20%): news + social NLP
Fundamentals (default 45%):
    Valuation  (33%): P/E, PEG, EV/EBITDA, price-to-target upside
    Financial  (33%): ROE, D/E, revenue growth, dividend
    Estimates  (34%): earnings growth, analyst recs, beat rate, fwd estimates
Technical    (default 35%):
    Trend      (60%): MACD, moving averages, relative-strength vs SPY
    Oscillators(40%): RSI, Stochastics, volume
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("GPAScorer")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe(val, multiplier: float = 1.0):
    try:
        return float(val) * multiplier if val is not None else None
    except Exception:
        return None


def _grade(gpa: float) -> str:
    if gpa >= 3.7:   return "A+"
    elif gpa >= 3.5: return "A"
    elif gpa >= 3.3: return "A-"
    elif gpa >= 3.0: return "B+"
    elif gpa >= 2.7: return "B"
    elif gpa >= 2.3: return "B-"
    elif gpa >= 2.0: return "C+"
    elif gpa >= 1.7: return "C"
    else:            return "D"


# ──────────────────────────────────────────────────────────────────────────────
# SENTIMENT  (0–4)
# ──────────────────────────────────────────────────────────────────────────────

def score_sentiment(sentiment: dict) -> dict:
    """
    Converts raw VADER/news sentiment dict to 0–4 score.
    4 = Headline positive + strong trend
    3 = Positive
    2 = Mixed / neutral
    1 = Negative
    """
    s = sentiment.get("combined_score", 0.0)
    v = sentiment.get("velocity", 0.0)

    if s >= 0.35:    base = 4.0
    elif s >= 0.10:  base = 3.0
    elif s >= -0.10: base = 2.0
    else:            base = 1.0

    velocity_bonus = 0.3 if v > 0.05 else (-0.3 if v < -0.05 else 0.0)
    reddit_boost   = 0.2 if sentiment.get("reddit_mentions", 0) >= 5 else 0.0
    final = max(1.0, min(4.0, base + velocity_bonus + reddit_boost))

    return {
        "score": round(final, 3),
        "detail": {
            "base_score":     base,
            "velocity_bonus": velocity_bonus,
            "reddit_boost":   reddit_boost,
            "combined_raw":   round(s, 3),
            "velocity":       round(v, 3),
            "mentions":       sentiment.get("reddit_mentions", 0),
            "headline_count": sentiment.get("headline_count", 0),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS SUB-CATEGORIES
# ──────────────────────────────────────────────────────────────────────────────

def score_valuation(f: dict) -> dict:
    """
    Valuation sub-category: P/E, PEG, EV/EBITDA, price-to-target upside
    """
    scores = []
    components = {}

    # P/E TTM
    pe = _safe(f.get("pe_ttm"))
    if pe is not None and pe > 0:
        if pe < 12:    pe_s = 4.0
        elif pe < 20:  pe_s = 3.0
        elif pe < 30:  pe_s = 2.0
        else:          pe_s = 1.0
        scores.append(pe_s)
        components["pe_ttm"] = {"value": round(pe, 1), "score": pe_s,
                                 "label": f"P/E {pe:.1f}"}

    # PEG ratio
    peg = _safe(f.get("peg"))
    if peg is not None and 0 < peg < 20:
        if peg < 1.0:   peg_s = 4.0
        elif peg < 1.5: peg_s = 3.0
        elif peg < 2.0: peg_s = 2.0
        else:           peg_s = 1.0
        scores.append(peg_s)
        components["peg"] = {"value": round(peg, 2), "score": peg_s,
                              "label": f"PEG {peg:.2f}"}

    # EV/EBITDA
    ev_ebitda = _safe(f.get("enterprise_to_ebitda"))
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 8:    ev_s = 4.0
        elif ev_ebitda < 14: ev_s = 3.0
        elif ev_ebitda < 20: ev_s = 2.0
        else:                ev_s = 1.0
        scores.append(ev_s)
        components["ev_ebitda"] = {"value": round(ev_ebitda, 1), "score": ev_s,
                                    "label": f"EV/EBITDA {ev_ebitda:.1f}"}

    # Analyst price target upside
    target  = _safe(f.get("target_price"))
    current = _safe(f.get("current_price"))
    if target and current and current > 0:
        upside = (target / current - 1) * 100
        if upside >= 20:   up_s = 4.0
        elif upside >= 10: up_s = 3.0
        elif upside >= 0:  up_s = 2.0
        else:              up_s = 1.0
        scores.append(up_s)
        components["price_target_upside"] = {
            "value": round(upside, 1), "score": up_s,
            "label": f"Target upside {upside:+.1f}%"
        }

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


def score_financial(f: dict) -> dict:
    """
    Financial sub-category: ROE, Debt/Equity, Revenue Growth, Dividend
    """
    scores = []
    components = {}

    # ROE (>18% best)
    roe = _safe(f.get("roe"), 100)
    if roe is not None:
        if roe >= 20:   roe_s = 4.0
        elif roe >= 12: roe_s = 3.0
        elif roe >= 6:  roe_s = 2.0
        else:           roe_s = 1.0
        scores.append(roe_s)
        components["roe"] = {"value": round(roe, 1), "score": roe_s,
                              "label": f"ROE {roe:.1f}%"}

    # Debt/Equity (lower = better)
    de = _safe(f.get("debt_equity"))
    if de is not None:
        de_adj = de / 100 if de > 10 else de
        if de_adj <= 0.2:   de_s = 4.0
        elif de_adj <= 0.5: de_s = 3.0
        elif de_adj <= 1.0: de_s = 2.0
        else:                de_s = 1.0
        scores.append(de_s)
        components["debt_equity"] = {"value": round(de_adj, 2), "score": de_s,
                                      "label": f"D/E {de_adj:.2f}"}

    # Revenue Growth YoY
    rg = _safe(f.get("revenue_growth"), 100)
    if rg is not None:
        if rg >= 20:   rg_s = 4.0
        elif rg >= 10: rg_s = 3.0
        elif rg >= 0:  rg_s = 2.0
        else:          rg_s = 1.0
        scores.append(rg_s)
        components["revenue_growth"] = {"value": round(rg, 1), "score": rg_s,
                                         "label": f"Rev Growth {rg:+.1f}%"}

    # Dividend Yield (bonus for income; 0 is neutral)
    # yfinance inconsistently returns yield as a decimal (0.042) or already as
    # a percentage (4.2).  If the raw value is >= 0.5 it's already in pct form.
    raw_div = _safe(f.get("dividend_yield"))
    if raw_div is not None and raw_div > 0:
        div = raw_div if raw_div >= 0.5 else raw_div * 100
        if div >= 4.0:   div_s = 4.0
        elif div >= 2.5: div_s = 3.5
        elif div >= 1.0: div_s = 3.0
        else:            div_s = 2.5
        scores.append(div_s)
        components["dividend_yield"] = {"value": round(div, 2), "score": div_s,
                                         "label": f"Div {div:.2f}%"}

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


def score_estimates(f: dict) -> dict:
    """
    Estimates sub-category: earnings growth, analyst recs, beat rate, fwd estimates
    """
    scores = []
    components = {}

    # Earnings Growth YoY
    eg = _safe(f.get("earnings_growth_yoy"), 100)
    if eg is not None:
        if eg >= 25:    eg_s = 4.0
        elif eg >= 15:  eg_s = 3.0
        elif eg >= 5:   eg_s = 2.0
        else:           eg_s = 1.0
        scores.append(eg_s)
        components["earnings_growth"] = {"value": round(eg, 1), "score": eg_s,
                                          "label": f"EPS Growth {eg:+.1f}%"}

    # Analyst Recommendation (1=Strong Buy, 5=Strong Sell from yfinance)
    rec = _safe(f.get("analyst_recommendation"))
    if rec is not None:
        if rec <= 1.5:   rec_s = 4.0
        elif rec <= 2.5: rec_s = 3.0
        elif rec <= 3.5: rec_s = 2.0
        else:            rec_s = 1.0
        # Map numeric to text label
        rec_text_map = {4.0: "Strong Buy", 3.0: "Buy", 2.0: "Hold", 1.0: "Sell"}
        scores.append(rec_s)
        components["analyst_rec"] = {
            "value": round(rec, 2), "score": rec_s,
            "label": f"Analyst: {rec_text_map.get(rec_s, 'Hold')}"
        }

    # EPS Beat Rate (% of quarters beating estimates)
    beat = _safe(f.get("beat_expectations"), 100)
    if beat is None:
        beat = _safe(f.get("beat_rate"), 100)
    if beat is not None:
        if beat >= 80:   beat_s = 4.0
        elif beat >= 65: beat_s = 3.0
        elif beat >= 50: beat_s = 2.0
        else:            beat_s = 1.0
        scores.append(beat_s)
        components["beat_rate"] = {"value": round(beat, 1), "score": beat_s,
                                    "label": f"Beat rate {beat:.0f}%"}

    # Forward earnings estimate growth (next year vs current year)
    est_curr = _safe(f.get("earnings_estimate_current_year"))
    est_next = _safe(f.get("earnings_estimate_next_year"))
    if est_curr and est_next and est_curr > 0:
        fwd_growth = (est_next / est_curr - 1) * 100
        if fwd_growth >= 20:   fwd_s = 4.0
        elif fwd_growth >= 10: fwd_s = 3.0
        elif fwd_growth >= 0:  fwd_s = 2.0
        else:                  fwd_s = 1.0
        scores.append(fwd_s)
        components["fwd_eps_growth"] = {
            "value": round(fwd_growth, 1), "score": fwd_s,
            "label": f"Fwd EPS growth {fwd_growth:+.1f}%"
        }

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


def score_fundamentals_full(f: dict, sub_weights: dict) -> dict:
    """
    Rolls up Valuation, Financial, Estimates into one Fundamentals score
    using the given sub-weights.
    """
    val_result  = score_valuation(f)
    fin_result  = score_financial(f)
    est_result  = score_estimates(f)

    w_val = sub_weights.get("valuation", 0.33)
    w_fin = sub_weights.get("financial", 0.33)
    w_est = sub_weights.get("estimates", 0.34)

    final = (
        val_result["score"] * w_val +
        fin_result["score"] * w_fin +
        est_result["score"] * w_est
    )

    return {
        "score": round(final, 3),
        "sub_weights": {"valuation": w_val, "financial": w_fin, "estimates": w_est},
        "sub_scores": {
            "valuation":  val_result,
            "financial":  fin_result,
            "estimates":  est_result,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL SUB-CATEGORIES
# ──────────────────────────────────────────────────────────────────────────────

def score_trend(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> dict:
    """
    Trend sub-category: MACD, Moving Averages, Relative Strength vs SPY
    """
    if df is None or len(df) < 20:
        return {"score": 2.0, "detail": {"error": "insufficient data"}}

    close  = df["Close"]
    volume = df["Volume"]
    price  = close.iloc[-1]
    scores = []
    components = {}

    # ── MACD ──────────────────────────────────────────────────────────────────
    if len(close) >= 26:
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        mv, sv = macd.iloc[-1], signal.iloc[-1]
        mp, sp = macd.iloc[-2], signal.iloc[-2]
        crossover = (mp <= sp) and (mv > sv)
        if crossover and mv > 0:   macd_s = 4.0
        elif mv > 0 and mv > sv:   macd_s = 3.0
        elif mv > 0:               macd_s = 2.0
        else:                      macd_s = 1.0
        scores.append(macd_s)
        components["macd"] = {
            "value": round(mv, 3), "crossover": crossover, "score": macd_s,
            "label": ("Bullish crossover" if crossover else
                      ("MACD positive" if mv > 0 else "MACD negative"))
        }

    # ── Moving Averages ────────────────────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean() if len(close) >= 50 else sma20
    sma200 = close.rolling(200).mean() if len(close) >= 200 else sma50
    sma20_v  = sma20.iloc[-1]
    sma50_v  = sma50.iloc[-1]
    sma200_v = sma200.iloc[-1]

    sma20_s = sma20.dropna()
    slope_pct = (sma20_s.iloc[-1] / sma20_s.iloc[-5] - 1) * 100 if len(sma20_s) >= 5 else 0

    above_20  = price > sma20_v
    above_50  = price > sma50_v
    above_200 = price > sma200_v

    if slope_pct > 2.0 and above_20 and above_50 and above_200:
        ma_s = 4.0
        ma_label = "Price above all MAs, strong uptrend"
    elif slope_pct > 0.5 and above_20 and above_50:
        ma_s = 3.0
        ma_label = "Price above SMA20/50, uptrend"
    elif slope_pct > -0.5 and above_20:
        ma_s = 2.0
        ma_label = "Price above SMA20, flat trend"
    else:
        ma_s = 1.0
        ma_label = "Price below moving averages"

    scores.append(ma_s)
    components["moving_averages"] = {
        "sma20": round(sma20_v, 2), "sma50": round(sma50_v, 2),
        "sma200": round(sma200_v, 2), "slope_5d_pct": round(slope_pct, 2),
        "score": ma_s, "label": ma_label,
    }

    # ── Relative Strength vs SPY ───────────────────────────────────────────────
    if spy_df is not None and len(spy_df) >= 21:
        try:
            sym_ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100
            spy_ret_1m = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-21] - 1) * 100
            rel_1m     = sym_ret_1m - spy_ret_1m

            sym_ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else sym_ret_1m
            spy_ret_3m = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-63] - 1) * 100 if len(spy_df) >= 63 else spy_ret_1m
            rel_3m     = sym_ret_3m - spy_ret_3m

            rs_combined = (rel_1m * 0.6) + (rel_3m * 0.4)
            if rs_combined >= 5.0:   rs_s = 4.0
            elif rs_combined >= 0.0: rs_s = 3.0
            elif rs_combined >= -5:  rs_s = 2.0
            else:                    rs_s = 1.0

            scores.append(rs_s)
            components["rel_strength"] = {
                "vs_spy_1m": round(rel_1m, 2), "vs_spy_3m": round(rel_3m, 2),
                "combined": round(rs_combined, 2), "score": rs_s,
                "label": f"vs SPY 1M: {rel_1m:+.1f}%, 3M: {rel_3m:+.1f}%"
            }
        except Exception:
            pass

    # ── Volume on up days ──────────────────────────────────────────────────────
    price_up  = close.iloc[-1] > close.iloc[-2]
    avg_vol   = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    if price_up and vol_ratio >= 1.5:   vol_s = 4.0
    elif price_up:                       vol_s = 3.0
    elif not price_up and vol_ratio < 1: vol_s = 2.5
    else:                                vol_s = 1.5
    scores.append(vol_s)
    components["volume_trend"] = {
        "ratio": round(vol_ratio, 2), "price_up": price_up, "score": vol_s,
        "label": f"Vol {vol_ratio:.1f}x avg, {'up' if price_up else 'down'} day"
    }

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


def score_oscillators(df: pd.DataFrame) -> dict:
    """
    Oscillators sub-category: RSI, Stochastics, ATR (volatility filter)
    """
    if df is None or len(df) < 14:
        return {"score": 2.0, "detail": {"error": "insufficient data"}}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    scores = []
    components = {}

    # ── RSI(14) ────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]

    # Oversold/building = 4, normal momentum = 3, overbought approach = 2, overbought = 1
    if 30 <= rsi <= 50:      rsi_s = 4.0
    elif 50 < rsi <= 65:     rsi_s = 3.0
    elif 20 <= rsi < 30:     rsi_s = 3.5  # bouncing from oversold
    elif 65 < rsi <= 75:     rsi_s = 2.0
    else:                    rsi_s = 1.0  # <20 or >75
    scores.append(rsi_s)
    components["rsi"] = {
        "value": round(rsi, 1), "score": rsi_s,
        "label": f"RSI {rsi:.0f}" + (" (oversold)" if rsi < 30 else " (overbought)" if rsi > 70 else "")
    }

    # ── Stochastics %K (14,3) ─────────────────────────────────────────────────
    if len(high) >= 14:
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k_pct  = ((close - low14) / (high14 - low14).replace(0, np.nan) * 100).iloc[-1]
        k_prev = ((close - low14) / (high14 - low14).replace(0, np.nan) * 100).iloc[-2]
        rising = k_pct > k_prev

        if k_pct < 20 and rising:   stoch_s = 4.0
        elif k_pct < 50 and rising: stoch_s = 3.0
        elif k_pct < 80:            stoch_s = 2.0
        else:                       stoch_s = 1.0
        scores.append(stoch_s)
        components["stochastics"] = {
            "k_pct": round(k_pct, 1), "rising": rising, "score": stoch_s,
            "label": f"Stoch %K {k_pct:.0f} ({'↑' if rising else '↓'})"
        }

    # ── ATR (volatility — prefer moderate 1–4%) ───────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14).mean().iloc[-1] / close.iloc[-1]) * 100

    if 1.0 <= atr_pct <= 4.0:       atr_s = 4.0
    elif 0.5 <= atr_pct < 1.0:      atr_s = 3.0
    elif 4.0 < atr_pct <= 6.0:      atr_s = 3.0
    elif atr_pct < 0.5:             atr_s = 2.0
    elif 6.0 < atr_pct <= 8.0:      atr_s = 2.0
    else:                           atr_s = 1.0
    scores.append(atr_s)
    components["atr"] = {
        "value": round(atr_pct, 2), "score": atr_s,
        "label": f"ATR {atr_pct:.1f}%"
    }

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


def score_technical_full(df: pd.DataFrame, spy_df: pd.DataFrame,
                          sub_weights: dict) -> dict:
    """
    Rolls up Trend and Oscillators into one Technical score.
    """
    trend_result = score_trend(df, spy_df)
    osc_result   = score_oscillators(df)

    w_trend = sub_weights.get("trend", 0.60)
    w_osc   = sub_weights.get("oscillators", 0.40)

    final = trend_result["score"] * w_trend + osc_result["score"] * w_osc

    return {
        "score": round(final, 3),
        "sub_weights": {"trend": w_trend, "oscillators": w_osc},
        "sub_scores": {
            "trend":       trend_result,
            "oscillators": osc_result,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# MASTER GPA ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class GPAEngine:
    """
    Orchestrates Sentiment / Fundamentals / Technical into a 0.0–4.0 GPA.
    Weights come from a strategy dict (see strategy_manager.py).
    """

    def __init__(self, strategy: dict = None):
        """
        strategy: dict with keys:
            weights = {"sentiment": 0.20, "fundamentals": 0.45, "technical": 0.35,
                       "fund_sub":  {"valuation": 0.33, "financial": 0.33, "estimates": 0.34},
                       "tech_sub":  {"trend": 0.60, "oscillators": 0.40}}
            thresholds = {"min_gpa_to_buy": 3.5, "min_gpa_to_show": 3.0, "max_gpa_to_sell": 2.5}
            auto_trade = False
        """
        from modules.strategy_manager import StrategyManager, DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS
        if strategy is None:
            strategy = {
                "weights":    DEFAULT_WEIGHTS,
                "thresholds": DEFAULT_THRESHOLDS,
                "auto_trade": False,
            }
        self.weights    = strategy.get("weights",    DEFAULT_WEIGHTS)
        self.thresholds = strategy.get("thresholds", DEFAULT_THRESHOLDS)
        self.auto_trade = strategy.get("auto_trade", False)
        self._spy_df    = None

    def set_spy_df(self, spy_df: pd.DataFrame):
        self._spy_df = spy_df

    def score(self, symbol: str, ohlcv_df: pd.DataFrame,
              fundamentals: dict, sentiment: dict) -> dict:
        """
        Returns full GPA report dict for one stock.
        Keys: symbol, gpa, grade, buy_signal, sell_signal,
              categories (sent/fund/tech with sub-scores), top_drivers,
              weights_used, thresholds, timestamp
        """
        w = self.weights
        fund_sub = w.get("fund_sub", {"valuation": 0.33, "financial": 0.33, "estimates": 0.34})
        tech_sub = w.get("tech_sub", {"trend": 0.60, "oscillators": 0.40})

        sent_result = score_sentiment(sentiment)
        fund_result = score_fundamentals_full(fundamentals, fund_sub)
        tech_result = score_technical_full(ohlcv_df, self._spy_df, tech_sub)

        w_sent = w.get("sentiment",    0.20)
        w_fund = w.get("fundamentals", 0.45)
        w_tech = w.get("technical",    0.35)

        gpa = round(
            sent_result["score"] * w_sent +
            fund_result["score"] * w_fund +
            tech_result["score"] * w_tech,
            3
        )

        grade = _grade(gpa)

        # Thresholds
        t           = self.thresholds
        buy_signal  = gpa >= t.get("min_gpa_to_buy", 3.5)
        sell_signal = gpa <= t.get("max_gpa_to_sell", 2.5)

        # Top drivers (by weighted contribution)
        category_contributions = {
            "Sentiment":    sent_result["score"] * w_sent,
            "Fundamentals": fund_result["score"] * w_fund,
            "Technical":    tech_result["score"] * w_tech,
        }
        sorted_cats = sorted(category_contributions.items(), key=lambda x: x[1], reverse=True)
        top_drivers = [c[0] for c in sorted_cats[:2]]
        detractors  = [c[0] for c in sorted_cats if c[1] < (gpa * w_sent)]  # below average

        return {
            "symbol":     symbol,
            "gpa":        gpa,
            "grade":      grade,
            "buy_signal":  buy_signal,
            "sell_signal": sell_signal,
            "categories": {
                "sentiment": {
                    "score":   sent_result["score"],
                    "weight":  w_sent,
                    "contribution": round(sent_result["score"] * w_sent, 3),
                    "detail":  sent_result["detail"],
                },
                "fundamentals": {
                    "score":      fund_result["score"],
                    "weight":     w_fund,
                    "contribution": round(fund_result["score"] * w_fund, 3),
                    "sub_weights": fund_result["sub_weights"],
                    "sub_scores":  fund_result["sub_scores"],
                },
                "technical": {
                    "score":      tech_result["score"],
                    "weight":     w_tech,
                    "contribution": round(tech_result["score"] * w_tech, 3),
                    "sub_weights": tech_result["sub_weights"],
                    "sub_scores":  tech_result["sub_scores"],
                },
            },
            "top_drivers":  top_drivers,
            "detractors":   detractors,
            "weights_used": w,
            "thresholds":   t,
            "timestamp":    pd.Timestamp.now().isoformat(),
        }

    # ── Backward compat shim for trading_agent.py ────────────────────────────
    # trading_agent.py calls engine.score() which now returns categories instead
    # of pillars — we keep the same interface, trading_agent just needs gpa/signals.
    # No changes needed there.
