"""
gpa_scorer.py — GPA Scoring Engine  (0.0 – 4.0 scale)
3-Category model: Sentiment / Fundamentals / Technical

Sentiment    (default 20%): Alpha Vantage news + StockTwits social
Fundamentals (default 45%):
    Valuation  (33%): P/E, PEG, EV/EBITDA, price-to-target upside
    Financial  (33%): ROE, D/E, revenue growth, dividend
    Estimates  (34%): earnings growth, analyst recs, beat rate, fwd estimates
Technical    (default 35%):
    Trend      (60%): support/resistance, moving averages, trend slope,
                       ATR buffer, Fibonacci, volume, relative strength vs SPY
    Oscillators(40%): RSI, Stochastics, CMF (Chaikin Money Flow), ATR

Thresholds match Steve's original Codes sheet (A=4, B=3, C=2, D=1).

RSI NOTE: Currently scored by absolute level (30-50 = Score 4).
          Pending decision on whether to switch to "trough = Buy" mode.
          See score_rsi() below for the toggle.
"""

import json
import logging
import time
import requests
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

log = logging.getLogger("GPAScorer")

# ──────────────────────────────────────────────────────────────────────────────
# DAILY SENTIMENT CACHE
# Saves Alpha Vantage results to disk keyed by TICKER_YYYY-MM-DD.
# Free tier = 25 calls/day, so we never re-call for the same ticker on the
# same calendar day.
# ──────────────────────────────────────────────────────────────────────────────
_SENTIMENT_CACHE_FILE = Path(__file__).parent.parent / "sentiment_cache.json"

def _load_sent_cache() -> dict:
    try:
        if _SENTIMENT_CACHE_FILE.exists():
            return json.loads(_SENTIMENT_CACHE_FILE.read_text())
    except Exception:
        pass
    return {}

def _save_sent_cache(cache: dict):
    try:
        _SENTIMENT_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass

AV_DAILY_LIMIT = 25   # free tier cap; change to 75 if on paid plan

def _sent_cache_key(ticker: str) -> str:
    return f"{ticker.upper()}_{date.today().isoformat()}"

def _get_cached_av(ticker: str):
    """Return cached AV result for today, or None if not cached."""
    cache = _load_sent_cache()
    return cache.get(_sent_cache_key(ticker))

def _store_av(ticker: str, result: dict):
    """Store AV result in today's cache slot."""
    cache = _load_sent_cache()
    today_prefix = date.today().isoformat()
    cache = {k: v for k, v in cache.items() if k.endswith(today_prefix)}
    cache[_sent_cache_key(ticker)] = result
    _save_sent_cache(cache)

def _av_calls_today() -> int:
    """Count how many real AV API calls have been made today (cached = free)."""
    cache = _load_sent_cache()
    today_prefix = date.today().isoformat()
    return sum(1 for k in cache if k.endswith(today_prefix))

def _av_quota_ok() -> bool:
    """Return True if we still have AV calls available today."""
    used = _av_calls_today()
    remaining = AV_DAILY_LIMIT - used
    if remaining <= 0:
        log.warning(f"AV daily quota reached ({AV_DAILY_LIMIT} calls). "
                    f"Sentiment will use cached/neutral values until tomorrow.")
        return False
    if remaining <= 5:
        log.info(f"AV quota low: {remaining} calls remaining today.")
    return True

# ── RSI mode ──────────────────────────────────────────────────────────────────
# "C" = Combined: direction is primary signal, absolute level is modifier.
#       Rising from below 50 = best (trough reversal).
#       Falling regardless of level = lower score.
# "B" = Trough only (direction only, ignore absolute level)
# False = Absolute level only (original behavior)
RSI_TROUGH_MODE = "C"

# ──────────────────────────────────────────────────────────────────────────────
# API FETCH HELPERS  (Alpha Vantage + StockTwits)
# ──────────────────────────────────────────────────────────────────────────────

AV_BASE = "https://www.alphavantage.co/query"
ST_BASE = "https://api.stocktwits.com/api/2/streams/symbol"

# Alpha Vantage label → numeric score (0-4 scale for blending)
_AV_LABEL_SCORE = {
    "Bullish":           4.0,
    "Somewhat-Bullish":  3.0,
    "Neutral":           2.0,
    "Somewhat-Bearish":  1.5,
    "Bearish":           1.0,
}


def fetch_alpha_vantage_sentiment(ticker: str, api_key: str,
                                  max_articles: int = 10) -> dict:
    """
    Pull the most recent news sentiment for ticker from Alpha Vantage.
    Free tier: 25 calls/day.  Returns a normalized dict for score_sentiment().

    Returns:
        {
          "av_score":          float,   # 0.0–4.0 blended score
          "av_label":          str,     # e.g. "Somewhat-Bullish"
          "av_article_count":  int,
          "av_ok":             bool,    # False if API error / quota exceeded
        }
    """
    if not api_key:
        return {"av_score": 2.0, "av_label": "Neutral",
                "av_article_count": 0, "av_ok": False}

    # Check daily cache first — free tier is 25 calls/day
    cached = _get_cached_av(ticker)
    if cached is not None:
        log.info(f"AV {ticker}: cached ({cached.get('av_article_count',0)} articles, {cached.get('av_label','?')})")
        return cached

    # Check quota before making a real API call
    if not _av_quota_ok():
        return {"av_score": 2.0, "av_label": "Quota reached",
                "av_article_count": 0, "av_ok": False}

    try:
        resp = requests.get(AV_BASE, params={
            "function": "NEWS_SENTIMENT",
            "tickers":  ticker,
            "limit":    max_articles,
            "apikey":   api_key,
        }, timeout=10)
        data = resp.json()

        feed = data.get("feed", [])
        if not feed:
            result = {"av_score": 2.0, "av_label": "Neutral",
                      "av_article_count": 0, "av_ok": True}
            _store_av(ticker, result)   # cache "no news" so we don't retry today
            return result

        # Weight each article by relevance score for this ticker
        total_weight = 0.0
        weighted_sum = 0.0
        labels = []

        for article in feed[:max_articles]:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    rel   = float(ts.get("relevance_score", 0.5))
                    score = float(ts.get("ticker_sentiment_score", 0))
                    label = ts.get("ticker_sentiment_label", "Neutral")
                    # AV scores range -1 to +1; map to 1-4
                    mapped = (score + 1) / 2 * 3 + 1   # -1→1, 0→2.5, +1→4
                    weighted_sum += mapped * rel
                    total_weight += rel
                    labels.append(label)

        if total_weight == 0:
            return {"av_score": 2.0, "av_label": "Neutral",
                    "av_article_count": len(feed), "av_ok": True}

        blended_score = round(weighted_sum / total_weight, 3)
        # Most common label
        from collections import Counter
        top_label = Counter(labels).most_common(1)[0][0] if labels else "Neutral"

        result = {
            "av_score":         min(4.0, max(1.0, blended_score)),
            "av_label":         top_label,
            "av_article_count": len(feed),
            "av_ok":            True,
        }
        _store_av(ticker, result)
        log.info(f"AV sentiment for {ticker}: {top_label} ({len(feed)} articles) → score {result['av_score']:.2f}")
        return result
    except Exception as e:
        log.warning(f"Alpha Vantage error for {ticker}: {e}")
        return {"av_score": 2.0, "av_label": "Neutral",
                "av_article_count": 0, "av_ok": False}


def fetch_stocktwits_sentiment(ticker: str) -> dict:
    """
    Pull the most recent 30 StockTwits messages for ticker.
    No API key needed for public reads.
    Note: StockTwits free unauthenticated API is unreliable as of 2025;
    returns st_ok=False silently when the API is unavailable.
    """
    _fail = {"st_bullish_ratio": 0.5, "st_message_count": 0,
             "st_tagged_count": 0, "st_ok": False}
    try:
        resp = requests.get(
            f"{ST_BASE}/{ticker}.json",
            headers={"User-Agent": "TaylorTradingAgent/1.0"},
            timeout=8,
        )
        # Non-200 or empty body = API unavailable / rate-limited
        if resp.status_code != 200 or not resp.content:
            return _fail

        try:
            data = resp.json()
        except Exception:
            # Empty or non-JSON body (common when rate-limited)
            return _fail

        messages = data.get("messages", [])
        if not messages:
            return _fail

        bullish = sum(1 for m in messages
                      if (m.get("entities", {}).get("sentiment") or {})
                         .get("basic") == "Bullish")
        bearish = sum(1 for m in messages
                      if (m.get("entities", {}).get("sentiment") or {})
                         .get("basic") == "Bearish")
        tagged = bullish + bearish
        ratio  = (bullish / tagged) if tagged > 0 else 0.5

        return {
            "st_bullish_ratio": round(ratio, 3),
            "st_message_count": len(messages),
            "st_tagged_count":  tagged,
            "st_ok":            True,
        }
    except Exception:
        # Swallow all errors silently — StockTwits is a nice-to-have, not critical
        return _fail


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
# SENTIMENT  (Codes sheet: "Positive News"  4=Headline&Great 3=Positive 2=Mixed 1=Neg)
# ──────────────────────────────────────────────────────────────────────────────

def score_sentiment(sentiment: dict) -> dict:
    """
    Scores news + social sentiment on the 1-4 GPA scale.

    Accepts a dict that may contain any combination of:
      - Alpha Vantage keys: av_score (1-4), av_label, av_article_count, av_ok
      - StockTwits keys:    st_bullish_ratio (0-1), st_message_count, st_tagged_count
      - Legacy VADER keys:  combined_score (-1 to +1), velocity, reddit_mentions
        (kept for backwards compatibility if AV/ST not available)

    Mapping to Codes sheet "Positive News":
      Score 4 = AV Bullish AND StockTwits >70% bullish  (Headline & Great)
      Score 3 = AV Somewhat-Bullish  OR  StockTwits 55-70%  (Positive)
      Score 2 = Neutral mixed signals                          (Mixed)
      Score 1 = AV Bearish  OR  StockTwits <35% bullish       (Negative)
    """
    scores    = []
    detail    = {}
    source    = "none"

    # ── Alpha Vantage ────────────────────────────────────────────────────────
    av_score = _safe(sentiment.get("av_score"))
    av_ok    = sentiment.get("av_ok", False)
    if av_score is not None and av_ok:
        scores.append(av_score)
        detail["alpha_vantage"] = {
            "score":        round(av_score, 3),
            "label":        sentiment.get("av_label", "Neutral"),
            "article_count": sentiment.get("av_article_count", 0),
        }
        source = "alpha_vantage"

    # ── StockTwits ───────────────────────────────────────────────────────────
    st_ratio  = _safe(sentiment.get("st_bullish_ratio"))
    st_count  = sentiment.get("st_message_count", 0)
    st_tagged = sentiment.get("st_tagged_count", 0)
    st_ok     = sentiment.get("st_ok", False)

    if st_ratio is not None and st_ok and st_tagged >= 3:
        # Convert bullish ratio (0-1) to 1-4 scale
        if st_ratio >= 0.70:   st_s = 4.0
        elif st_ratio >= 0.55: st_s = 3.0
        elif st_ratio >= 0.40: st_s = 2.0
        else:                  st_s = 1.0
        # Volume bonus: lots of activity = stronger signal
        if st_count >= 20 and st_s >= 3.0:
            st_s = min(4.0, st_s + 0.25)
        scores.append(st_s)
        detail["stocktwits"] = {
            "score":         round(st_s, 3),
            "bullish_ratio": round(st_ratio, 3),
            "messages":      st_count,
            "tagged":        st_tagged,
        }
        source = source + "+stocktwits" if source != "none" else "stocktwits"

    # ── Legacy VADER / Reddit fallback (backwards compat) ───────────────────
    combined = _safe(sentiment.get("combined_score", 0.0))
    if combined is not None and not (av_ok or (st_ok and st_tagged >= 3)):
        if combined >= 0.35:    base = 4.0
        elif combined >= 0.10:  base = 3.0
        elif combined >= -0.10: base = 2.0
        else:                   base = 1.0
        velocity_bonus = 0.3 if sentiment.get("velocity", 0) > 0.05 else (
                        -0.3 if sentiment.get("velocity", 0) < -0.05 else 0.0)
        reddit_boost = 0.2 if sentiment.get("reddit_mentions", 0) >= 5 else 0.0
        base = max(1.0, min(4.0, base + velocity_bonus + reddit_boost))
        scores.append(base)
        detail["legacy_vader"] = {
            "score":      round(base, 3),
            "combined":   round(combined, 3),
            "velocity":   round(sentiment.get("velocity", 0), 3),
            "mentions":   sentiment.get("reddit_mentions", 0),
        }
        source = "vader_legacy"

    final = round(np.mean(scores), 3) if scores else 2.0
    final = max(1.0, min(4.0, final))

    return {
        "score": final,
        "detail": {
            "source": source,
            "headline_count": sentiment.get("av_article_count",
                              sentiment.get("headline_count", 0)),
            **detail,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS — VALUATION
# Codes sheet: P/E 4=<10, 3=10-20, 2=20-30, 1>=30
#               PEG 4=<1,  3=1-1.5, 2=1.6-2, 1>2
#               EV/EBITDA 4=<10, 3=10-15, 2=15-20, 1>20
# ──────────────────────────────────────────────────────────────────────────────

def score_valuation(f: dict) -> dict:
    scores     = []
    components = {}

    # ── P/E TTM (Codes: 4=<10, 3=10-20, 2=20-30, 1>=30) ─────────────────────
    pe = _safe(f.get("pe_ttm"))
    if pe is not None and pe > 0:
        if pe < 10:    pe_s = 4.0
        elif pe < 20:  pe_s = 3.0
        elif pe < 30:  pe_s = 2.0
        else:          pe_s = 1.0
        scores.append(pe_s)
        components["pe_ttm"] = {"value": round(pe, 1), "score": pe_s,
                                 "label": f"P/E {pe:.1f}"}

    # ── PEG (Codes: 4=<1, 3=1-1.5, 2=1.6-2, 1>2) ────────────────────────────
    peg = _safe(f.get("peg"))
    if peg is not None and 0 < peg < 20:
        if peg < 1.0:   peg_s = 4.0
        elif peg < 1.5: peg_s = 3.0
        elif peg < 2.0: peg_s = 2.0
        else:           peg_s = 1.0
        scores.append(peg_s)
        components["peg"] = {"value": round(peg, 2), "score": peg_s,
                              "label": f"PEG {peg:.2f}"}

    # ── EV/EBITDA (Codes: 4=<10, 3=10-15, 2=15-20, 1>20) ────────────────────
    ev_ebitda = _safe(f.get("enterprise_to_ebitda"))
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 10:   ev_s = 4.0
        elif ev_ebitda < 15: ev_s = 3.0
        elif ev_ebitda < 20: ev_s = 2.0
        else:                ev_s = 1.0
        scores.append(ev_s)
        components["ev_ebitda"] = {"value": round(ev_ebitda, 1), "score": ev_s,
                                    "label": f"EV/EBITDA {ev_ebitda:.1f}"}

    # ── Analyst price target upside (bonus metric — not in Codes sheet) ───────
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


# ──────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS — FINANCIAL
# Codes sheet: ROE 4=>25, 3=15-25, 2=5-15, 1=<5
#               D/E 4=near 0, 3=<0.5, 2=0.5-1, 1>1
#               Dividend 4=>4%, 3=2-4%, 2=0-2%, 1=no div
# ──────────────────────────────────────────────────────────────────────────────

def score_financial(f: dict) -> dict:
    scores     = []
    components = {}

    # ── ROE (Codes: 4=>25%, 3=15-25%, 2=5-15%, 1=<5%) ───────────────────────
    roe = _safe(f.get("roe"), 100)
    if roe is not None:
        if roe >= 25:   roe_s = 4.0
        elif roe >= 15: roe_s = 3.0
        elif roe >= 5:  roe_s = 2.0
        else:           roe_s = 1.0
        scores.append(roe_s)
        components["roe"] = {"value": round(roe, 1), "score": roe_s,
                              "label": f"ROE {roe:.1f}%"}

    # ── Debt/Equity (Codes: 4=~0, 3=<0.5, 2=0.5-1, 1>1) ────────────────────
    de = _safe(f.get("debt_equity"))
    if de is not None:
        de_adj = de / 100 if de > 10 else de
        if de_adj <= 0.10:   de_s = 4.0    # Codes: "4=0" — near-zero debt
        elif de_adj <= 0.50: de_s = 3.0
        elif de_adj <= 1.00: de_s = 2.0
        else:                de_s = 1.0
        scores.append(de_s)
        components["debt_equity"] = {"value": round(de_adj, 2), "score": de_s,
                                      "label": f"D/E {de_adj:.2f}"}

    # ── Revenue Growth YoY ────────────────────────────────────────────────────
    rg = _safe(f.get("revenue_growth"), 100)
    if rg is not None:
        if rg >= 20:   rg_s = 4.0
        elif rg >= 10: rg_s = 3.0
        elif rg >= 0:  rg_s = 2.0
        else:          rg_s = 1.0
        scores.append(rg_s)
        components["revenue_growth"] = {"value": round(rg, 1), "score": rg_s,
                                         "label": f"Rev Growth {rg:+.1f}%"}

    # ── Dividend Yield (Codes: 4=>4%, 3=2-4%, 2=0-2%, 1=no div) ─────────────
    # yfinance inconsistently returns decimal (0.042) or percent (4.2)
    raw_div = _safe(f.get("dividend_yield"))
    if raw_div is not None and raw_div > 0:
        div = raw_div if raw_div >= 0.5 else raw_div * 100
        if div >= 4.0:   div_s = 4.0
        elif div >= 2.0: div_s = 3.0    # Codes threshold: 2%, not 2.5%
        elif div > 0:    div_s = 2.0
        else:            div_s = 1.0
        scores.append(div_s)
        components["dividend_yield"] = {"value": round(div, 2), "score": div_s,
                                         "label": f"Div {div:.2f}%"}

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


# ──────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS — ESTIMATES
# Codes: Analyst 4=<1.5, 3=1.6-2.0, 2=2.1-2.6, 1>2.6
#         Earnings growth 4=>25, 3=15-25, 2=5-15, 1<5
#         Fwd estimates 4=>20%, 3=10-20%, 2=0-10%, 1=negative
# ──────────────────────────────────────────────────────────────────────────────

def score_estimates(f: dict) -> dict:
    scores     = []
    components = {}

    # ── Earnings Growth YoY (Codes: 4=>25%, 3=15-25%, 2=5-15%, 1=<5%) ───────
    eg = _safe(f.get("earnings_growth_yoy"), 100)
    if eg is not None:
        if eg >= 25:   eg_s = 4.0
        elif eg >= 15: eg_s = 3.0
        elif eg >= 5:  eg_s = 2.0
        else:          eg_s = 1.0
        scores.append(eg_s)
        components["earnings_growth"] = {"value": round(eg, 1), "score": eg_s,
                                          "label": f"EPS Growth {eg:+.1f}%"}

    # ── Analyst Recommendation (Codes: 4=<1.5, 3=1.5-2.0, 2=2.1-2.6, 1>2.6)
    # yfinance uses 1=Strong Buy → 5=Strong Sell scale
    rec = _safe(f.get("analyst_recommendation"))
    if rec is not None:
        if rec < 1.5:    rec_s = 4.0
        elif rec <= 2.0: rec_s = 3.0
        elif rec <= 2.6: rec_s = 2.0
        else:            rec_s = 1.0
        rec_text = {4.0: "Strong Buy", 3.0: "Buy", 2.0: "Hold", 1.0: "Sell"}
        scores.append(rec_s)
        components["analyst_rec"] = {
            "value": round(rec, 2), "score": rec_s,
            "label": f"Analyst: {rec_text.get(rec_s, 'Hold')}"
        }

    # ── EPS Beat Rate (% of quarters beating estimates, from yfinance) ────────
    beat = _safe(f.get("beat_expectations"), 100)
    if beat is None:
        beat = _safe(f.get("beat_rate"), 100)
    if beat is not None:
        if beat >= 80:   beat_s = 4.0    # Codes "Beat Expectations: 4=Above"
        elif beat >= 65: beat_s = 3.0
        elif beat >= 50: beat_s = 2.0
        else:            beat_s = 1.0
        scores.append(beat_s)
        components["beat_rate"] = {"value": round(beat, 1), "score": beat_s,
                                    "label": f"Beat rate {beat:.0f}%"}

    # ── Forward EPS Growth: next year vs current year ─────────────────────────
    # Codes: Est Next Yr 4=>20%, 3=10-20%, 2=0-10%, 1=negative
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
    val_result = score_valuation(f)
    fin_result = score_financial(f)
    est_result = score_estimates(f)

    w_val = sub_weights.get("valuation", 0.33)
    w_fin = sub_weights.get("financial", 0.33)
    w_est = sub_weights.get("estimates", 0.34)

    final = (val_result["score"] * w_val +
             fin_result["score"] * w_fin +
             est_result["score"] * w_est)

    return {
        "score":       round(final, 3),
        "sub_weights": {"valuation": w_val, "financial": w_fin, "estimates": w_est},
        "sub_scores":  {
            "valuation": val_result,
            "financial": fin_result,
            "estimates": est_result,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL HELPERS  (inline from support_resistance.py — no import needed)
# All take a pre-fetched OHLCV DataFrame, matching Codes sheet parameters.
# ──────────────────────────────────────────────────────────────────────────────

_CLUSTER_WINDOW   = 0.02   # 2% price band = same level
_VOLUME_HIGH_MULT = 1.5    # >1.5x 50-day avg = "High Volume"
_FLAT_THRESHOLD   = 0.03   # <3% move = flat


def _score_value_of_support(df: pd.DataFrame) -> float:
    """
    Codes: 4=Solid and New, 3=Mixed and Old, 2=Thin Lower Low, 1=No Support-New Low
    Uses: 52W low proximity, price cluster density below, weekly pivot S1.
    """
    if df is None or len(df) < 20:
        return 2.0
    close   = df["Close"].values.astype(float)
    high    = df["High"].values.astype(float)
    low     = df["Low"].values.astype(float)
    curr    = close[-1]
    low52   = low.min()
    high52  = high.max()
    span    = high52 - low52 if high52 > low52 else 1.0

    # Signal 1: cushion above 52W low (0=at low, 1=at high)
    pct = (curr - low52) / span
    if pct < 0.03:   sig1 = 1.0
    elif pct < 0.15: sig1 = 1.5
    elif pct < 0.35: sig1 = 2.5
    elif pct < 0.60: sig1 = 3.0
    else:            sig1 = 4.0

    # Signal 2: historical price cluster below current (more = stronger floor)
    zone_lo = curr * (1 - _CLUSTER_WINDOW * 3)
    zone_hi = curr * (1 - _CLUSTER_WINDOW * 0.1)
    n_cluster = int(np.sum((close >= zone_lo) & (close <= zone_hi)))
    if n_cluster >= 15:   sig2 = 4.0
    elif n_cluster >= 8:  sig2 = 3.0
    elif n_cluster >= 3:  sig2 = 2.5
    elif n_cluster >= 1:  sig2 = 2.0
    else:                 sig2 = 1.5

    # Signal 3: weekly pivot S1
    if len(df) >= 5:
        w_hi  = high[-5:].max()
        w_lo  = low[-5:].min()
        pivot = (w_hi + w_lo + close[-1]) / 3
        s1    = 2 * pivot - w_hi
        s2    = pivot - (w_hi - w_lo)
        if curr > pivot:  sig3 = 4.0
        elif curr > s1:   sig3 = 3.0
        elif curr > s2:   sig3 = 2.0
        else:             sig3 = 1.0
    else:
        sig3 = 2.5

    return round(min(4.0, max(1.0, sig1 * 0.40 + sig2 * 0.35 + sig3 * 0.25)), 2)


def _score_lack_of_resistance(df: pd.DataFrame) -> float:
    """
    Codes: 4=No Resistance (new H), 3=Thin Higher High, 2=Thick and Old,
           1=Thick and New.
    Uses: 52W high proximity, cluster density above, weekly pivot R1.
    """
    if df is None or len(df) < 20:
        return 2.0
    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    curr  = close[-1]
    high52 = high.max()

    # Signal 1: how close to 52W high
    pct_below = (high52 - curr) / high52 if high52 > 0 else 0
    if pct_below <= 0.02:   sig1 = 4.0   # basically at new high
    elif pct_below <= 0.05: sig1 = 3.5
    elif pct_below <= 0.15: sig1 = 2.5
    elif pct_below <= 0.30: sig1 = 1.5
    else:                   sig1 = 1.0

    # Signal 2: cluster density ABOVE (more = thicker resistance)
    zone_lo = curr * (1 + _CLUSTER_WINDOW * 0.1)
    zone_hi = curr * (1 + _CLUSTER_WINDOW * 4)
    n_cluster = int(np.sum((close >= zone_lo) & (close <= zone_hi)))
    if n_cluster >= 15:   sig2 = 1.0
    elif n_cluster >= 8:  sig2 = 2.0
    elif n_cluster >= 3:  sig2 = 3.0
    else:                 sig2 = 4.0

    # Signal 3: weekly pivot R1
    if len(df) >= 5:
        w_hi  = high[-5:].max()
        w_lo  = low[-5:].min()
        pivot = (w_hi + w_lo + close[-1]) / 3
        r1    = 2 * pivot - w_lo
        pct_to_r1 = (r1 - curr) / curr if r1 > curr else 0
        if curr >= r1:          sig3 = 4.0
        elif pct_to_r1 < 0.02:  sig3 = 3.5
        elif pct_to_r1 < 0.05:  sig3 = 3.0
        elif pct_to_r1 < 0.10:  sig3 = 2.0
        else:                   sig3 = 1.5
    else:
        sig3 = 2.5

    return round(min(4.0, max(1.0, sig1 * 0.45 + sig2 * 0.35 + sig3 * 0.20)), 2)


def _score_slope_trend(df: pd.DataFrame, days: int) -> float:
    """
    Codes: 3Mo 4=>45deg Up, 3=Up<45, 2=Flat, 1=Down
           1Yr 4=>30deg Up, 3=Up<30, 2=Flat, 1=Down
    Implemented as % price change + regression slope (no manual angle estimation).
    """
    if df is None:
        return 2.0
    n = min(days, len(df))
    if n < 5:
        return 2.0
    subset = df["Close"].iloc[-n:].values.astype(float)
    pct = (subset[-1] - subset[0]) / subset[0] * 100
    x   = np.arange(len(subset))
    slope, _ = np.polyfit(x, subset, 1)
    slope_pct_day = slope / subset[0] * 100

    if pct > 30:    base = 4.0
    elif pct > 5:   base = 3.0
    elif pct > -5:  base = 2.0
    else:           base = 1.0

    # Regression slope confirmation bonus
    if slope_pct_day > 0.05 and base >= 3.0:
        base = min(4.0, base + 0.25)
    elif slope_pct_day < -0.05 and base <= 2.0:
        base = max(1.0, base - 0.25)

    return round(base, 2)


def _score_ma_position(df: pd.DataFrame, ma_period: int) -> float:
    """
    Codes: 4=Uptrend and Above, 3=Above or Up/Near, 2=Below but Uptrend,
           1=Below and Downtrend.
    """
    if df is None or len(df) < ma_period + 5:
        return 2.0
    closes   = df["Close"].values.astype(float)
    ma       = np.mean(closes[-ma_period:])
    ma_prev  = np.mean(closes[-(ma_period + 5):-5])
    curr     = closes[-1]
    ma_up    = ma > ma_prev
    pct_vs   = (curr - ma) / ma * 100

    if ma_up and pct_vs > 0:
        s = 4.0
    elif pct_vs > 0:
        s = 3.0
    elif ma_up and pct_vs > -3:
        s = 3.0   # near MA but uptrending
    elif ma_up:
        s = 2.0   # below MA but MA uptrending
    else:
        s = 1.0   # below AND MA downtrending

    # Don't over-reward extreme extension (>10% above MA = reversion risk)
    if s == 4.0 and pct_vs > 10:
        s = 3.5
    return round(s, 2)


def _score_volume_trend(df: pd.DataFrame, days: int = 20) -> float:
    """
    Codes: 4=H Vol Up, 3=N Vol Up, 2=N Vol Down, 1=H Vol Down.
    """
    if df is None or len(df) < days + 10:
        return 2.0
    closes = df["Close"].values.astype(float)
    vols   = df["Volume"].values.astype(float)
    ref_n  = min(50, len(df))
    avg_vol    = np.mean(vols[-ref_n:])
    recent_vol = np.mean(vols[-days:])
    up   = closes[-1] > closes[-days] * (1 + _FLAT_THRESHOLD)
    down = closes[-1] < closes[-days] * (1 - _FLAT_THRESHOLD)
    high_vol = recent_vol > avg_vol * _VOLUME_HIGH_MULT

    if high_vol and up:    return 4.0
    if not high_vol and up: return 3.0
    if not high_vol and down: return 2.0
    if high_vol and down:  return 1.0
    return 2.0


def _score_atr_buffer(df: pd.DataFrame) -> float:
    """
    NEW: How many ATR (14-day) units of cushion above the 52W low.
    More ATRs = thicker shock absorber = higher score.
    """
    if df is None or len(df) < 20:
        return 2.0
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = min(14, len(df) - 1)
    tr = np.maximum(high[-n:] - low[-n:],
         np.maximum(np.abs(high[-n:] - close[-n-1:-1]),
                    np.abs(low[-n:] - close[-n-1:-1])))
    atr = np.mean(tr) if len(tr) > 0 else 1.0
    buf = (close[-1] - low.min()) / atr if atr > 0 else 0
    if buf >= 5.0: return 4.0
    elif buf >= 3.0: return 3.0
    elif buf >= 1.5: return 2.0
    else: return 1.0


def _score_fibonacci(df: pd.DataFrame) -> float:
    """
    NEW: Where is the current price in the 52W fib ladder?
    Above 61.8% of range (golden ratio) = strong uptrend = 4.
    """
    if df is None or len(df) < 50:
        return 2.0
    high52 = float(df["High"].max())
    low52  = float(df["Low"].min())
    curr   = float(df["Close"].iloc[-1])
    span   = high52 - low52
    if span == 0:
        return 2.0
    pos = (curr - low52) / span
    if pos >= 0.618: return 4.0
    elif pos >= 0.382: return 3.0
    elif pos >= 0.236: return 2.0
    else: return 1.0


def _score_cmf(df: pd.DataFrame, period: int = 20) -> float:
    """
    Chaikin Money Flow — buying vs selling pressure.
    Codes/Weighted sheet includes CMF as an oscillator.
    CMF > +0.20 = strong accumulation = 4, < -0.20 = distribution = 1.
    """
    if df is None or len(df) < period:
        return 2.0
    sub   = df.iloc[-period:]
    high  = sub["High"].values.astype(float)
    low   = sub["Low"].values.astype(float)
    close = sub["Close"].values.astype(float)
    vol   = sub["Volume"].values.astype(float)
    hl    = high - low
    hl[hl == 0] = 0.001
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * vol
    cmf = np.sum(mfv) / np.sum(vol) if np.sum(vol) > 0 else 0
    if cmf >= 0.20:    return 4.0
    elif cmf >= 0.05:  return 3.5
    elif cmf >= 0:     return 3.0
    elif cmf >= -0.10: return 2.0
    elif cmf >= -0.20: return 1.5
    else:              return 1.0


def score_rsi(close_series: pd.Series) -> tuple:
    """
    RSI(14) scoring.  Mode controlled by RSI_TROUGH_MODE at top of file.
    Returns (score: float, rsi_value: float, label: str).

    Mode "C" — Combined (Steve's choice): direction is PRIMARY, level is MODIFIER.
    ┌──────────────────────┬─────────────────┬──────────────────────┐
    │ RSI Zone             │ Rising          │ Falling              │
    ├──────────────────────┼─────────────────┼──────────────────────┤
    │ 30–50  (building)    │ 4.0  ← best     │ 2.0  declining       │
    │ <30    (deep oversld)│ 3.5  reversal   │ 1.5  still in freefall│
    │ 50–65  (healthy)     │ 3.5  momentum   │ 2.0  fading          │
    │ 65–75  (extended)    │ 2.5  caution    │ 1.5  rolling over    │
    │ >75    (overbought)  │ 2.0  very risky │ 1.0  ← worst         │
    └──────────────────────┴─────────────────┴──────────────────────┘

    Mode "B" — Trough only: direction is everything, level ignored.
    Mode False — Absolute level only (legacy behavior).
    """
    delta = close_series.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_series = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi   = float(rsi_series.iloc[-1])
    rsi_p = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else rsi
    rising = rsi > rsi_p

    if RSI_TROUGH_MODE == "C":
        # Combined: direction primary, level modifier
        if 30 <= rsi <= 50 and rising:      s = 4.0   # trough reversal — best
        elif rsi < 30 and rising:           s = 3.5   # deep oversold reversal
        elif 50 < rsi <= 65 and rising:     s = 3.5   # healthy uptrend momentum
        elif 65 < rsi <= 75 and rising:     s = 2.5   # extended, proceed with caution
        elif rsi > 75 and rising:           s = 2.0   # very overbought even if rising
        elif 30 <= rsi <= 65 and not rising: s = 2.0  # fading from good zone
        elif rsi < 30 and not rising:       s = 1.5   # still in free-fall
        elif 65 < rsi <= 75 and not rising: s = 1.5   # rolling over from overbought
        else:                               s = 1.0   # overbought and reversing (>75, falling)

        if rsi < 30 and rising:
            label = f"RSI {rsi:.0f} - Trough reversal from oversold"
        elif 30 <= rsi <= 50 and rising:
            label = f"RSI {rsi:.0f} - Building from trough"
        elif rising:
            label = f"RSI {rsi:.0f} - Momentum rising"
        elif rsi > 70:
            label = f"RSI {rsi:.0f} - Overbought, rolling over"
        else:
            label = f"RSI {rsi:.0f} - Momentum fading"

    elif RSI_TROUGH_MODE == "B":
        # Trough only: direction is everything
        if rsi <= 50 and rising:            s = 4.0
        elif 50 < rsi <= 65 and rising:     s = 3.0
        elif rsi > 65 and rising:           s = 2.5
        elif rsi > 50 and not rising:       s = 2.0
        else:                               s = 1.0
        label = ("Trough reversal" if (rsi <= 50 and rising) else
                 "Momentum rising" if rising else "Momentum fading")

    else:
        # Absolute level (legacy default)
        if 30 <= rsi <= 50:      s = 4.0
        elif 20 <= rsi < 30:     s = 3.5
        elif 50 < rsi <= 65:     s = 3.0
        elif 65 < rsi <= 75:     s = 2.0
        else:                    s = 1.0
        label = (f"RSI {rsi:.0f}" +
                 (" (oversold)" if rsi < 30 else
                  " (overbought)" if rsi > 70 else ""))

    return round(s, 2), round(rsi, 1), label


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL — TREND  (Codes sheet Trend parameters, all auto-computed)
# ──────────────────────────────────────────────────────────────────────────────

def score_trend(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> dict:
    """
    Trend sub-category.  Replaces MACD+simple-MA with Codes-sheet parameters:
      - Value of Support       (Codes: Solid/New to New Low)
      - Lack of Resistance     (Codes: New High to Thick/New)
      - 3-Month Trend slope    (Codes: >45deg to Down)
      - 1-Year Trend slope     (Codes: >30deg to Down)
      - Above 50d MA           (Codes: Up+Above to Down+Below)
      - Above 200d MA          (Codes: Up+Above to Down+Below)
      - Volume Trend           (Codes: H Vol Up to H Vol Down)
      - ATR Support Buffer     (NEW: volatility-adjusted cushion)
      - Fibonacci Position     (NEW: where in 52W range)
    Also retains Relative Strength vs SPY as a bonus metric.
    """
    if df is None or len(df) < 20:
        return {"score": 2.0, "detail": {"error": "insufficient data"}}

    scores     = []
    components = {}

    # ── Support & Resistance (from Codes sheet Trend section) ─────────────────
    support_s = _score_value_of_support(df)
    scores.append(support_s)
    components["value_of_support"] = {
        "score": support_s,
        "label": ("Solid/New" if support_s >= 3.5 else
                  "Mixed/Old" if support_s >= 2.5 else
                  "Thin/Lower" if support_s >= 1.5 else "New Low")
    }

    resist_s = _score_lack_of_resistance(df)
    scores.append(resist_s)
    components["lack_of_resistance"] = {
        "score": resist_s,
        "label": ("New High" if resist_s >= 3.5 else
                  "Thin Resist" if resist_s >= 2.5 else
                  "Thick/Old" if resist_s >= 1.5 else "Thick/New")
    }

    # ── Trend Slopes ──────────────────────────────────────────────────────────
    t3mo = _score_slope_trend(df, 63)
    scores.append(t3mo)
    pct_3mo = ((df["Close"].iloc[-1] / df["Close"].iloc[-min(63, len(df))] - 1)
               * 100 if len(df) >= 5 else 0)
    components["trend_3mo"] = {
        "score": t3mo, "pct_change": round(pct_3mo, 1),
        "label": f"3Mo: {pct_3mo:+.1f}%"
    }

    t1yr = _score_slope_trend(df, 252)
    scores.append(t1yr)
    pct_1yr = ((df["Close"].iloc[-1] / df["Close"].iloc[-min(252, len(df))] - 1)
               * 100 if len(df) >= 5 else 0)
    components["trend_1yr"] = {
        "score": t1yr, "pct_change": round(pct_1yr, 1),
        "label": f"1Yr: {pct_1yr:+.1f}%"
    }

    # ── Moving Averages ───────────────────────────────────────────────────────
    ma50_s = _score_ma_position(df, 50)
    scores.append(ma50_s)
    ma50_v = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else float(df["Close"].mean())
    components["ma_50"] = {
        "score": ma50_s, "ma_value": round(ma50_v, 2),
        "label": f"50MA: {ma50_v:.2f}"
    }

    ma200_s = _score_ma_position(df, 200)
    scores.append(ma200_s)
    ma200_v = float(df["Close"].rolling(200).mean().iloc[-1]) if len(df) >= 200 else float(df["Close"].mean())
    components["ma_200"] = {
        "score": ma200_s, "ma_value": round(ma200_v, 2),
        "label": f"200MA: {ma200_v:.2f}"
    }

    # ── Volume Trend ──────────────────────────────────────────────────────────
    vol_s = _score_volume_trend(df)
    scores.append(vol_s)
    components["volume_trend"] = {"score": vol_s}

    # ── ATR Support Buffer (NEW) ──────────────────────────────────────────────
    atr_s = _score_atr_buffer(df)
    scores.append(atr_s)
    components["atr_support_buffer"] = {"score": atr_s, "label": "NEW"}

    # ── Fibonacci Position (NEW) ──────────────────────────────────────────────
    fib_s = _score_fibonacci(df)
    scores.append(fib_s)
    curr_p = float(df["Close"].iloc[-1])
    low52  = float(df["Low"].min())
    high52 = float(df["High"].max())
    fib_pos = (curr_p - low52) / (high52 - low52) * 100 if high52 > low52 else 50
    components["fibonacci"] = {
        "score": fib_s, "position_pct": round(fib_pos, 1),
        "label": f"Fib {fib_pos:.0f}% of 52W range  [NEW]"
    }

    # ── Relative Strength vs SPY (bonus, not in Codes sheet) ─────────────────
    if spy_df is not None and len(spy_df) >= 21 and len(df) >= 21:
        try:
            close = df["Close"]
            sym_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100
            spy_1m = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-21] - 1) * 100
            rel_1m = sym_1m - spy_1m
            sym_3m = (close.iloc[-1] / close.iloc[-min(63, len(close))] - 1) * 100
            spy_3m = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-min(63, len(spy_df))] - 1) * 100
            rel_3m = sym_3m - spy_3m
            rs = rel_1m * 0.6 + rel_3m * 0.4
            rs_s = 4.0 if rs >= 5 else (3.0 if rs >= 0 else (2.0 if rs >= -5 else 1.0))
            scores.append(rs_s)
            components["rel_strength_spy"] = {
                "score": rs_s, "vs_spy_1m": round(rel_1m, 2),
                "label": f"vs SPY 1M: {rel_1m:+.1f}%"
            }
        except Exception:
            pass

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL — OSCILLATORS  (RSI, Stochastics, CMF, ATR)
# ──────────────────────────────────────────────────────────────────────────────

def score_oscillators(df: pd.DataFrame) -> dict:
    """
    Oscillators sub-category.
    Codes/Weighted sheet: RSI trough, Stochastics trough, MACD trough, CMF.
    RSI mode: controlled by RSI_TROUGH_MODE flag (pending Steve's decision).
    CMF added.  ATR retained as volatility quality filter.
    """
    if df is None or len(df) < 14:
        return {"score": 2.0, "detail": {"error": "insufficient data"}}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    scores = []
    components = {}

    # ── RSI(14) — mode set by RSI_TROUGH_MODE ────────────────────────────────
    rsi_s, rsi_v, rsi_label = score_rsi(close)
    scores.append(rsi_s)
    components["rsi"] = {
        "value": rsi_v, "score": rsi_s, "label": rsi_label,
        "mode": "trough" if RSI_TROUGH_MODE else "absolute_level"
    }

    # ── Stochastics %K (14,3) — trough = Buy (matches Codes sheet) ───────────
    if len(high) >= 14:
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k_raw  = (close - low14) / (high14 - low14).replace(0, np.nan) * 100
        k_pct  = float(k_raw.iloc[-1])
        k_prev = float(k_raw.iloc[-2])
        rising = k_pct > k_prev
        # Trough = Buy: below 20 and rising = best signal
        if k_pct < 20 and rising:    stoch_s = 4.0
        elif k_pct < 50 and rising:  stoch_s = 3.0
        elif k_pct < 80:             stoch_s = 2.0
        else:                        stoch_s = 1.0
        scores.append(stoch_s)
        components["stochastics"] = {
            "k_pct": round(k_pct, 1), "rising": rising, "score": stoch_s,
            "label": f"Stoch %K {k_pct:.0f} ({'rising' if rising else 'falling'})"
        }

    # ── CMF (Chaikin Money Flow) — from Codes Weighted sheet ─────────────────
    cmf_s = _score_cmf(df)
    scores.append(cmf_s)
    # Compute raw CMF value for display
    if len(df) >= 20:
        sub   = df.iloc[-20:]
        h_arr = sub["High"].values.astype(float)
        l_arr = sub["Low"].values.astype(float)
        c_arr = sub["Close"].values.astype(float)
        v_arr = sub["Volume"].values.astype(float)
        hl    = h_arr - l_arr
        hl[hl == 0] = 0.001
        mfm  = ((c_arr - l_arr) - (h_arr - c_arr)) / hl
        cmf_raw = float(np.sum(mfm * v_arr) / np.sum(v_arr)) if np.sum(v_arr) > 0 else 0
    else:
        cmf_raw = 0.0
    components["cmf"] = {
        "value": round(cmf_raw, 3), "score": cmf_s,
        "label": (f"CMF {cmf_raw:+.3f} " +
                  ("accumulation" if cmf_raw > 0.1 else
                   "distribution" if cmf_raw < -0.1 else "neutral"))
    }

    # ── MACD (trough = Buy, from Weighted sheet) ──────────────────────────────
    if len(close) >= 26:
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        mv, sv = float(macd.iloc[-1]), float(signal.iloc[-1])
        mp, sp = float(macd.iloc[-2]), float(signal.iloc[-2])
        crossover = (mp <= sp) and (mv > sv)   # bullish trough crossover
        if crossover and mv > 0:   macd_s = 4.0
        elif mv > 0 and mv > sv:   macd_s = 3.0
        elif mv > 0:               macd_s = 2.5
        elif crossover:            macd_s = 2.0   # crossing up but still negative
        else:                      macd_s = 1.0
        scores.append(macd_s)
        components["macd"] = {
            "value": round(mv, 3), "crossover": crossover, "score": macd_s,
            "label": ("Bullish crossover" if crossover else
                      ("MACD+/above signal" if (mv > 0 and mv > sv) else
                       ("MACD+" if mv > 0 else "MACD negative")))
        }

    # ── ATR % (volatility quality filter — prefer moderate 1-4%) ─────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_pct = float(tr.rolling(14).mean().iloc[-1] / close.iloc[-1]) * 100
    if 1.0 <= atr_pct <= 4.0:       atr_s = 4.0
    elif 4.0 < atr_pct <= 6.0:      atr_s = 3.0
    elif 0.5 <= atr_pct < 1.0:      atr_s = 3.0
    elif 6.0 < atr_pct <= 8.0:      atr_s = 2.0
    elif atr_pct < 0.5:             atr_s = 2.0
    else:                           atr_s = 1.0
    scores.append(atr_s)
    components["atr_volatility"] = {
        "value": round(atr_pct, 2), "score": atr_s,
        "label": f"ATR {atr_pct:.1f}% of price"
    }

    final = round(np.mean(scores), 3) if scores else 2.0
    return {"score": final, "detail": components}


def score_technical_full(df: pd.DataFrame, spy_df: pd.DataFrame,
                          sub_weights: dict) -> dict:
    trend_result = score_trend(df, spy_df)
    osc_result   = score_oscillators(df)

    w_trend = sub_weights.get("trend",       0.60)
    w_osc   = sub_weights.get("oscillators", 0.40)

    final = trend_result["score"] * w_trend + osc_result["score"] * w_osc
    return {
        "score":       round(final, 3),
        "sub_weights": {"trend": w_trend, "oscillators": w_osc},
        "sub_scores":  {"trend": trend_result, "oscillators": osc_result},
    }


# ──────────────────────────────────────────────────────────────────────────────
# MASTER GPA ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class GPAEngine:
    """
    Orchestrates Sentiment / Fundamentals / Technical into a 0.0-4.0 GPA.
    Weights come from a strategy dict (see strategy_manager.py).
    API keys are passed in via set_api_keys() — called once at app startup.
    """

    def __init__(self, strategy: dict = None):
        from modules.strategy_manager import DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS
        if strategy is None:
            strategy = {"weights": DEFAULT_WEIGHTS,
                        "thresholds": DEFAULT_THRESHOLDS,
                        "auto_trade": False}
        self.weights    = strategy.get("weights",    DEFAULT_WEIGHTS)
        self.thresholds = strategy.get("thresholds", DEFAULT_THRESHOLDS)
        self.auto_trade = strategy.get("auto_trade", False)
        self._spy_df    = None
        self._av_key    = ""    # set via set_api_keys()

    def set_spy_df(self, spy_df: pd.DataFrame):
        self._spy_df = spy_df

    def set_api_keys(self, av_key: str = "", **_):
        """Pass in the Alpha Vantage API key (and any other future keys)."""
        self._av_key = av_key or ""

    def score(self, symbol: str, ohlcv_df: pd.DataFrame,
              fundamentals: dict, sentiment: dict) -> dict:
        """
        Returns full GPA report for one stock.
        sentiment dict may contain pre-fetched AV/StockTwits data,
        or the engine will fetch live if av_key is set.
        """
        # ── Enrich sentiment if keys available and data not yet fetched ───────
        if self._av_key and not sentiment.get("av_ok"):
            av_data = fetch_alpha_vantage_sentiment(symbol, self._av_key)
            sentiment = {**sentiment, **av_data}
            time.sleep(0.05)

        if not sentiment.get("st_ok"):
            st_data = fetch_stocktwits_sentiment(symbol)
            sentiment = {**sentiment, **st_data}

        w = self.weights
        fund_sub = w.get("fund_sub", {"valuation": 0.33, "financial": 0.33,
                                       "estimates": 0.34})
        tech_sub = w.get("tech_sub", {"trend": 0.60, "oscillators": 0.40})

        sent_result = score_sentiment(sentiment)
        fund_result = score_fundamentals_full(fundamentals, fund_sub)
        tech_result = score_technical_full(ohlcv_df, self._spy_df, tech_sub)

        w_sent = w.get("sentiment",    0.20)
        w_fund = w.get("fundamentals", 0.45)
        w_tech = w.get("technical",    0.35)

        gpa = round(sent_result["score"] * w_sent +
                    fund_result["score"] * w_fund +
                    tech_result["score"] * w_tech, 3)
        grade = _grade(gpa)

        t           = self.thresholds
        buy_signal  = gpa >= t.get("min_gpa_to_buy",  3.5)
        sell_signal = gpa <= t.get("max_gpa_to_sell", 2.5)

        category_contributions = {
            "Sentiment":    sent_result["score"] * w_sent,
            "Fundamentals": fund_result["score"] * w_fund,
            "Technical":    tech_result["score"] * w_tech,
        }
        sorted_cats = sorted(category_contributions.items(),
                             key=lambda x: x[1], reverse=True)
        top_drivers = [c[0] for c in sorted_cats[:2]]
        detractors  = [c[0] for c in sorted_cats
                       if c[1] < (gpa * min(w_sent, w_fund, w_tech))]

        return {
            "symbol":      symbol,
            "gpa":         gpa,
            "grade":       grade,
            "buy_signal":  buy_signal,
            "sell_signal": sell_signal,
            "categories": {
                "sentiment": {
                    "score":        sent_result["score"],
                    "weight":       w_sent,
                    "contribution": round(sent_result["score"] * w_sent, 3),
                    "detail":       sent_result["detail"],
                },
                "fundamentals": {
                    "score":        fund_result["score"],
                    "weight":       w_fund,
                    "contribution": round(fund_result["score"] * w_fund, 3),
                    "sub_weights":  fund_result["sub_weights"],
                    "sub_scores":   fund_result["sub_scores"],
                },
                "technical": {
                    "score":        tech_result["score"],
                    "weight":       w_tech,
                    "contribution": round(tech_result["score"] * w_tech, 3),
                    "sub_weights":  tech_result["sub_weights"],
                    "sub_scores":   tech_result["sub_scores"],
                },
            },
            "top_drivers":  top_drivers,
            "detractors":   detractors,
            "weights_used": w,
            "thresholds":   t,
            "timestamp":    pd.Timestamp.now().isoformat(),
        }
