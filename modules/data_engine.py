"""
data_engine.py — Market data, news, and sentiment ingestion
Supports: Alpaca (primary), yfinance (fallback), Reddit, NewsAPI, RSS feeds
"""

import time
import logging
import feedparser
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

log = logging.getLogger("DataEngine")

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_DATA_AVAILABLE = True
except ImportError:
    ALPACA_DATA_AVAILABLE = False


# ==============================================================================
# MARKET DATA
# ==============================================================================

class MarketDataEngine:
    """Fetches OHLCV, fundamentals, beta, market cap."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.alpaca_client = None
        if ALPACA_DATA_AVAILABLE and cfg["data_sources"]["primary_market_data"] == "alpaca":
            try:
                self.alpaca_client = StockHistoricalDataClient(
                    api_key=cfg["alpaca"]["api_key"],
                    secret_key=cfg["alpaca"]["secret_key"],
                )
                log.info("Alpaca market data client initialized")
            except Exception as e:
                log.warning(f"Alpaca data client failed: {e}, falling back to yfinance")

    def get_bars(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """Returns daily OHLCV DataFrame."""
        if self.alpaca_client:
            try:
                return self._alpaca_bars(symbol, days)
            except Exception as e:
                log.debug(f"Alpaca bars failed for {symbol}: {e}, trying yfinance")

        if YF_AVAILABLE:
            return self._yf_bars(symbol, days)
        return None

    def _alpaca_bars(self, symbol: str, days: int) -> pd.DataFrame:
        end   = datetime.now()
        start = end - timedelta(days=days + 30)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = self.alpaca_client.get_stock_bars(req)
        df = bars.df
        if hasattr(df.index, 'levels'):  # multi-index
            df = df.xs(symbol, level='symbol') if symbol in df.index.get_level_values('symbol') else df
        df.index = pd.to_datetime(df.index)
        df.rename(columns={"open":"Open","high":"High","low":"Low",
                            "close":"Close","volume":"Volume"}, inplace=True)
        return df.tail(days)

    def _yf_bars(self, symbol: str, days: int) -> pd.DataFrame:
        period = f"{max(days + 30, 90)}d"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        return df.tail(days) if not df.empty else None

    def get_fundamentals(self, symbol: str) -> dict:
        """Returns fundamentals dict using yfinance (free tier sufficient)."""
        result = {
            "roe": None, "earnings_growth_yoy": None, "debt_equity": None,
            "pe_ttm": None, "peg": None, "market_cap": None, "beta": None,
            "dividend_yield": None, "sector": None, "industry": None,
            "eps_growth_5y": None, "revenue_growth": None,
            # Extended fields for new GPA model
            "enterprise_to_ebitda": None, "analyst_recommendation": None,
            "earnings_estimate_current_year": None, "earnings_estimate_next_year": None,
            "five_year_avg_growth": None, "beat_expectations": None,
            "target_price": None, "current_price": None,
        }
        if not YF_AVAILABLE:
            return result
        try:
            ticker = yf.Ticker(symbol)
            info   = ticker.info
            result.update({
                "roe":               info.get("returnOnEquity"),
                "earnings_growth_yoy": info.get("earningsGrowth"),
                "debt_equity":       info.get("debtToEquity"),
                "pe_ttm":            info.get("trailingPE"),
                "peg":               info.get("pegRatio"),
                "market_cap":        info.get("marketCap"),
                "beta":              info.get("beta"),
                "dividend_yield":    info.get("dividendYield"),
                "sector":            info.get("sector"),
                "industry":          info.get("industry"),
                "eps_growth_5y":     info.get("earningsQuarterlyGrowth"),
                "revenue_growth":    info.get("revenueGrowth"),
                "forward_pe":        info.get("forwardPE"),
                "name":              info.get("longName", symbol),
                # Extended
                "enterprise_to_ebitda":           info.get("enterpriseToEbitda"),
                "analyst_recommendation":          info.get("recommendationMean"),
                "five_year_avg_growth":            info.get("earningsGrowth", 0) * 100
                                                   if info.get("earningsGrowth") else None,
                "target_price":                    info.get("targetMeanPrice"),
                "current_price":                   info.get("currentPrice") or info.get("regularMarketPrice"),
                "earnings_estimate_current_year":  self._safe_est_growth(info, "currentYear"),
                "earnings_estimate_next_year":     self._safe_est_growth(info, "nextYear"),
            })
            # Beat expectations: % of last 4 quarters where EPS beat estimate
            try:
                hist = ticker.earnings_history
                if hist is not None and not hist.empty and len(hist) >= 2:
                    beats = (hist["epsActual"] > hist["epsEstimate"]).sum()
                    result["beat_expectations"] = round(beats / len(hist), 2)
            except Exception:
                pass
        except Exception as e:
            log.debug(f"Fundamentals failed for {symbol}: {e}")
        return result

    @staticmethod
    def _safe_est_growth(info: dict, period: str) -> float:
        """Try to extract forward earnings growth % from yfinance info."""
        try:
            # yfinance sometimes has these in earningsGrowth or forwardEpsGrowth
            if period == "currentYear":
                v = info.get("earningsGrowth") or info.get("revenueGrowth")
            else:
                v = info.get("forwardEpsGrowth") or info.get("earningsGrowth")
            return round(float(v) * 100, 1) if v is not None else None
        except Exception:
            return None

    def get_snapshot(self, symbol: str) -> dict:
        """Gets current price snapshot."""
        if self.alpaca_client:
            try:
                req  = StockSnapshotRequest(symbol_or_symbols=symbol)
                snap = self.alpaca_client.get_stock_snapshot(req)
                s    = snap[symbol]
                return {
                    "price": s.latest_trade.price,
                    "volume": s.daily_bar.volume,
                    "prev_close": s.previous_daily_bar.close,
                    "change_pct": (s.latest_trade.price / s.previous_daily_bar.close - 1) * 100,
                }
            except Exception as e:
                log.debug(f"Alpaca snapshot failed: {e}")

        if YF_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                hist   = ticker.history(period="2d")
                if len(hist) >= 2:
                    price      = hist["Close"].iloc[-1]
                    prev_close = hist["Close"].iloc[-2]
                    return {
                        "price": price,
                        "volume": hist["Volume"].iloc[-1],
                        "prev_close": prev_close,
                        "change_pct": (price / prev_close - 1) * 100,
                    }
            except Exception as e:
                log.debug(f"yfinance snapshot failed: {e}")
        return {"price": 0, "volume": 0, "prev_close": 0, "change_pct": 0}


# ==============================================================================
# SENTIMENT ENGINE
# ==============================================================================

class SentimentEngine:
    """Fetches and scores news + social media sentiment."""

    NEWS_RSS = [
        "https://finance.yahoo.com/rss/topstories",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
        "https://feeds.marketwatch.com/marketwatch/topstories",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    ]

    BULLISH_TERMS = {
        "beat", "beats", "surge", "surges", "rally", "soar", "record high",
        "strong earnings", "upgrade", "upgraded", "buy rating", "outperform",
        "profit growth", "raised guidance", "breakout", "momentum", "beat estimates",
        "earnings beat", "dividend increase", "share buyback", "new contract",
        "partnership", "acquisition", "market share", "growth accelerates",
    }
    BEARISH_TERMS = {
        "miss", "misses", "decline", "crash", "downgrade", "downgraded",
        "layoffs", "guidance cut", "lowered guidance", "loss", "debt concern",
        "investigation", "lawsuit", "recall", "miss estimates", "sell rating",
        "underperform", "guidance miss", "revenue miss", "profit warning",
        "restructuring", "write-down", "impairment", "default",
    }

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.reddit = None
        self._init_reddit()

    def _init_reddit(self):
        if not REDDIT_AVAILABLE:
            return
        try:
            self.reddit = praw.Reddit(
                client_id=self.cfg["data_sources"]["reddit_client_id"],
                client_secret=self.cfg["data_sources"]["reddit_client_secret"],
                user_agent=self.cfg["data_sources"]["reddit_user_agent"],
            )
            log.info("Reddit API initialized")
        except Exception as e:
            log.warning(f"Reddit init failed: {e}")

    def score_text(self, text: str) -> float:
        """Returns sentiment in [-1.0, +1.0]. Uses VADER if available."""
        if VADER_AVAILABLE:
            return vader.polarity_scores(text)["compound"]
        # Fallback: keyword scoring
        t = text.lower()
        bull = sum(1 for w in self.BULLISH_TERMS if w in t)
        bear = sum(1 for w in self.BEARISH_TERMS if w in t)
        total = bull + bear
        if total == 0:
            return 0.0
        return (bull - bear) / total

    def get_news_sentiment(self, symbol: str) -> dict:
        """Fetches headlines and scores them. Returns aggregated result."""
        scores, headlines, velocity = [], [], []

        # Yahoo Finance ticker RSS
        try:
            url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                score = self.score_text(title + " " + entry.get("summary", ""))
                pub   = entry.get("published_parsed")
                headlines.append({"title": title, "score": score})
                scores.append(score)
                # Recency bonus: articles in last 4 hours get double weight
                if pub:
                    pub_dt = datetime(*pub[:6])
                    hours_old = (datetime.now() - pub_dt).total_seconds() / 3600
                    if hours_old < 4:
                        scores.append(score)  # count twice
        except Exception as e:
            log.debug(f"Yahoo RSS failed for {symbol}: {e}")

        # NewsAPI (if key provided)
        news_api_key = self.cfg["data_sources"].get("news_api_key", "")
        if news_api_key and "YOUR_" not in news_api_key:
            try:
                url = (
                    f"https://newsapi.org/v2/everything"
                    f"?q={symbol}&sortBy=publishedAt&pageSize=10"
                    f"&apiKey={news_api_key}"
                )
                resp = requests.get(url, timeout=5)
                if resp.ok:
                    for article in resp.json().get("articles", [])[:8]:
                        text  = (article.get("title") or "") + " " + (article.get("description") or "")
                        score = self.score_text(text)
                        headlines.append({"title": article.get("title", ""), "score": score})
                        scores.append(score)
            except Exception as e:
                log.debug(f"NewsAPI failed: {e}")

        # Compute weighted average and velocity
        avg   = float(np.mean(scores)) if scores else 0.0
        stdev = float(np.std(scores))  if len(scores) > 1 else 0.0
        # Velocity: difference between recent vs older scores
        if len(scores) >= 4:
            recent = np.mean(scores[:len(scores)//2])
            older  = np.mean(scores[len(scores)//2:])
            velocity_score = float(recent - older)
        else:
            velocity_score = 0.0

        return {
            "avg_score":      avg,
            "velocity":       velocity_score,
            "stdev":          stdev,
            "sample_count":   len(scores),
            "top_headlines":  headlines[:5],
            "raw_scores":     scores,
        }

    def get_reddit_sentiment(self, symbol: str) -> dict:
        """Scans WSB and r/stocks for mentions and sentiment."""
        if not self.reddit:
            return {"avg_score": 0.0, "mention_count": 0, "velocity": 0.0}
        scores, mention_count = [], 0
        try:
            for sub in ["wallstreetbets", "stocks", "investing"]:
                subreddit = self.reddit.subreddit(sub)
                for submission in subreddit.search(symbol, sort="new", time_filter="day", limit=15):
                    text  = submission.title + " " + (submission.selftext or "")[:200]
                    score = self.score_text(text)
                    scores.append(score)
                    mention_count += 1
        except Exception as e:
            log.debug(f"Reddit failed for {symbol}: {e}")
        return {
            "avg_score":    float(np.mean(scores)) if scores else 0.0,
            "mention_count": mention_count,
            "velocity":     float(np.std(scores)) if len(scores) > 1 else 0.0,
        }

    def get_combined_sentiment(self, symbol: str) -> dict:
        """Merges news + reddit into a single sentiment package."""
        news   = self.get_news_sentiment(symbol)
        reddit = self.get_reddit_sentiment(symbol)
        # Weight: 60% news, 40% reddit (reddit can be noisy)
        combined = (news["avg_score"] * 0.60) + (reddit["avg_score"] * 0.40)
        velocity = (news["velocity"] * 0.70) + (reddit["velocity"] * 0.30)
        return {
            "combined_score":  combined,
            "velocity":        velocity,
            "news_score":      news["avg_score"],
            "reddit_score":    reddit["avg_score"],
            "news_count":      news["sample_count"],
            "reddit_mentions": reddit["mention_count"],
            "top_headlines":   news["top_headlines"],
        }


# ==============================================================================
# TRENDING STOCK SCANNER
# ==============================================================================

class TrendingScanner:
    """Identifies 10-15 trending stocks via news frequency and social velocity."""

    TRENDING_RSS = [
        "https://finance.yahoo.com/rss/topstories",
        "https://feeds.marketwatch.com/marketwatch/topstories",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.reuters.com/reuters/businessNews",
    ]

    def __init__(self, seed_tickers: list):
        self.seeds = seed_tickers

    def scan_rss_mentions(self) -> dict:
        """Counts how many times each known ticker appears in top headlines."""
        mention_counts = {t: 0 for t in self.seeds}
        for url in self.TRENDING_RSS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    text = (entry.get("title","") + " " + entry.get("summary","")).upper()
                    for ticker in self.seeds:
                        # Match whole word (avoid false positives)
                        if f" {ticker} " in f" {text} " or f"${ticker}" in text:
                            mention_counts[ticker] += 1
            except Exception as e:
                log.debug(f"RSS scan failed for {url}: {e}")
        return mention_counts

    def get_unusual_volume(self, tickers: list, data_engine: MarketDataEngine) -> dict:
        """Returns volume ratio (today vs 20-day avg) for each ticker."""
        vol_ratios = {}
        for ticker in tickers:
            try:
                df = data_engine.get_bars(ticker, days=25)
                if df is not None and len(df) >= 5:
                    avg_vol = df["Volume"].iloc[:-1].mean()
                    today_vol = df["Volume"].iloc[-1]
                    vol_ratios[ticker] = today_vol / avg_vol if avg_vol > 0 else 1.0
            except Exception:
                vol_ratios[ticker] = 1.0
        return vol_ratios

    def get_top_trending(self, data_engine: MarketDataEngine, n: int = 15) -> list:
        """Returns top N tickers by combined trending score."""
        mentions  = self.scan_rss_mentions()
        vol_ratios = self.get_unusual_volume(self.seeds, data_engine)

        scores = {}
        for t in self.seeds:
            m = mentions.get(t, 0)
            v = vol_ratios.get(t, 1.0)
            # Trending score: mentions * 2 + log(vol_ratio) capped
            scores[t] = m * 2 + min(np.log(v + 0.01) * 3, 5)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top    = [t for t, s in ranked[:n]]
        log.info(f"Top trending: {top[:8]}")
        return top


# ==============================================================================
# DATAENGINE — unified wrapper used by app.py
# ==============================================================================

class DataEngine:
    """
    Convenience wrapper so app.py can do:
        de = DataEngine()
        de.get_fundamentals(symbol)
        de.get_sentiment(symbol)
    Loads config automatically; falls back gracefully if config is missing.
    """

    _DEFAULT_CFG = {
        "data_sources": {
            "primary_market_data": "yfinance",
            "reddit_client_id":     "",
            "reddit_client_secret": "",
            "reddit_user_agent":    "TaylorTradingAgent/1.0",
        },
        "alpaca": {"api_key": "", "secret_key": ""},
    }

    def __init__(self):
        cfg_path = Path(__file__).parent.parent / "config.json"
        try:
            import json
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            # Merge with defaults so missing keys don't cause crashes
            self._cfg = {**self._DEFAULT_CFG, **cfg}
            # Ensure nested keys exist
            self._cfg.setdefault("data_sources", self._DEFAULT_CFG["data_sources"])
            self._cfg.setdefault("alpaca",       self._DEFAULT_CFG["alpaca"])
        except Exception:
            self._cfg = self._DEFAULT_CFG

        self._market = MarketDataEngine(self._cfg)
        self._sentiment = SentimentEngine(self._cfg)

    def get_fundamentals(self, symbol: str) -> dict:
        return self._market.get_fundamentals(symbol)

    def get_sentiment(self, symbol: str) -> dict:
        return self._sentiment.get_combined_sentiment(symbol)

    def get_bars(self, symbol: str, days: int = 90):
        return self._market.get_bars(symbol, days)
