"""
backtester.py — Historical backtesting with GPA scoring and S&P 500 benchmark
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("Backtester")


class Backtester:
    """
    Runs a simplified event-driven backtest using historical OHLCV data.
    Simulates the GPA scoring logic at daily granularity.
    """

    def __init__(self, cfg: dict, gpa_engine, data_engine):
        self.cfg         = cfg
        self.gpa_engine  = gpa_engine
        self.data_engine = data_engine

    def run(self, symbols: list, lookback_days: int = 252,
            initial_capital: float = 14516.50) -> dict:
        """
        Backtests strategy over lookback_days.
        Returns performance metrics and per-trade log.
        """
        log.info(f"Starting backtest: {len(symbols)} symbols, {lookback_days} days")

        # Pull historical data
        all_data = {}
        spy_df   = self.data_engine.get_bars("SPY", days=lookback_days + 60)
        if spy_df is None:
            log.error("Could not fetch SPY data for backtest")
            return {}

        self.gpa_engine.set_spy_df(spy_df)

        for sym in symbols:
            df = self.data_engine.get_bars(sym, days=lookback_days + 60)
            if df is not None and len(df) > 60:
                all_data[sym] = df

        if not all_data:
            log.error("No data available for backtest")
            return {}

        # Simulation
        capital     = initial_capital
        cash        = initial_capital
        positions   = {}   # symbol -> {shares, entry, entry_date}
        trades      = []
        equity_curve = []

        stop_pct   = self.cfg["risk"]["stop_loss_pct"] / 100
        target_pct = self.cfg["risk"]["take_profit_pct"] / 100
        pos_pct    = self.cfg["risk"]["position_size_pct"] / 100
        min_gpa    = self.cfg["trading"]["min_gpa_to_buy"]
        max_gpa_sell = self.cfg["trading"]["max_gpa_to_sell"]

        # Simulate day by day (using 20-day windows for scoring)
        dates = sorted(spy_df.index)[-lookback_days:]

        for i, date in enumerate(dates):
            if i < 20:  # need history for indicators
                equity_curve.append({"date": str(date.date()), "value": capital})
                continue

            # Update portfolio value
            portfolio_value = cash
            for sym, pos in list(positions.items()):
                sym_df = all_data.get(sym)
                if sym_df is None:
                    continue
                sym_dates = sym_df.index[sym_df.index <= date]
                if len(sym_dates) == 0:
                    continue
                current_price = sym_df.loc[sym_dates[-1], "Close"]
                portfolio_value += pos["shares"] * current_price

                # Check stop-loss
                if current_price <= pos["entry"] * (1 - stop_pct):
                    sell_val = pos["shares"] * current_price
                    cash += sell_val
                    pnl  = (current_price - pos["entry"]) * pos["shares"]
                    trades.append({
                        "date": str(date.date()), "symbol": sym, "action": "SELL",
                        "shares": pos["shares"], "price": current_price,
                        "pnl": round(pnl, 2), "reason": "stop_loss",
                    })
                    del positions[sym]
                    continue

                # Check take-profit
                if current_price >= pos["entry"] * (1 + target_pct):
                    sell_val = pos["shares"] * current_price
                    cash += sell_val
                    pnl  = (current_price - pos["entry"]) * pos["shares"]
                    trades.append({
                        "date": str(date.date()), "symbol": sym, "action": "SELL",
                        "shares": pos["shares"], "price": current_price,
                        "pnl": round(pnl, 2), "reason": "take_profit",
                    })
                    del positions[sym]

            capital = portfolio_value
            equity_curve.append({"date": str(date.date()), "value": round(capital, 2)})

            # Scan for new entries (only on ~20% of days to simulate "medium" frequency)
            if i % 5 != 0:
                continue

            for sym, sym_df in all_data.items():
                if sym in positions:
                    continue
                if cash < capital * pos_pct:
                    continue

                # Get historical slice up to this date
                slice_df = sym_df[sym_df.index <= date].tail(60)
                if len(slice_df) < 20:
                    continue

                # Quick technical score (simplified for backtest speed)
                close  = slice_df["Close"]
                delta  = close.diff()
                gain   = delta.clip(lower=0).rolling(14).mean()
                loss   = (-delta.clip(upper=0)).rolling(14).mean()
                rsi    = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]
                sma20  = close.rolling(20).mean().iloc[-1]
                price  = close.iloc[-1]
                ret5d  = (price / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0

                # Simplified GPA proxy (0-4)
                tech = 0
                if 30 <= rsi <= 65: tech += 2
                if price > sma20:   tech += 1
                if ret5d > 2:       tech += 1
                # Use tech score as proxy GPA (scale to 0-4)
                proxy_gpa = tech  # 0-4

                if proxy_gpa >= min_gpa - 0.5:  # slightly relaxed for backtest
                    entry_price = price
                    shares = max(1, int((capital * pos_pct) / entry_price))
                    cost   = shares * entry_price
                    if cash >= cost:
                        cash -= cost
                        positions[sym] = {
                            "shares": shares,
                            "entry":  entry_price,
                            "entry_date": str(date.date()),
                        }
                        trades.append({
                            "date": str(date.date()), "symbol": sym, "action": "BUY",
                            "shares": shares, "price": round(entry_price, 2),
                            "pnl": 0, "reason": f"GPA proxy {proxy_gpa:.1f}",
                        })

        # Close all remaining positions at end
        last_date = dates[-1]
        for sym, pos in positions.items():
            sym_df = all_data.get(sym)
            if sym_df is not None:
                sym_dates = sym_df.index[sym_df.index <= last_date]
                if len(sym_dates) > 0:
                    price = sym_df.loc[sym_dates[-1], "Close"]
                    pnl   = (price - pos["entry"]) * pos["shares"]
                    trades.append({
                        "date": str(last_date.date()), "symbol": sym, "action": "SELL",
                        "shares": pos["shares"], "price": round(price, 2),
                        "pnl": round(pnl, 2), "reason": "end_of_backtest",
                    })

        return self._compute_metrics(
            equity_curve, trades, initial_capital, spy_df, lookback_days
        )

    def _compute_metrics(self, equity_curve: list, trades: list,
                         initial_capital: float, spy_df: pd.DataFrame,
                         lookback_days: int) -> dict:
        if not equity_curve:
            return {}

        eq_values = [e["value"] for e in equity_curve]
        final_val = eq_values[-1]
        total_ret = (final_val / initial_capital - 1) * 100

        # SPY benchmark return
        spy_ret = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-lookback_days] - 1) * 100 \
                  if len(spy_df) >= lookback_days else 0

        # Daily returns
        daily_rets = pd.Series(eq_values).pct_change().dropna()
        sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) \
                 if daily_rets.std() > 0 else 0

        # Max drawdown
        peak   = pd.Series(eq_values).cummax()
        dd     = (pd.Series(eq_values) / peak - 1) * 100
        max_dd = dd.min()

        # Trade stats
        buy_trades  = [t for t in trades if t["action"] == "BUY"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        pnls        = [t["pnl"] for t in sell_trades if t["pnl"] != 0]
        wins        = [p for p in pnls if p > 0]
        losses      = [p for p in pnls if p <= 0]
        win_rate    = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win     = np.mean(wins)   if wins   else 0
        avg_loss    = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float("inf")

        metrics = {
            "initial_capital":    initial_capital,
            "final_value":        round(final_val, 2),
            "total_return_pct":   round(total_ret, 2),
            "spy_return_pct":     round(spy_ret, 2),
            "alpha_pct":          round(total_ret - spy_ret, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "max_drawdown_pct":   round(max_dd, 2),
            "total_trades":       len(buy_trades),
            "win_rate_pct":       round(win_rate, 2),
            "avg_win":            round(avg_win, 2),
            "avg_loss":           round(avg_loss, 2),
            "profit_factor":      round(profit_factor, 3),
            "equity_curve":       equity_curve[-50:],  # last 50 for dashboard
            "trades":             trades[-30:],
        }

        log.info(f"Backtest complete: return={total_ret:.1f}% | "
                 f"SPY={spy_ret:.1f}% | alpha={total_ret-spy_ret:.1f}% | "
                 f"Sharpe={sharpe:.2f} | win_rate={win_rate:.1f}%")
        return metrics
