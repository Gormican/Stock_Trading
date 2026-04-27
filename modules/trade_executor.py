"""
trade_executor.py — Alpaca paper/live order execution with full logging
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("TradeExecutor")
TRADE_LOG = Path.home() / "taylor_trade_log.json"

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        LimitOrderRequest, StopOrderRequest,
        MarketOrderRequest, TrailingStopOrderRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    ALPACA_OK = True
except ImportError:
    ALPACA_OK = False


class TradeExecutor:
    def __init__(self, cfg: dict, dry_run: bool = False):
        self.dry_run = dry_run or cfg["trading"]["mode"] == "paper"
        self.client  = None
        if ALPACA_OK:
            try:
                self.client = TradingClient(
                    api_key    = cfg["alpaca"]["api_key"],
                    secret_key = cfg["alpaca"]["secret_key"],
                    paper      = True,  # always paper for safety
                )
                acct = self.client.get_account()
                log.info(f"Alpaca connected | status={acct.status} | "
                         f"cash=${float(acct.cash):,.2f}")
            except Exception as e:
                log.error(f"Alpaca connect failed: {e}")

    def get_portfolio_value(self) -> float:
        if not self.client:
            return 14516.50  # fallback to CSV snapshot
        try:
            acct = self.client.get_account()
            return float(acct.portfolio_value)
        except Exception:
            return 0.0

    def get_cash(self) -> float:
        if not self.client:
            return 2376.86
        try:
            return float(self.client.get_account().cash)
        except Exception:
            return 0.0

    def get_positions(self) -> dict:
        """Returns {symbol: {qty, market_value, avg_cost, unrealized_pnl}}"""
        if not self.client:
            return {}
        try:
            positions = self.client.get_all_positions()
            return {
                p.symbol: {
                    "qty":            float(p.qty),
                    "market_value":   float(p.market_value),
                    "avg_cost":       float(p.avg_entry_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pct": float(p.unrealized_plpc) * 100,
                }
                for p in positions
            }
        except Exception as e:
            log.warning(f"get_positions failed: {e}")
            return {}

    # ── Orders ────────────────────────────────────────────────────────────────

    def buy_limit(self, symbol: str, shares: int, price: float,
                  gpa: float = 0, reason: str = "") -> Optional[str]:
        limit = round(price * 1.002, 2)
        self._log_trade("BUY", symbol, shares, limit, gpa, reason)
        if self.dry_run or not self.client:
            log.info(f"  [PAPER] BUY  {shares}x {symbol} @ ${limit:.2f} = "
                     f"${shares*limit:,.2f} | GPA {gpa:.2f}")
            return f"PAPER-BUY-{symbol}"
        try:
            req = LimitOrderRequest(
                symbol=symbol, qty=shares, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, limit_price=limit,
            )
            order = self.client.submit_order(req)
            log.info(f"  BUY  {shares}x {symbol} @ ${limit:.2f} | ID {order.id}")
            return str(order.id)
        except Exception as e:
            log.error(f"  BUY failed {symbol}: {e}")
            return None

    def sell_limit(self, symbol: str, shares: int, price: float,
                   gpa: float = 0, reason: str = "") -> Optional[str]:
        limit = round(price * 0.998, 2)
        self._log_trade("SELL", symbol, shares, limit, gpa, reason)
        if self.dry_run or not self.client:
            log.info(f"  [PAPER] SELL {shares}x {symbol} @ ${limit:.2f} | {reason}")
            return f"PAPER-SELL-{symbol}"
        try:
            req = LimitOrderRequest(
                symbol=symbol, qty=shares, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY, limit_price=limit,
            )
            order = self.client.submit_order(req)
            log.info(f"  SELL {shares}x {symbol} @ ${limit:.2f} | ID {order.id}")
            return str(order.id)
        except Exception as e:
            log.error(f"  SELL failed {symbol}: {e}")
            return None

    def place_stop_loss(self, symbol: str, shares: int, stop: float):
        if self.dry_run or not self.client:
            log.info(f"  [PAPER] STOP-LOSS {symbol} @ ${stop:.2f}")
            return
        try:
            req = StopOrderRequest(
                symbol=symbol, qty=shares, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC, stop_price=round(stop, 2),
            )
            self.client.submit_order(req)
            log.info(f"  Stop-loss set: {symbol} @ ${stop:.2f}")
        except Exception as e:
            log.warning(f"  Stop-loss failed {symbol}: {e}")

    def cancel_all_orders(self):
        if self.client:
            try:
                self.client.cancel_orders()
                log.info("All open orders cancelled")
            except Exception as e:
                log.warning(f"Cancel orders failed: {e}")

    # ── Trade log ─────────────────────────────────────────────────────────────

    def _log_trade(self, action: str, symbol: str, shares: int,
                   price: float, gpa: float, reason: str):
        history = json.loads(TRADE_LOG.read_text()) if TRADE_LOG.exists() else []
        history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action, "symbol": symbol, "shares": shares,
            "price": price, "value": round(shares * price, 2),
            "gpa": gpa, "reason": reason,
            "mode": "paper" if self.dry_run else "live",
        })
        TRADE_LOG.write_text(json.dumps(history, indent=2))

    def get_trade_history(self) -> list:
        if TRADE_LOG.exists():
            try:
                return json.loads(TRADE_LOG.read_text())
            except Exception:
                return []
        return []
