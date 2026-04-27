"""
risk_manager.py — Position sizing, stop-loss, kill switch, sector limits
"""
import json
import logging
from datetime import datetime, date
from pathlib import Path

log = logging.getLogger("RiskManager")

STATE_FILE = Path.home() / "taylor_risk_state.json"


class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg              = cfg
        self.portfolio_value  = 0.0
        self.daily_start_value = 0.0
        self.open_positions   = {}   # symbol -> {qty, entry_price, sector}
        self.sector_exposure  = {}   # sector -> total $ value
        self.daily_pnl        = 0.0
        self.killed           = False
        self._load_state()

    # ── State persistence ─────────────────────────────────────────────────────

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                s = json.loads(STATE_FILE.read_text())
                if s.get("date") == str(date.today()):
                    self.daily_pnl        = s.get("daily_pnl", 0.0)
                    self.daily_start_value = s.get("daily_start_value", 0.0)
                    self.killed           = s.get("killed", False)
                    self.open_positions   = s.get("open_positions", {})
                    self.sector_exposure  = s.get("sector_exposure", {})
            except Exception as e:
                log.debug(f"State load failed: {e}")

    def save_state(self):
        state = {
            "date":              str(date.today()),
            "daily_pnl":         self.daily_pnl,
            "daily_start_value": self.daily_start_value,
            "killed":            self.killed,
            "open_positions":    self.open_positions,
            "sector_exposure":   self.sector_exposure,
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def update_portfolio_value(self, value: float):
        self.portfolio_value = value
        if self.daily_start_value == 0:
            self.daily_start_value = value

    # ── Kill switch ──────────────────────────────────────────────────────────

    def check_kill_switch(self) -> bool:
        """Returns True if daily loss limit exceeded → halt all trading."""
        if self.killed:
            log.warning("KILL SWITCH ACTIVE — trading halted for today")
            return True
        if self.daily_start_value > 0 and self.portfolio_value > 0:
            daily_loss_pct = (self.portfolio_value / self.daily_start_value - 1) * 100
            limit = -abs(self.cfg["risk"]["daily_loss_limit_pct"])
            if daily_loss_pct <= limit:
                self.killed = True
                self.save_state()
                log.critical(
                    f"KILL SWITCH TRIGGERED: daily loss {daily_loss_pct:.2f}% "
                    f"exceeded limit {limit:.2f}%"
                )
                return True
        return False

    # ── Position sizing ───────────────────────────────────────────────────────

    def position_size_shares(self, price: float, risk_slider: float = None) -> int:
        """
        risk_slider overrides config when provided (1.0 = min, 5.0 = max).
        Returns whole number of shares.
        """
        if price <= 0:
            return 0
        pct = risk_slider if risk_slider else self.cfg["risk"]["position_size_pct"]
        pct = max(1.0, min(5.0, pct)) / 100
        dollars = self.portfolio_value * pct
        # Cap at available cash minus 5% buffer
        dollars = min(dollars, self.portfolio_value * 0.95)
        import math
        return max(1, math.floor(dollars / price))

    # ── Stop loss / take profit ───────────────────────────────────────────────

    def stop_price(self, entry: float) -> float:
        return round(entry * (1 - self.cfg["risk"]["stop_loss_pct"] / 100), 2)

    def take_profit_price(self, entry: float) -> float:
        return round(entry * (1 + self.cfg["risk"]["take_profit_pct"] / 100), 2)

    # ── Sector exposure ───────────────────────────────────────────────────────

    def can_add_sector_exposure(self, sector: str, trade_value: float) -> bool:
        """Returns False if sector would exceed max_sector_exposure_pct."""
        limit_pct = self.cfg["risk"]["max_sector_exposure_pct"] / 100
        max_sector = self.portfolio_value * limit_pct
        current    = self.sector_exposure.get(sector or "Unknown", 0.0)
        if current + trade_value > max_sector:
            log.warning(
                f"Sector cap: {sector} at ${current:,.0f} + ${trade_value:,.0f} "
                f"> max ${max_sector:,.0f}"
            )
            return False
        return True

    def max_positions_reached(self) -> bool:
        n = len(self.open_positions)
        limit = self.cfg["trading"]["max_open_positions"]
        if n >= limit:
            log.info(f"Max open positions reached ({n}/{limit})")
            return True
        return False

    # ── Record a new trade ────────────────────────────────────────────────────

    def record_buy(self, symbol: str, qty: int, price: float, sector: str = None):
        value = qty * price
        self.open_positions[symbol] = {
            "qty": qty, "entry_price": price,
            "sector": sector or "Unknown", "value": value,
            "stop": self.stop_price(price),
            "target": self.take_profit_price(price),
        }
        self.sector_exposure[sector or "Unknown"] = \
            self.sector_exposure.get(sector or "Unknown", 0.0) + value
        self.save_state()

    def record_sell(self, symbol: str, price: float):
        pos = self.open_positions.pop(symbol, None)
        if pos:
            pnl = (price - pos["entry_price"]) * pos["qty"]
            self.daily_pnl += pnl
            sector = pos.get("sector", "Unknown")
            self.sector_exposure[sector] = max(
                0, self.sector_exposure.get(sector, 0) - pos["value"]
            )
            self.save_state()
            return pnl
        return 0.0

    # ── Pre-trade check ───────────────────────────────────────────────────────

    def pre_trade_ok(self, symbol: str, price: float, sector: str,
                     direction: str = "long") -> tuple:
        """Returns (ok: bool, reason: str)."""
        if self.check_kill_switch():
            return False, "Kill switch active"
        if self.max_positions_reached():
            return False, "Max open positions"
        shares = self.position_size_shares(price)
        value  = shares * price
        if not self.can_add_sector_exposure(sector, value):
            return False, f"Sector cap: {sector}"
        if direction == "short" and self.cfg["trading"]["direction"] == "long_only":
            return False, "Long-only mode"
        return True, "OK"
