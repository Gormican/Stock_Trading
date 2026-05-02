"""
account_manager.py — Multiple Alpaca account management
Supports: named paper accounts + live account
Remembers last selected account across sessions
"""
import json
from pathlib import Path

LAST_ACCOUNT_FILE = Path.home() / "taylor_last_account.json"

ACCOUNT_DEFAULTS = {
    "Taylor": {
        "name": "Taylor",
        "label": "Taylor (Paper)",
        "type": "paper",
        "api_key": "",
        "secret_key": "",
        "base_url": "https://paper-api.alpaca.markets/v2",
        "paper": True,
        "color": "#3b82f6",
        "icon": "📘",
    },
    "$100K": {
        "name": "$100K",
        "label": "$100K (Paper)",
        "type": "paper",
        "api_key": "",
        "secret_key": "",
        "base_url": "https://paper-api.alpaca.markets/v2",
        "paper": True,
        "color": "#8b5cf6",
        "icon": "📗",
    },
    "Live": {
        "name": "Live",
        "label": "⚠️ Live Account (Real Money)",
        "type": "live",
        "api_key": "",
        "secret_key": "",
        "base_url": "https://api.alpaca.markets/v2",
        "paper": False,
        "color": "#ef4444",
        "icon": "🔴",
    },
}


class AccountManager:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path

    def get_all(self) -> dict:
        """Return all accounts from config, merged with defaults."""
        cfg = self._load_cfg()
        saved = cfg.get("accounts", {})
        result = {}
        for name, defaults in ACCOUNT_DEFAULTS.items():
            acct = dict(defaults)
            if name in saved:
                acct.update(saved[name])
            result[name] = acct
        return result

    def get(self, name: str) -> dict:
        return self.get_all().get(name, ACCOUNT_DEFAULTS.get(name, {}))

    def _load_prefs(self) -> dict:
        """Load the preferences file (last account + per-account strategies)."""
        try:
            if LAST_ACCOUNT_FILE.exists():
                return json.loads(LAST_ACCOUNT_FILE.read_text())
        except Exception:
            pass
        return {}

    def _save_prefs(self, prefs: dict):
        try:
            LAST_ACCOUNT_FILE.write_text(json.dumps(prefs, indent=2))
        except Exception:
            pass

    def get_last(self) -> str:
        return self._load_prefs().get("last", "Taylor")

    def set_last(self, name: str):
        prefs = self._load_prefs()
        prefs["last"] = name
        self._save_prefs(prefs)

    def get_strategy(self, account_name: str) -> str:
        """Return the strategy that was last used with this account."""
        return self._load_prefs().get("strategies", {}).get(account_name, "Default")

    def save_strategy(self, account_name: str, strategy_name: str):
        """Persist the strategy selection for this account."""
        prefs = self._load_prefs()
        if "strategies" not in prefs:
            prefs["strategies"] = {}
        prefs["strategies"][account_name] = strategy_name
        self._save_prefs(prefs)

    def save_keys(self, name: str, api_key: str, secret_key: str):
        cfg = self._load_cfg()
        if "accounts" not in cfg:
            cfg["accounts"] = {}
        if name not in cfg["accounts"]:
            cfg["accounts"][name] = dict(ACCOUNT_DEFAULTS.get(name, {}))
        cfg["accounts"][name]["api_key"] = api_key
        cfg["accounts"][name]["secret_key"] = secret_key
        self._save_cfg(cfg)

    def is_configured(self, name: str) -> bool:
        acct = self.get(name)
        k = acct.get("api_key", "")
        s = acct.get("secret_key", "")
        return bool(k and s and "YOUR_" not in k and len(k) > 10)

    def account_names(self) -> list:
        return list(ACCOUNT_DEFAULTS.keys())

    def _load_cfg(self) -> dict:
        if self.cfg_path.exists():
            try:
                return json.loads(self.cfg_path.read_text())
            except Exception:
                pass
        return {}

    def _save_cfg(self, cfg: dict):
        self.cfg_path.write_text(json.dumps(cfg, indent=2))
