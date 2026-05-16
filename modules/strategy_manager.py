"""
strategy_manager.py — Named GPA strategy management
Save, load, delete named weight strategies with auto-normalization
"""
import json
from pathlib import Path
from datetime import datetime

STRATEGIES_FILE = Path(__file__).parent.parent / "strategies.json"
STRATEGIES_XLSX = Path(__file__).parent.parent / "Strategies.xlsx"

DEFAULT_WEIGHTS = {
    "sentiment":    0.20,
    "fundamentals": 0.45,
    "technical":    0.35,
    "fund_sub": {
        "valuation": 0.33,
        "financial":  0.33,
        "estimates":  0.34,
    },
    "tech_sub": {
        "trend":       0.60,
        "oscillators": 0.40,
    },
}

DEFAULT_THRESHOLDS = {
    "min_gpa_to_buy":   3.5,
    "min_gpa_to_show":  3.0,
    "max_gpa_to_sell":  2.5,
    "min_gpa_to_alert": 3.5,
}

BUILTIN_STRATEGIES = {
    "Default": {
        "name": "Default", "created": "built-in",
        "weights": DEFAULT_WEIGHTS,
        "thresholds": DEFAULT_THRESHOLDS,
        "auto_trade": False,
        "description": "Balanced across all categories",
    },
    "Growth": {
        "name": "Growth", "created": "built-in",
        "weights": {
            "sentiment": 0.20, "fundamentals": 0.40, "technical": 0.40,
            "fund_sub": {"valuation": 0.25, "financial": 0.30, "estimates": 0.45},
            "tech_sub": {"trend": 0.65, "oscillators": 0.35},
        },
        "thresholds": {**DEFAULT_THRESHOLDS, "min_gpa_to_buy": 3.3},
        "auto_trade": False,
        "description": "Emphasizes earnings growth and technical momentum",
    },
    "Income": {
        "name": "Income", "created": "built-in",
        "weights": {
            "sentiment": 0.15, "fundamentals": 0.55, "technical": 0.30,
            "fund_sub": {"valuation": 0.30, "financial": 0.50, "estimates": 0.20},
            "tech_sub": {"trend": 0.55, "oscillators": 0.45},
        },
        "thresholds": {**DEFAULT_THRESHOLDS, "min_gpa_to_buy": 3.2},
        "auto_trade": False,
        "description": "Emphasizes dividends, ROE, and financial strength",
    },
    "Momentum": {
        "name": "Momentum", "created": "built-in",
        "weights": {
            "sentiment": 0.30, "fundamentals": 0.25, "technical": 0.45,
            "fund_sub": {"valuation": 0.20, "financial": 0.30, "estimates": 0.50},
            "tech_sub": {"trend": 0.70, "oscillators": 0.30},
        },
        "thresholds": {**DEFAULT_THRESHOLDS, "min_gpa_to_buy": 3.5},
        "auto_trade": False,
        "description": "Heavy technical and sentiment weighting for momentum plays",
    },
}


class StrategyManager:

    def load_all(self) -> dict:
        """Returns built-in + user-saved strategies."""
        result = dict(BUILTIN_STRATEGIES)
        if STRATEGIES_FILE.exists():
            try:
                user = json.loads(STRATEGIES_FILE.read_text())
                result.update(user)
            except Exception:
                pass
        return result

    def save(self, name: str, weights: dict, thresholds: dict, auto_trade: bool,
             description: str = ""):
        """Save a user strategy. Cannot overwrite built-ins."""
        user = {}
        if STRATEGIES_FILE.exists():
            try:
                user = json.loads(STRATEGIES_FILE.read_text())
            except Exception:
                pass
        user[name] = {
            "name": name,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "weights": self.normalize(weights),
            "thresholds": thresholds,
            "auto_trade": auto_trade,
            "description": description,
        }
        STRATEGIES_FILE.write_text(json.dumps(user, indent=2))

    def delete(self, name: str):
        """Delete a user strategy (cannot delete built-ins)."""
        if name in BUILTIN_STRATEGIES:
            return
        if STRATEGIES_FILE.exists():
            try:
                user = json.loads(STRATEGIES_FILE.read_text())
                user.pop(name, None)
                STRATEGIES_FILE.write_text(json.dumps(user, indent=2))
            except Exception:
                pass

    def get(self, name: str) -> dict:
        return self.load_all().get(name, BUILTIN_STRATEGIES["Default"])

    def names(self) -> list:
        return list(self.load_all().keys())

    def is_builtin(self, name: str) -> bool:
        return name in BUILTIN_STRATEGIES

    def normalize(self, weights: dict) -> dict:
        """Ensure all weight groups sum to 1.0."""
        import copy
        w = copy.deepcopy(weights)

        # Main weights
        main_keys = ["sentiment", "fundamentals", "technical"]
        main_total = sum(w.get(k, 0) for k in main_keys)
        if main_total > 0:
            for k in main_keys:
                w[k] = round(w.get(k, 0) / main_total, 4)

        # Fund sub-weights
        if "fund_sub" in w:
            sub = w["fund_sub"]
            t = sum(sub.values())
            if t > 0:
                w["fund_sub"] = {k: round(v / t, 4) for k, v in sub.items()}

        # Tech sub-weights
        if "tech_sub" in w:
            sub = w["tech_sub"]
            t = sum(sub.values())
            if t > 0:
                w["tech_sub"] = {k: round(v / t, 4) for k, v in sub.items()}

        return w

    # ── Excel import / export ─────────────────────────────────────────────────
    def import_from_xlsx(self, xlsx_path: Path | str = STRATEGIES_XLSX) -> dict:
        """
        Read Strategies.xlsx and persist every registered strategy to strategies.json.
        The rich category/subcategory/component tree is preserved under the `tree`
        key for use by the Config tab; the legacy `weights` field is also written
        so the existing GPAEngine keeps working unchanged.

        Returns:
            {
              "imported": ["Default", ...],
              "active":   "Default" or None,
              "skipped":  ["Custom1", ...],  # sheets in workbook but not in Strategies index
              "errors":   [],
            }
        """
        from modules import strategy_xlsx  # local import to avoid hard dep at startup
        xlsx_path = Path(xlsx_path)
        result = {"imported": [], "active": None, "skipped": [], "errors": []}
        if not xlsx_path.exists():
            result["errors"].append(f"File not found: {xlsx_path}")
            return result

        try:
            data = strategy_xlsx.load_xlsx(xlsx_path)
        except Exception as e:
            result["errors"].append(f"Parse error: {e}")
            return result

        # Existing user strategies on disk
        user: dict = {}
        if STRATEGIES_FILE.exists():
            try:
                user = json.loads(STRATEGIES_FILE.read_text())
            except Exception:
                user = {}

        result["active"] = data.get("_active")
        for entry in data.get("_index", []):
            name = entry["name"]
            rich = data.get(name)
            if not rich:
                continue
            # Built-ins keep their hard-coded defaults; xlsx version is saved as
            # a regular user strategy so it can be edited/deleted.
            user[name] = {
                "name":         name,
                "created":      f"imported from xlsx {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "weights":      self.normalize(rich["legacy_weights"]),
                "thresholds":   dict(DEFAULT_THRESHOLDS),
                "auto_trade":   False,
                "description":  rich.get("description", ""),
                "active":       bool(rich.get("active")),
                "tree":         rich.get("tree", {}),  # rich tree preserved
                "source":       "xlsx",
            }
            result["imported"].append(name)

        STRATEGIES_FILE.write_text(json.dumps(user, indent=2))
        return result

    def active_strategy_from_xlsx(self, xlsx_path: Path | str = STRATEGIES_XLSX) -> str | None:
        """Return the name of the strategy marked ACTIVE in the xlsx, or None."""
        from modules import strategy_xlsx
        xlsx_path = Path(xlsx_path)
        if not xlsx_path.exists():
            return None
        try:
            data = strategy_xlsx.load_xlsx(xlsx_path)
            return data.get("_active")
        except Exception:
            return None
