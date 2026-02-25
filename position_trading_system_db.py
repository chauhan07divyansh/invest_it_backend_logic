"""
position_trading_system_db.py
==============================
Drop-in replacement for EnhancedPositionTradingSystem that replaces
the hardcoded initialize_stock_database() and get_stock_info_from_db()
with live SQLite reads via StockDB.

Only the two changed methods + __init__ are shown here.
Paste them over the originals in your main file, or
inherit from this class.
"""

import logging
from pathlib import Path

# Import the DB layer
from stock_db import StockDB, get_stock_db, DB_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MIXIN — paste these methods into EnhancedPositionTradingSystem
# ─────────────────────────────────────────────────────────────────────────────

class StockDBMixin:
    """
    Replace initialize_stock_database() and get_stock_info_from_db()
    with these database-backed versions.

    In your __init__, change:
        self.initialize_stock_database()
    to:
        self._init_stock_db()
    """

    def _init_stock_db(self, db_path: Path = DB_PATH):
        """
        Connect to the SQLite stock database.
        Call this from __init__ instead of initialize_stock_database().
        """
        try:
            self._stock_db: StockDB = get_stock_db(db_path)
            stats = self._stock_db.get_stats()
            total = stats["total_active"]
            caps  = stats["by_market_cap"]

            logger.info("✅ Stock database loaded: %d active symbols", total)
            logger.info("   Large Cap : %d", caps.get("Large", 0))
            logger.info("   Mid Cap   : %d", caps.get("Mid",   0))
            logger.info("   Small Cap : %d", caps.get("Small", 0))

        except FileNotFoundError:
            logger.critical(
                "❌ stocks.db not found. Run `python stock_db_init.py` first."
            )
            raise
        except Exception as e:
            logger.error("❌ Failed to connect to stock database: %s", e)
            raise

    # ── Drop-in replacement for get_stock_info_from_db() ─────────────────────

    def get_stock_info_from_db(self, symbol: str) -> dict:
        """
        Fetch stock metadata from SQLite.
        Returns a dict compatible with the original hardcoded version:
            {name, sector, market_cap, div_yield}
        Plus bonus fields from the DB:
            {index_name, sentiment_bias, sector_score, preference}
        """
        try:
            if not symbol:
                raise ValueError("Empty symbol")
            info = self._stock_db.get_stock_info(symbol)
            return info
        except Exception as e:
            logger.error("get_stock_info_from_db error for %s: %s", symbol, e)
            return {
                "symbol":         str(symbol).upper(),
                "name":           str(symbol).upper(),
                "sector":         "Unknown",
                "market_cap":     "Unknown",
                "index_name":     "Unknown",
                "div_yield":      0.010,
                "sentiment_bias": 1.00,
                "sector_score":   60,
                "preference":     "Low",
            }

    # ── Drop-in replacement for get_all_stock_symbols() ──────────────────────

    def get_all_stock_symbols(
        self,
        market_cap: str = None,
        index_name: str = None,
        sector: str = None,
    ) -> list[str]:
        """
        Get symbols from DB with optional filters.
        All filters are optional — calling with no args returns everything.

        Examples:
            self.get_all_stock_symbols()                        # all active
            self.get_all_stock_symbols(market_cap="Large")      # Nifty50 + Next50
            self.get_all_stock_symbols(index_name="Nifty50")    # only Nifty50
            self.get_all_stock_symbols(sector="Banking")        # banking stocks
        """
        try:
            if sector:
                return self._stock_db.get_symbols_by_sector(sector)
            if index_name:
                return self._stock_db.get_symbols_by_index(index_name)
            if market_cap:
                return self._stock_db.get_symbols_by_cap(market_cap)
            return self._stock_db.get_all_symbols()
        except Exception as e:
            logger.error("get_all_stock_symbols error: %s", e)
            return ["RELIANCE", "TCS", "HDFCBANK"]  # minimal fallback

    # ── Bonus: use DB sector score in analyze_market_cycles() ────────────────

    def _get_sector_score_from_db(self, sector_name: str) -> int:
        """
        Returns the sector_score stored in the DB instead of
        the hardcoded if/elif chain in analyze_market_cycles().
        """
        try:
            info = self._stock_db.get_sector_info(sector_name)
            return info["sector_score"] if info else 60
        except Exception:
            return 60

    def _get_sentiment_bias_from_db(self, sector_name: str) -> float:
        """
        Returns sentiment_bias from DB instead of the hardcoded
        sector_sentiment_bias dict in calculate_position_trading_score().
        """
        try:
            info = self._stock_db.get_sector_info(sector_name)
            return info["sentiment_bias"] if info else 1.00
        except Exception:
            return 1.00


# ─────────────────────────────────────────────────────────────────────────────
# FULL UPDATED __init__ snippet
# ─────────────────────────────────────────────────────────────────────────────
#
# Replace the __init__ block in EnhancedPositionTradingSystem with this:
#
#   def __init__(self, data_provider=None, mda_processor=None, redis_client=None,
#                db_path: Path = DB_PATH):
#       try:
#           ...existing code...
#
#           # ── REPLACE THIS LINE ──────────────────────────────────────
#           # self.initialize_stock_database()         ← DELETE THIS
#           # ── WITH THIS ─────────────────────────────────────────────
#           self._init_stock_db(db_path)               # ← ADD THIS
#           # ──────────────────────────────────────────────────────────
#
#           ...rest of __init__...
#
# Then delete the entire initialize_stock_database() method — it's no longer needed.
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED analyze_market_cycles() snippet
# ─────────────────────────────────────────────────────────────────────────────
#
# Replace the hardcoded if/elif sector_score block with:
#
#   sector_score = self._get_sector_score_from_db(sector)
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED calculate_position_trading_score() snippet
# ─────────────────────────────────────────────────────────────────────────────
#
# Replace the hardcoded sector_sentiment_bias dict with:
#
#   sentiment_multiplier = self._get_sentiment_bias_from_db(sector)
#   sentiment_score = min(100, sentiment_score * sentiment_multiplier)
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db = get_stock_db()

    print("\n── Stats ──────────────────────────────────")
    print(json.dumps(db.get_stats(), indent=2))

    print("\n── RELIANCE info ──────────────────────────")
    print(json.dumps(db.get_stock_info("RELIANCE"), indent=2))

    print("\n── Alias lookup: BAJAJ-AUTO ───────────────")
    print(json.dumps(db.get_stock_info("BAJAJ-AUTO"), indent=2))

    print("\n── Nifty50 symbols (first 10) ─────────────")
    print(db.get_symbols_by_index("Nifty50")[:10])

    print("\n── Banking sector symbols ──────────────────")
    print(db.get_symbols_by_sector("Banking"))

    print("\n── Search 'hdfc' ────────────────────────────")
    results = db.search_stocks("hdfc")
    for r in results:
        print(f"  {r['symbol']:15s} {r['name']}")
