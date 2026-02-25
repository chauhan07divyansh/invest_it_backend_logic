"""
stock_db.py
===========
All database read/write operations for the stock universe.
Replaces the hardcoded `initialize_stock_database()` and
`get_stock_info_from_db()` methods in EnhancedPositionTradingSystem.

Usage:
    from stock_db import StockDB

    db = StockDB()                          # uses stocks.db by default
    info = db.get_stock_info("RELIANCE")    # → dict with name/sector/etc.
    symbols = db.get_all_symbols()          # → ["RELIANCE", "TCS", ...]
    symbols = db.get_symbols_by_cap("Mid")  # → midcap symbols only
    symbols = db.get_symbols_by_index("Nifty50")
"""

import sqlite3
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("stocks.db")

# Default returned when a symbol is not found
_DEFAULT_STOCK = {
    "symbol":     "UNKNOWN",
    "name":       "Unknown",
    "sector":     "Unknown",
    "market_cap": "Unknown",
    "index_name": "Unknown",
    "div_yield":  0.010,
    "sentiment_bias": 1.00,
    "sector_score":   60,
    "preference":     "Low",
}


class StockDB:
    """
    Thin repository layer over the stocks SQLite database.
    Thread-safe for read operations (SQLite WAL mode).
    Write operations (add/update/deactivate) should be called
    from a single thread or protected externally.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        if not db_path.exists():
            raise FileNotFoundError(
                f"Stock database not found at {db_path.resolve()}. "
                "Run `python stock_db_init.py` first."
            )
        self._verify_connection()
        logger.info("✅ StockDB connected: %s", db_path.resolve())

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION HELPER
    # ─────────────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _verify_connection(self):
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1 FROM stocks LIMIT 1")
        except Exception as e:
            raise RuntimeError(f"StockDB connection failed: {e}") from e

    # ─────────────────────────────────────────────────────────────────────────
    # READ OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def get_stock_info(self, symbol: str) -> dict:
        """Fetch stock info with simple instance-level cache."""
        clean = self._clean_symbol(symbol)
        if not hasattr(self, "_info_cache"):
            self._info_cache = {}
        if clean in self._info_cache:
            return self._info_cache[clean]
        row = self._fetch_stock_row(clean)
        if row is None:
            resolved = self._resolve_alias(clean)
            if resolved:
                row = self._fetch_stock_row(resolved)
        if row is None:
            logger.warning("Symbol not found in DB: %s", symbol)
            result = _DEFAULT_STOCK.copy()
            result["symbol"] = clean
            result["name"] = clean
        else:
            result = self._row_to_dict(row)
        self._info_cache[clean] = result
        return result

    def _clear_cache(self):
        """Clear in-memory cache after write operations."""
        self._info_cache = {}

    def get_all_symbols(self, active_only: bool = True) -> list[str]:
        """Return all stock symbols, optionally filtering to active only."""
        query = "SELECT symbol FROM stocks"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY symbol"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [r["symbol"] for r in rows]

    def get_symbols_by_cap(self, market_cap: str, active_only: bool = True) -> list[str]:
        """market_cap: 'Large' | 'Mid' | 'Small'"""
        query = "SELECT symbol FROM stocks WHERE market_cap = ?"
        params = [market_cap]
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY symbol"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [r["symbol"] for r in rows]

    def get_symbols_by_index(self, index_name: str, active_only: bool = True) -> list[str]:
        """index_name: 'Nifty50' | 'NiftyNext50' | 'NiftyMidcap100' | 'Smallcap'"""
        query = "SELECT symbol FROM stocks WHERE index_name = ?"
        params = [index_name]
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY symbol"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [r["symbol"] for r in rows]

    def get_symbols_by_sector(self, sector_name: str, active_only: bool = True) -> list[str]:
        """Return all symbols belonging to a given sector name."""
        query = """
            SELECT s.symbol FROM stocks s
            JOIN sectors sec ON s.sector_id = sec.id
            WHERE sec.name = ?
        """
        params = [sector_name]
        if active_only:
            query += " AND s.is_active = 1"
        query += " ORDER BY s.symbol"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [r["symbol"] for r in rows]

    def get_sector_info(self, sector_name: str) -> Optional[dict]:
        """Return sector metadata dict or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sectors WHERE name = ?", (sector_name,)
            ).fetchone()
        return dict(row) if row else None

    def get_all_sectors(self) -> list[dict]:
        """Return all sectors as list of dicts."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM sectors ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def search_stocks(self, query: str, limit: int = 20) -> list[dict]:
        """
        Full-text search across symbol and name.
        Useful for a search-bar / autocomplete endpoint.
        """
        pattern = f"%{query.upper()}%"
        sql = """
            SELECT s.symbol, s.name, sec.name AS sector, s.market_cap, s.index_name
            FROM stocks s
            JOIN sectors sec ON s.sector_id = sec.id
            WHERE (UPPER(s.symbol) LIKE ? OR UPPER(s.name) LIKE ?)
              AND s.is_active = 1
            ORDER BY
                CASE WHEN UPPER(s.symbol) = ? THEN 0
                     WHEN UPPER(s.symbol) LIKE ? THEN 1
                     ELSE 2 END,
                s.symbol
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(
                sql,
                (pattern, f"%{query}%", query.upper(), f"{query.upper()}%", limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Return database statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM stocks WHERE is_active=1").fetchone()[0]
            by_cap = conn.execute(
                "SELECT market_cap, COUNT(*) as cnt FROM stocks WHERE is_active=1 GROUP BY market_cap"
            ).fetchall()
            by_index = conn.execute(
                "SELECT index_name, COUNT(*) as cnt FROM stocks WHERE is_active=1 GROUP BY index_name"
            ).fetchall()
        return {
            "total_active": total,
            "by_market_cap": {r["market_cap"]: r["cnt"] for r in by_cap},
            "by_index": {r["index_name"]: r["cnt"] for r in by_index},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # WRITE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def add_stock(
        self,
        symbol: str,
        name: str,
        sector_name: str,
        market_cap: str,
        index_name: str = "Other",
    ) -> bool:
        """
        Add a new stock. Returns True on success, False if already exists.
        Automatically creates the sector if it doesn't exist.
        """
        clean = self._clean_symbol(symbol)
        try:
            with self._connect() as conn:
                # Ensure sector exists
                sector_id = self._get_or_create_sector(conn, sector_name)
                conn.execute(
                    """INSERT INTO stocks (symbol, name, sector_id, market_cap, index_name)
                       VALUES (?, ?, ?, ?, ?)""",
                    (clean, name, sector_id, market_cap, index_name),
                )
                conn.commit()
            logger.info("Added stock: %s (%s)", clean, name)
            self.get_stock_info.cache_clear()
            return True
        except sqlite3.IntegrityError:
            logger.warning("Stock already exists: %s", clean)
            return False

    def update_stock(self, symbol: str, **kwargs) -> bool:
        """
        Update fields on an existing stock.
        Allowed kwargs: name, market_cap, index_name, is_active, sector_name.
        """
        clean = self._clean_symbol(symbol)
        allowed = {"name", "market_cap", "index_name", "is_active"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if "sector_name" in kwargs:
            with self._connect() as conn:
                sector_id = self._get_or_create_sector(conn, kwargs["sector_name"])
            updates["sector_id"] = sector_id

        if not updates:
            logger.warning("update_stock: no valid fields provided")
            return False

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        set_clause += ", updated_at = datetime('now')"
        values = list(updates.values()) + [clean]

        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE stocks SET {set_clause} WHERE symbol = ?", values
            )
            conn.commit()

        if cur.rowcount == 0:
            logger.warning("update_stock: symbol not found: %s", clean)
            return False

        self.get_stock_info.cache_clear()
        logger.info("Updated stock: %s → %s", clean, updates)
        return True

    def deactivate_stock(self, symbol: str) -> bool:
        """Soft-delete: sets is_active=0."""
        return self.update_stock(symbol, is_active=0)

    def add_alias(self, alias: str, symbol: str) -> bool:
        """Register an alternate ticker spelling."""
        clean_symbol = self._clean_symbol(symbol)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO stock_aliases (alias, symbol) VALUES (?, ?)",
                    (alias.upper(), clean_symbol),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("add_alias failed: %s → %s: %s", alias, symbol, e)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """Normalise: strip exchange suffix, upper-case, strip whitespace."""
        return str(symbol).split(".")[0].upper().strip()

    def get_stock_info(self, symbol: str) -> dict:
        """
        Returns full stock info dict for a symbol (with simple in-memory cache).
        Call _clear_cache() after any write operation.
        """
        clean = self._clean_symbol(symbol)
        # Check instance-level cache
        if not hasattr(self, "_cache"):
            self._cache = {}
        if clean in self._cache:
            return self._cache[clean]
        result = self._get_stock_info_uncached(clean)
        self._cache[clean] = result
        return result

    def _clear_cache(self):
        """Clear the in-memory lookup cache after write operations."""
        self._cache = {}

    def _get_stock_info_uncached(self, symbol: str) -> dict:
        """Internal: fetch without cache, resolve aliases if needed."""
        row = self._fetch_stock_row(symbol)
        if row is None:
            resolved = self._resolve_alias(symbol)
            if resolved:
                row = self._fetch_stock_row(resolved)
        if row is None:
            logger.warning("Symbol not found in DB: %s", symbol)
            default = _DEFAULT_STOCK.copy()
            default["symbol"] = symbol
            default["name"] = symbol
            return default
        return self._row_to_dict(row)

    def _fetch_stock_row(self, symbol: str):
        sql = """
            SELECT s.symbol, s.name, sec.name AS sector, s.market_cap,
                   s.index_name, sec.default_div_yield AS div_yield,
                   sec.sentiment_bias, sec.sector_score, sec.preference
            FROM stocks s
            JOIN sectors sec ON s.sector_id = sec.id
            WHERE s.symbol = ? AND s.is_active = 1
        """
        with self._connect() as conn:
            return conn.execute(sql, (symbol,)).fetchone()

    def _resolve_alias(self, alias: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT symbol FROM stock_aliases WHERE alias = ?", (alias,)
            ).fetchone()
        return row["symbol"] if row else None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "symbol":         row["symbol"],
            "name":           row["name"],
            "sector":         row["sector"],
            "market_cap":     row["market_cap"],
            "index_name":     row["index_name"],
            "div_yield":      row["div_yield"],
            "sentiment_bias": row["sentiment_bias"],
            "sector_score":   row["sector_score"],
            "preference":     row["preference"],
        }

    @staticmethod
    def _get_or_create_sector(conn: sqlite3.Connection, sector_name: str) -> int:
        row = conn.execute(
            "SELECT id FROM sectors WHERE name = ?", (sector_name,)
        ).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            "INSERT INTO sectors (name) VALUES (?)", (sector_name,)
        )
        conn.commit()
        return cur.lastrowid


# Convenience singleton — import and use directly
# from stock_db import stock_db
# stock_db.get_stock_info("TCS")
stock_db = None  # Initialised lazily on first import usage

def get_stock_db(db_path: Path = DB_PATH) -> StockDB:
    """Return module-level singleton StockDB instance."""
    global stock_db
    if stock_db is None:
        stock_db = StockDB(db_path)
    return stock_db
