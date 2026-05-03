# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QuantForge — Disk Cache (Parquet)                                       ║
# ║  Avoids repeated API calls; TTL-based invalidation                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from backend.config import CACHE_DIR, CACHE_TTL_HOURS, SYMBOL, DURATION_YEARS
from backend.data.ingestion import fetch_all_candles, resample_to_timeframe

log = logging.getLogger("quantforge.data.cache")


class DataCache:
    """
    Parquet-based disk cache for OHLCV data.
    - Fetches once, caches for CACHE_TTL_HOURS
    - Provides multi-timeframe views via resampling
    - Memory-efficient: loads only when requested
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, pd.DataFrame] = {}

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol}_{interval}.parquet"

    def _is_fresh(self, path: Path) -> bool:
        """Check if cached file is within TTL."""
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < CACHE_TTL_HOURS

    def get_data(
        self,
        symbol: str = SYMBOL,
        interval: str = "1h",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get OHLCV data — from memory cache, disk cache, or fresh fetch.
        """
        cache_key = f"{symbol}_{interval}"

        # 1. Memory cache (fastest)
        if not force_refresh and cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # 2. Disk cache
        path = self._cache_path(symbol, interval)
        if not force_refresh and self._is_fresh(path):
            log.info(f"Loading {symbol} {interval} from disk cache")
            df = pd.read_parquet(path)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            self._memory_cache[cache_key] = df
            return df

        # 3. Fresh fetch (always fetch 1h, resample if needed)
        base_interval = "1h"
        base_key = f"{symbol}_{base_interval}"
        base_path = self._cache_path(symbol, base_interval)

        if force_refresh or not self._is_fresh(base_path):
            log.info(f"Fetching fresh data from Bybit for {symbol}")
            df_base = fetch_all_candles(
                symbol=symbol,
                interval=base_interval,
                duration_years=DURATION_YEARS,
            )
            df_base.to_parquet(base_path, index=False)
            self._memory_cache[base_key] = df_base
        else:
            df_base = pd.read_parquet(base_path)
            df_base["datetime"] = pd.to_datetime(df_base["datetime"], utc=True)
            self._memory_cache[base_key] = df_base

        # Resample if needed
        if interval != base_interval:
            df = resample_to_timeframe(df_base, interval)
            df.to_parquet(path, index=False)
            self._memory_cache[cache_key] = df
        else:
            df = self._memory_cache[base_key]

        log.info(f"Data ready: {symbol} {interval} — {len(df):,} candles")
        return df

    def clear_memory(self):
        """Release in-memory DataFrames to free RAM."""
        self._memory_cache.clear()
        log.info("Memory cache cleared")

    def clear_all(self):
        """Delete all cached parquet files."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        self._memory_cache.clear()
        log.info("All caches cleared")
