# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QuantForge — Bybit Data Ingestion                                      ║
# ║  Fetches historical OHLCV data with pagination + caching                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from backend.config import BYBIT_BASE_URL, SYMBOL

log = logging.getLogger("quantforge.data.ingestion")

# ── Interval mapping: human → Bybit v5 format ────────────────────────────────
_INTERVAL_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15",
    "30m": "30", "1h": "60", "2h": "120", "4h": "240",
    "6h": "360", "12h": "720", "1d": "D", "1w": "W", "1M": "M",
}

# ── Interval → milliseconds (for resampling calculations) ────────────────────
_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
}


def bybit_interval(interval: str) -> str:
    """Convert human-readable interval to Bybit v5 format."""
    mapped = _INTERVAL_MAP.get(interval)
    if mapped is None:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Valid options: {list(_INTERVAL_MAP.keys())}"
        )
    return mapped


def fetch_all_candles(
    symbol: str = SYMBOL,
    interval: str = "1h",
    duration_years: int = 5,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch ALL historical klines from Bybit v5 in 1000-candle pages.
    Returns a DataFrame sorted oldest → newest with columns:
    [time, open, high, low, close, volume, turnover, datetime]
    """
    bv_interval = bybit_interval(interval)
    now = datetime.utcnow()
    start_ms = int((now - timedelta(days=duration_years * 365)).timestamp() * 1_000)
    end_ms = int(now.timestamp() * 1_000)
    all_rows = []

    log.info(f"Fetching {symbol} {interval} data ({duration_years}y)...")

    while True:
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    BYBIT_BASE_URL,
                    params={
                        "category": "linear",
                        "symbol": symbol,
                        "interval": bv_interval,
                        "start": start_ms,
                        "end": end_ms,
                        "limit": 1_000,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Bybit API failed after {max_retries} retries: {e}")
                wait = 2 ** attempt
                log.warning(f"Retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                time.sleep(wait)

        if payload.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit API error {payload.get('retCode')}: {payload.get('retMsg')}"
            )

        rows = payload["result"]["list"]  # newest-first
        if not rows:
            break

        rows_asc = list(reversed(rows))
        all_rows.extend(rows_asc)

        if len(rows) < 1_000:
            break

        oldest_ts = int(rows_asc[0][0])
        if oldest_ts <= start_ms:
            break
        end_ms = oldest_ts - 1
        time.sleep(0.2)  # rate limit courtesy

    log.info(f"[OK] Fetched {len(all_rows):,} candles for {symbol} {interval}")

    if not all_rows:
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "volume", "turnover", "datetime"]
        )

    df = pd.DataFrame(
        all_rows,
        columns=["time", "open", "high", "low", "close", "volume", "turnover"],
    ).astype(float)
    df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def resample_to_timeframe(df_1h: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample 1h OHLCV data to a higher timeframe (e.g. 4h, 1d).
    Input must have a 'datetime' column.
    """
    if target_interval == "1h":
        return df_1h.copy()

    rule_map = {
        "2h": "2h", "4h": "4h", "6h": "6h",
        "12h": "12h", "1d": "1D", "1w": "1W",
    }
    rule = rule_map.get(target_interval)
    if rule is None:
        raise ValueError(f"Cannot resample to '{target_interval}'")

    df = df_1h.set_index("datetime").copy()
    resampled = df.resample(rule).agg({
        "time": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "turnover": "sum",
    }).dropna(subset=["time"])

    resampled = resampled.reset_index()
    return resampled
