# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QuantForge — Indicator Computation                                      ║
# ║  Vectorized indicator library — computed once, referenced by strategies   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("quantforge.data.indicators")


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands: upper, middle, lower."""
    middle = sma(series, period)
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0)
    plus_dm = plus_dm.where(plus_dm > 0, 0.0)

    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0)
    minus_dm = minus_dm.where(minus_dm > 0, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(span=period, adjust=False).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator %K and %D."""
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume / Volume SMA ratio."""
    vol_ma = volume.rolling(period).mean()
    return volume / vol_ma.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

# Registry of all available indicators and their column names
INDICATOR_REGISTRY = {
    # Trend
    "ema_8": ("ema", 8), "ema_13": ("ema", 13), "ema_21": ("ema", 21),
    "ema_34": ("ema", 34), "ema_55": ("ema", 55), "ema_89": ("ema", 89),
    "ema_200": ("ema", 200),
    "sma_20": ("sma", 20), "sma_50": ("sma", 50), "sma_200": ("sma", 200),
    "adx_14": ("adx", 14),
    # Momentum
    "rsi_14": ("rsi", 14), "rsi_7": ("rsi", 7), "rsi_21": ("rsi", 21),
    "macd_line": ("macd_line",), "macd_signal": ("macd_signal",), "macd_hist": ("macd_hist",),
    "stoch_k": ("stoch_k",), "stoch_d": ("stoch_d",),
    # Volatility
    "atr_14": ("atr", 14),
    "bb_upper": ("bb_upper",), "bb_middle": ("bb_middle",), "bb_lower": ("bb_lower",),
    "atr_pct": ("atr_pct",),
    # Volume
    "volume_ma_20": ("volume_ma", 20),
    "volume_ratio": ("volume_ratio",),
    "obv": ("obv",),
    # Custom
    "candle_body_ratio": ("candle_body_ratio",),
}

# Columns that strategies can reference
ALL_INDICATOR_COLUMNS = list(INDICATOR_REGISTRY.keys()) + [
    "open", "high", "low", "close", "volume",
]


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ALL indicators at once. Returns enriched DataFrame.
    This is called ONCE per data load — strategies reference columns by name.
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Trend ──────────────────────────────────────────────────────────────
    for period in [8, 13, 21, 34, 55, 89, 200]:
        df[f"ema_{period}"] = ema(c, period)
    for period in [20, 50, 200]:
        df[f"sma_{period}"] = sma(c, period)
    df["adx_14"] = adx(h, l, c, 14)

    # ── Momentum ───────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        df[f"rsi_{period}"] = rsi(c, period)
    ml, sl, mh = macd(c)
    df["macd_line"] = ml
    df["macd_signal"] = sl
    df["macd_hist"] = mh
    sk, sd = stochastic(h, l, c)
    df["stoch_k"] = sk
    df["stoch_d"] = sd

    # ── Volatility ─────────────────────────────────────────────────────────
    df["atr_14"] = atr(h, l, c, 14)
    bb_u, bb_m, bb_l = bollinger_bands(c)
    df["bb_upper"] = bb_u
    df["bb_middle"] = bb_m
    df["bb_lower"] = bb_l
    df["atr_pct"] = df["atr_14"] / c

    # ── Volume ─────────────────────────────────────────────────────────────
    df["volume_ma_20"] = sma(v, 20)
    df["volume_ratio"] = volume_ratio(v, 20)
    df["obv"] = obv(c, v)

    # ── Custom ─────────────────────────────────────────────────────────────
    body = (c - df["open"]).abs()
    wick = h - l
    df["candle_body_ratio"] = body / wick.replace(0, np.nan)
    df["candle_body_ratio"] = df["candle_body_ratio"].fillna(0.5)

    # ── Warmup NaN handling ────────────────────────────────────────────────
    df = df.iloc[200:].reset_index(drop=True)

    log.info(f"Indicators computed: {len(df)} rows, {len(df.columns)} columns")
    return df
