# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QuantForge — Multi-Timeframe Indicator Computation                      ║
# ║  Vectorized indicator library — computed once, referenced by strategies   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("quantforge.data.indicators")


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE INDICATOR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    middle = sma(series, period)
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    vol_ma = volume.rolling(period).mean()
    return volume / vol_ma.replace(0, np.nan)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - tp_sma) / (0.015 * mad).replace(0, np.nan)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * volume
    delta = tp.diff()
    pos_flow = rmf.where(delta > 0, 0.0).rolling(period).sum()
    neg_flow = rmf.where(delta < 0, 0.0).rolling(period).sum()
    mr = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + mr))


def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 20) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Simplified Supertrend indicator. Returns the supertrend line."""
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)  # 1 = up, -1 = down

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            st.iloc[i] = lower_band.iloc[i]
        else:
            st.iloc[i] = upper_band.iloc[i]
    return st


def _linear_slope(series: pd.Series, period: int) -> pd.Series:
    """Rolling linear regression slope."""
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    if x_var == 0:
        return pd.Series(0.0, index=series.index)
    def _slope(y):
        if len(y) < period:
            return np.nan
        y_mean = y.mean()
        return ((x - x_mean) * (y - y_mean)).sum() / x_var
    return series.rolling(period).apply(_slope, raw=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTE INDICATORS FOR A SINGLE TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_tf_indicators(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute all indicators for one timeframe, prefixing column names."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # ── Trend ──────────────────────────────────────────────────────────────
    for period in [8, 13, 21, 34, 55, 89, 200]:
        out[f"{prefix}ema_{period}"] = ema(c, period)
    for period in [20, 50, 200]:
        out[f"{prefix}sma_{period}"] = sma(c, period)
    out[f"{prefix}adx_14"] = adx(h, l, c, 14)

    # ── Momentum ───────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        out[f"{prefix}rsi_{period}"] = rsi(c, period)
    ml, sl_line, mh = macd(c)
    out[f"{prefix}macd_line"] = ml
    out[f"{prefix}macd_signal"] = sl_line
    out[f"{prefix}macd_hist"] = mh
    sk, sd = stochastic(h, l, c)
    out[f"{prefix}stoch_k"] = sk
    out[f"{prefix}stoch_d"] = sd
    out[f"{prefix}roc_10"] = c.pct_change(10) * 100

    # ── Volatility ─────────────────────────────────────────────────────────
    out[f"{prefix}atr_14"] = atr(h, l, c, 14)
    out[f"{prefix}atr_pct"] = out[f"{prefix}atr_14"] / c
    bb_u, bb_m, bb_l = bollinger_bands(c)
    out[f"{prefix}bb_upper"] = bb_u
    out[f"{prefix}bb_middle"] = bb_m
    out[f"{prefix}bb_lower"] = bb_l
    out[f"{prefix}bb_width"] = (bb_u - bb_l) / bb_m

    # ── Volume ─────────────────────────────────────────────────────────────
    out[f"{prefix}volume_ma_20"] = sma(v, 20)
    out[f"{prefix}volume_ratio"] = volume_ratio(v, 20)
    out[f"{prefix}obv"] = obv(c, v)
    out[f"{prefix}obv_slope_5"] = _linear_slope(out[f"{prefix}obv"], 5)
    out[f"{prefix}volume_expanding"] = (v > v.shift(1)).fillna(False).astype(int)

    # ── Advanced oscillators ───────────────────────────────────────────────
    out[f"{prefix}willr_14"] = williams_r(h, l, c, 14)
    out[f"{prefix}cci_20"] = cci(h, l, c, 20)
    out[f"{prefix}mfi_14"] = mfi(h, l, c, v, 14)
    out[f"{prefix}cmf_20"] = cmf(h, l, c, v, 20)

    # ── Supertrend ─────────────────────────────────────────────────────────
    out[f"{prefix}supertrend_10_3"] = supertrend(h, l, c, 10, 3.0)

    # ── VWAP deviation ─────────────────────────────────────────────────────
    vwap_num = (c * v).rolling(20).sum()
    vwap_den = v.rolling(20).sum().replace(0, np.nan)
    vwap_val = vwap_num / vwap_den
    out[f"{prefix}vwap_dev"] = (c / vwap_val - 1) * 100

    # ── Price structure ────────────────────────────────────────────────────
    for n in [10, 20, 50]:
        out[f"{prefix}high_{n}"] = h.rolling(n).max()
        out[f"{prefix}low_{n}"] = l.rolling(n).min()

    # ── Regime ─────────────────────────────────────────────────────────────
    adx_trend = (out[f"{prefix}adx_14"] > 25).fillna(False)
    bull_regime = (adx_trend & (c > out[f"{prefix}ema_200"]).fillna(False)).astype(int)
    bear_regime = (adx_trend & (c < out[f"{prefix}ema_200"]).fillna(False)).astype(int)
    out[f"{prefix}regime"] = bull_regime - bear_regime
    out[f"{prefix}ema_200_slope"] = _linear_slope(out[f"{prefix}ema_200"], 5)

    # ── Candle structure ───────────────────────────────────────────────────
    body = (c - o).abs()
    wick = h - l
    wick_safe = wick.replace(0, np.nan)
    out[f"{prefix}body_ratio"] = (body / wick_safe).fillna(0.5)
    out[f"{prefix}upper_wick_ratio"] = ((h - pd.concat([c, o], axis=1).max(axis=1)) / wick_safe).fillna(0)
    out[f"{prefix}lower_wick_ratio"] = ((pd.concat([c, o], axis=1).min(axis=1) - l) / wick_safe).fillna(0)
    out[f"{prefix}is_bullish"] = (c > o).fillna(False).astype(int)
    out[f"{prefix}is_bearish"] = (c < o).fillna(False).astype(int)

    # Consecutive candles
    bull = out[f"{prefix}is_bullish"].astype(bool)
    out[f"{prefix}consec_bullish_2"] = (bull & bull.shift(1).fillna(False)).astype(int)
    out[f"{prefix}consec_bullish_3"] = (bull & bull.shift(1).fillna(False) & bull.shift(2).fillna(False)).astype(int)
    bear = out[f"{prefix}is_bearish"].astype(bool)
    out[f"{prefix}consec_bearish_2"] = (bear & bear.shift(1).fillna(False)).astype(int)
    out[f"{prefix}consec_bearish_3"] = (bear & bear.shift(1).fillna(False) & bear.shift(2).fillna(False)).astype(int)

    # Hammer: lower wick > 2x body, upper wick < 0.3x body
    out[f"{prefix}is_hammer"] = (
        (out[f"{prefix}lower_wick_ratio"] > 0.6).fillna(False) &
        (out[f"{prefix}upper_wick_ratio"] < 0.15).fillna(False) &
        (out[f"{prefix}body_ratio"] < 0.35).fillna(False)
    ).astype(int)
    out[f"{prefix}is_shooting_star"] = (
        (out[f"{prefix}upper_wick_ratio"] > 0.6).fillna(False) &
        (out[f"{prefix}lower_wick_ratio"] < 0.15).fillna(False) &
        (out[f"{prefix}body_ratio"] < 0.35).fillna(False)
    ).astype(int)

    # Engulfing patterns
    prev_body = (c.shift(1) - o.shift(1))
    curr_body = (c - o)
    out[f"{prefix}is_engulfing_bull"] = (
        (prev_body < 0).fillna(False) & (curr_body > 0).fillna(False) &
        (o < c.shift(1)).fillna(False) & (c > o.shift(1)).fillna(False)
    ).astype(int)
    out[f"{prefix}is_engulfing_bear"] = (
        (prev_body > 0).fillna(False) & (curr_body < 0).fillna(False) &
        (o > c.shift(1)).fillna(False) & (c < o.shift(1)).fillna(False)
    ).astype(int)

    # ── Lag features ───────────────────────────────────────────────────────
    for col_base in ["rsi_14", "macd_hist", "ema_21", "stoch_k", "adx_14", "cci_20"]:
        src = out[f"{prefix}{col_base}"]
        out[f"{prefix}prev_1_{col_base}"] = src.shift(1)
        out[f"{prefix}prev_3_{col_base}"] = src.shift(3)

    # ── Distance from 52w high ─────────────────────────────────────────────
    high_52w = h.rolling(min(365, len(h))).max()
    out[f"{prefix}dist_from_52w_high"] = (high_52w - c) / c * 100

    # ── EMA vs SMA ─────────────────────────────────────────────────────────
    out[f"{prefix}ema_vs_sma_20"] = out[f"{prefix}ema_21"] - out[f"{prefix}sma_20"]

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  RESAMPLE 1H → HIGHER TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def _resample_ohlcv(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1h OHLCV to a higher timeframe, then forward-fill back to 1h index."""
    df = df_1h.set_index("datetime").copy()
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return resampled


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER COMPUTATION — MULTI-TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════

# Keep a flat list of all indicator columns for strategy references
ALL_INDICATOR_COLUMNS = []


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ALL indicators across multiple timeframes.
    Primary timeframe is 1h. Higher timeframes (4h, 1d) are resampled
    from the 1h data and forward-filled back to the 1h index.
    Returns enriched DataFrame aligned on the 1h index.
    """
    global ALL_INDICATOR_COLUMNS

    df = df.copy()
    if "datetime" not in df.columns:
        log.warning("No datetime column found, computing single-TF only")
        result = df.copy()
        tf_indicators = _compute_tf_indicators(df, "tf_1h_")
        result = pd.concat([result, tf_indicators], axis=1)
        result = result.iloc[200:].reset_index(drop=True)
        ALL_INDICATOR_COLUMNS = [c for c in result.columns if c.startswith("tf_")]
        log.info(f"Indicators computed: {len(result)} rows, {len(result.columns)} columns")
        return result

    # Ensure datetime is proper
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    result = df.copy()

    # ── 1H indicators (primary) ───────────────────────────────────────────
    tf_1h = _compute_tf_indicators(df, "tf_1h_")
    result = pd.concat([result, tf_1h], axis=1)

    # Add hour/day features to 1h
    result["tf_1h_hour_utc"] = result["datetime"].dt.hour
    result["tf_1h_day_of_week"] = result["datetime"].dt.dayofweek

    # ── Higher timeframes ─────────────────────────────────────────────────
    htf_configs = [
        ("4h", "4h"),
        ("1d", "1D"),
    ]

    for tf_label, resample_rule in htf_configs:
        try:
            htf_ohlcv = _resample_ohlcv(df, resample_rule)
            htf_indicators = _compute_tf_indicators(htf_ohlcv, f"tf_{tf_label}_")

            # Forward-fill to 1h index
            htf_indicators.index = pd.to_datetime(htf_indicators.index, utc=True)
            df_dt_index = pd.to_datetime(result["datetime"], utc=True)
            htf_reindexed = htf_indicators.reindex(df_dt_index, method="ffill")
            htf_reindexed.index = result.index

            result = pd.concat([result, htf_reindexed], axis=1)
        except Exception as e:
            log.warning(f"Failed to compute {tf_label} indicators: {e}")

    # ── Legacy compatibility columns ──────────────────────────────────────
    # Map old column names to tf_1h_ prefixed ones for backward compatibility
    legacy_map = {
        "ema_8": "tf_1h_ema_8", "ema_13": "tf_1h_ema_13", "ema_21": "tf_1h_ema_21",
        "ema_34": "tf_1h_ema_34", "ema_55": "tf_1h_ema_55", "ema_89": "tf_1h_ema_89",
        "ema_200": "tf_1h_ema_200",
        "sma_20": "tf_1h_sma_20", "sma_50": "tf_1h_sma_50", "sma_200": "tf_1h_sma_200",
        "adx_14": "tf_1h_adx_14",
        "rsi_7": "tf_1h_rsi_7", "rsi_14": "tf_1h_rsi_14", "rsi_21": "tf_1h_rsi_21",
        "macd_line": "tf_1h_macd_line", "macd_signal": "tf_1h_macd_signal",
        "macd_hist": "tf_1h_macd_hist",
        "stoch_k": "tf_1h_stoch_k", "stoch_d": "tf_1h_stoch_d",
        "atr_14": "tf_1h_atr_14", "atr_pct": "tf_1h_atr_pct",
        "bb_upper": "tf_1h_bb_upper", "bb_middle": "tf_1h_bb_middle",
        "bb_lower": "tf_1h_bb_lower", "bb_width": "tf_1h_bb_width",
        "volume_ma_20": "tf_1h_volume_ma_20", "volume_ratio": "tf_1h_volume_ratio",
        "obv": "tf_1h_obv",
        "candle_body_ratio": "tf_1h_body_ratio",
        "roc_10": "tf_1h_roc_10",
        "regime": "tf_1h_regime",
    }
    for old_name, new_name in legacy_map.items():
        if new_name in result.columns and old_name not in result.columns:
            result[old_name] = result[new_name]

    # ── Warmup NaN handling ────────────────────────────────────────────────
    result = result.iloc[200:].reset_index(drop=True)

    ALL_INDICATOR_COLUMNS = [c for c in result.columns if c.startswith("tf_")]

    log.info(f"Indicators computed: {len(result)} rows, {len(result.columns)} columns (multi-TF)")
    return result
