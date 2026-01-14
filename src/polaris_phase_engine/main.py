#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify

load_dotenv()

# =========================
# logging
# =========================
logger = logging.getLogger("polaris-phase-engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# =========================
# helpers
# =========================
def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on", "y")


def _now_ts() -> int:
    return int(time.time())


def _iso_utc(ts_sec: int) -> str:
    return datetime.fromtimestamp(int(ts_sec), tz=timezone.utc).isoformat()


def _interval_to_ms(interval: str) -> int:
    # supports: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d
    interval = (interval or "").strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1]) * 60_000
    if interval.endswith("h"):
        return int(interval[:-1]) * 3_600_000
    if interval.endswith("d"):
        return int(interval[:-1]) * 86_400_000
    # fallback (assume hours)
    return 3_600_000


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# =========================
# config
# =========================
HTTP_HOST = _env_str("PPE_HTTP_HOST", "127.0.0.1")
HTTP_PORT = _env_int("PPE_HTTP_PORT", 8001)

SYMBOLS = [s.strip().upper() for s in _env_str(
    "PPE_SYMBOLS",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,ADAUSDT,TONUSDT",
).split(",") if s.strip()]

INTERVAL_4H = _env_str("PPE_INTERVAL", "4h")
KLINES_LIMIT_4H = _env_int("PPE_KLINES_LIMIT", 500)
MIN_BARS_IN_PHASE_FOR_PRE = _env_int("PPE_MIN_BARS_IN_PHASE_FOR_SIGNAL", 2)

CACHE_ENABLED = _env_bool("PPE_CACHE_ENABLED", True)
CACHE_REFRESH_SEC = _env_int("PPE_CACHE_REFRESH_SEC", 60)

# phase thresholds (same “mvp” logic)
TREND_THR = _env_float("PPE_TREND_THR", 0.0015)
PH_ATR_Z_THR = _env_float("PPE_PHASE_ATR_Z_THR", 0.7)
PH_VOL_Z_THR = _env_float("PPE_PHASE_VOL_Z_THR", 0.5)
PH_ACC_ATR_Z_THR = _env_float("PPE_PHASE_ACC_ATR_Z_THR", -0.7)
PH_ACC_VOL_Z_THR = _env_float("PPE_PHASE_ACC_VOL_Z_THR", -0.5)
Z_WINDOW = _env_int("PPE_Z_WINDOW", 100)

# CONF (per TЗ)
CONFIRM_ENABLED = _env_bool("PPE_CONFIRM_ENABLED", True)
CONF_INTERVAL_1H = _env_str("PPE_CONFIRM_INTERVAL", "1h")
CONF_WINDOW_BARS = _env_int("PPE_CONFIRM_WINDOW_BARS", 6)          # N
CONF_TRIGGER_BARS = _env_int("PPE_CONFIRM_TRIGGER_BARS", 3)        # 2–3

PRICE_FLAT_PCT = _env_float("PPE_CONF_PRICE_FLAT_PCT", 0.003)      # 0.3%
PRICE_MOVE_PCT = _env_float("PPE_CONF_PRICE_MOVE_PCT", 0.007)      # 0.7%
OI_MOVE_PCT = _env_float("PPE_CONF_OI_MOVE_PCT", 0.01)             # +1%
DELTA_RATIO_THR = _env_float("PPE_CONF_DELTA_RATIO_THR", 0.15)     # 0.15

# CVD
CVD_ENABLED = _env_bool("PPE_CVD_ENABLED", True)

# Binance
BINANCE_FAPI_BASE = _env_str("PPE_BINANCE_FAPI_BASE", "https://fapi.binance.com").rstrip("/")
BINANCE_TIMEOUT_SEC = _env_int("PPE_BINANCE_TIMEOUT_SEC", 10)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polaris-phase-engine/3.0"})


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = BINANCE_TIMEOUT_SEC) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url} params={params} err={last_exc}")


def _binance_fapi_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    fapi/v1/klines returns:
    [
      open_time, open, high, low, close, volume, close_time,
      quote_asset_volume, number_of_trades,
      taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
    ]
    """
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines"
    data = _http_get_json(url, params={"symbol": symbol, "interval": interval, "limit": limit})
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for k in data:
        if not isinstance(k, list) or len(k) < 11:
            continue
        rows.append({
            "open_time_ms": int(k[0]),
            "open": _safe_float(k[1]),
            "high": _safe_float(k[2]),
            "low": _safe_float(k[3]),
            "close": _safe_float(k[4]),
            "volume": _safe_float(k[5]),
            "close_time_ms": int(k[6]),
            "quote_volume": _safe_float(k[7]),
            "trades": int(_safe_float(k[8])),
            "taker_buy_base": _safe_float(k[9]),
            "taker_buy_quote": _safe_float(k[10]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # drop still-forming candle if present (close_time in the future)
    now_ms = int(time.time() * 1000)
    df = df[df["close_time_ms"] <= now_ms].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def _binance_oi_hist(symbol: str, period: str, limit: int) -> pd.DataFrame:
    """
    /futures/data/openInterestHist
    returns list of:
    { "symbol": "...", "sumOpenInterest": "...", "sumOpenInterestValue": "...", "timestamp": 123 }
    """
    url = f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist"
    data = _http_get_json(url, params={"symbol": symbol, "period": period, "limit": limit})
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        rows.append({
            "ts_ms": int(_safe_float(it.get("timestamp", 0))),
            "oi": _safe_float(it.get("sumOpenInterest", 0.0)),
            "oi_value": _safe_float(it.get("sumOpenInterestValue", 0.0)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    now_ms = int(time.time() * 1000)
    df = df[df["ts_ms"] <= now_ms].copy()
    df.sort_values("ts_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _binance_ticker_24h(symbol: str) -> Dict[str, Any]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/ticker/24hr"
    data = _http_get_json(url, params={"symbol": symbol})
    return data if isinstance(data, dict) else {}


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    z = (s - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)


def _phase_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      ema20, ema50, trendDir, atr14, atrZ, volZ, phase
    """
    if df.empty or len(df) < 60:
        return df

    close = df["close"]
    vol = df["volume"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    trend = (ema20 - ema50) / ema50.replace(0, np.nan)
    trend = trend.fillna(0.0)

    trend_dir = np.where(trend > TREND_THR, 1, np.where(trend < -TREND_THR, -1, 0))

    atr14 = _compute_atr(df, 14).fillna(method="bfill").fillna(0.0)
    atr_z = _zscore(atr14, Z_WINDOW)
    vol_z = _zscore(vol, Z_WINDOW)

    phase = np.select(
        [
            (trend_dir == 1) & (atr_z > PH_ATR_Z_THR) & (vol_z > PH_VOL_Z_THR),
            (trend_dir == -1) & (atr_z > PH_ATR_Z_THR) & (vol_z > PH_VOL_Z_THR),
            (trend_dir == 0) & (atr_z < PH_ACC_ATR_Z_THR) & (vol_z < PH_ACC_VOL_Z_THR),
        ],
        ["PUMP", "DUMP", "ACCUM"],
        default="CORR",
    )

    out = df.copy()
    out["ema20"] = ema20
    out["ema50"] = ema50
    out["trend"] = trend
    out["trendDir"] = trend_dir.astype(int)
    out["atr14"] = atr14
    out["atrZ"] = atr_z
    out["volZ"] = vol_z
    out["phase"] = phase
    return out


def _bars_in_same_phase(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    last = str(df["phase"].iloc[-1])
    n = 0
    for i in range(len(df) - 1, -1, -1):
        if str(df["phase"].iloc[i]) != last:
            break
        n += 1
    return n


def _get_symbol_phase(symbol: str) -> Dict[str, Any]:
    df = _binance_fapi_klines(symbol, INTERVAL_4H, KLINES_LIMIT_4H)
    if df.empty or len(df) < 80:
        return {"ok": False, "symbol": symbol, "interval": INTERVAL_4H, "error": "not enough klines"}

    df = _phase_series(df)
    if df.empty:
        return {"ok": False, "symbol": symbol, "interval": INTERVAL_4H, "error": "phase calc failed"}

    last = df.iloc[-1]
    bars = _bars_in_same_phase(df)

    phase = str(last["phase"])
    pump_count = int(bars if phase == "PUMP" else 0)
    dump_count = int(bars if phase == "DUMP" else 0)

    close_time_ms = int(last["close_time_ms"])
    close_ts = int(close_time_ms // 1000)

    ema20 = float(last["ema20"])
    atr14 = float(last["atr14"])
    close = float(last["close"])

    price_not_too_high = bool(close <= (ema20 + 2.0 * atr14)) if atr14 > 0 else True
    price_not_too_low = bool(close >= (ema20 - 2.0 * atr14)) if atr14 > 0 else True

    return {
        "ok": True,
        "symbol": symbol,
        "interval": INTERVAL_4H,
        "time": _iso_utc(close_ts),
        "ts": close_ts,
        "phase": phase,
        "trendDir": int(last["trendDir"]),
        "atrZ": float(last["atrZ"]),
        "volZ": float(last["volZ"]),
        "barsInPhase": int(bars),
        "pumpCount": pump_count,
        "dumpCount": dump_count,
        "close": close,
        "ema20": ema20,
        "atr14": atr14,
        "priceNotTooHigh": price_not_too_high,
        "priceNotTooLow": price_not_too_low,
    }


def _make_pre_signal(ph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not ph.get("ok"):
        return None

    phase = str(ph.get("phase", ""))
    trend_dir = int(ph.get("trendDir", 0))
    bars = int(ph.get("barsInPhase", 0))

    if phase == "PUMP" and trend_dir == 1 and bars >= MIN_BARS_IN_PHASE_FOR_PRE and bool(ph.get("priceNotTooHigh", True)):
        return {
            "symbol": ph["symbol"],
            "side": "LONG",
            "kind": "PRE",
            "phase": "PUMP",
            "interval": INTERVAL_4H,
            "trendDir": trend_dir,
            "barsInPhase": bars,
            "generatedAt": int(ph.get("ts") or 0),
            "price": float(ph.get("close") or 0.0),
        }

    if phase == "DUMP" and trend_dir == -1 and bars >= MIN_BARS_IN_PHASE_FOR_PRE and bool(ph.get("priceNotTooLow", True)):
        return {
            "symbol": ph["symbol"],
            "side": "SHORT",
            "kind": "PRE",
            "phase": "DUMP",
            "interval": INTERVAL_4H,
            "trendDir": trend_dir,
            "barsInPhase": bars,
            "generatedAt": int(ph.get("ts") or 0),
            "price": float(ph.get("close") or 0.0),
        }

    return None


def _price_change_pct(df_win: pd.DataFrame) -> float:
    c0 = float(df_win["close"].iloc[0])
    c1 = float(df_win["close"].iloc[-1])
    return (c1 - c0) / c0 if c0 else 0.0


def _oi_change_pct(oi_win: pd.DataFrame) -> float:
    o0 = float(oi_win["oi"].iloc[0])
    o1 = float(oi_win["oi"].iloc[-1])
    return (o1 - o0) / o0 if o0 else 0.0


def _cvd_metrics(df_win: pd.DataFrame) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns:
      cvd_delta_base (sum delta over window),
      cvd_delta_quote (sum delta_quote over window),
      delta_ratio (abs(sum(delta))/sum(volume)),
      last_delta_base,
      last_delta_quote,
      buy_vol_last,
      taker_buy_pct_last
    """
    vol = df_win["volume"].astype(float)
    buy = df_win["taker_buy_base"].astype(float)

    # delta per bar: buy - sell = buy - (vol-buy) = 2*buy - vol
    delta_base = (2.0 * buy) - vol
    close = df_win["close"].astype(float)
    delta_quote = delta_base * close

    cvd_delta_base = float(delta_base.sum())
    cvd_delta_quote = float(delta_quote.sum())

    vol_sum = float(vol.sum())
    delta_ratio = (abs(cvd_delta_base) / vol_sum) if vol_sum > 0 else 0.0

    last_delta_base = float(delta_base.iloc[-1])
    last_delta_quote = float(delta_quote.iloc[-1])
    buy_vol_last = float(buy.iloc[-1])
    vol_last = float(vol.iloc[-1])
    taker_buy_pct_last = (buy_vol_last / vol_last) if vol_last > 0 else 0.0
    return (
        cvd_delta_base,
        cvd_delta_quote,
        delta_ratio,
        last_delta_base,
        last_delta_quote,
        buy_vol_last,
        taker_buy_pct_last,
    )


def _break_up(df: pd.DataFrame, bars: int) -> bool:
    if df.empty or len(df) < bars + 1:
        return False
    c = float(df["close"].iloc[-1])
    prev_high = float(df["high"].iloc[-(bars + 1):-1].max())
    return c > prev_high


def _break_down(df: pd.DataFrame, bars: int) -> bool:
    if df.empty or len(df) < bars + 1:
        return False
    c = float(df["close"].iloc[-1])
    prev_low = float(df["low"].iloc[-(bars + 1):-1].min())
    return c < prev_low


def _conf_context(ph: Dict[str, Any]) -> Dict[str, Any]:
    phase = str(ph.get("phase", ""))
    return {
        "phase": phase,
        "trendDir": int(ph.get("trendDir", 0)),
        "pumpCount": int(ph.get("pumpCount", 0)),
        "dumpCount": int(ph.get("dumpCount", 0)),
        "barsInPhase": int(ph.get("barsInPhase", 0)),
    }


def _compute_conf_signals(symbol: str, ph: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not CONFIRM_ENABLED:
        return []
    if not CVD_ENABLED:
        return []

    w = max(2, CONF_WINDOW_BARS)
    trig_bars = max(2, CONF_TRIGGER_BARS)

    # Need a bit more than window for trigger check and safety
    need_kl = max(80, w + trig_bars + 5)
    df1 = _binance_fapi_klines(symbol, CONF_INTERVAL_1H, need_kl)
    if df1.empty or len(df1) < (w + trig_bars + 1):
        return []

    # use last w + trig_bars bars as “current”
    df_tail = df1.iloc[-(w + trig_bars + 1):].copy()

    # window for metrics: last w bars ending at the last closed candle
    df_win = df_tail.iloc[-w:].copy()

    # OI aligned by count (good enough)
    try:
        oi_df = _binance_oi_hist(symbol, CONF_INTERVAL_1H, max(50, w + 5))
        if oi_df.empty or len(oi_df) < 2:
            return []
        oi_win = oi_df.iloc[-(w + 1):].copy() if len(oi_df) >= (w + 1) else oi_df.copy()
        oi_chg = _oi_change_pct(oi_win)
    except Exception as exc:
        logger.warning("CONF: OI not available for %s: %s", symbol, exc)
        return []

    pchg = _price_change_pct(df_win)

    (
        cvd_delta_base,
        cvd_delta_quote,
        delta_ratio,
        last_delta_base,
        last_delta_quote,
        buy_vol_last,
        taker_buy_pct_last,
    ) = _cvd_metrics(df_win)

    now_ts = int(int(df_tail["close_time_ms"].iloc[-1]) // 1000)

    # Conditions (per TЗ)
    oi_up = oi_chg >= OI_MOVE_PCT
    oi_pos = oi_chg > 0.0

    price_flat_up = pchg <= PRICE_FLAT_PCT           # “не выросла > 0.3%”
    price_flat_dn = pchg >= -PRICE_FLAT_PCT          # “не просела > 0.3%”
    price_up = pchg >= PRICE_MOVE_PCT
    price_dn = pchg <= -PRICE_MOVE_PCT

    cvd_pos = cvd_delta_base > 0
    cvd_neg = cvd_delta_base < 0
    delta_ok = delta_ratio >= DELTA_RATIO_THR

    trig_up = _break_up(df_tail, trig_bars)
    trig_dn = _break_down(df_tail, trig_bars)

    ctx = _conf_context(ph)
    phase4h = str(ctx.get("phase", ""))

    signals: List[Dict[str, Any]] = []

    # LONG_1_ABSORB (dump context) + micro-break up
    if (phase4h == "DUMP" or int(ctx.get("dumpCount", 0)) >= 2) and price_flat_dn and oi_up and cvd_neg and delta_ok and trig_up:
        signals.append({
            "symbol": symbol,
            "side": "LONG",
            "kind": "CONF",
            "confirmType": "LONG_1_ABSORB",
            "phase": phase4h,
            "interval": INTERVAL_4H,
            "triggerInterval": CONF_INTERVAL_1H,
            "barsInPhase": int(ctx.get("dumpCount", 0)) or int(ctx.get("barsInPhase", 0)),
            "trendDir": int(ctx.get("trendDir", 0)),
            "generatedAt": now_ts,
            "lookbackBars": w,
            "triggerBars": trig_bars,
            "priceChangePct": pchg,
            "oiChangePct": oi_chg,
            "cvdDeltaBase": cvd_delta_base,
            "cvdDeltaQuote": cvd_delta_quote,
            "deltaRatio": delta_ratio,
            "info": {
                "cvdLastBase": last_delta_base,
                "cvdLastQuote": last_delta_quote,
                "buyVolLast": buy_vol_last,
                "sellVolLast": float(df_win["volume"].iloc[-1]) - buy_vol_last,
                "takerBuyPctLast": taker_buy_pct_last,
                "volLast": float(df_win["volume"].iloc[-1]),
            },
        })

    # SHORT_1_ABSORB (pump context) + micro-break down
    if (phase4h == "PUMP" or int(ctx.get("pumpCount", 0)) >= 2) and price_flat_up and oi_up and cvd_pos and delta_ok and trig_dn:
        signals.append({
            "symbol": symbol,
            "side": "SHORT",
            "kind": "CONF",
            "confirmType": "SHORT_1_ABSORB",
            "phase": phase4h,
            "interval": INTERVAL_4H,
            "triggerInterval": CONF_INTERVAL_1H,
            "barsInPhase": int(ctx.get("pumpCount", 0)) or int(ctx.get("barsInPhase", 0)),
            "trendDir": int(ctx.get("trendDir", 0)),
            "generatedAt": now_ts,
            "lookbackBars": w,
            "triggerBars": trig_bars,
            "priceChangePct": pchg,
            "oiChangePct": oi_chg,
            "cvdDeltaBase": cvd_delta_base,
            "cvdDeltaQuote": cvd_delta_quote,
            "deltaRatio": delta_ratio,
            "info": {
                "cvdLastBase": last_delta_base,
                "cvdLastQuote": last_delta_quote,
                "buyVolLast": buy_vol_last,
                "sellVolLast": float(df_win["volume"].iloc[-1]) - buy_vol_last,
                "takerBuyPctLast": taker_buy_pct_last,
                "volLast": float(df_win["volume"].iloc[-1]),
            },
        })

    # LONG_2_BREAK (participation up) + break up
    if price_up and oi_pos and cvd_pos and delta_ok and trig_up:
        signals.append({
            "symbol": symbol,
            "side": "LONG",
            "kind": "CONF",
            "confirmType": "LONG_2_BREAK",
            "phase": phase4h,
            "interval": INTERVAL_4H,
            "triggerInterval": CONF_INTERVAL_1H,
            "barsInPhase": int(ctx.get("barsInPhase", 0)),
            "trendDir": int(ctx.get("trendDir", 0)),
            "generatedAt": now_ts,
            "lookbackBars": w,
            "triggerBars": trig_bars,
            "priceChangePct": pchg,
            "oiChangePct": oi_chg,
            "cvdDeltaBase": cvd_delta_base,
            "cvdDeltaQuote": cvd_delta_quote,
            "deltaRatio": delta_ratio,
            "info": {
                "cvdLastBase": last_delta_base,
                "cvdLastQuote": last_delta_quote,
                "buyVolLast": buy_vol_last,
                "sellVolLast": float(df_win["volume"].iloc[-1]) - buy_vol_last,
                "takerBuyPctLast": taker_buy_pct_last,
                "volLast": float(df_win["volume"].iloc[-1]),
            },
        })

    # SHORT_2_BREAK (participation down) + break down
    if price_dn and oi_pos and cvd_neg and delta_ok and trig_dn:
        signals.append({
            "symbol": symbol,
            "side": "SHORT",
            "kind": "CONF",
            "confirmType": "SHORT_2_BREAK",
            "phase": phase4h,
            "interval": INTERVAL_4H,
            "triggerInterval": CONF_INTERVAL_1H,
            "barsInPhase": int(ctx.get("barsInPhase", 0)),
            "trendDir": int(ctx.get("trendDir", 0)),
            "generatedAt": now_ts,
            "lookbackBars": w,
            "triggerBars": trig_bars,
            "priceChangePct": pchg,
            "oiChangePct": oi_chg,
            "cvdDeltaBase": cvd_delta_base,
            "cvdDeltaQuote": cvd_delta_quote,
            "deltaRatio": delta_ratio,
            "info": {
                "cvdLastBase": last_delta_base,
                "cvdLastQuote": last_delta_quote,
                "buyVolLast": buy_vol_last,
                "sellVolLast": float(df_win["volume"].iloc[-1]) - buy_vol_last,
                "takerBuyPctLast": taker_buy_pct_last,
                "volLast": float(df_win["volume"].iloc[-1]),
            },
        })

    return signals


# =========================
# cache
# =========================
CACHE_LOCK = threading.Lock()
CACHE: Dict[str, Any] = {
    "updatedAt": 0,
    "durationMs": 0,
    "error": None,
    "phases": {},
    "signals": [],
}


def _refresh_cache_once() -> None:
    t0 = time.time()
    err: Optional[str] = None
    phases: Dict[str, Any] = {}
    signals: List[Dict[str, Any]] = []

    try:
        # phases + PRE
        for sym in SYMBOLS:
            ph = _get_symbol_phase(sym)
            phases[sym] = ph
            pre = _make_pre_signal(ph)
            if pre:
                signals.append(pre)

        # CONF
        if CONFIRM_ENABLED:
            for sym in SYMBOLS:
                ph = phases.get(sym) or {}
                if ph.get("ok"):
                    signals.extend(_compute_conf_signals(sym, ph))

    except Exception as exc:
        err = str(exc)
        logger.exception("Cache refresh failed: %s", exc)

    dt_ms = int((time.time() - t0) * 1000)
    with CACHE_LOCK:
        CACHE["updatedAt"] = _now_ts()
        CACHE["durationMs"] = dt_ms
        CACHE["error"] = err
        CACHE["phases"] = phases
        CACHE["signals"] = signals

    logger.info("Cache refreshed: symbols=%d signals=%d %dms", len(SYMBOLS), len(signals), dt_ms)


def _cache_loop() -> None:
    while True:
        _refresh_cache_once()
        time.sleep(max(5, CACHE_REFRESH_SEC))


# =========================
# HTTP
# =========================
app = Flask(__name__)


@app.get("/health")
def health() -> Any:
    with CACHE_LOCK:
        updated_at = int(CACHE.get("updatedAt") or 0)
        duration_ms = int(CACHE.get("durationMs") or 0)
        err = CACHE.get("error")

    age = max(0, _now_ts() - updated_at) if updated_at else 0

    payload = {
        "status": "ok",
        "symbols": SYMBOLS,
        "interval": INTERVAL_4H,
        "confirmEnabled": bool(CONFIRM_ENABLED),
        "confirmInterval": CONF_INTERVAL_1H,
        "cvdEnabled": bool(CVD_ENABLED),
        "cvdWindowBars": int(CONF_WINDOW_BARS),
        "cacheEnabled": bool(CACHE_ENABLED),
        "cacheRefreshSec": int(CACHE_REFRESH_SEC),
        "cacheUpdatedAt": updated_at,
        "cacheAgeSec": age,
        "cacheDurationMs": duration_ms,
        "cacheError": err,
    }
    return jsonify(payload)


@app.get("/phases")
def phases() -> Any:
    with CACHE_LOCK:
        ph = CACHE.get("phases") or {}
    return jsonify(ph)


@app.get("/signals")
def signals() -> Any:
    with CACHE_LOCK:
        sigs = CACHE.get("signals") or []
    return jsonify({"signals": sigs})


def main() -> None:
    logger.info(
        "Starting Polaris Phase Engine on %s:%s symbols=%s interval=%s confirm=%s/%s cvd=%s window=%s cache=%s refresh=%ss",
        HTTP_HOST,
        HTTP_PORT,
        ",".join(SYMBOLS),
        INTERVAL_4H,
        "on" if CONFIRM_ENABLED else "off",
        CONF_INTERVAL_1H,
        "on" if CVD_ENABLED else "off",
        CONF_WINDOW_BARS,
        "on" if CACHE_ENABLED else "off",
        CACHE_REFRESH_SEC,
    )

    if CACHE_ENABLED:
        th = threading.Thread(target=_cache_loop, name="cache-loop", daemon=True)
        th.start()
    else:
        _refresh_cache_once()

    app.run(host=HTTP_HOST, port=HTTP_PORT)


if __name__ == "__main__":
    main()
