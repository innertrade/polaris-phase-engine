#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify

# ===================== CONFIG / ENV ===================== #

load_dotenv()

BINANCE_FAPI = os.getenv("PPE_BINANCE_FAPI", "https://fapi.binance.com")

_raw_symbols = os.getenv("PPE_SYMBOLS", "BTCUSDT")
SYMBOLS: List[str] = [s.strip().upper() for s in _raw_symbols.split(",") if s.strip()]

# Base TF for phases
INTERVAL = os.getenv("PPE_INTERVAL", "4h")
KLINES_LIMIT = int(os.getenv("PPE_KLINES_LIMIT", "500"))

# Phase params
EMA_FAST = int(os.getenv("PPE_EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("PPE_EMA_SLOW", "50"))
TREND_UP_THR = float(os.getenv("PPE_TREND_UP_THR", "0.0015"))
TREND_DN_THR = float(os.getenv("PPE_TREND_DN_THR", "-0.0015"))

ATR_LEN = int(os.getenv("PPE_ATR_LEN", "14"))
ATR_Z_LEN = int(os.getenv("PPE_ATR_Z_LEN", "100"))
ATR_QUIET = float(os.getenv("PPE_ATR_QUIET", "-0.7"))
ATR_HOT = float(os.getenv("PPE_ATR_HOT", "0.7"))

VOL_Z_LEN = int(os.getenv("PPE_VOL_Z_LEN", "100"))
VOL_QUIET = float(os.getenv("PPE_VOL_QUIET", "-0.5"))
VOL_HOT = float(os.getenv("PPE_VOL_HOT", "0.5"))

MIN_BARS_IN_PHASE_FOR_SIGNAL = int(os.getenv("PPE_MIN_BARS_IN_PHASE_FOR_SIGNAL", "2"))

# ---- CONF v2 (CVD + OI + Price) ----
CONF_ENABLED = os.getenv("PPE_CONFIRM_ENABLED", "true").lower() in ("1", "true", "yes", "on")
CONF_INTERVAL = os.getenv("PPE_CONFIRM_INTERVAL", "1h")
CONF_KLINES_LIMIT = int(os.getenv("PPE_CONFIRM_KLINES_LIMIT", "200"))

# window for CONF analysis in bars (of CONF_INTERVAL)
CONF_WINDOW_BARS = int(os.getenv("PPE_CONF_WINDOW_BARS", "12"))

# price thresholds
CONF_PRICE_FLAT_PCT = float(os.getenv("PPE_CONF_PRICE_FLAT_PCT", "0.003"))  # 0.3% ~ "flat"
CONF_PRICE_MOVE_PCT = float(os.getenv("PPE_CONF_PRICE_MOVE_PCT", "0.007"))  # 0.7% ~ "move"

# delta ratio threshold (delta_sum / total_volume)
CONF_DELTA_RATIO_THR = float(os.getenv("PPE_CONF_DELTA_RATIO_THR", "0.15"))

# open interest settings
CONF_OI_VALUE_MODE = os.getenv("PPE_CONF_OI_VALUE_MODE", "value").strip().lower()  # "value" or "contracts"
CONF_OI_MOVE_PCT = float(os.getenv("PPE_CONF_OI_MOVE_PCT", "0.01"))  # 1% minimal OI rise

# ---- CACHE to make /signals instant ----
CACHE_ENABLED = os.getenv("PPE_CACHE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
CACHE_REFRESH_SEC = int(os.getenv("PPE_CACHE_REFRESH_SEC", "60"))
CACHE_STARTUP_REFRESH = os.getenv("PPE_CACHE_STARTUP_REFRESH", "true").lower() in ("1", "true", "yes", "on")

HTTP_HOST = os.getenv("PPE_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("PPE_HTTP_PORT", "8001"))

# ===================== LOGGER / HTTP ===================== #

logger = logging.getLogger("polaris-phase-engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PolarisPhaseEngine/2.1-confv2-cache"})

# ===================== BINANCE HELPERS ===================== #


def _binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Load klines from Binance Futures (USDT-M)."""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    resp = SESSION.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"No klines for {symbol} interval={interval}")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)

    num_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for c in num_cols:
        df[c] = df[c].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = df.sort_values("close_time").reset_index(drop=True)
    return df


def _binance_open_interest_hist(symbol: str, period: str, limit: int) -> pd.DataFrame:
    """
    Binance OI history endpoint (USDT-M):
    GET /futures/data/openInterestHist?symbol=...&period=...&limit=...
    """
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": int(limit)}
    resp = SESSION.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"No openInterestHist for {symbol} period={period}")

    df = pd.DataFrame(data)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"Bad openInterestHist payload for {symbol}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ("sumOpenInterest", "sumOpenInterestValue"):
        if c in df.columns:
            df[c] = df[c].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ===================== INDICATORS ===================== #


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(length).mean()


def _zscore(series: pd.Series, length: int) -> pd.Series:
    roll = series.rolling(length)
    mean = roll.mean()
    std = roll.std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def _delta_ratio(df_win: pd.DataFrame) -> float:
    """
    delta_ratio = sum(delta) / sum(volume), where delta ≈ (buy - sell) by aggressor:
      buy  = taker_buy_base
      sell = volume - taker_buy_base
      delta = buy - sell = 2*taker_buy_base - volume
    Roughly in [-1..1].
    """
    if df_win is None or df_win.empty:
        return 0.0
    total_vol = float(df_win["volume"].sum())
    if not np.isfinite(total_vol) or total_vol <= 0:
        return 0.0
    delta_sum = float((2.0 * df_win["taker_buy_base"] - df_win["volume"]).sum())
    return delta_sum / total_vol


def _price_change_pct(df_win: pd.DataFrame) -> float:
    if df_win is None or len(df_win) < 2:
        return 0.0
    p0 = float(df_win["close"].iloc[0])
    p1 = float(df_win["close"].iloc[-1])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return 0.0
    return (p1 - p0) / p0


def _oi_change_pct(oi_df: pd.DataFrame) -> float:
    if oi_df is None or oi_df.empty or len(oi_df) < 2:
        return 0.0

    col = "sumOpenInterestValue" if CONF_OI_VALUE_MODE == "value" else "sumOpenInterest"
    if col not in oi_df.columns:
        col = "sumOpenInterestValue" if "sumOpenInterestValue" in oi_df.columns else "sumOpenInterest"
    if col not in oi_df.columns:
        return 0.0

    v0 = float(oi_df[col].iloc[0])
    v1 = float(oi_df[col].iloc[-1])
    if not np.isfinite(v0) or v0 <= 0 or not np.isfinite(v1):
        return 0.0
    return (v1 - v0) / v0


def _drop_unclosed_last_kline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binance часто возвращает текущую (ещё НЕ закрытую) свечу как последнюю.
    У неё close_time в будущем. Её выкидываем.
    """
    if df is None or df.empty:
        return df
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    if pd.Timestamp(df["close_time"].iloc[-1]).tz_convert("UTC") > now_utc:
        df = df.iloc[:-1].copy()
    return df


# ===================== PHASES 4H ===================== #


def _classify_phases(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ema_fast = _ema(close, EMA_FAST)
    ema_slow = _ema(close, EMA_SLOW)

    slope = (ema_fast - ema_slow) / ema_slow
    trend_dir = np.where(
        slope > TREND_UP_THR,
        1,
        np.where(slope < TREND_DN_THR, -1, 0),
    )

    atr = _atr(df, ATR_LEN)
    atr_z = _zscore(atr, ATR_Z_LEN)
    vol_z = _zscore(df["volume"], VOL_Z_LEN)

    regime = np.where(
        (atr_z < ATR_QUIET) & (vol_z < VOL_QUIET),
        "QUIET",
        np.where((atr_z > ATR_HOT) & (vol_z > VOL_HOT), "HOT", "NORMAL"),
    )

    dist_fast = (close - ema_fast) / ema_fast

    phase = np.full(len(df), "CORR", dtype=object)

    # PUMP
    phase[(trend_dir == 1) & (regime == "HOT") & (dist_fast > 0)] = "PUMP"
    # DUMP
    phase[(trend_dir == -1) & (regime == "HOT") & (dist_fast < 0)] = "DUMP"
    # ACCUM
    phase[(trend_dir >= 0) & (regime == "QUIET")] = "ACCUM"
    # RANGE
    phase[(regime == "QUIET") & (trend_dir == 0)] = "RANGE"

    pump_count = np.zeros(len(df), dtype=int)
    dump_count = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if i == 0:
            pump_count[i] = 1 if phase[i] == "PUMP" else 0
            dump_count[i] = 1 if phase[i] == "DUMP" else 0
        else:
            if phase[i] == "PUMP":
                pump_count[i] = pump_count[i - 1] + 1
                dump_count[i] = 0
            elif phase[i] == "DUMP":
                dump_count[i] = dump_count[i - 1] + 1
                pump_count[i] = 0
            else:
                pump_count[i] = 0
                dump_count[i] = 0

    phased = pd.DataFrame(
        {
            "close_time": df["close_time"],
            "close": close,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "trend_dir": trend_dir,
            "atr": atr,
            "atr_z": atr_z,
            "vol_z": vol_z,
            "regime": regime,
            "phase": phase,
            "pumpCount": pump_count,
            "dumpCount": dump_count,
        }
    )
    return phased


def compute_phase_for_symbol(symbol: str) -> Dict[str, Any]:
    try:
        df = _binance_klines(symbol, INTERVAL, KLINES_LIMIT)
    except Exception as exc:
        logger.exception("Failed to load klines for %s: %s", symbol, exc)
        return {"symbol": symbol, "error": str(exc)}

    phased = _classify_phases(df)
    last = phased.iloc[-1]

    last_close_time = df["close_time"].iloc[-1]
    last_high = df["high"].iloc[-1]
    last_low = df["low"].iloc[-1]

    return {
        "symbol": symbol,
        "interval": INTERVAL,
        "phase": str(last["phase"]),
        "trendDir": int(last["trend_dir"]),
        "regime": str(last["regime"]),
        "pumpCount": int(last["pumpCount"]),
        "dumpCount": int(last["dumpCount"]),
        "close": float(last["close"]),
        "emaFast": float(last["ema_fast"]),
        "emaSlow": float(last["ema_slow"]),
        "atr": float(last["atr"]) if not pd.isna(last["atr"]) else None,
        "atrZ": float(last["atr_z"]) if not pd.isna(last["atr_z"]) else None,
        "volZ": float(last["vol_z"]) if not pd.isna(last["vol_z"]) else None,
        "timestamp": int(last_close_time.timestamp()),
        "high4h": float(last_high),
        "low4h": float(last_low),
    }


def compute_all_phases() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for symbol in SYMBOLS:
        out[symbol] = compute_phase_for_symbol(symbol)
    return out


# ===================== SIGNALS: PRE ===================== #


def _compute_pre_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []

    for symbol, info in phases.items():
        if not isinstance(info, dict):
            continue
        if info.get("error"):
            continue

        phase = info.get("phase")
        pump_count = int(info.get("pumpCount") or 0)
        dump_count = int(info.get("dumpCount") or 0)

        if phase == "PUMP" and pump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL:
            signals.append(
                {
                    "symbol": symbol,
                    "side": "LONG",
                    "kind": "PRE",
                    "phase": phase,
                    "barsInPhase": pump_count,
                    "interval": info.get("interval"),
                    "trendDir": info.get("trendDir"),
                    "generatedAt": info.get("timestamp"),
                }
            )
        elif phase == "DUMP" and dump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL:
            signals.append(
                {
                    "symbol": symbol,
                    "side": "SHORT",
                    "kind": "PRE",
                    "phase": phase,
                    "barsInPhase": dump_count,
                    "interval": info.get("interval"),
                    "trendDir": info.get("trendDir"),
                    "generatedAt": info.get("timestamp"),
                }
            )

    return signals


# ===================== SIGNALS: CONF v2 (LONG-1/2 SHORT-1/2) ===================== #


def _conf_context(ph: Dict[str, Any]) -> Optional[str]:
    """Return 'LONG' or 'SHORT' if PRE context is active right now, else None."""
    if not ph or ph.get("error"):
        return None

    phase = ph.get("phase")
    trend_dir = int(ph.get("trendDir") or 0)
    pump_count = int(ph.get("pumpCount") or 0)
    dump_count = int(ph.get("dumpCount") or 0)

    if phase == "PUMP" and trend_dir > 0 and pump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL:
        return "LONG"
    if phase == "DUMP" and trend_dir < 0 and dump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL:
        return "SHORT"
    return None


def _compute_conf_signals_v2(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not CONF_ENABLED:
        return []

    signals: List[Dict[str, Any]] = []
    w = max(3, int(CONF_WINDOW_BARS))

    need_klines = max(int(CONF_KLINES_LIMIT), w + 10)
    need_oi = w + 10

    for symbol, ph in phases.items():
        ctx = _conf_context(ph)
        if ctx is None:
            continue

        try:
            df = _binance_klines(symbol, CONF_INTERVAL, need_klines)
            df = _drop_unclosed_last_kline(df)
        except Exception as exc:
            logger.exception("CONF v2: failed to load %s %s klines: %s", symbol, CONF_INTERVAL, exc)
            continue

        if df is None or df.empty or len(df) < w + 2:
            continue

        df_win = df.iloc[-w:].copy()

        pchg = _price_change_pct(df_win)
        dr = _delta_ratio(df_win)

        # OI history
        try:
            oi_df = _binance_open_interest_hist(symbol, CONF_INTERVAL, need_oi)
            if oi_df is None or oi_df.empty or len(oi_df) < 2:
                continue
            oi_df_win = oi_df.iloc[-w:].copy() if len(oi_df) >= w else oi_df.copy()
            oi_chg = _oi_change_pct(oi_df_win)
        except Exception as exc:
            logger.warning("CONF v2: OI not available for %s: %s", symbol, exc)
            continue

        oi_up = oi_chg >= CONF_OI_MOVE_PCT
        price_flat = abs(pchg) <= CONF_PRICE_FLAT_PCT
        price_up = pchg >= CONF_PRICE_MOVE_PCT
        price_dn = pchg <= -CONF_PRICE_MOVE_PCT
        delta_pos = dr >= CONF_DELTA_RATIO_THR
        delta_neg = dr <= -CONF_DELTA_RATIO_THR

        now_ts = int(df_win["close_time"].iloc[-1].timestamp())

        def _emit(side: str, ctype: str) -> Dict[str, Any]:
            return {
                "symbol": symbol,
                "side": side,
                "kind": "CONF",
                "confirmType": ctype,
                "phase": ph.get("phase"),
                "interval": ph.get("interval"),           # 4h context
                "triggerInterval": CONF_INTERVAL,         # confirm TF
                "barsInPhase": int(ph.get("pumpCount") or ph.get("dumpCount") or 0),
                "trendDir": int(ph.get("trendDir") or 0),
                "generatedAt": now_ts,
                # чтобы TG не путался:
                "lookbackBars": w,
                "windowBars": w,
                "priceChangePct": float(pchg),
                "oiChangePct": float(oi_chg),
                "deltaRatio": float(dr),
                "oiMode": CONF_OI_VALUE_MODE,
            }

        if ctx == "LONG":
            # LONG-1: продолжение импульса (OI↑ + Price↑ + Delta+)
            if oi_up and price_up and delta_pos:
                signals.append(_emit("LONG", "LONG-1"))

            # LONG-2: абсорбция (OI↑ + Price ~ flat + Delta-)
            if oi_up and price_flat and delta_neg:
                signals.append(_emit("LONG", "LONG-2"))

        if ctx == "SHORT":
            # SHORT-1: продолжение падения (OI↑ + Price↓ + Delta-)
            if oi_up and price_dn and delta_neg:
                signals.append(_emit("SHORT", "SHORT-1"))

            # SHORT-2: распределение (OI↑ + Price ~ flat + Delta+)
            if oi_up and price_flat and delta_pos:
                signals.append(_emit("SHORT", "SHORT-2"))

    return signals


def compute_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    pre = _compute_pre_signals(phases)
    conf = _compute_conf_signals_v2(phases)
    return pre + conf


# ===================== CACHE (phases + signals) ===================== #

_cache_lock = threading.Lock()
_cache_state: Dict[str, Any] = {
    "phases": {},
    "signals": [],
    "updatedAt": 0,
    "durationMs": None,
    "error": None,
}
_refresh_in_progress = False


def _cache_get() -> Dict[str, Any]:
    with _cache_lock:
        return dict(_cache_state)


def _cache_set(phases: Dict[str, Any], signals: List[Dict[str, Any]], duration_ms: int) -> None:
    with _cache_lock:
        _cache_state["phases"] = phases
        _cache_state["signals"] = signals
        _cache_state["updatedAt"] = int(time.time())
        _cache_state["durationMs"] = int(duration_ms)
        _cache_state["error"] = None


def _cache_set_error(err: str, duration_ms: int) -> None:
    with _cache_lock:
        _cache_state["updatedAt"] = int(time.time())
        _cache_state["durationMs"] = int(duration_ms)
        _cache_state["error"] = err


def _refresh_cache_once(force: bool = False) -> None:
    global _refresh_in_progress

    if not CACHE_ENABLED:
        return

    with _cache_lock:
        if _refresh_in_progress and not force:
            return
        _refresh_in_progress = True

    t0 = time.time()
    try:
        phases = compute_all_phases()
        signals = compute_signals(phases)
        dur = int((time.time() - t0) * 1000)
        _cache_set(phases, signals, dur)
        logger.info("Cache refreshed: symbols=%d signals=%d %dms", len(phases), len(signals), dur)
    except Exception as exc:
        dur = int((time.time() - t0) * 1000)
        logger.exception("Cache refresh failed: %s", exc)
        _cache_set_error(str(exc), dur)
    finally:
        with _cache_lock:
            _refresh_in_progress = False


def _cache_loop() -> None:
    if CACHE_STARTUP_REFRESH:
        _refresh_cache_once(force=True)

    while True:
        time.sleep(max(5, CACHE_REFRESH_SEC))
        _refresh_cache_once()


# ===================== FLASK HTTP ===================== #

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Any:
    st = _cache_get() if CACHE_ENABLED else None
    age = None
    if st and st.get("updatedAt"):
        age = int(time.time() - int(st["updatedAt"]))

    return jsonify(
        {
            "status": "ok",
            "symbols": SYMBOLS,
            "interval": INTERVAL,
            "confirmEnabled": CONF_ENABLED,
            "confirmInterval": CONF_INTERVAL,
            "cacheEnabled": CACHE_ENABLED,
            "cacheRefreshSec": CACHE_REFRESH_SEC if CACHE_ENABLED else None,
            "cacheUpdatedAt": st.get("updatedAt") if st else None,
            "cacheAgeSec": age,
            "cacheDurationMs": st.get("durationMs") if st else None,
            "cacheError": st.get("error") if st else None,
        }
    )


@app.route("/phases", methods=["GET"])
def phases_endpoint() -> Any:
    if CACHE_ENABLED:
        st = _cache_get()
        if not st.get("updatedAt"):
            _refresh_cache_once(force=True)
            st = _cache_get()
        return jsonify(st.get("phases", {}))

    phases = compute_all_phases()
    return jsonify(phases)


@app.route("/signals", methods=["GET"])
def signals_endpoint() -> Any:
    if CACHE_ENABLED:
        st = _cache_get()
        if not st.get("updatedAt"):
            _refresh_cache_once(force=True)
            st = _cache_get()
        return jsonify({"signals": st.get("signals", [])})

    phases = compute_all_phases()
    signals = compute_signals(phases)
    return jsonify({"signals": signals})


def main() -> None:
    logger.info(
        "Starting Polaris Phase Engine on %s:%s symbols=%s interval=%s confirm=%s/%s cache=%s refresh=%ss",
        HTTP_HOST,
        HTTP_PORT,
        ",".join(SYMBOLS),
        INTERVAL,
        "on" if CONF_ENABLED else "off",
        CONF_INTERVAL,
        "on" if CACHE_ENABLED else "off",
        CACHE_REFRESH_SEC,
    )

    if CACHE_ENABLED:
        t = threading.Thread(target=_cache_loop, name="ppe-cache", daemon=True)
        t.start()

    app.run(host=HTTP_HOST, port=HTTP_PORT)


if __name__ == "__main__":
    main()
