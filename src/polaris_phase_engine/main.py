#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify

# ===================== КОНФИГ / ENV ===================== #

load_dotenv()

BINANCE_FAPI = os.getenv("PPE_BINANCE_FAPI", "https://fapi.binance.com")

_raw_symbols = os.getenv("PPE_SYMBOLS", "BTCUSDT")
SYMBOLS: List[str] = [s.strip().upper() for s in _raw_symbols.split(",") if s.strip()]

# базовый ТФ фаз – 4H
INTERVAL = os.getenv("PPE_INTERVAL", "4h")
KLINES_LIMIT = int(os.getenv("PPE_KLINES_LIMIT", "500"))

# Параметры фаз (синхронны с PolarisPhaseMVP)
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

# --- CONF v2 (OI + CVD + Price) на младшем ТФ (по умолчанию 1h) ---

CONF_ENABLED = os.getenv("PPE_CONFIRM_ENABLED", "true").lower() in ("1", "true", "yes", "on")
CONF_INTERVAL = os.getenv("PPE_CONFIRM_INTERVAL", "1h")
CONF_KLINES_LIMIT = int(os.getenv("PPE_CONFIRM_KLINES_LIMIT", "200"))

CONF_LOOKBACK_BARS = int(os.getenv("PPE_CONF_LOOKBACK_BARS", "12"))  # 12h на 1h
CONF_OI_CHANGE_PCT_THR = float(os.getenv("PPE_CONF_OI_CHANGE_PCT_THR", "0.015"))  # 1.5%
CONF_DELTA_RATIO_THR = float(os.getenv("PPE_CONF_DELTA_RATIO_THR", "0.15"))        # 15%
CONF_PRICE_FLAT_PCT = float(os.getenv("PPE_CONF_PRICE_FLAT_PCT", "0.003"))         # 0.3%
CONF_PRICE_MOVE_PCT = float(os.getenv("PPE_CONF_PRICE_MOVE_PCT", "0.007"))         # 0.7%

# Можно считать OI в USD (как в TV) или в контрактах
CONF_OI_VALUE_MODE = os.getenv("PPE_CONF_OI_VALUE_MODE", "value").lower()  # "value" | "contracts"

HTTP_HOST = os.getenv("PPE_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("PPE_HTTP_PORT", "8001"))

# ===================== ЛОГГЕР / HTTP ===================== #

logger = logging.getLogger("polaris-phase-engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PolarisPhaseEngine/2.0-conf-v2"})

# ===================== УТИЛИТЫ ===================== #

def _binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = SESSION.get(url, params=params, timeout=10)
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
        "open", "high", "low", "close", "volume", "quote_volume",
        "taker_buy_base", "taker_buy_quote",
    ]
    for c in num_cols:
        df[c] = df[c].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def _binance_open_interest_hist(symbol: str, period: str, limit: int) -> pd.DataFrame:
    """
    Binance Futures Open Interest History.
    period: "5m","15m","30m","1h","2h","4h","6h","12h","1d"
    """
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    resp = SESSION.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"No openInterestHist for {symbol} period={period}")

    df = pd.DataFrame(data)
    # timestamp приходит в ms
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)

    # sumOpenInterest / sumOpenInterestValue обычно строки
    for c in ["sumOpenInterest", "sumOpenInterestValue"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(length).mean()


def _zscore(series: pd.Series, length: int) -> pd.Series:
    roll = series.rolling(length)
    mean = roll.mean()
    std = roll.std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def _delta_ratio(df: pd.DataFrame, bars: int) -> float:
    """
    Агрессорная дельта за окно (нормированная):
    buy = taker_buy_base
    sell = volume - buy
    delta = buy - sell = 2*buy - volume
    ratio = sum(delta) / sum(volume)
    """
    if len(df) < bars:
        return 0.0
    tail = df.tail(bars)
    vol = float(tail["volume"].sum())
    if vol <= 0:
        return 0.0
    buy = float(tail["taker_buy_base"].sum())
    delta = (2.0 * buy) - vol
    return float(delta / vol)


def _price_change_pct(df: pd.DataFrame, bars: int) -> float:
    if len(df) < bars + 1:
        return 0.0
    c0 = float(df["close"].iloc[-(bars + 1)])
    c1 = float(df["close"].iloc[-1])
    if c0 <= 0:
        return 0.0
    return float((c1 / c0) - 1.0)


def _oi_change_pct(oi: pd.DataFrame, bars: int, value_mode: str) -> float:
    """
    value_mode:
      - "value" => sumOpenInterestValue (USD)
      - "contracts" => sumOpenInterest (contracts)
    """
    col = "sumOpenInterestValue" if value_mode == "value" else "sumOpenInterest"
    if col not in oi.columns:
        # fallback
        col = "sumOpenInterestValue" if "sumOpenInterestValue" in oi.columns else "sumOpenInterest"

    if len(oi) < bars + 1:
        return 0.0
    v0 = float(oi[col].iloc[-(bars + 1)])
    v1 = float(oi[col].iloc[-1])
    if v0 <= 0:
        return 0.0
    return float((v1 / v0) - 1.0)


# ===================== РАСЧЁТ ФАЗ 4H ===================== #

def _classify_phases(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ema_fast = _ema(close, EMA_FAST)
    ema_slow = _ema(close, EMA_SLOW)

    slope = (ema_fast - ema_slow) / ema_slow
    trend_dir = np.where(
        slope > TREND_UP_THR, 1,
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

    phase[(trend_dir == 1) & (regime == "HOT") & (dist_fast > 0)] = "PUMP"
    phase[(trend_dir == -1) & (regime == "HOT") & (dist_fast < 0)] = "DUMP"
    phase[(trend_dir >= 0) & (regime == "QUIET")] = "ACCUM"
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

    return pd.DataFrame(
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
    return {symbol: compute_phase_for_symbol(symbol) for symbol in SYMBOLS}


# ===================== СИГНАЛЫ: PRE ===================== #

def _compute_pre_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    for symbol, info in phases.items():
        if not isinstance(info, dict) or info.get("error"):
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


# ===================== CONF v2: LONG-1/LONG-2/SHORT-1/SHORT-2 ===================== #

def _compute_conf_signals_v2(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not CONF_ENABLED:
        return []

    signals: List[Dict[str, Any]] = []

    # фильтруем кандидатов по 4h контексту (чтобы не DDOSить Binance)
    candidates: List[Dict[str, Any]] = []
    for symbol, info in phases.items():
        if not isinstance(info, dict) or info.get("error"):
            continue
        phase = info.get("phase")
        pump_count = int(info.get("pumpCount") or 0)
        dump_count = int(info.get("dumpCount") or 0)

        if phase == "PUMP" and pump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL:
            candidates.append({"symbol": symbol, "ctx": "LONG", "info": info})
        elif phase == "DUMP" and dump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL:
            candidates.append({"symbol": symbol, "ctx": "SHORT", "info": info})

    if not candidates:
        return []

    # period для OI hist совпадает по смыслу с interval
    oi_period = CONF_INTERVAL  # "1h" по умолчанию

    for item in candidates:
        symbol = item["symbol"]
        ctx = item["ctx"]
        info = item["info"]

        try:
            df = _binance_klines(symbol, CONF_INTERVAL, CONF_KLINES_LIMIT)
            oi = _binance_open_interest_hist(symbol, oi_period, CONF_KLINES_LIMIT)
        except Exception as exc:
            logger.exception("CONF v2 data load failed for %s: %s", symbol, exc)
            continue

        lookback = CONF_LOOKBACK_BARS
        if len(df) < lookback + 1 or len(oi) < lookback + 1:
            continue

        price_chg = _price_change_pct(df, lookback)
        dr = _delta_ratio(df, lookback)
        oi_chg = _oi_change_pct(oi, lookback, CONF_OI_VALUE_MODE)

        # базовый фильтр: OI должен расти (иначе эти 4 типа теряют смысл)
        if oi_chg < CONF_OI_CHANGE_PCT_THR:
            continue

        # тайм штамп = close_time последней свечи на CONF_INTERVAL
        gen_ts = int(df["close_time"].iloc[-1].timestamp())

        payload_base = {
            "symbol": symbol,
            "kind": "CONF",
            "phase": info.get("phase"),              # 4h контекст
            "interval": info.get("interval"),        # 4h interval
            "triggerInterval": CONF_INTERVAL,        # 1h trigger
            "trendDir": int(info.get("trendDir") or 0),
            "barsInPhase": int(info.get("pumpCount") or info.get("dumpCount") or 0),
            "generatedAt": gen_ts,
            "lookbackBars": lookback,
            "priceChangePct": float(price_chg),
            "oiChangePct": float(oi_chg),
            "deltaRatio": float(dr),
        }

        # LONG ctx: LONG-1 / LONG-2
        if ctx == "LONG":
            # LONG-1: OI↑ + delta↓ + price не падает
            if (dr <= -CONF_DELTA_RATIO_THR) and (price_chg >= -CONF_PRICE_FLAT_PCT):
                signals.append(
                    {
                        **payload_base,
                        "side": "LONG",
                        "confirmType": "LONG-1",
                    }
                )
                continue

            # LONG-2: OI↑ + delta↑ + price↑
            if (dr >= CONF_DELTA_RATIO_THR) and (price_chg >= CONF_PRICE_MOVE_PCT):
                signals.append(
                    {
                        **payload_base,
                        "side": "LONG",
                        "confirmType": "LONG-2",
                    }
                )
                continue

        # SHORT ctx: SHORT-1 / SHORT-2
        if ctx == "SHORT":
            # SHORT-1: OI↑ + delta↑ + price не растет
            if (dr >= CONF_DELTA_RATIO_THR) and (price_chg <= CONF_PRICE_FLAT_PCT):
                signals.append(
                    {
                        **payload_base,
                        "side": "SHORT",
                        "confirmType": "SHORT-1",
                    }
                )
                continue

            # SHORT-2: OI↑ + delta↓ + price↓
            if (dr <= -CONF_DELTA_RATIO_THR) and (price_chg <= -CONF_PRICE_MOVE_PCT):
                signals.append(
                    {
                        **payload_base,
                        "side": "SHORT",
                        "confirmType": "SHORT-2",
                    }
                )
                continue

    return signals


def compute_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    pre = _compute_pre_signals(phases)
    conf = _compute_conf_signals_v2(phases)
    return pre + conf


# ===================== FLASK HTTP ===================== #

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "symbols": SYMBOLS,
            "interval": INTERVAL,
            "confirmEnabled": CONF_ENABLED,
            "confirmInterval": CONF_INTERVAL,
        }
    )

@app.route("/phases", methods=["GET"])
def phases_endpoint() -> Any:
    phases = compute_all_phases()
    return jsonify(phases)

@app.route("/signals", methods=["GET"])
def signals_endpoint() -> Any:
    phases = compute_all_phases()
    signals = compute_signals(phases)
    return jsonify({"signals": signals})

def main() -> None:
    logger.info(
        "Starting Polaris Phase Engine on %s:%s symbols=%s interval=%s conf=%s/%s",
        HTTP_HOST, HTTP_PORT, ",".join(SYMBOLS), INTERVAL, CONF_ENABLED, CONF_INTERVAL
    )
    app.run(host=HTTP_HOST, port=HTTP_PORT)

if __name__ == "__main__":
    main()
