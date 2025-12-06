#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
from typing import Any, Dict, List

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

# Параметры, синхронные с PolarisPhaseMVP
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

MIN_BARS_IN_PHASE_FOR_SIGNAL = int(
    os.getenv("PPE_MIN_BARS_IN_PHASE_FOR_SIGNAL", "2")
)

HTTP_HOST = os.getenv("PPE_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("PPE_HTTP_PORT", "8001"))

# ===================== ЛОГГЕР / HTTP ===================== #

logger = logging.getLogger("polaris-phase-engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PolarisPhaseEngine/1.0"})

# ===================== УТИЛИТЫ ===================== #


def _binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Загрузка свечей с Binance Futures (USDT-M)."""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = SESSION.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"No klines for {symbol}")

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

    return df


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
    std = roll.std(ddof=0)
    std = std.replace(0, np.nan)
    return (series - mean) / std


# ===================== РАСЧЁТ ФАЗ ===================== #


def _classify_phases(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ema_fast = _ema(close, EMA_FAST)
    ema_slow = _ema(close, EMA_SLOW)

    # тренд по отношению EMA fast / slow
    slope = (ema_fast - ema_slow) / ema_slow
    trend_dir = np.where(
        slope > TREND_UP_THR,
        1,
        np.where(slope < TREND_DN_THR, -1, 0),
    )

    atr = _atr(df, ATR_LEN)
    atr_z = _zscore(atr, ATR_Z_LEN)
    vol_z = _zscore(df["volume"], VOL_Z_LEN)

    # режим волы/объёма
    regime = np.where(
        (atr_z < ATR_QUIET) & (vol_z < VOL_QUIET),
        "QUIET",
        np.where(
            (atr_z > ATR_HOT) & (vol_z > VOL_HOT),
            "HOT",
            "NORMAL",
        ),
    )

    dist_fast = (close - ema_fast) / ema_fast

    phase = np.full(len(df), "CORR", dtype=object)

    # PUMP: ап-тренд + горячий режим + цена над быстрой EMA
    phase[
        (trend_dir == 1)
        & (regime == "HOT")
        & (dist_fast > 0)
    ] = "PUMP"

    # DUMP: даун-тренд + горячий режим + цена под быстрой EMA
    phase[
        (trend_dir == -1)
        & (regime == "HOT")
        & (dist_fast < 0)
    ] = "DUMP"

    # ACCUM: более спокойный ап-контекст + тихий режим
    phase[
        (trend_dir >= 0)
        & (regime == "QUIET")
    ] = "ACCUM"

    # RANGE: флет при тихом режиме
    phase[
        (regime == "QUIET") & (trend_dir == 0)
    ] = "RANGE"

    pump_count = np.zeros(len(df), dtype=int)
    dump_count = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if i == 0:
            if phase[i] == "PUMP":
                pump_count[i] = 1
            elif phase[i] == "DUMP":
                dump_count[i] = 1
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
        "timestamp": int(df["close_time"].iloc[-1].timestamp()),
    }


def compute_all_phases() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for symbol in SYMBOLS:
        out[symbol] = compute_phase_for_symbol(symbol)
    return out


# ===================== СИГНАЛЫ ===================== #


def compute_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
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


# ===================== FLASK HTTP ===================== #

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "symbols": SYMBOLS,
            "interval": INTERVAL,
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
        "Starting Polaris Phase Engine HTTP server on %s:%s symbols=%s interval=%s",
        HTTP_HOST,
        HTTP_PORT,
        ",".join(SYMBOLS),
        INTERVAL,
    )
    app.run(host=HTTP_HOST, port=HTTP_PORT)


if __name__ == "__main__":
    main()
