#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime, timezone
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

# --- CONF-логика ---

CONF_ENABLED = os.getenv("PPE_CONFIRM_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# рабочий ТФ для подтверждения (1h)
CONF_INTERVAL = os.getenv("PPE_CONFIRM_INTERVAL", "1h")
CONF_KLINES_LIMIT = int(os.getenv("PPE_CONFIRM_KLINES_LIMIT", "200"))

# максимум "возраст" PRE (часы) для CONF
CONF_MAX_AGE_HOURS = float(os.getenv("PPE_CONFIRM_MAX_AGE_HOURS", "24"))

# максимум длины фазы (в барах 4H), после которой CONF уже не даём
CONF_MAX_PHASE_BARS = int(os.getenv("PPE_CONFIRM_MAX_PHASE_BARS", "6"))

# сколько 1h-баров смотрим назад для отката
CONF_PULLBACK_LOOKBACK = int(os.getenv("PPE_CONFIRM_PULLBACK_LOOKBACK", "6"))

# объём на откате не должен быть больше этого множителя от медианы
CONF_PULLBACK_VOL_MAX_MULT = float(
    os.getenv("PPE_CONFIRM_PULLBACK_VOL_MAX_MULT", "1.5")
)

# объём на подтверждающей свече должен быть больше этого множителя от медианы
CONF_CONFIRM_VOL_MIN_MULT = float(
    os.getenv("PPE_CONFIRM_CONFIRM_VOL_MIN_MULT", "1.2")
)

# допуск по цене относительно EMA (0.002 = 0.2%)
CONF_PRICE_EPS = float(os.getenv("PPE_CONFIRM_PRICE_EPS", "0.002"))

# на сколько максимум цена может уйти за хай/лоу 4H-бара PRE (0.015 = 1.5%)
CONF_MAX_BREAKOUT_4H = float(os.getenv("PPE_CONFIRM_MAX_BREAKOUT_4H", "0.015"))

HTTP_HOST = os.getenv("PPE_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("PPE_HTTP_PORT", "8001"))

# ===================== ЛОГГЕР / HTTP ===================== #

logger = logging.getLogger("polaris-phase-engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PolarisPhaseEngine/1.1"})


# ===================== УТИЛИТЫ ===================== #


def _binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Загрузка свечей с Binance Futures (USDT-M)."""
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


# ===================== РАСЧЁТ ФАЗ 4H ===================== #


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


# ===================== СИГНАЛЫ: PRE + CONF ===================== #


def _compute_pre_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """PRE-сигналы по 4H-фазам PUMP/DUMP."""
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


def _try_confirm_long(
    symbol: str,
    info: Dict[str, Any],
    df_1h: pd.DataFrame,
    vol_med: float,
) -> Dict[str, Any] | None:
    """Попытка найти CONF LONG по 1H на основе PRE LONG на 4H."""
    if len(df_1h) < 10:
        return None

    df = df_1h.copy()
    df["ema_fast"] = _ema(df["close"], EMA_FAST)
    df["ema_slow"] = _ema(df["close"], EMA_SLOW)

    last = df.iloc[-1]
    ema_fast_last = float(df["ema_fast"].iloc[-1])

    # Подтверждающая бычья свеча
    is_bull = last["close"] > last["open"]
    close_above_ema = last["close"] > ema_fast_last
    volume_ok = last["volume"] >= vol_med * CONF_CONFIRM_VOL_MIN_MULT

    if not (is_bull and close_above_ema and volume_ok):
        return None

    # Ищем откатную зону в последних N барах до подтверждающего
    lookback = min(CONF_PULLBACK_LOOKBACK, len(df) - 1)
    start_idx = len(df) - 1 - lookback
    pullback_indices: List[int] = []

    for i in range(start_idx, len(df) - 1):
        low_i = df["low"].iat[i]
        ema_fast_i = df["ema_fast"].iat[i]
        ema_slow_i = df["ema_slow"].iat[i]
        vol_i = df["volume"].iat[i]

        # зона около EMA20/EMA50
        in_zone = (
            low_i <= ema_fast_i * (1 + CONF_PRICE_EPS)
            and low_i >= ema_slow_i * (1 - CONF_PRICE_EPS)
        )
        vol_ok = vol_i <= vol_med * CONF_PULLBACK_VOL_MAX_MULT

        if in_zone and vol_ok:
            pullback_indices.append(i)

    if not pullback_indices:
        return None

    pullback_low = float(min(df["low"].iat[i] for i in pullback_indices))
    pullback_high = float(max(df["high"].iat[i] for i in pullback_indices))

    # подтверждающая свеча должна закрыться выше максимума отката
    if last["close"] <= pullback_high:
        return None

    # не запрыгиваем слишком высоко относительно хай 4H-бара PRE
    high4h = info.get("high4h")
    if high4h is not None:
        try:
            high4h_f = float(high4h)
            if last["close"] > high4h_f * (1 + CONF_MAX_BREAKOUT_4H):
                return None
        except Exception:
            pass

    last_close_time: datetime = df["close_time"].iloc[-1]
    pre_ts = int(info.get("timestamp") or 0)

    return {
        "symbol": symbol,
        "side": "LONG",
        "kind": "CONF",
        "phase": info.get("phase"),
        "interval": info.get("interval"),  # 4h-контекст
        "triggerInterval": CONF_INTERVAL,  # 1h-триггер
        "barsInPhase": int(info.get("pumpCount") or 0),
        "trendDir": int(info.get("trendDir") or 0),
        "generatedAt": int(last_close_time.timestamp()),
        "entryClose": float(last["close"]),
        "pullbackLow": pullback_low,
        "pullbackHigh": pullback_high,
        "preTimestamp": pre_ts,
    }


def _try_confirm_short(
    symbol: str,
    info: Dict[str, Any],
    df_1h: pd.DataFrame,
    vol_med: float,
) -> Dict[str, Any] | None:
    """Попытка найти CONF SHORT по 1H на основе PRE SHORT на 4H."""
    if len(df_1h) < 10:
        return None

    df = df_1h.copy()
    df["ema_fast"] = _ema(df["close"], EMA_FAST)
    df["ema_slow"] = _ema(df["close"], EMA_SLOW)

    last = df.iloc[-1]
    ema_fast_last = float(df["ema_fast"].iloc[-1])

    # Подтверждающая медвежья свеча
    is_bear = last["close"] < last["open"]
    close_below_ema = last["close"] < ema_fast_last
    volume_ok = last["volume"] >= vol_med * CONF_CONFIRM_VOL_MIN_MULT

    if not (is_bear and close_below_ema and volume_ok):
        return None

    # Ищем откат вверх в зону EMA20/EMA50
    lookback = min(CONF_PULLBACK_LOOKBACK, len(df) - 1)
    start_idx = len(df) - 1 - lookback
    pullback_indices: List[int] = []

    for i in range(start_idx, len(df) - 1):
        high_i = df["high"].iat[i]
        ema_fast_i = df["ema_fast"].iat[i]
        ema_slow_i = df["ema_slow"].iat[i]
        vol_i = df["volume"].iat[i]

        in_zone = (
            high_i >= ema_fast_i * (1 - CONF_PRICE_EPS)
            and high_i <= ema_slow_i * (1 + CONF_PRICE_EPS)
        )
        vol_ok = vol_i <= vol_med * CONF_PULLBACK_VOL_MAX_MULT

        if in_zone and vol_ok:
            pullback_indices.append(i)

    if not pullback_indices:
        return None

    pullback_low = float(min(df["low"].iat[i] for i in pullback_indices))
    pullback_high = float(max(df["high"].iat[i] for i in pullback_indices))

    # подтверждающая свеча должна закрыться ниже минимума отката
    if last["close"] >= pullback_low:
        return None

    # не шортим в самой дыре относительно лоу 4H-бара PRE
    low4h = info.get("low4h")
    if low4h is not None:
        try:
            low4h_f = float(low4h)
            if last["close"] < low4h_f * (1 - CONF_MAX_BREAKOUT_4H):
                return None
        except Exception:
            pass

    last_close_time: datetime = df["close_time"].iloc[-1]
    pre_ts = int(info.get("timestamp") or 0)

    return {
        "symbol": symbol,
        "side": "SHORT",
        "kind": "CONF",
        "phase": info.get("phase"),
        "interval": info.get("interval"),  # 4h-контекст
        "triggerInterval": CONF_INTERVAL,  # 1h-триггер
        "barsInPhase": int(info.get("dumpCount") or 0),
        "trendDir": int(info.get("trendDir") or 0),
        "generatedAt": int(last_close_time.timestamp()),
        "entryClose": float(last["close"]),
        "pullbackLow": pullback_low,
        "pullbackHigh": pullback_high,
        "preTimestamp": pre_ts,
    }


def _compute_confirm_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CONF-сигналы по 1H на основе PRE-контекста 4H."""
    if not CONF_ENABLED:
        return []

    signals: List[Dict[str, Any]] = []

    for symbol, info in phases.items():
        if not isinstance(info, dict):
            continue
        if info.get("error"):
            continue

        phase = info.get("phase")
        trend_dir = int(info.get("trendDir") or 0)
        regime = info.get("regime")
        pump_count = int(info.get("pumpCount") or 0)
        dump_count = int(info.get("dumpCount") or 0)
        pre_ts = info.get("timestamp")

        if pre_ts is None:
            continue

        # Проверяем, что PRE вообще имеет смысл
        is_long_ctx = (
            phase == "PUMP"
            and trend_dir > 0
            and pump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL
            and pump_count <= CONF_MAX_PHASE_BARS
        )

        is_short_ctx = (
            phase == "DUMP"
            and trend_dir < 0
            and dump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL
            and dump_count <= CONF_MAX_PHASE_BARS
        )

        if not (is_long_ctx or is_short_ctx):
            continue

        # Можно дополнительно фильтровать по regime, но PUMP/DUMP уже достаточно
        try:
            df_1h = _binance_klines(symbol, CONF_INTERVAL, CONF_KLINES_LIMIT)
        except Exception as exc:
            logger.exception(
                "Failed to load %s %s klines for CONF: %s",
                symbol,
                CONF_INTERVAL,
                exc,
            )
            continue

        if df_1h.empty:
            continue

        last_close_time: datetime = df_1h["close_time"].iloc[-1]
        age_hours = (last_close_time.timestamp() - float(pre_ts)) / 3600.0
        if age_hours < 0:
            age_hours = 0.0

        if age_hours > CONF_MAX_AGE_HOURS:
            # PRE слишком старый для CONF
            continue

        vol_med = float(df_1h["volume"].tail(50).median())
        if not np.isfinite(vol_med) or vol_med <= 0:
            vol_med = float(df_1h["volume"].median())
        if vol_med <= 0:
            continue

        if is_long_ctx:
            conf = _try_confirm_long(symbol, info, df_1h, vol_med)
            if conf is not None:
                signals.append(conf)

        if is_short_ctx:
            conf = _try_confirm_short(symbol, info, df_1h, vol_med)
            if conf is not None:
                signals.append(conf)

    return signals


def compute_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Общий генератор сигналов: PRE + CONF."""
    pre = _compute_pre_signals(phases)
    conf = _compute_confirm_signals(phases)
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
