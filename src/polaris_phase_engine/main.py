#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

MIN_BARS_IN_PHASE_FOR_SIGNAL = int(
    os.getenv("PPE_MIN_BARS_IN_PHASE_FOR_SIGNAL", "2")
)

# --- Wyckoff CONF на младшем ТФ (по умолчанию 1h) ---

CONF_ENABLED = os.getenv("PPE_CONFIRM_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CONF_INTERVAL = os.getenv("PPE_CONFIRM_INTERVAL", "1h")
CONF_KLINES_LIMIT = int(os.getenv("PPE_CONFIRM_KLINES_LIMIT", "200"))
CONF_MAX_AGE_HOURS = float(os.getenv("PPE_CONFIRM_MAX_AGE_HOURS", "24"))

# TR/спринг/UT параметры
WYC_TR_BARS = int(os.getenv("PPE_WYC_TR_BARS", "24"))          # размер окна TR (в барах CONF_INTERVAL)
WYC_MIN_TR_BARS = int(os.getenv("PPE_WYC_MIN_TR_BARS", "8"))   # минимум баров в TR
WYC_TR_MAX_RANGE_PCT = float(os.getenv("PPE_WYC_TR_MAX_RANGE_PCT", "0.03"))  # макс. высота TR, напр. 3%

WYC_SPRING_BREAK_PCT = float(os.getenv("PPE_WYC_SPRING_BREAK_PCT", "0.002"))  # насколько низко выходить ниже TR-низа
WYC_SPRING_WICK_RATIO = float(os.getenv("PPE_WYC_SPRING_WICK_RATIO", "0.6"))  # доля нижней тени от спреда
WYC_SPRING_VOL_MULT = float(os.getenv("PPE_WYC_SPRING_VOL_MULT", "1.5"))      # объём на спринг-баре

WYC_UT_BREAK_PCT = float(os.getenv("PPE_WYC_UT_BREAK_PCT", "0.002"))          # выход выше TR-верха
WYC_UT_WICK_RATIO = float(os.getenv("PPE_WYC_UT_WICK_RATIO", "0.6"))          # верхняя тень
WYC_UT_VOL_MULT = float(os.getenv("PPE_WYC_UT_VOL_MULT", "1.5"))

WYC_CONFIRM_VOL_MULT = float(os.getenv("PPE_WYC_CONFIRM_VOL_MULT", "1.2"))    # объём на подтверждающем баре

HTTP_HOST = os.getenv("PPE_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("PPE_HTTP_PORT", "8001"))

# ===================== ЛОГГЕР / HTTP ===================== #

logger = logging.getLogger("polaris-phase-engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PolarisPhaseEngine/1.3-wyc-1h"})


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


def _compute_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Простейший CVD по агрессорам:
    delta = buy - sell = taker_buy_base - (volume - taker_buy_base)
    """
    buy = df["taker_buy_base"]
    sell = df["volume"] - df["taker_buy_base"]
    delta = buy - sell
    return delta.cumsum()


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


# ===================== СИГНАЛЫ: PRE ===================== #


def _compute_pre_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """PRE-сигналы по 4H-фазам PUMP/DUMP. Логика как и была."""
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


# ===================== Wyckoff CONF на 1h (TR + Spring / UT) ===================== #


def _find_wyckoff_confirm_long(
    df: pd.DataFrame,
    pre_ts: int,
) -> Optional[Dict[str, Any]]:
    """
    Wyckoff CONF LONG на младшем ТФ:
    - TR (диапазон) в последние WYC_TR_BARS баров
    - внутри TR есть спринг (вынос вниз + возврат)
    - последняя свеча – бычья, с объёмом, закрывается в верхней части TR
    """
    if len(df) < WYC_MIN_TR_BARS + 2:
        return None

    # берём последнее окно
    window = min(WYC_TR_BARS + 2, len(df))
    df_win = df.iloc[-window:].copy()

    # TR ядро: все кроме последнего бара (подтверждающий)
    if len(df_win) < WYC_MIN_TR_BARS + 1:
        return None
    df_tr = df_win.iloc[:-1]
    conf_row = df_win.iloc[-1]

    # диапазон TR
    tr_low = float(df_tr["low"].min())
    tr_high = float(df_tr["high"].max())
    if tr_low <= 0:
        return None
    tr_range = tr_high - tr_low
    tr_range_pct = tr_range / tr_low
    if tr_range_pct > WYC_TR_MAX_RANGE_PCT:
        return None

    # объём
    vol_med = float(df_tr["volume"].median())
    if not np.isfinite(vol_med) or vol_med <= 0:
        vol_med = float(df_win["volume"].median())
    if vol_med <= 0:
        return None

    # ищем последний спринг в TR
    spring_idx: Optional[int] = None
    for i in range(len(df_tr) - 1, -1, -1):
        row = df_tr.iloc[i]
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        spread = hi - lo
        if spread <= 0:
            continue

        # выход ниже TR-низа
        if lo > tr_low * (1 - WYC_SPRING_BREAK_PCT):
            continue
        # закрытие обратно в диапазон
        if cl <= tr_low:
            continue
        # длинная нижняя тень
        if (cl - lo) / spread < WYC_SPRING_WICK_RATIO:
            continue
        # объём на спринг-баре
        if float(row["volume"]) < vol_med * WYC_SPRING_VOL_MULT:
            continue

        spring_idx = df_tr.index[i]
        break

    if spring_idx is None:
        return None

    spring_row = df.loc[spring_idx]

    # подтверждающая свеча (последний бар окна)
    hi_c = float(conf_row["high"])
    lo_c = float(conf_row["low"])
    op_c = float(conf_row["open"])
    cl_c = float(conf_row["close"])
    spread_c = hi_c - lo_c
    vol_c = float(conf_row["volume"])

    if spread_c <= 0:
        return None

    # бычья свеча
    if cl_c <= op_c:
        return None

    # закрытие в верхней части диапазона
    tr_mid = tr_low + tr_range * 0.5
    if cl_c <= tr_mid:
        return None

    # желательно выше хая спринга
    if cl_c <= float(spring_row["high"]):
        return None

    # объём выше медианы
    if vol_c < vol_med * WYC_CONFIRM_VOL_MULT:
        return None

    # возраст относительно PRE
    conf_time: datetime = conf_row["close_time"]
    conf_ts = int(conf_time.timestamp())
    age_hours = (conf_ts - float(pre_ts)) / 3600.0
    if age_hours < 0 or age_hours > CONF_MAX_AGE_HOURS:
        return None

    # CVD по окну
    cvd = _compute_cvd(df_win)
    try:
        cvd_delta = float(cvd.iloc[-1] - cvd.iloc[0])
    except Exception:
        cvd_delta = 0.0

    return {
        "entryClose": cl_c,
        "rangeLow": tr_low,
        "rangeHigh": tr_high,
        "springLow": float(spring_row["low"]),
        "springTime": int(spring_row["close_time"].timestamp()),
        "confirmTime": conf_ts,
        "cvdDelta": cvd_delta,
    }


def _find_wyckoff_confirm_short(
    df: pd.DataFrame,
    pre_ts: int,
) -> Optional[Dict[str, Any]]:
    """
    Wyckoff CONF SHORT на младшем ТФ:
    - TR в последние WYC_TR_BARS баров
    - внутри TR есть upthrust (вынос вверх + возврат)
    - последняя свеча – медвежья, с объёмом, закрывается в нижней части TR
    """
    if len(df) < WYC_MIN_TR_BARS + 2:
        return None

    window = min(WYC_TR_BARS + 2, len(df))
    df_win = df.iloc[-window:].copy()

    if len(df_win) < WYC_MIN_TR_BARS + 1:
        return None
    df_tr = df_win.iloc[:-1]
    conf_row = df_win.iloc[-1]

    tr_low = float(df_tr["low"].min())
    tr_high = float(df_tr["high"].max())
    if tr_low <= 0:
        return None
    tr_range = tr_high - tr_low
    tr_range_pct = tr_range / tr_low
    if tr_range_pct > WYC_TR_MAX_RANGE_PCT:
        return None

    vol_med = float(df_tr["volume"].median())
    if not np.isfinite(vol_med) or vol_med <= 0:
        vol_med = float(df_win["volume"].median())
    if vol_med <= 0:
        return None

    # ищем последний upthrust в TR
    ut_idx: Optional[int] = None
    for i in range(len(df_tr) - 1, -1, -1):
        row = df_tr.iloc[i]
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        spread = hi - lo
        if spread <= 0:
            continue

        # выход выше TR-верха
        if hi < tr_high * (1 + WYC_UT_BREAK_PCT):
            continue
        # закрытие обратно в диапазон
        if cl >= tr_high:
            continue
        # длинная верхняя тень
        if (hi - cl) / spread < WYC_UT_WICK_RATIO:
            continue
        # объём на UT-баре
        if float(row["volume"]) < vol_med * WYC_UT_VOL_MULT:
            continue

        ut_idx = df_tr.index[i]
        break

    if ut_idx is None:
        return None

    ut_row = df.loc[ut_idx]

    # подтверждающая свеча (последний бар окна)
    hi_c = float(conf_row["high"])
    lo_c = float(conf_row["low"])
    op_c = float(conf_row["open"])
    cl_c = float(conf_row["close"])
    spread_c = hi_c - lo_c
    vol_c = float(conf_row["volume"])

    if spread_c <= 0:
        return None

    # медвежья свеча
    if cl_c >= op_c:
        return None

    # закрытие в нижней части диапазона
    tr_mid = tr_low + tr_range * 0.5
    if cl_c >= tr_mid:
        return None

    # желательно ниже low UT-бара
    if cl_c >= float(ut_row["low"]):
        return None

    # объём выше медианы
    if vol_c < vol_med * WYC_CONFIRM_VOL_MULT:
        return None

    # возраст относительно PRE
    conf_time: datetime = conf_row["close_time"]
    conf_ts = int(conf_time.timestamp())
    age_hours = (conf_ts - float(pre_ts)) / 3600.0
    if age_hours < 0 or age_hours > CONF_MAX_AGE_HOURS:
        return None

    # CVD по окну
    cvd = _compute_cvd(df_win)
    try:
        cvd_delta = float(cvd.iloc[-1] - cvd.iloc[0])
    except Exception:
        cvd_delta = 0.0

    return {
        "entryClose": cl_c,
        "rangeLow": tr_low,
        "rangeHigh": tr_high,
        "upthrustHigh": float(ut_row["high"]),
        "upthrustTime": int(ut_row["close_time"].timestamp()),
        "confirmTime": conf_ts,
        "cvdDelta": cvd_delta,
    }


def _compute_confirm_signals_wyckoff(
    phases: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """CONF-сигналы Wyckoff на младшем ТФ (TR + спринг/UT) в контексте 4h PRE."""
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
        pump_count = int(info.get("pumpCount") or 0)
        dump_count = int(info.get("dumpCount") or 0)
        pre_ts = info.get("timestamp")

        if pre_ts is None:
            continue

        # контекст PRE
        is_long_ctx = (
            phase == "PUMP"
            and trend_dir > 0
            and pump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL
        )

        is_short_ctx = (
            phase == "DUMP"
            and trend_dir < 0
            and dump_count >= MIN_BARS_IN_PHASE_FOR_SIGNAL
        )

        if not (is_long_ctx or is_short_ctx):
            continue

        try:
            df_1h = _binance_klines(symbol, CONF_INTERVAL, CONF_KLINES_LIMIT)
        except Exception as exc:
            logger.exception(
                "Failed to load %s %s klines for Wyckoff CONF: %s",
                symbol,
                CONF_INTERVAL,
                exc,
            )
            continue

        if df_1h.empty:
            continue

        if is_long_ctx:
            conf = _find_wyckoff_confirm_long(df_1h, int(pre_ts))
            if conf is not None:
                signals.append(
                    {
                        "symbol": symbol,
                        "side": "LONG",
                        "kind": "CONF",
                        "confirmType": "WYC_TR_SPRING",
                        "phase": phase,
                        "interval": info.get("interval"),  # 4h-контекст
                        "triggerInterval": CONF_INTERVAL,   # 1h-триггер
                        "barsInPhase": pump_count,
                        "trendDir": trend_dir,
                        "generatedAt": conf["confirmTime"],
                        "entryClose": conf["entryClose"],
                        "rangeLow": conf["rangeLow"],
                        "rangeHigh": conf["rangeHigh"],
                        "springLow": conf["springLow"],
                        "springTime": conf["springTime"],
                        "preTimestamp": int(pre_ts),
                        "cvdDelta": conf["cvdDelta"],
                    }
                )

        if is_short_ctx:
            conf = _find_wyckoff_confirm_short(df_1h, int(pre_ts))
            if conf is not None:
                signals.append(
                    {
                        "symbol": symbol,
                        "side": "SHORT",
                        "kind": "CONF",
                        "confirmType": "WYC_TR_UT",
                        "phase": phase,
                        "interval": info.get("interval"),  # 4h-контекст
                        "triggerInterval": CONF_INTERVAL,   # 1h-триггер
                        "barsInPhase": dump_count,
                        "trendDir": trend_dir,
                        "generatedAt": conf["confirmTime"],
                        "entryClose": conf["entryClose"],
                        "rangeLow": conf["rangeLow"],
                        "rangeHigh": conf["rangeHigh"],
                        "upthrustHigh": conf["upthrustHigh"],
                        "upthrustTime": conf["upthrustTime"],
                        "preTimestamp": int(pre_ts),
                        "cvdDelta": conf["cvdDelta"],
                    }
                )

    return signals


def compute_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """PRE (4h PUMP/DUMP) + Wyckoff CONF (1h TR + спринг/UT)."""
    pre = _compute_pre_signals(phases)
    wyc_conf = _compute_confirm_signals_wyckoff(phases)
    return pre + wyc_conf


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
