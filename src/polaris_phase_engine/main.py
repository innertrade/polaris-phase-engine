cd /opt/polaris-phase-engine
source .venv/bin/activate

cat > src/polaris_phase_engine/main.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Any, Dict, List, Tuple

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

# минимальное кол-во баров в фазе, условие можно будет доработать
MIN_BARS_IN_PHASE_FOR_SIGNAL = int(
    os.getenv("PPE_MIN_BARS_IN_PHASE_FOR_SIGNAL", "2")
)

HTTP_HOST = os.getenv("PPE_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("PPE_HTTP_PORT", "8001"))

# ===================== ЛОГГЕР / HTTP ===================== #

logger = logging.getLogger("polaris-phase-engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PolarisPhaseEngine/1.0"})


# ===================== УТИЛИТЫ ===================== #

def _binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Клайны Binance USDT-перпов.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    try:
        resp = SESSION.get(
            url,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("fetch klines failed for %s: %s", symbol, e)
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    cols = [
        "openTime", "open", "high", "low", "close", "volume",
        "closeTime", "qav", "ntrades", "tbbav", "tbqav", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["openTime"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df[["openTime", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("openTime").reset_index(drop=True)
    return df


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=length, adjust=False).mean()
    return atr


def _zscore(series: pd.Series, length: int) -> pd.Series:
    ma = series.rolling(length).mean()
    sd = series.rolling(length).std(ddof=0)
    z = (series - ma) / sd.replace(0, np.nan)
    return z.fillna(0.0)


# ===================== РАСЧЁТ ФАЗ ===================== #

def compute_phase_for_symbol(symbol: str) -> Dict[str, Any]:
    df = _binance_klines(symbol, INTERVAL, KLINES_LIMIT)
    if df.empty or len(df) < max(EMA_SLOW, ATR_Z_LEN, VOL_Z_LEN) + 5:
        return {
            "symbol": symbol,
            "ok": False,
            "error": "not_enough_data",
        }

    # EMA / тренд
    df["emaFast"] = _ema(df["close"], EMA_FAST)
    df["emaSlow"] = _ema(df["close"], EMA_SLOW)
    df["trendRaw"] = (df["emaFast"] - df["emaSlow"]) / df["emaSlow"].replace(0, np.nan)

    df["trendDir"] = 0
    df.loc[df["trendRaw"] > TREND_UP_THR, "trendDir"] = 1
    df.loc[df["trendRaw"] < TREND_DN_THR, "trendDir"] = -1

    # ATR Z
    df["atr"] = _atr(df, ATR_LEN)
    df["atrZ"] = _zscore(df["atr"], ATR_Z_LEN)

    # Volume Z
    df["volZ"] = _zscore(df["volume"], VOL_Z_LEN)

    last = df.iloc[-1]

    trendDir = int(last["trendDir"])
    atrZ = float(last["atrZ"])
    volZ = float(last["volZ"])

    # Логика фазы как в Pine:
    # PUMP / DUMP / ACCUM / CORR
    if trendDir == 1 and atrZ > ATR_HOT and volZ > VOL_HOT:
        phase = "PUMP"
    elif trendDir == -1 and atrZ > ATR_HOT and volZ > VOL_HOT:
        phase = "DUMP"
    elif trendDir == 0 and atrZ < ATR_QUIET and volZ < VOL_QUIET:
        phase = "ACCUM"
    else:
        phase = "CORR"

    return {
        "symbol": symbol,
        "ok": True,
        "time": last["openTime"].isoformat(),
        "trendDir": trendDir,
        "atrZ": round(atrZ, 2),
        "volZ": round(volZ, 2),
        "phase": phase,
    }


def compute_all_phases() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sym in SYMBOLS:
        try:
            out[sym] = compute_phase_for_symbol(sym)
        except Exception as e:
            logger.exception("phase calc failed for %s: %s", sym, e)
            out[sym] = {"symbol": sym, "ok": False, "error": "exception"}
    return out


# ===================== СИГНАЛЫ ===================== #

def compute_signals(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Простейшая логика сигналов:
    - если фаза PUMP и trendDir=1 → LONG
    - если фаза DUMP и trendDir=-1 → SHORT
    Историю фаз пока не трогаем – смотрим только последнюю точку.
    """
    signals: List[Dict[str, Any]] = []

    for sym, info in phases.items():
        if not info.get("ok"):
            continue

        phase = info.get("phase")
        trend = info.get("trendDir")

        if phase == "PUMP" and trend == 1:
            side = "LONG"
        elif phase == "DUMP" and trend == -1:
            side = "SHORT"
        else:
            continue

        signals.append(
            {
                "symbol": sym,
                "side": side,
                "tf": INTERVAL,
                "phase": phase,
                "trendDir": trend,
                "atrZ": info.get("atrZ"),
                "volZ": info.get("volZ"),
                "time": info.get("time"),
            }
        )

    return signals


# ===================== FLASK HTTP ===================== #

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "engine": "polaris-phase-mvp",
            "status": "ok",
            "interval": INTERVAL,
            "symbols": SYMBOLS,
        }
    )


@app.route("/phases", methods=["GET"])
def phases_endpoint() -> Any:
    phases = compute_all_phases()
    return jsonify(phases)


@app.route("/signals", methods=["GET"])
def signals_endpoint() -> Any:
    phases = compute_all_phases()
    sigs = compute_signals(phases)
    return jsonify({"signals": sigs, "count": len(sigs)})


def main() -> None:
    logger.info(
        "Starting Polaris Phase Engine HTTP server... symbols=%s interval=%s",
        SYMBOLS,
        INTERVAL,
    )
    app.run(host=HTTP_HOST, port=HTTP_PORT)


if __name__ == "__main__":
    main()
EOF
