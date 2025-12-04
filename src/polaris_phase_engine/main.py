cat > src/polaris_phase_engine/main.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify

# ---------------------- ENV ---------------------- #

# грузим .env из корня проекта, если есть
load_dotenv()

HTTP_PORT: int = int(os.getenv("HTTP_PORT", "8088"))

# таймфрейм движка (минуты) – эталон 4H
INTERVAL_MIN: int = int(os.getenv("INTERVAL_MIN", "240"))

# сколько баров тянуть с биржи
WINDOW_BARS: int = int(os.getenv("WINDOW_BARS", "300"))

# как часто пересчитывать вселенную
POLL_SEC: int = int(os.getenv("POLL_SEC", "60"))

HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "8.0"))
SLEEP_BETWEEN_CALLS: float = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.2"))

# вселенная тикеров (Binance Futures, напр. BTCUSDT, ETHUSDT,...)
_UNIVERSE_ENV = os.getenv("UNIVERSE", "").strip()
if _UNIVERSE_ENV:
    UNIVERSE_DEFAULT: List[str] = [
        s.strip().upper() for s in _UNIVERSE_ENV.split(",") if s.strip()
    ]
else:
    # запасной вариант, чтобы не было пусто
    UNIVERSE_DEFAULT = ["BTCUSDT"]

# параметры PolarisPhaseMVP (один в один с Pine по умолчанию)
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))

ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_ZLEN = int(os.getenv("ATR_ZLEN", "100"))

VOL_ZLEN = int(os.getenv("VOL_ZLEN", "100"))

TREND_UP_THR = float(os.getenv("TREND_UP_THR", "0.0015"))
TREND_DN_THR = float(os.getenv("TREND_DN_THR", "-0.0015"))

ATR_QUIET = float(os.getenv("ATR_QUIET", "-0.7"))
ATR_HOT = float(os.getenv("ATR_HOT", "0.7"))

VOL_QUIET = float(os.getenv("VOL_QUIET", "-0.5"))
VOL_HOT = float(os.getenv("VOL_HOT", "0.5"))

# Binance USDT-M Futures API
BINANCE_FAPI_URL = os.getenv("BINANCE_FAPI_URL", "https://fapi.binance.com")

# ---------------------- LOGGING ---------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("polaris_phase_engine")

# ---------------------- HTTP APP -------------------- #

app = Flask(__name__)

_STATE: Dict[str, Any] = {
    "rows": [],              # список строк по символам
    "last_update": 0,        # epoch seconds
    "universe": UNIVERSE_DEFAULT,
    "last_phase": {},        # symbol -> str
    "_bg_started": False,
}

_session = requests.Session()
_session.headers.update({"User-Agent": "PolarisPhaseEngine/1.0"})

# ---------------------- HELPERS --------------------- #


def _interval_to_str(mins: int) -> str:
    """Конверсия минут в строку интервала Binance."""
    mapping = {
        1: "1m",
        3: "3m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "1h",
        120: "2h",
        240: "4h",
        360: "6h",
        480: "8h",
        720: "12h",
        1440: "1d",
    }
    if mins in mapping:
        return mapping[mins]
    if mins % 1440 == 0:
        return f"{mins // 1440}d"
    if mins % 60 == 0:
        return f"{mins // 60}h"
    return f"{mins}m"


def _get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    try:
        r = _session.get(url, params=params, timeout=HTTP_TIMEOUT)
        if r.ok:
            return r.json()
        log.warning(
            "HTTP %s for %s params=%s body=%s",
            r.status_code,
            url,
            params,
            r.text[:200],
        )
    except Exception as e:
        log.warning("HTTP GET fail %s: %s", url, e)
    return None


# ----------------- BINANCE KLINES ------------------ #


def _fetch_futures_klines(
    symbol: str,
    interval_min: int,
    limit: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Binance USDT-M futures klines (последние N баров).
    /fapi/v1/klines
    """
    interval_str = _interval_to_str(interval_min)
    # max 1500, нам столько не надо
    limit = max(10, min(limit, 1000))

    url = f"{BINANCE_FAPI_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval_str,
        "limit": str(limit),
    }
    data = _get(url, params)
    if not isinstance(data, list) or not data:
        log.warning("No klines for %s (%s)", symbol, interval_str)
        return None

    out: List[Dict[str, Any]] = []
    # [0] openTime, [1] open, [2] high, [3] low, [4] close,
    # [5] volume (base), [6] closeTime, [7] quoteAssetVolume, ...
    for row in data:
        try:
            out.append(
                {
                    "open_time": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume_base": float(row[5]),
                    "close_time": int(row[6]),
                    "volume_quote": float(row[7]),
                }
            )
        except Exception:
            log.warning("Bad kline row for %s: %s", symbol, row)
            return None

    return out


# ----------------- MATH UTILITIES ------------------ #


def _ema_series(values: List[float], length: int) -> List[float]:
    if not values:
        return []
    if length <= 1:
        return list(values)

    alpha = 2.0 / (length + 1.0)
    ema = values[0]
    out: List[float] = []
    for v in values:
        ema = alpha * v + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _true_range_series(
    highs: List[float],
    lows: List[float],
    closes: List[float],
) -> List[float]:
    res: List[float] = []
    prev_close = closes[0]
    for i in range(len(closes)):
        h = highs[i]
        l = lows[i]
        c = closes[i]
        if i == 0:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        res.append(tr)
        prev_close = c
    return res


def _rma(values: List[float], length: int) -> List[float]:
    """
    Wilders RMA как в Pine (ta.rma), используется в ATR.
    """
    n = len(values)
    if n == 0:
        return []
    if length <= 1:
        return list(values)

    out: List[float] = [0.0] * n
    if n < length:
        # мало данных — просто EMA
        return _ema_series(values, length)

    # первое значение — SMA по первой выборке
    first = sum(values[:length]) / float(length)
    for i in range(length):
        out[i] = first

    prev = first
    for i in range(length, n):
        v = values[i]
        prev = (prev * (length - 1) + v) / float(length)
        out[i] = prev

    return out


def _atr_series(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    length: int,
) -> List[float]:
    tr = _true_range_series(highs, lows, closes)
    return _rma(tr, length)


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / float(n)
    var = sum((x - m) ** 2 for x in vals) / float(max(1, n - 1))
    return m, math.sqrt(var)


def _zscore(curr: float, hist: List[float]) -> Optional[float]:
    # убираем NaN/None
    arr = [x for x in hist if x is not None]
    if len(arr) < 5:
        return None
    mean, std = _mean_std(arr)
    if std <= 0:
        return None
    return (curr - mean) / std


# --------------- PHASE CALC PER SYMBOL --------------- #


def _calc_phase_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    # сколько баров минимум нужно под EMA/ATR/Z
    min_needed = max(EMA_SLOW + ATR_ZLEN, VOL_ZLEN + 10, 150)

    klines = _fetch_futures_klines(
        symbol,
        INTERVAL_MIN,
        limit=max(WINDOW_BARS, min_needed),
    )
    if not klines or len(klines) < min_needed:
        log.warning("Not enough klines for %s: got %s",
                    symbol, len(klines) if klines else 0)
        return None

    closes = [k["close"] for k in klines]
    highs = [k["high"] for k in klines]
    lows = [k["low"] for k in klines]
    # используем quote volume (в USDT)
    vols = [k["volume_quote"] for k in klines]

    ema_fast = _ema_series(closes, EMA_FAST)
    ema_slow = _ema_series(closes, EMA_SLOW)

    fast_last = ema_fast[-1]
    slow_last = ema_slow[-1]

    trend_raw = (fast_last - slow_last) / slow_last if slow_last != 0 else 0.0
    if trend_raw > TREND_UP_THR:
        trend_dir = 1
    elif trend_raw < TREND_DN_THR:
        trend_dir = -1
    else:
        trend_dir = 0

    atr_ser = _atr_series(highs, lows, closes, ATR_LEN)
    atr_curr = atr_ser[-1]
    atr_hist = atr_ser[-(ATR_ZLEN + 1):-1]  # предыдущие ATR_ZLEN баров
    atr_z = _zscore(atr_curr, atr_hist)

    vol_curr = vols[-1]
    vol_hist = vols[-(VOL_ZLEN + 1):-1]
    vol_z = _zscore(vol_curr, vol_hist)

    # Фаза по логике PolarisPhaseMVP
    if (
        trend_dir == 1
        and atr_z is not None
        and vol_z is not None
        and atr_z > ATR_HOT
        and vol_z > VOL_HOT
    ):
        phase = "PUMP"
    elif (
        trend_dir == -1
        and atr_z is not None
        and vol_z is not None
        and atr_z > ATR_HOT
        and vol_z > VOL_HOT
    ):
        phase = "DUMP"
    elif (
        trend_dir == 0
        and atr_z is not None
        and vol_z is not None
        and atr_z < ATR_QUIET
        and vol_z < VOL_QUIET
    ):
        phase = "ACCUM"
    else:
        phase = "CORR"

    prev_phase = _STATE["last_phase"].get(symbol)
    _STATE["last_phase"][symbol] = phase

    # PRE-сигнал в лонг: первый бар PUMP после не-PUMP
    long_ctx = phase == "PUMP" and trend_dir == 1
    long_signal = bool(long_ctx and prev_phase != "PUMP")

    last_bar = klines[-1]
    bar_ts = last_bar["open_time"]  # ms
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(bar_ts // 1000))

    tv_symbol = f"BINANCE:{symbol}"
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

    return {
        "ts": ts_str,
        "symbol": symbol,
        "close": last_bar["close"],
        "phase": phase,
        "trendDir": trend_dir,
        "atrZ": round(atr_z, 2) if atr_z is not None else None,
        "volZ": round(vol_z, 2) if vol_z is not None else None,
        "emaFast": fast_last,
        "emaSlow": slow_last,
        "bar_ts": bar_ts,
        "long_ctx": long_ctx,
        "long_signal": long_signal,
        "tv_link": tv_link,
    }


# --------------- BACKGROUND WORKER ----------------- #


def _rebuild_loop() -> None:
    log.info(
        "background worker start | universe=%d, interval=%dm",
        len(_STATE["universe"]),
        INTERVAL_MIN,
    )
    while True:
        start = time.time()
        rows: List[Dict[str, Any]] = []
        for i, sym in enumerate(_STATE["universe"], 1):
            try:
                row = _calc_phase_for_symbol(sym)
                if row:
                    rows.append(row)
            except Exception as e:
                log.exception("calc failed for %s: %s", sym, e)
            time.sleep(SLEEP_BETWEEN_CALLS)
        _STATE["rows"] = rows
        _STATE["last_update"] = int(time.time())
        took = time.time() - start
        log.info("phases rebuilt: %d symbols in %.1fs", len(rows), took)
        time.sleep(max(2, POLL_SEC))


def _start_background_once() -> None:
    if _STATE.get("_bg_started"):
        return
    _STATE["_bg_started"] = True
    t = threading.Thread(target=_rebuild_loop, daemon=True)
    t.start()


# стартуем воркер при импорте
_start_background_once()


# ---------------------- HTTP ----------------------- #


@app.get("/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "interval_min": INTERVAL_MIN,
            "window_bars": WINDOW_BARS,
            "universe_size": len(_STATE["universe"]),
            "last_update": _STATE["last_update"],
            "poll_sec": POLL_SEC,
        }
    )


@app.get("/signals")
def signals() -> Any:
    return jsonify(
        {
            "data": _STATE["rows"],
            "count": len(_STATE["rows"]),
            "last_update": _STATE["last_update"],
            "interval_min": INTERVAL_MIN,
        }
    )


# ---------------------- ENTRYPOINT ----------------- #


def main() -> None:
    log.info(
        "Polaris Phase Engine HTTP starting on port %d | interval=%dm | universe=%d",
        HTTP_PORT,
        INTERVAL_MIN,
        len(_STATE["universe"]),
    )
    # host=0.0.0.0 – слушаем снаружи
    app.run(host="0.0.0.0", port=HTTP_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
EOF
