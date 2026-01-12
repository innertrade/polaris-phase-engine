import os
import time
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify
import requests

load_dotenv()

logger = logging.getLogger("polaris-phase-engine")
logging.basicConfig(
    level=os.getenv("PPE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

HTTP_HOST = os.getenv("PPE_HTTP_HOST", "127.0.0.1").strip()
HTTP_PORT = int(os.getenv("PPE_HTTP_PORT", "8001"))

INTERVAL = os.getenv("PPE_INTERVAL", "4h").strip()
KLINES_LIMIT = int(os.getenv("PPE_KLINES_LIMIT", "500"))

SYMBOLS = [s.strip() for s in os.getenv("PPE_SYMBOLS", "BTCUSDT").split(",") if s.strip()]

# caching
CACHE_ENABLED = os.getenv("PPE_CACHE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
CACHE_REFRESH_SEC = int(os.getenv("PPE_CACHE_REFRESH_SEC", "60"))

# keep for /health
CONFIRM_ENABLED = os.getenv("PPE_CONFIRM_ENABLED", "true").lower() in ("1", "true", "yes", "on")
CONFIRM_INTERVAL = os.getenv("PPE_CONFIRM_INTERVAL", "1h").strip()

# thresholds (simple detector)
CONF_PRICE_FLAT_PCT = float(os.getenv("PPE_CONF_PRICE_FLAT_PCT", "0.003"))
CONF_PRICE_MOVE_PCT = float(os.getenv("PPE_CONF_PRICE_MOVE_PCT", "0.007"))

MIN_BARS_IN_PHASE_FOR_SIGNAL = int(os.getenv("PPE_MIN_BARS_IN_PHASE_FOR_SIGNAL", "2"))

# CVD
CVD_ENABLED = os.getenv("PPE_CVD_ENABLED", "true").lower() in ("1", "true", "yes", "on")
CVD_WINDOW_BARS = int(os.getenv("PPE_CVD_WINDOW_BARS", "120"))

# binance
BINANCE_FAPI_BASE = os.getenv("PPE_BINANCE_FAPI_BASE", "https://fapi.binance.com").strip()

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polaris-phase-engine/1.1"})

app = Flask(__name__)

_CACHE: Dict[str, Any] = {
    "updated_at": 0,
    "signals": [],
    "error": None,
    "duration_ms": None,
}


def _now_ts() -> int:
    return int(time.time())


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 12) -> Any:
    r = SESSION.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _binance_fapi_klines(symbol: str, interval: str, limit: int) -> List[List[Any]]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines"
    return _http_get_json(url, params={"symbol": symbol, "interval": interval, "limit": limit})


def _binance_fapi_open_interest(symbol: str) -> Dict[str, Any]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/openInterest"
    return _http_get_json(url, params={"symbol": symbol})


def _binance_fapi_ticker_24h(symbol: str) -> Dict[str, Any]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/ticker/24hr"
    return _http_get_json(url, params={"symbol": symbol})


def _calc_cvd(kl: List[List[Any]], window: int) -> Dict[str, Any]:
    """
    Binance Futures klines:
      [5]  volume (base)
      [7]  quoteAssetVolume
      [9]  takerBuyBaseAssetVolume
      [10] takerBuyQuoteAssetVolume

    CVD (base)  ~= Σ( buy_base - sell_base )  = Σ( 2*takerBuyBase - volume )
    CVD (quote) ~= Σ( 2*takerBuyQuote - quoteAssetVolume )
    """
    if not kl:
        return {
            "cvdWindow": window,
            "cvdBase": 0.0,
            "cvdQuote": 0.0,
            "cvdLastBase": 0.0,
            "cvdLastQuote": 0.0,
            "takerBuyPctLast": None,
            "buyVolLast": 0.0,
            "sellVolLast": 0.0,
        }

    window = max(1, int(window or 1))
    slice_kl = kl[-window:] if len(kl) > window else kl

    cvd_base = 0.0
    cvd_quote = 0.0

    for k in slice_kl:
        vol_base = _safe_float(k[5]) if len(k) > 5 else 0.0
        vol_quote = _safe_float(k[7]) if len(k) > 7 else 0.0
        buy_base = _safe_float(k[9]) if len(k) > 9 else 0.0
        buy_quote = _safe_float(k[10]) if len(k) > 10 else 0.0

        cvd_base += (2.0 * buy_base - vol_base)
        cvd_quote += (2.0 * buy_quote - vol_quote)

    last = kl[-1]
    vol_last = _safe_float(last[5]) if len(last) > 5 else 0.0
    buy_last = _safe_float(last[9]) if len(last) > 9 else 0.0
    sell_last = max(vol_last - buy_last, 0.0)

    volq_last = _safe_float(last[7]) if len(last) > 7 else 0.0
    buyq_last = _safe_float(last[10]) if len(last) > 10 else 0.0

    cvd_last_base = 2.0 * buy_last - vol_last
    cvd_last_quote = 2.0 * buyq_last - volq_last

    taker_buy_pct = (buy_last / vol_last) if vol_last > 0 else None

    return {
        "cvdWindow": window,
        "cvdBase": cvd_base,
        "cvdQuote": cvd_quote,
        "cvdLastBase": cvd_last_base,
        "cvdLastQuote": cvd_last_quote,
        "takerBuyPctLast": taker_buy_pct,
        "buyVolLast": buy_last,
        "sellVolLast": sell_last,
    }


@dataclass
class PhaseSignal:
    symbol: str
    side: str      # LONG / SHORT
    phase: str     # range / impulse
    ts: int
    price: float
    info: Dict[str, Any]


def _detect_phase_signal(symbol: str) -> Optional[PhaseSignal]:
    kl = _binance_fapi_klines(symbol, INTERVAL, KLINES_LIMIT)
    if not kl or len(kl) < max(5, MIN_BARS_IN_PHASE_FOR_SIGNAL):
        return None

    closes = [_safe_float(k[4]) for k in kl]
    highs = [_safe_float(k[2]) for k in kl]
    lows = [_safe_float(k[3]) for k in kl]
    vols = [_safe_float(k[5]) for k in kl]

    last_close = closes[-1]
    last_high = highs[-1]
    last_low = lows[-1]

    prev_close = closes[-2]
    price_delta = last_close - prev_close
    price_delta_pct = (price_delta / prev_close) if prev_close else 0.0

    rng = ((last_high - last_low) / last_close) if last_close else 0.0

    # отсекаем супер-шум
    if abs(price_delta_pct) < CONF_PRICE_MOVE_PCT and rng < CONF_PRICE_FLAT_PCT:
        return None

    oi_raw = _binance_fapi_open_interest(symbol)
    oi_val = _safe_float(oi_raw.get("openInterest", 0.0))

    t24 = _binance_fapi_ticker_24h(symbol)
    last_price_24 = _safe_float(t24.get("lastPrice", last_close))

    side = "LONG" if price_delta > 0 else "SHORT"
    phase = "impulse" if abs(price_delta_pct) >= CONF_PRICE_MOVE_PCT else "range"

    info: Dict[str, Any] = {
        "interval": INTERVAL,
        "klineCount": len(kl),
        "lastPrice24h": last_price_24,
        "oi": oi_val,
        "priceDelta": price_delta,
        "priceDeltaPct": price_delta_pct,
        "rangePct": rng,
        "volLast": vols[-1] if vols else 0.0,
    }

    if CVD_ENABLED:
        info.update(_calc_cvd(kl, CVD_WINDOW_BARS))

    return PhaseSignal(
        symbol=symbol,
        side=side,
        phase=phase,
        ts=_now_ts(),
        price=last_close,
        info=info,
    )


def _build_signals() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in SYMBOLS:
        try:
            sig = _detect_phase_signal(s)
            if not sig:
                continue
            out.append(
                {
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "phase": sig.phase,
                    "ts": sig.ts,
                    "price": sig.price,
                    "info": sig.info,
                }
            )
        except Exception as e:
            logger.error("Signal build failed for %s: %s", s, e)
            logger.debug(traceback.format_exc())
    return out


def _refresh_cache() -> None:
    t0 = time.time()
    err = None
    sigs: List[Dict[str, Any]] = []
    try:
        sigs = _build_signals()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error("Cache refresh failed: %s", err)
        logger.debug(traceback.format_exc())

    dt_ms = int((time.time() - t0) * 1000)
    _CACHE["updated_at"] = _now_ts()
    _CACHE["signals"] = sigs
    _CACHE["error"] = err
    _CACHE["duration_ms"] = dt_ms

    logger.info("Cache refreshed: symbols=%d signals=%d %dms", len(SYMBOLS), len(sigs), dt_ms)


def _maybe_refresh_cache() -> None:
    if not CACHE_ENABLED:
        return
    updated = int(_CACHE.get("updated_at", 0) or 0)
    age = _now_ts() - updated if updated else 10**9
    if updated == 0 or age >= CACHE_REFRESH_SEC:
        _refresh_cache()


@app.get("/health")
def health():
    _maybe_refresh_cache()
    updated = int(_CACHE.get("updated_at", 0) or 0)
    cache_age = None if updated == 0 else (_now_ts() - updated)

    return jsonify(
        {
            "status": "ok",
            "interval": INTERVAL,
            "symbols": SYMBOLS,
            "cacheEnabled": CACHE_ENABLED,
            "cacheRefreshSec": CACHE_REFRESH_SEC,
            "cacheUpdatedAt": updated,
            "cacheAgeSec": cache_age,
            "cacheDurationMs": _CACHE.get("duration_ms"),
            "cacheError": _CACHE.get("error"),
            "confirmEnabled": CONFIRM_ENABLED,
            "confirmInterval": CONFIRM_INTERVAL,
            "cvdEnabled": CVD_ENABLED,
            "cvdWindowBars": CVD_WINDOW_BARS,
        }
    )


@app.get("/signals")
def signals():
    _maybe_refresh_cache()
    if CACHE_ENABLED:
        return jsonify({"signals": _CACHE.get("signals", []) or []})

    try:
        return jsonify({"signals": _build_signals()})
    except Exception as e:
        return jsonify({"signals": [], "error": f"{type(e).__name__}: {e}"}), 500


@app.post("/refresh")
def refresh():
    _refresh_cache()
    return jsonify({"ok": True, "signals": _CACHE.get("signals", []), "updatedAt": _CACHE.get("updated_at")})


def main():
    logger.info(
        "Starting Polaris Phase Engine on %s:%s symbols=%s interval=%s confirm=%s/%s cache=%s refresh=%ss cvd=%s window=%d",
        HTTP_HOST,
        HTTP_PORT,
        ",".join(SYMBOLS),
        INTERVAL,
        "on" if CONFIRM_ENABLED else "off",
        CONFIRM_INTERVAL,
        "on" if CACHE_ENABLED else "off",
        CACHE_REFRESH_SEC,
        "on" if CVD_ENABLED else "off",
        CVD_WINDOW_BARS,
    )

    if CACHE_ENABLED:
        try:
            _refresh_cache()
        except Exception:
            logger.debug(traceback.format_exc())

    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False)


if __name__ == "__main__":
    main()
