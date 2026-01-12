#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("polaris-tg-forwarder")
logging.basicConfig(
    level=os.getenv("PPE_TG_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TG_BOT_TOKEN = os.getenv("PPE_TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("PPE_TG_CHAT_ID", "").strip()

POLL_INTERVAL = int(os.getenv("PPE_TG_POLL_INTERVAL", "60"))
SIGNALS_URL = os.getenv("PPE_TG_SIGNALS_URL", "http://127.0.0.1:8001/signals").strip()

TV_LINK_TEMPLATE = os.getenv(
    "PPE_TG_TV_LINK_TEMPLATE",
    "https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}",
).strip()

STATE_PATH = os.getenv("PPE_TG_STATE_PATH", "/opt/polaris-phase-engine/.tg_forwarder_state.json").strip()
STATE_MAX_UIDS = int(os.getenv("PPE_TG_STATE_MAX", "5000"))
STATE_TTL_SEC = int(os.getenv("PPE_TG_STATE_TTL_SEC", str(7 * 24 * 3600)))  # 7 days

ENGINE_MODE = os.getenv("PPE_TG_ENGINE_MODE", "changes").strip().lower()  # "changes" | "all"
ENGINE_SEND_STARTUP = os.getenv("PPE_TG_ENGINE_SEND_STARTUP", "true").lower() in ("1", "true", "yes", "on")

HTTP_TIMEOUT_SEC = int(os.getenv("PPE_TG_HTTP_TIMEOUT", "15"))

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polaris-tg-forwarder/2.3"})


def _tv_link(symbol: str) -> str:
    try:
        return TV_LINK_TEMPLATE.format(symbol=symbol)
    except Exception:
        return TV_LINK_TEMPLATE


def _ts_to_str(ts: Optional[int]) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ""


def _fmt_num(x: Any, digits: int = 2, suffix: bool = True) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if not suffix:
        return f"{v:.{digits}f}"
    sign = "-" if v < 0 else ""
    v = abs(v)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= div:
            return f"{sign}{v / div:.{digits}f}{unit}"
    return f"{sign}{v:.{digits}f}"


def _fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "n/a"


def _load_state() -> Dict[str, Any]:
    """
    new format:
      { "uids": {uid: ts}, "last": {symbol: {"fp": "...", "ts": ...}} }
    legacy format:
      { uid: ts, ... }
    """
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "uids" in raw and "last" in raw:
            uids = raw.get("uids", {}) if isinstance(raw.get("uids", {}), dict) else {}
            last = raw.get("last", {}) if isinstance(raw.get("last", {}), dict) else {}

            uids2 = {str(k): int(v) for k, v in uids.items() if isinstance(v, (int, float, str))}
            last2: Dict[str, Dict[str, Any]] = {}
            for sym, v in last.items():
                if not isinstance(v, dict):
                    continue
                fp = str(v.get("fp", "")).strip()
                ts = int(v.get("ts", 0) or 0)
                if sym and fp:
                    last2[str(sym).upper()] = {"fp": fp, "ts": ts}
            return {"uids": uids2, "last": last2}

        if isinstance(raw, dict):
            uids = {}
            for k, v in raw.items():
                try:
                    uids[str(k)] = int(v)
                except Exception:
                    continue
            return {"uids": uids, "last": {}}
    except Exception:
        pass
    return {"uids": {}, "last": {}}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        tmp = {"uids": state.get("uids", {}), "last": state.get("last", {})}
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(tmp, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to save state: %s", e)


def _prune_state(state: Dict[str, Any]) -> Dict[str, Any]:
    now = int(time.time())
    uids: Dict[str, int] = state.get("uids", {}) if isinstance(state.get("uids", {}), dict) else {}
    last: Dict[str, Dict[str, Any]] = state.get("last", {}) if isinstance(state.get("last", {}), dict) else {}

    uids = {k: int(v) for k, v in uids.items() if (now - int(v)) <= STATE_TTL_SEC}

    if len(uids) > STATE_MAX_UIDS:
        items = sorted(uids.items(), key=lambda kv: kv[1], reverse=True)[:STATE_MAX_UIDS]
        uids = dict(items)

    return {"uids": uids, "last": last}


def _is_engine_signal(sig: Dict[str, Any]) -> bool:
    return "symbol" in sig and "side" in sig and "phase" in sig and isinstance(sig.get("info"), dict)


def _make_uid_engine(sig: Dict[str, Any]) -> str:
    symbol = str(sig.get("symbol", "")).upper()
    side = str(sig.get("side", "")).upper()
    phase = str(sig.get("phase", "")).lower()
    ts = sig.get("ts") or ""
    return f"ENGINE|{symbol}|{side}|{phase}|{ts}"


def _engine_fingerprint(sig: Dict[str, Any]) -> str:
    side = str(sig.get("side", "")).upper()
    phase = str(sig.get("phase", "")).lower()
    return f"{side}|{phase}"


def _format_engine(sig: Dict[str, Any]) -> str:
    symbol = str(sig.get("symbol", "")).upper()
    side = str(sig.get("side", "")).upper()
    phase = str(sig.get("phase", "")).lower()
    price = sig.get("price", None)
    ts = sig.get("ts", None)
    info = sig.get("info", {}) if isinstance(sig.get("info", {}), dict) else {}

    interval = info.get("interval", "4h")
    oi = info.get("oi", None)
    pd_pct = info.get("priceDeltaPct", None)
    rng_pct = info.get("rangePct", None)
    vol_last = info.get("volLast", None)

    # CVD (prefer quote)
    cvd_w = info.get("cvdWindow", None)
    cvd_q = info.get("cvdQuote", None)
    cvd_lq = info.get("cvdLastQuote", None)
    tb_pct = info.get("takerBuyPctLast", None)

    link = _tv_link(symbol)
    t = _ts_to_str(ts)

    lines = [
        f"📡 ENGINE  {symbol}  {side}  {phase}",
        f"tf: {interval} | price: {_fmt_num(price, digits=2, suffix=False)}",
        f"ΔP: {_fmt_pct(pd_pct)} | OI: {_fmt_num(oi)} | vol: {_fmt_num(vol_last)} | rng: {_fmt_pct(rng_pct)}",
    ]

    if cvd_w is not None and (cvd_q is not None or cvd_lq is not None):
        cvd_line = f"CVD({cvd_w}): {_fmt_num(cvd_q)}"
        if cvd_lq is not None:
            cvd_line += f" | ΔCVD(last): {_fmt_num(cvd_lq)}"
        if tb_pct is not None:
            cvd_line += f" | takerBuy: {_fmt_pct(tb_pct)}"
        lines.append(cvd_line)

    if t:
        lines.append(f"time: {t}")

    lines.append(f"TV: {link}")
    return "\n".join(lines)


def _fetch_signals() -> List[Dict[str, Any]]:
    r = SESSION.get(SIGNALS_URL, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    sigs = data.get("signals", [])
    if isinstance(sigs, list):
        return [s for s in sigs if isinstance(s, dict)]
    return []


def _send_to_telegram(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("PPE_TG_BOT_TOKEN or PPE_TG_CHAT_ID is not set")

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    r = SESSION.post(url, json=payload, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()


def main() -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        logger.error("PPE_TG_BOT_TOKEN is not set or PPE_TG_CHAT_ID is not set")
        return

    logger.info(
        "Starting TG forwarder: url=%s chat_id=%s interval=%ss mode=%s",
        SIGNALS_URL,
        TG_CHAT_ID,
        POLL_INTERVAL,
        ENGINE_MODE,
    )

    state = _prune_state(_load_state())
    uids: Dict[str, int] = state.get("uids", {})
    last: Dict[str, Dict[str, Any]] = state.get("last", {})

    first_cycle = True

    while True:
        try:
            sigs = _fetch_signals()
            logger.info("Fetched %d signals", len(sigs))

            now = int(time.time())
            sent_any = False

            for sig in sigs:
                if not _is_engine_signal(sig):
                    continue

                symbol = str(sig.get("symbol", "")).upper()
                uid = _make_uid_engine(sig)
                fp = _engine_fingerprint(sig)

                if uid in uids:
                    continue

                if ENGINE_MODE == "changes":
                    prev_fp = (last.get(symbol, {}) or {}).get("fp")
                    if prev_fp == fp:
                        uids[uid] = now
                        continue

                    if first_cycle and not ENGINE_SEND_STARTUP and prev_fp is None:
                        last[symbol] = {"fp": fp, "ts": now}
                        uids[uid] = now
                        continue

                msg = _format_engine(sig)
                _send_to_telegram(msg)

                uids[uid] = now
                last[symbol] = {"fp": fp, "ts": now}
                sent_any = True

                logger.info("Sent ENGINE signal to Telegram: %s %s %s", symbol, sig.get("side"), sig.get("phase"))

            first_cycle = False

            if sent_any:
                state = _prune_state({"uids": uids, "last": last})
                uids = state["uids"]
                last = state["last"]
                _save_state(state)

        except Exception as e:
            logger.error("Failed cycle: %s", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
