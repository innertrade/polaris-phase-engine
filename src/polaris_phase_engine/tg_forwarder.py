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
    level=logging.INFO,
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
STATE_MAX = int(os.getenv("PPE_TG_STATE_MAX", "5000"))
STATE_TTL_SEC = int(os.getenv("PPE_TG_STATE_TTL_SEC", str(7 * 24 * 3600)))  # 7 days

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polaris-tg-forwarder/2.2"})


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


def _human_num(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    sgn = "-" if v < 0 else ""
    v = abs(v)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= div:
            return f"{sgn}{v/div:.2f}{unit}"
    return f"{sgn}{v:.2f}"


def _pct(x: Any) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "n/a"


def _num(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def _load_state() -> Dict[str, int]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        pass
    return {}


def _save_state(state: Dict[str, int]) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to save state: %s", e)


def _prune_state(state: Dict[str, int]) -> Dict[str, int]:
    now = int(time.time())
    state = {k: v for k, v in state.items() if (now - int(v)) <= STATE_TTL_SEC}
    if len(state) > STATE_MAX:
        items = sorted(state.items(), key=lambda kv: kv[1], reverse=True)[:STATE_MAX]
        state = dict(items)
    return state


def _make_uid(sig: Dict[str, Any]) -> str:
    kind = str(sig.get("kind", "")).upper()
    symbol = str(sig.get("symbol", "")).upper()
    side = str(sig.get("side", "")).upper()
    ctype = str(sig.get("confirmType", "")).upper()
    ts = sig.get("generatedAt") or sig.get("ts") or sig.get("timestamp") or ""
    # kind|type|symbol|side|ts is enough for de-dupe
    return f"{kind}|{ctype}|{symbol}|{side}|{ts}"


def _format_pre(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "")
    side = sig.get("side", "")
    phase = sig.get("phase", "")
    interval = sig.get("interval", "4h")
    bars = sig.get("barsInPhase", "")
    trend = sig.get("trendDir", "")
    t = _ts_to_str(sig.get("generatedAt"))
    link = _tv_link(symbol)

    trend_txt = "UP ↑" if int(trend or 0) > 0 else ("DOWN ↓" if int(trend or 0) < 0 else "FLAT →")

    lines = [
        f"🟡 *PRE {side}*  `{symbol}`",
        f"_ctx_: {interval} {phase} | bars: {bars} | trend: {trend_txt}",
        "⚠️ вход: *только после ретеста на 15m* (не по первой свече).",
    ]
    if t:
        lines.append(f"_time_: {t}")
    lines.append(f"🔗 TV: {link}")
    return "\n".join(lines)


def _format_conf(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "")
    side = sig.get("side", "")
    ctype = sig.get("confirmType", "CONF")
    phase = sig.get("phase", "")
    interval = sig.get("interval", "4h")
    trig = sig.get("triggerInterval", "1h")
    bars = sig.get("barsInPhase", "")
    lb = sig.get("lookbackBars", "")
    t = _ts_to_str(sig.get("generatedAt"))
    link = _tv_link(symbol)

    pc = sig.get("priceChangePct", None)
    oi = sig.get("oiChangePct", None)
    dr = sig.get("deltaRatio", None)
    cvd_q = sig.get("cvdDeltaQuote", None)

    lines = [
        f"✅ *CONF* `{ctype}`  `{symbol}`  → *{side}*",
        f"_ctx_: {interval} {phase} | bars: {bars} | _tf_: {trig} | N: {lb}",
        f"ΔP: {_pct(pc)} | ΔOI: {_pct(oi)} | CVDΔ: {_human_num(cvd_q)} | ΔR: {_num(dr, 3)}",
        "🎯 вход: *только ретест 15m* (pullback), не “вжух” по первой.",
    ]
    if t:
        lines.append(f"_time_: {t}")
    lines.append(f"🔗 TV: {link}")
    return "\n".join(lines)


def _format_message(sig: Dict[str, Any]) -> str:
    kind = str(sig.get("kind", "")).upper()
    if kind == "PRE":
        return _format_pre(sig)
    if kind == "CONF":
        return _format_conf(sig)
    # fallback
    return f"ℹ️ `{sig}`"


def _fetch_signals() -> List[Dict[str, Any]]:
    r = SESSION.get(SIGNALS_URL, timeout=10)
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
    r = SESSION.post(url, json=payload, timeout=10)
    r.raise_for_status()


def main() -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        logger.error("PPE_TG_BOT_TOKEN is not set or PPE_TG_CHAT_ID is not set")
        return

    logger.info("Starting TG forwarder: url=%s chat_id=%s interval=%ss", SIGNALS_URL, TG_CHAT_ID, POLL_INTERVAL)

    state = _prune_state(_load_state())

    while True:
        try:
            sigs = _fetch_signals()
            logger.info("Fetched %d signals", len(sigs))

            sent_any = False
            now = int(time.time())

            for sig in sigs:
                uid = _make_uid(sig)
                if uid in state:
                    continue

                msg = _format_message(sig)
                _send_to_telegram(msg)

                state[uid] = now
                sent_any = True
                logger.info("Sent signal to Telegram: %s %s %s", sig.get("symbol"), sig.get("kind"), sig.get("confirmType"))

            if sent_any:
                state = _prune_state(state)
                _save_state(state)

        except Exception as e:
            logger.error("Failed cycle: %s", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
