#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polaris Phase Engine -> Telegram forwarder.

Consumes /signals JSON from Polaris Phase Engine and forwards PRE/CONF/ENGINE signals
to a Telegram chat.

Environment:
  PPE_TG_BOT_TOKEN
  PPE_TG_CHAT_ID
  PPE_TG_POLL_INTERVAL (sec, default 60)
  PPE_TG_SIGNALS_URL (default http://127.0.0.1:8001/signals)
  PPE_TG_TV_LINK_TEMPLATE (default https://www.tradingview.com/chart/?symbol=BINANCE:{symbol})
  PPE_TG_STATE_PATH (default /opt/polaris-phase-engine/.tg_forwarder_state.json)
  PPE_TG_STATE_MAX (default 5000)
  PPE_TG_STATE_TTL_SEC (default 7 days)
  PPE_TG_MODE (default changes)  # changes|all
"""

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
MODE = os.getenv("PPE_TG_MODE", "changes").strip().lower()  # changes|all

SIGNALS_URL = os.getenv("PPE_TG_SIGNALS_URL", "http://127.0.0.1:8001/signals").strip()

TV_LINK_TEMPLATE = os.getenv(
    "PPE_TG_TV_LINK_TEMPLATE",
    "https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}",
).strip()

STATE_PATH = os.getenv("PPE_TG_STATE_PATH", "/opt/polaris-phase-engine/.tg_forwarder_state.json").strip()
STATE_MAX = int(os.getenv("PPE_TG_STATE_MAX", "5000"))
STATE_TTL_SEC = int(os.getenv("PPE_TG_STATE_TTL_SEC", str(7 * 24 * 3600)))  # 7 days

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polaris-tg-forwarder/3.0"})


# ---------- utils ----------

def _tv_link(symbol: str) -> str:
    try:
        return TV_LINK_TEMPLATE.format(symbol=symbol)
    except Exception:
        return TV_LINK_TEMPLATE


def _ts_to_str(ts: Optional[int]) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ""


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{x * 100:+.2f}%"
    except Exception:
        return "n/a"


def _fmt_ratio(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{x:+.2f}"
    except Exception:
        return "n/a"


def _fmt_huge(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
        sign = "+" if v >= 0 else "-"
        v = abs(v)
        for suf, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if v >= div:
                return f"{sign}{v / div:.2f}{suf}"
        return f"{sign}{v:.0f}"
    except Exception:
        return "n/a"


def _trend_arrow(trend_dir: Optional[int]) -> str:
    if trend_dir is None:
        return "?"
    return "UP ↑" if trend_dir > 0 else "DOWN ↓"


# ---------- state ----------

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
    items = [(k, v) for k, v in state.items() if now - int(v) <= STATE_TTL_SEC]
    items.sort(key=lambda kv: kv[1], reverse=True)
    if len(items) > STATE_MAX:
        items = items[:STATE_MAX]
    return {k: int(v) for k, v in items}


# ---------- formatting ----------

_CONF_HINTS = {
    "LONG_1_ABSORB": "контртренд/реверс: CVD↓ + OI↑ + цена держится",
    "LONG_2_BREAKOUT": "по тренду: CVD↑ + OI↑ + цена растёт",
    "SHORT_1_ABSORB": "контртренд/реверс: CVD↑ + OI↑ + цена держится",
    "SHORT_2_BREAKOUT": "по тренду: CVD↓ + OI↑ + цена падает",
}


def _format_pre(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "")
    side = sig.get("side", "")
    interval = sig.get("interval") or sig.get("tf") or "4h"
    phase = sig.get("phase", "")
    bars = sig.get("barsInPhase")
    trend_dir = sig.get("trendDir")
    ts = sig.get("generatedAt") or sig.get("ts")

    lines = [
        f"🟡 PRE {side}  {symbol}",
        f"ctx: {interval} {phase} | bars: {bars} | trend: {_trend_arrow(trend_dir)}",
        "⚠️ вход: только после ретеста на 15m (не по первой свече).",
    ]
    t = _ts_to_str(ts)
    if t:
        lines.append(f"time: {t}")
    lines.append(f"🔗 TV: {_tv_link(symbol)}")
    return "\n".join(lines)


def _format_conf(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "")
    side = sig.get("side", "")
    confirm_type = sig.get("confirmType", "")
    interval = sig.get("interval") or "4h"
    phase = sig.get("phase", "")
    info = sig.get("info") or {}

    trig = info.get("triggerInterval") or info.get("triggerTf") or "1h"
    lb = info.get("lookbackBars")

    d_p = info.get("priceChangePct")
    d_oi = info.get("oiChangePct")
    cvd_d = info.get("cvdDeltaQuote")
    dr = info.get("deltaRatio")
    tb = info.get("takerBuyPctLast")

    ts = sig.get("generatedAt") or sig.get("ts")

    hint = _CONF_HINTS.get(str(confirm_type), "")

    lines = [
        f"✅ CONF `{confirm_type}`  {symbol}  → {side}",
        f"tf: {interval} {phase} | trig: {trig} | lb: {lb}",
        f"ΔP: {_fmt_pct(d_p)} | ΔOI: {_fmt_pct(d_oi)} | CVDΔ: {_fmt_huge(cvd_d)} | DR: {_fmt_ratio(dr)}",
    ]
    if tb is not None:
        try:
            lines.append(f"takerBuy(last): {float(tb) * 100:.2f}%")
        except Exception:
            pass
    if hint:
        lines.append(f"hint: {hint}")

    t = _ts_to_str(ts)
    if t:
        lines.append(f"time: {t}")
    lines.append(f"🔗 TV: {_tv_link(symbol)}")
    return "\n".join(lines)


def _format_engine(sig: Dict[str, Any]) -> str:
    """Backward-compatible: older engine signals that include price/OI/vol/CVD."""
    symbol = sig.get("symbol", "")
    side = sig.get("side", "")
    phase = sig.get("phase", "")
    price = sig.get("price")
    info = sig.get("info") or {}
    ts = sig.get("ts") or sig.get("generatedAt")

    interval = info.get("interval") or sig.get("interval") or "4h"
    dp = info.get("priceDeltaPct")
    oi = info.get("oi")
    vol_last = info.get("volLast")
    rng = info.get("rangePct")

    cvd_quote = info.get("cvdQuote")
    cvd_last_quote = info.get("cvdLastQuote")
    cvd_win = info.get("cvdWindow")
    tb = info.get("takerBuyPctLast")

    lines = [f"📡 ENGINE  {symbol}  {side}  {phase}"]
    if price is not None:
        lines.append(f"tf: {interval} | price: {float(price):.2f}")
    else:
        lines.append(f"tf: {interval}")

    parts = []
    if dp is not None:
        parts.append(f"ΔP: {float(dp) * 100:+.2f}%")
    if oi is not None:
        # show OI as raw if already scaled
        try:
            parts.append(f"OI: {float(oi):.2f}")
        except Exception:
            parts.append(f"OI: {oi}")
    if vol_last is not None:
        parts.append(f"vol: {_fmt_huge(vol_last)}")
    if rng is not None:
        parts.append(f"rng: {float(rng) * 100:.2f}%")
    if parts:
        lines.append(" | ".join(parts))

    if cvd_quote is not None or cvd_last_quote is not None:
        win = f"({cvd_win})" if cvd_win is not None else ""
        line = f"CVD{win}: {_fmt_huge(cvd_quote)} | ΔCVD(last): {_fmt_huge(cvd_last_quote)}"
        if tb is not None:
            try:
                line += f" | takerBuy: {float(tb) * 100:.2f}%"
            except Exception:
                pass
        lines.append(line)

    t = _ts_to_str(ts)
    if t:
        lines.append(f"time: {t}")
    lines.append(f"TV: {_tv_link(symbol)}")
    return "\n".join(lines)


def _format_signal(sig: Dict[str, Any]) -> str:
    kind = (sig.get("kind") or "ENGINE").upper()
    if kind == "PRE":
        return _format_pre(sig)
    if kind == "CONF":
        return _format_conf(sig)
    return _format_engine(sig)


def _make_uid(sig: Dict[str, Any]) -> str:
    # if engine provides a uid/id - use it
    for k in ("uid", "id"):
        v = sig.get(k)
        if v:
            return str(v)

    kind = (sig.get("kind") or "ENGINE").upper()
    symbol = sig.get("symbol") or ""
    side = sig.get("side") or ""
    phase = sig.get("phase") or ""
    ct = sig.get("confirmType") or ""
    ts = sig.get("generatedAt") or sig.get("ts") or ""
    return f"{kind}|{symbol}|{side}|{phase}|{ct}|{ts}"


# ---------- network ----------

def _fetch_signals() -> List[Dict[str, Any]]:
    r = SESSION.get(SIGNALS_URL, timeout=10)
    r.raise_for_status()
    data = r.json()
    sigs = data.get("signals", [])
    if not isinstance(sigs, list):
        return []
    return [s for s in sigs if isinstance(s, dict)]


def _send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    r = SESSION.post(url, json=payload, timeout=15)
    r.raise_for_status()


def main() -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("PPE_TG_BOT_TOKEN or PPE_TG_CHAT_ID is not set")

    logger.info(
        "Starting TG forwarder: url=%s chat_id=%s interval=%ss mode=%s",
        SIGNALS_URL,
        TG_CHAT_ID,
        POLL_INTERVAL,
        MODE,
    )

    state = _load_state()

    while True:
        try:
            sigs = _fetch_signals()
            logger.info("Fetched %d signals", len(sigs))

            now = int(time.time())
            state = _prune_state(state)

            for sig in sigs:
                uid = _make_uid(sig)
                if MODE != "all" and uid in state:
                    continue

                msg = _format_signal(sig)
                if msg:
                    _send_telegram(msg)
                    state[uid] = now
                    logger.info(
                        "Sent %s signal to Telegram: %s %s",
                        (sig.get("kind") or "ENGINE"),
                        sig.get("symbol"),
                        sig.get("confirmType") or sig.get("phase") or "",
                    )

            _save_state(state)

        except Exception as e:
            logger.error("Failed cycle: %s", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
