#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

PPE_TG_BOT_TOKEN = os.getenv("PPE_TG_BOT_TOKEN")
PPE_TG_CHAT_ID = os.getenv("PPE_TG_CHAT_ID")
PPE_TG_POLL_INTERVAL = int(os.getenv("PPE_TG_POLL_INTERVAL", "60"))
PPE_TG_SIGNALS_URL = os.getenv("PPE_TG_SIGNALS_URL", "http://127.0.0.1:8001/signals")

if not PPE_TG_BOT_TOKEN:
    raise RuntimeError("PPE_TG_BOT_TOKEN is not set")
if not PPE_TG_CHAT_ID:
    raise RuntimeError("PPE_TG_CHAT_ID is not set")

TG_SEND_URL = f"https://api.telegram.org/bot{PPE_TG_BOT_TOKEN}/sendMessage"

logger = logging.getLogger("polaris-tg-forwarder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

session = requests.Session()
session.headers.update({"User-Agent": "PolarisTGForwarder/wyc-1"})


# ----- helpers --------------------------------------------------------------


def _trend_str(trend_dir: Any) -> str:
    try:
        d = int(trend_dir)
    except Exception:
        return "нет данных"
    if d > 0:
        return "UP ↑"
    if d < 0:
        return "DOWN ↓"
    return "FLAT →"


def _tv_link(symbol: str, exchange: str = "BINANCE") -> str:
    sym = f"{exchange}:{symbol.upper()}"
    return f"https://www.tradingview.com/chart/?symbol={quote_plus(sym)}"


def _make_signal_id(sig: Dict[str, Any]) -> str:
    """
    Уникальный id сигнала, чтобы не слать дубли.
    Привязан к символу, типу, стороне и времени генерации.
    """
    parts = [
        str(sig.get("symbol", "")),
        str(sig.get("kind", "")),
        str(sig.get("side", "")),
        str(sig.get("phase", "")),
        str(sig.get("confirmType", "")),
        str(sig.get("generatedAt", "")),
    ]
    key = "|".join(parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# ----- formatting -----------------------------------------------------------


def _format_pre(sig: Dict[str, Any]) -> str:
    """
    Сообщение для PRE-сигнала (4h PUMP/DUMP).
    """
    symbol = sig.get("symbol", "?")
    side = sig.get("side", "").upper()  # LONG / SHORT
    phase = sig.get("phase", "?")
    interval = sig.get("interval", "4h")
    bars = sig.get("barsInPhase", "?")
    trend = _trend_str(sig.get("trendDir"))

    if side == "LONG":
        emoji = "🚀"
    elif side == "SHORT":
        emoji = "🛑"
    else:
        emoji = "⚠️"

    tv = _tv_link(symbol)

    text = (
        f"{emoji} [PRE {side}] {symbol} {interval} {phase}\n\n"
        f"Фаза {interval}: {phase} (баров в фазе: {bars}, тренд: {trend})\n\n"
        f"TV: {tv}"
    )
    return text


def _format_conf(sig: Dict[str, Any]) -> str:
    """
    Сообщение для CONF Wyckoff:
    LONG: confirmType = WYC_TR_SPRING
    SHORT: confirmType = WYC_TR_UT
    """
    symbol = sig.get("symbol", "?")
    side = sig.get("side", "").upper()  # LONG / SHORT
    phase = sig.get("phase", "?")
    interval = sig.get("interval", "4h")          # базовый ТФ фаз
    trigger_interval = sig.get("triggerInterval", "1h")  # ТФ триггера (1h)
    bars = sig.get("barsInPhase", "?")
    trend = _trend_str(sig.get("trendDir"))
    confirm_type = sig.get("confirmType", "")

    range_low = sig.get("rangeLow")
    range_high = sig.get("rangeHigh")
    entry_close = sig.get("entryClose")

    cvd_delta = sig.get("cvdDelta")
    cvd_str = ""
    if cvd_delta is not None:
        try:
            cvd_val = float(cvd_delta)
            cvd_str = f"{cvd_val:+.0f}"
        except Exception:
            cvd_str = "n/a"

    if side == "LONG":
        emoji = "✅"
        label = "WYC SPRING"
        spring_low = sig.get("springLow")
        extra_line = f"Spring: {spring_low:.6f}" if isinstance(spring_low, (float, int)) else "Spring: n/a"
    else:
        emoji = "⛔️"
        label = "WYC UT"
        ut_high = sig.get("upthrustHigh")
        extra_line = f"Upthrust: {ut_high:.6f}" if isinstance(ut_high, (float, int)) else "Upthrust: n/a"

    tv = _tv_link(symbol)

    # TR строка
    if isinstance(range_low, (float, int)) and isinstance(range_high, (float, int)):
        tr_line = f"TR {trigger_interval}: {range_low:.6f}–{range_high:.6f}"
    else:
        tr_line = f"TR {trigger_interval}: n/a"

    # entry строка
    if isinstance(entry_close, (float, int)):
        entry_line = f"Вход (close {trigger_interval}): {entry_close:.6f}"
    else:
        entry_line = f"Вход (close {trigger_interval}): n/a"

    # CVD строка
    if cvd_str:
        cvd_line = f"CVD Δ (окно TR): {cvd_str}"
    else:
        cvd_line = "CVD Δ (окно TR): n/a"

    text = (
        f"{emoji} [CONF {side} • {label}] {symbol} {interval}→{trigger_interval}\n\n"
        f"Фаза {interval}: {phase} (баров в фазе: {bars}, тренд: {trend})\n"
        f"{tr_line}\n"
        f"{extra_line}\n"
        f"{entry_line}\n"
        f"{cvd_line}\n\n"
        f"TV: {tv}"
    )
    return text


def _format_signal(sig: Dict[str, Any]) -> Optional[str]:
    kind = sig.get("kind")
    if kind == "PRE":
        return _format_pre(sig)
    if kind == "CONF":
        # сейчас у нас только Wyckoff-CONF
        return _format_conf(sig)
    # на всякий пожарный — неизвестный тип не шлём
    return None


# ----- telegram -------------------------------------------------------------


def _send_telegram(text: str) -> None:
    try:
        resp = session.post(
            TG_SEND_URL,
            json={
                "chat_id": PPE_TG_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",  # вдруг захотим стили в будущем
                "disable_web_page_preview": False,
            },
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Sent Telegram message (%d bytes)", len(text))
    except Exception as exc:
        logger.error("Failed to send Telegram message: %s", exc)


# ----- main loop ------------------------------------------------------------


def main() -> None:
    logger.info(
        "Starting Telegram forwarder: signals_url=%s chat_id=%s interval=%ss",
        PPE_TG_SIGNALS_URL,
        PPE_TG_CHAT_ID,
        PPE_TG_POLL_INTERVAL,
    )

    sent: set[str] = set()

    while True:
        try:
            resp = session.get(PPE_TG_SIGNALS_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            signals: List[Dict[str, Any]] = data.get("signals", [])
        except Exception as exc:
            logger.error("Failed to fetch signals: %s", exc)
            time.sleep(PPE_TG_POLL_INTERVAL)
            continue

        for sig in signals:
            sig_id = _make_signal_id(sig)
            if sig_id in sent:
                continue

            text = _format_signal(sig)
            if not text:
                continue

            _send_telegram(text)
            sent.add(sig_id)

        time.sleep(PPE_TG_POLL_INTERVAL)


if __name__ == "__main__":
    main()
