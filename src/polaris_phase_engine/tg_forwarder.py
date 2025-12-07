#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import time
from typing import Any, Dict, List, Set

import requests
from dotenv import load_dotenv

# ===================== КОНФИГ / ENV ===================== #

load_dotenv()

PPE_TG_BOT_TOKEN = os.getenv("PPE_TG_BOT_TOKEN", "").strip()
PPE_TG_CHAT_ID = os.getenv("PPE_TG_CHAT_ID", "").strip()
PPE_TG_POLL_INTERVAL = int(os.getenv("PPE_TG_POLL_INTERVAL", "60"))
PPE_TG_SIGNALS_URL = os.getenv(
    "PPE_TG_SIGNALS_URL", "http://127.0.0.1:8001/signals"
).strip()

if not PPE_TG_BOT_TOKEN:
    raise SystemExit("PPE_TG_BOT_TOKEN is not set")
if not PPE_TG_CHAT_ID:
    raise SystemExit("PPE_TG_CHAT_ID is not set")

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{PPE_TG_BOT_TOKEN}"

logger = logging.getLogger("polaris-tg-forwarder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ===================== УТИЛИТЫ ===================== #


def fetch_signals() -> List[Dict[str, Any]]:
    """Получить список сигналов с фазового движка."""
    try:
        resp = requests.get(PPE_TG_SIGNALS_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        signals = data.get("signals", [])
        if not isinstance(signals, list):
            logger.warning("Unexpected signals format: %r", data)
            return []
        return signals
    except Exception as exc:
        logger.error("Failed to fetch signals: %s", exc)
        return []


def trend_dir_to_str(trend_dir: Any) -> str:
    try:
        td = int(trend_dir)
    except Exception:
        return "?"
    if td > 0:
        return "UP ↑"
    if td < 0:
        return "DOWN ↓"
    return "FLAT →"


def format_signal_message(sig: Dict[str, Any]) -> str:
    """Сформировать текст сообщения для Telegram по одному сигналу."""
    symbol = sig.get("symbol", "?")
    side = sig.get("side", "?")  # LONG / SHORT
    kind = sig.get("kind", "?")  # PRE / CONF (пока только PRE)
    phase = sig.get("phase", "?")  # PUMP / DUMP
    interval = sig.get("interval", "?")
    bars = sig.get("barsInPhase", "?")
    trend = trend_dir_to_str(sig.get("trendDir"))

    # simple, но информативный формат
    # пример:
    # [PRE LONG] BTCUSDT 4h PUMP
    # bars in phase: 3, trend: UP ↑
    lines = [
        f"[{kind} {side}] {symbol} {interval} {phase}",
        f"bars in phase: {bars}, trend: {trend}",
    ]
    return "\n".join(lines)


def send_telegram_message(text: str) -> None:
    """Отправить текстовое сообщение в Telegram."""
    url = f"{TELEGRAM_API_BASE}/sendMessage"
    payload = {
        "chat_id": PPE_TG_CHAT_ID,
        "text": text,
        # без parse_mode, чтобы не париться с экранированием
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logger.error(
                "Telegram sendMessage failed: %s %s",
                resp.status_code,
                resp.text,
            )
        else:
            logger.info("Sent message to Telegram")
    except Exception as exc:
        logger.error("Error sending message to Telegram: %s", exc)


def build_signal_id(sig: Dict[str, Any]) -> str:
    """
    Уникальный ID сигнала для дедупликации.
    Если движок вернет тот же сигнал несколько раз,
    мы не будем его спамить в Телегу.
    """
    symbol = str(sig.get("symbol", ""))
    side = str(sig.get("side", ""))
    kind = str(sig.get("kind", ""))
    interval = str(sig.get("interval", ""))
    phase = str(sig.get("phase", ""))
    generated_at = str(sig.get("generatedAt", ""))

    return "|".join([symbol, side, kind, interval, phase, generated_at])


def main_loop() -> None:
    logger.info(
        "Starting Telegram forwarder: signals_url=%s chat_id=%s interval=%ss",
        PPE_TG_SIGNALS_URL,
        PPE_TG_CHAT_ID,
        PPE_TG_POLL_INTERVAL,
    )

    sent_ids: Set[str] = set()

    while True:
        try:
            signals = fetch_signals()
            if signals:
                logger.info("Fetched %d signals", len(signals))
            for sig in signals:
                sig_id = build_signal_id(sig)
                if sig_id in sent_ids:
                    continue  # уже отправляли

                text = format_signal_message(sig)
                send_telegram_message(text)
                sent_ids.add(sig_id)

                # чтобы set не рос бесконечно
                if len(sent_ids) > 1000:
                    # грубо, но достаточно: обнуляем
                    logger.info("Reset sent_ids set (size > 1000)")
                    sent_ids.clear()

        except Exception as exc:
            logger.error("Error in main loop: %s", exc)

        time.sleep(PPE_TG_POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
