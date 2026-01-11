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
STATE_TTL_SEC = int(os.getenv("PPE_TG_STATE_TTL_SEC", str(7 * 24 * 3600)))  # 7 дней

# Для нового формата /signals:
# - changes (default): шлём только если изменились phase/side (первый раз — отправим)
# - impulse_only: шлём только impulse и только на изменениях
# - all: шлём каждый цикл (осторожно: спам)
ENGINE_MODE = os.getenv("PPE_TG_ENGINE_MODE", "changes").strip().lower()

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


# ---------- STATE (v2) ----------
# state = {"v":2, "sent": {"KEY": {"ts": int, "sig": str}}}
def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Новый формат
        if isinstance(raw, dict) and raw.get("v") == 2 and isinstance(raw.get("sent"), dict):
            sent: Dict[str, Dict[str, Any]] = {}
            for k, rec in raw["sent"].items():
                if not isinstance(k, str) or not isinstance(rec, dict):
                    continue
                ts = int(rec.get("ts", 0) or 0)
                sig = str(rec.get("sig", "") or "")
                if ts > 0:
                    sent[k] = {"ts": ts, "sig": sig}
            return {"v": 2, "sent": sent}

        # Старый формат: dict[str,int]
        if isinstance(raw, dict):
            sent = {}
            ok = True
            for k, v in raw.items():
                if not isinstance(k, str):
                    ok = False
                    break
                try:
                    ts = int(v)
                except Exception:
                    ok = False
                    break
                sent[k] = {"ts": ts, "sig": ""}

            if ok:
                return {"v": 2, "sent": sent}

    except Exception:
        pass

    return {"v": 2, "sent": {}}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to save state: %s", e)


def _prune_state(state: Dict[str, Any]) -> Dict[str, Any]:
    now = int(time.time())
    sent = state.get("sent", {})
    if not isinstance(sent, dict):
        sent = {}

    # TTL
    kept: Dict[str, Dict[str, Any]] = {}
    for k, rec in sent.items():
        if not isinstance(k, str) or not isinstance(rec, dict):
            continue
        ts = int(rec.get("ts", 0) or 0)
        if ts <= 0:
            continue
        if (now - ts) <= STATE_TTL_SEC:
            kept[k] = {"ts": ts, "sig": str(rec.get("sig", "") or "")}

    # MAX
    if len(kept) > STATE_MAX:
        items = sorted(kept.items(), key=lambda kv: kv[1]["ts"], reverse=True)[:STATE_MAX]
        kept = dict(items)

    return {"v": 2, "sent": kept}


# ---------- legacy PRE/CONF ----------
def _make_uid(sig: Dict[str, Any]) -> str:
    kind = str(sig.get("kind", "")).upper()
    symbol = str(sig.get("symbol", "")).upper()
    side = str(sig.get("side", "")).upper()
    ctype = str(sig.get("confirmType", "")).upper()
    ts = sig.get("generatedAt") or sig.get("timestamp") or sig.get("ts") or ""
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
        f"🟡 PRE {side}  {symbol}",
        f"ctx: {interval} {phase} | bars: {bars} | trend: {trend_txt}",
    ]
    if t:
        lines.append(f"time: {t}")
    lines.append(f"TV: {link}")
    return "\n".join(lines)


def _format_conf(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "")
    side = sig.get("side", "")
    ctype = sig.get("confirmType", "CONF")
    phase = sig.get("phase", "")
    interval = sig.get("interval", "4h")
    trig = sig.get("triggerInterval", "1h")
    bars = sig.get("barsInPhase", "")
    t = _ts_to_str(sig.get("generatedAt"))
    link = _tv_link(symbol)

    oi = sig.get("oiChangePct", None)
    dr = sig.get("deltaRatio", None)
    pc = sig.get("priceChangePct", None)
    lb = sig.get("lookbackBars", None)
    if lb is None:
        lb = sig.get("windowBars", "")

    def pct(x: Any) -> str:
        try:
            return f"{float(x) * 100:.2f}%"
        except Exception:
            return "n/a"

    def num(x: Any) -> str:
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "n/a"

    lines = [
        f"✅ CONF {ctype}  {symbol} -> {side}",
        f"ctx: {interval} {phase} | bars: {bars} | tf: {trig} | lb: {lb}",
        f"OI: {pct(oi)} | Δ: {num(dr)} | Price: {pct(pc)}",
    ]
    if t:
        lines.append(f"time: {t}")
    lines.append(f"TV: {link}")
    return "\n".join(lines)


# ---------- engine /signals ----------
def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "n/a"


def _fmt_num(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"

    a = abs(v)
    if a >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"{v / 1_000:.2f}K"
    if a < 1:
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _is_engine_signal(sig: Dict[str, Any]) -> bool:
    return (
        isinstance(sig, dict)
        and "kind" not in sig
        and "symbol" in sig
        and "side" in sig
        and "phase" in sig
        and "ts" in sig
        and "info" in sig
    )


def _engine_signature(sig: Dict[str, Any]) -> str:
    phase = str(sig.get("phase", "")).lower()
    side = str(sig.get("side", "")).upper()
    return f"{phase}|{side}"


def _format_engine(sig: Dict[str, Any]) -> str:
    symbol = str(sig.get("symbol", "")).upper()
    side = str(sig.get("side", "")).upper()
    phase = str(sig.get("phase", "")).lower()
    price = sig.get("price", None)
    ts = sig.get("ts", None)
    info = sig.get("info", {}) or {}

    interval = str(info.get("interval", "") or "")
    oi = info.get("oi", None)
    pd = info.get("priceDelta", None)
    pdp = info.get("priceDeltaPct", None)
    rng = info.get("rangePct", None)
    vol = info.get("volLast", None)
    last24 = info.get("lastPrice24h", None)

    arrow = "🟢" if side == "LONG" else "🔴"
    ph = "⚡ impulse" if phase == "impulse" else "🧱 range"
    t = _ts_to_str(ts)

    lines = [f"{arrow} {symbol} {side}  ({ph}{' ' + interval if interval else ''})"]
    lines.append(f"Price: {_fmt_num(price)} | Δ24h: {_fmt_num(pd)} ({_fmt_pct(pdp)}) | last24h: {_fmt_num(last24)}")
    lines.append(f"OI: {_fmt_num(oi)} | Vol(last): {_fmt_num(vol)} | Range: {_fmt_pct(rng)}")
    if t:
        lines.append(f"Time: {t}")
    lines.append(f"TV: {_tv_link(symbol)}")
    return "\n".join(lines)


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

    logger.info(
        "Starting TG forwarder: url=%s chat_id=%s interval=%ss mode=%s",
        SIGNALS_URL,
        TG_CHAT_ID,
        POLL_INTERVAL,
        ENGINE_MODE,
    )

    state = _prune_state(_load_state())

    while True:
        try:
            sigs = _fetch_signals()
            logger.info("Fetched %d signals", len(sigs))

            sent_any = False
            now = int(time.time())
            sent = state.get("sent", {})
            if not isinstance(sent, dict):
                sent = {}

            for sig in sigs:
                # NEW engine format
                if _is_engine_signal(sig):
                    phase = str(sig.get("phase", "")).lower()
                    if ENGINE_MODE == "impulse_only" and phase != "impulse":
                        continue

                    symbol = str(sig.get("symbol", "")).upper()
                    key = f"ENG|{symbol}"
                    sigval = _engine_signature(sig)

                    rec = sent.get(key, {})
                    if ENGINE_MODE != "all" and isinstance(rec, dict) and rec.get("sig") == sigval:
                        continue

                    _send_to_telegram(_format_engine(sig))
                    sent[key] = {"ts": now, "sig": sigval}
                    sent_any = True
                    logger.info("Sent ENGINE signal to Telegram: %s %s %s", symbol, sig.get("side"), sig.get("phase"))
                    continue

                # legacy PRE/CONF
                uid = _make_uid(sig)
                if uid in sent:
                    continue

                kind = str(sig.get("kind", "")).upper()
                if kind == "PRE":
                    msg = _format_pre(sig)
                elif kind == "CONF":
                    msg = _format_conf(sig)
                else:
                    msg = f"INFO: {sig}"

                _send_to_telegram(msg)
                sent[uid] = {"ts": now, "sig": ""}
                sent_any = True
                logger.info("Sent legacy message to Telegram")

            if sent_any:
                state["sent"] = sent
                state = _prune_state(state)
                _save_state(state)

        except Exception as e:
            logger.error("Failed cycle: %s", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
