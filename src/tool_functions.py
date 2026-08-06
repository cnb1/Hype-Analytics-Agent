import json
import time

import requests

HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"


def _fetch_raw_candles(lookback_days: int, interval: str) -> list[dict]:
    """Fetch raw candle data from Hyperliquid. Shared by all tools that need price data."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (lookback_days * 24 * 3600 * 1000)

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": "HYPE",
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    resp = requests.post(HYPERLIQUID_URL, json=payload, timeout=10)
    resp.raise_for_status()
    candles = resp.json()

    cleaned = []
    for c in candles:
        cleaned.append({
            "time": time.strftime("%Y-%m-%d %H:%M", time.gmtime(c["t"] / 1000)),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        })
    return cleaned


def fetch_hype_candles(lookback_days: int = 5, interval: str = "4h") -> str:
    """Fetch HYPE candlestick data from Hyperliquid's Info API."""
    try:
        cleaned = _fetch_raw_candles(lookback_days, interval)

        if not cleaned:
            return json.dumps({"error": "No candle data returned"})

        peak = max(cleaned, key=lambda c: c["high"])
        trough = min(cleaned, key=lambda c: c["low"])
        last_close = cleaned[-1]["close"]

        summary = {
            "period_high": peak["high"],
            "period_high_time": peak["time"],
            "period_low": trough["low"],
            "period_low_time": trough["time"],
            "first_open": cleaned[0]["open"],
            "last_close": last_close,
            "last_close_time": cleaned[-1]["time"],
            "pct_from_period_high": round(
                (last_close - peak["high"]) / peak["high"] * 100, 2
            ),
            "pct_from_period_low": round(
                (last_close - trough["low"]) / trough["low"] * 100, 2
            ),
            "pct_change_over_period": round(
                (last_close - cleaned[0]["open"]) / cleaned[0]["open"] * 100, 2
            ),
            "total_volume": round(sum(c["volume"] for c in cleaned), 4),
        }

        return json.dumps({
            "coin": "HYPE",
            "interval": interval,
            "lookback_days": lookback_days,
            "candle_count": len(cleaned),
            "summary": summary,
            "candles": cleaned,
        })

    except requests.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})


INTERVAL_TO_HOURS = {
    "1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5,
    "1h": 1, "4h": 4, "1d": 24,
}


def calculate_mfi(interval: str = "4h", period: int = 10) -> str:
    """Calculate the Money Flow Index (MFI) for HYPE using the trailing N candles."""
    try:
        hours_per_candle = INTERVAL_TO_HOURS.get(interval, 4)
        needed_candles = period + 1
        lookback_days = max(1, int((needed_candles * hours_per_candle) / 24) + 2)

        candles = _fetch_raw_candles(lookback_days, interval)

        if not candles:
            return json.dumps({"error": "No candle data returned"})

        if len(candles) < needed_candles:
            return json.dumps({
                "error": (
                    f"Need at least {needed_candles} candles for MFI-{period}, "
                    f"got {len(candles)}."
                )
            })

        candles = candles[-needed_candles:]

        typical_prices = [
            (c["high"] + c["low"] + c["close"]) / 3 for c in candles
        ]
        raw_money_flows = [
            typical_prices[i] * candles[i]["volume"]
            for i in range(len(candles))
        ]

        positive_flows = []
        negative_flows = []
        for i in range(1, len(candles)):
            if typical_prices[i] > typical_prices[i - 1]:
                positive_flows.append(raw_money_flows[i])
                negative_flows.append(0.0)
            else:
                positive_flows.append(0.0)
                negative_flows.append(raw_money_flows[i])

        pos_sum = sum(positive_flows)
        neg_sum = sum(negative_flows)

        if neg_sum == 0:
            mfi = 100.0
        else:
            mfi = 100.0 - (100.0 / (1.0 + pos_sum / neg_sum))

        mfi = round(mfi, 2)

        if mfi < 30:
            signal = "BUY"
            reason = f"MFI {mfi} is below 30 — oversold territory"
        elif mfi > 70:
            signal = "SELL"
            reason = f"MFI {mfi} is above 70 — overbought territory"
        else:
            signal = "NEUTRAL"
            reason = f"MFI {mfi} is between 30 and 70 — no strong signal"

        return json.dumps({
            "coin": "HYPE",
            "indicator": "MFI",
            "period": period,
            "interval": interval,
            "trailing_candles": [
                {"time": c["time"], "high": c["high"], "low": c["low"],
                 "close": c["close"], "volume": c["volume"]}
                for c in candles
            ],
            "mfi": mfi,
            "signal": signal,
            "reason": reason,
        })

    except requests.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})


TOOL_FUNCTIONS = {
    "fetch_hype_candles": fetch_hype_candles,
    "calculate_mfi": calculate_mfi,
}
