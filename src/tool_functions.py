import json
import time

import requests

HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"


def fetch_hype_candles(lookback_days: int = 5, interval: str = "4h") -> str:
    """Fetch HYPE candlestick data from Hyperliquid's Info API."""
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

    try:
        resp = requests.post(HYPERLIQUID_URL, json=payload, timeout=10)
        resp.raise_for_status()
        candles = resp.json()

        if not candles:
            return json.dumps({"error": "No candle data returned"})

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


TOOL_FUNCTIONS = {
    "fetch_hype_candles": fetch_hype_candles,
}
