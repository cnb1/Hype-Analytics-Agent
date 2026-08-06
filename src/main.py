import os
import json
import time
import argparse
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / "resources" / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"

# ---------- Defaults (override at runtime with --days / --interval) ----------

DEFAULT_LOOKBACK_DAYS = 5
DEFAULT_INTERVAL = "4h"


# ---------- Tool: Fetch HYPE candle data from Hyperliquid ----------

def fetch_hype_candles(lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                       interval: str = DEFAULT_INTERVAL) -> str:
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

        # Clean up the data for the LLM — convert to readable format
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

        # Compute extremes here rather than leaving the model to scan the array —
        # it reliably picks the wrong row when eyeballing max/min over many candles.
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


# ---------- Tool definition for OpenAI function calling ----------

tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_hype_candles",
            "description": "Fetch HYPE candlestick (OHLCV) data from Hyperliquid. Use this to get price history for analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lookback_days": {
                        "type": "integer",
                        "description": (
                            "Number of days of historical data to fetch. "
                            f"Default {DEFAULT_LOOKBACK_DAYS}."
                        ),
                        "default": DEFAULT_LOOKBACK_DAYS,
                    },
                    "interval": {
                        "type": "string",
                        "description": (
                            "Candle interval. Options: 1m, 5m, 15m, 30m, 1h, 4h, 1d. "
                            f"Default {DEFAULT_INTERVAL}."
                        ),
                        "default": DEFAULT_INTERVAL,
                    },
                },
                "required": [],
            },
        },
    }
]

# ---------- Map tool names to functions ----------

tool_functions = {
    "fetch_hype_candles": fetch_hype_candles,
}


# ---------- Agent loop ----------

def run_agent(user_message: str,
              lookback_days: int = DEFAULT_LOOKBACK_DAYS,
              interval: str = DEFAULT_INTERVAL):
    """Run the agent with a user message, handling tool calls in a loop."""

    system_prompt = f"""You are a crypto market analyst specializing in HYPE, a token on the Hyperliquid exchange.

        You have access to a tool that fetches real candlestick (OHLCV) data for HYPE from Hyperliquid's API.
        
        When asked about HYPE's price action, market structure, or any analysis:
        1. Use the fetch_hype_candles tool to pull the data you need
        2. Analyze the returned candlestick data
        3. Provide a clear, concise answer
        
        Unless the user explicitly asks for a different window, call the tool with
        lookback_days={lookback_days} and interval="{interval}".
        
        The tool result contains a "summary" object with the period high/low, the latest
        close, and percentage moves already computed exactly. ALWAYS take these figures
        from "summary" — never re-derive them by scanning the "candles" array yourself.
        Use "candles" only for describing shape and sequence (trend, consolidation,
        where volume clustered).
        
        Note that a period high or low is the extreme of the intraday wick (the "high"
        and "low" fields), not a candle's closing price. Say which you mean if you cite
        a close instead.
        
        Keep your analysis grounded in the actual data returned. Be specific with prices, percentages, and timeframes."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'=' * 60}")
    print(f"User: {user_message}")
    print(f"{'=' * 60}\n")

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        choice = response.choices[0]

        # If the model wants to call tool(s)
        if choice.finish_reason == "tool_calls":
            assistant_msg = choice.message
            messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"[Tool Call] {fn_name}({fn_args})")

                # Execute the tool
                fn = tool_functions[fn_name]
                result = fn(**fn_args)

                print(f"[Tool Result] {len(json.loads(result).get('candles', []))} candles returned\n")

                # Append the tool result back to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # If the model is done (no more tool calls)
        elif choice.finish_reason == "stop":
            final_response = choice.message.content
            print(f"Agent:\n{final_response}\n")
            return final_response

        else:
            print(f"Unexpected finish reason: {choice.finish_reason}")
            break


# ---------- Main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYPE market analysis agent.")
    parser.add_argument(
        "--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help=f"Days of price history to analyze (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--interval", default=DEFAULT_INTERVAL,
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help=f"Candle interval (default: {DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Custom question to ask the agent. Defaults to the peak-vs-current analysis.",
    )
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1")

    prompt = args.prompt or (
        f"Fetch the last {args.days} days of HYPE {args.interval} candle data. "
        "What was the recent peak price and when did it occur? "
        "How far is the current price from that peak?"
    )

    run_agent(prompt, lookback_days=args.days, interval=args.interval)
