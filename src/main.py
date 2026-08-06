import os
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"

# ---------- Tool: Fetch HYPE candle data from Hyperliquid ----------

def fetch_hype_candles(lookback_days: int = 10, interval: str = "4h") -> str:
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

        return json.dumps({
            "coin": "HYPE",
            "interval": interval,
            "lookback_days": lookback_days,
            "candle_count": len(cleaned),
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
                        "description": "Number of days of historical data to fetch. Default 10.",
                        "default": 10,
                    },
                    "interval": {
                        "type": "string",
                        "description": "Candle interval. Options: 1m, 5m, 15m, 30m, 1h, 4h, 1d. Default 4h.",
                        "default": "4h",
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

def run_agent(user_message: str):
    """Run the agent with a user message, handling tool calls in a loop."""

    system_prompt = """You are a crypto market analyst specializing in HYPE, a token on the Hyperliquid exchange.

You have access to a tool that fetches real candlestick (OHLCV) data for HYPE from Hyperliquid's API.

When asked about HYPE's price action, market structure, or any analysis:
1. Use the fetch_hype_candles tool to pull the data you need
2. Analyze the returned candlestick data
3. Provide a clear, concise answer

Keep your analysis grounded in the actual data returned. Be specific with prices, percentages, and timeframes."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}\n")

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
    run_agent("Fetch the last 10 days of HYPE 4h candle data. What was the recent peak price and when did it occur? How far is the current price from that peak?")