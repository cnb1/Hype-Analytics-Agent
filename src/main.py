import os
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from tool_functions import TOOL_FUNCTIONS

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

load_dotenv(RESOURCES / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_LOOKBACK_DAYS = 5
DEFAULT_INTERVAL = "4h"


def load_tools() -> list:
    with open(RESOURCES / "tools.json") as f:
        return json.load(f)


def run_agent(user_message: str,
              lookback_days: int = DEFAULT_LOOKBACK_DAYS,
              interval: str = DEFAULT_INTERVAL):
    """Run the agent with a user message, handling tool calls in a loop."""

    tools = load_tools()

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

        if choice.finish_reason == "tool_calls":
            assistant_msg = choice.message
            messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"[Tool Call] {fn_name}({fn_args})")

                fn = TOOL_FUNCTIONS[fn_name]
                result = fn(**fn_args)

                print(f"[Tool Result] {len(json.loads(result).get('candles', []))} candles returned\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        elif choice.finish_reason == "stop":
            final_response = choice.message.content
            print(f"Agent:\n{final_response}\n")
            return final_response

        else:
            print(f"Unexpected finish reason: {choice.finish_reason}")
            break


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
