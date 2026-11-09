"""
Fetch a grouped session trace from Logfire, convert it to an AgentRunResult
and calculate its approximate cost.

Usage (from the project root):
    uv run python src/grigotel_ai_assistant/session_cost.py <trace_id>

Requires LOGFIRE_READ_TOKEN in .env.
"""

import os
import sys

from dotenv import load_dotenv
from logfire.query_client import LogfireQueryClient
from pydantic_ai.usage import RunUsage

try:  # works when the package is installed (uv init --package)
    from grigotel_ai_assistant.trace_replay import fetch_trace, trace_to_run_result
except ImportError:  # works when run as a script from this folder
    from trace_replay import fetch_trace, trace_to_run_result

load_dotenv()

MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # USD per 1M tokens
}


def calculate_cost(model_name, input_tokens, output_tokens):
    prices = MODEL_PRICES[model_name.lower()]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def main(trace_id: str, model_name: str = "gpt-4o-mini") -> None:
    with LogfireQueryClient(read_token=os.environ["LOGFIRE_READ_TOKEN"]) as client:
        trace = fetch_trace(trace_id, client)

    result = trace_to_run_result(trace)
    # `usage` is a method in older PydanticAI versions and a property in newer ones
    usage = result.usage if isinstance(result.usage, RunUsage) else result.usage()

    print(f"Messages in session: {len(result.all_messages())}")
    print(f"Input tokens:        {usage.input_tokens:,}")
    print(f"Output tokens:       {usage.output_tokens:,}")
    cost = calculate_cost(model_name, usage.input_tokens, usage.output_tokens)
    print(f"Approximate cost:    ${cost:.6f}  ({model_name})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python session_cost.py <trace_id>")
    main(sys.argv[1])
