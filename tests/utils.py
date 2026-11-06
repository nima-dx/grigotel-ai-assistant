"""
Test helpers for the GrigoTel agent.

Location: tests/utils.py
"""

import json


def collect_tools(messages: list) -> list[dict]:
    """
    Extract the tool calls from an agent's message history, in the order they were made.

    Works on GrigoTelAgent.messages, where each tool call is stored as
    {"type": "function_call", "name": ..., "arguments": "<json>", "call_id": ...}.

    Returns a list of {"name": str, "arguments": dict}.
    """
    calls = []
    for message in messages:
        if isinstance(message, dict) and message.get("type") == "function_call":
            calls.append({
                "name": message["name"],
                "arguments": json.loads(message.get("arguments") or "{}"),
            })
    return calls
