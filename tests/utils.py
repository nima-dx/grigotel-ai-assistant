"""
Test helpers for the GrigoTel agent.

  - collect_tools(messages)   the tool calls in the agent's message history, as ToolCall(name, args)
  - get_model_name(agent)     "provider:model" of the agent or the judge, e.g. "openai:gpt-4o-mini"
  - run_agent_test(agent, q)  sends one message to the agent and returns an AgentTestResult
                              (answer, tool calls, tool results, order, token usage)
  - capture_usage(model, ...) prints the tokens and cost of a run and adds them to a running total

Plus helpers that check the answers against the database and the documents.

Location: tests/utils.py
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict

from grigotel_ai_assistant.agent import MODEL, AssistantResponse, GrigoTelAgent
from grigotel_ai_assistant.build_index import clean_text
from grigotel_ai_assistant.session_cost import calculate_cost


# ── Tool calls ───────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


def collect_tools(messages: list) -> list[ToolCall]:
    """
    The tool calls in an agent's message history, in the order they were made.

    GrigoTelAgent stores each call as
    {"type": "function_call", "name": ..., "arguments": "<json>", "call_id": ...}.
    """
    tool_calls = []
    for m in messages:
        if isinstance(m, dict) and m.get("type") == "function_call":
            tool_calls.append(ToolCall(m["name"], json.loads(m.get("arguments") or "{}")))
    return tool_calls


# ── Running the agent ────────────────────────────────────────────────────

@dataclass
class AgentTestResult:
    """Everything one customer message produced (like PydanticAI's AgentRunResult)."""
    output: AssistantResponse
    tool_calls: list[ToolCall]
    tool_results: list[dict]  # [{"name": ..., "result": {...}}, ...] in call order
    order: dict | None
    usage: dict  # {"input_tokens": ..., "output_tokens": ...}
    messages: list = field(default_factory=list)  # the history entries added by this message

    def new_messages(self) -> list:
        return self.messages

    def tool_trace(self) -> list[dict]:
        """Each tool call with its result: the judge's ground truth."""
        return [
            {"name": call.name, "arguments": call.args, "result": result["result"]}
            for call, result in zip(self.tool_calls, self.tool_results)
        ]


def get_model_name(agent) -> str:
    """'provider:model' for a GrigoTelAgent or a PydanticAI agent (the judge)."""
    if isinstance(agent, GrigoTelAgent):
        return f"openai:{MODEL}"
    return f"{agent.model.system}:{agent.model.model_name}"


def _run(agent: GrigoTelAgent, user_prompt: str) -> AgentTestResult:
    history_length = len(agent.messages)
    events = list(agent.chat_stream(user_prompt))
    final = next(e for e in events if e["type"] == "final")
    new_messages = agent.messages[history_length:]
    return AgentTestResult(
        output=final["response"],
        tool_calls=collect_tools(new_messages),
        tool_results=[e for e in events if e["type"] == "tool_result"],
        order=final.get("order"),
        usage=dict(agent.last_usage),
        messages=new_messages,
    )


async def run_agent_test(agent: GrigoTelAgent, user_prompt: str) -> AgentTestResult:
    """
    Send one message to the agent (keeping its conversation) and return the result.

    GrigoTelAgent is synchronous, so it runs in a worker thread; this lets the async
    tests await it without blocking the event loop.
    """
    result = await asyncio.to_thread(_run, agent, user_prompt)
    capture_usage(get_model_name(agent), result.usage["input_tokens"], result.usage["output_tokens"])
    return result


# ── Cost tracking ────────────────────────────────────────────────────────

USAGE_TOTALS: dict[str, dict] = {}  # model -> {"input_tokens", "output_tokens", "cost"}


def capture_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Add one run's tokens (and the cost, if we know the model's price) to the totals."""
    model_name = model.split(":", 1)[-1]
    try:
        cost = calculate_cost(model_name, input_tokens, output_tokens)
    except KeyError:  # no price in session_cost.MODEL_PRICES
        cost = 0.0

    totals = USAGE_TOTALS.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0})
    totals["input_tokens"] += input_tokens
    totals["output_tokens"] += output_tokens
    totals["cost"] += cost
    print(f"[usage] {model}: {input_tokens} in / {output_tokens} out, ${cost:.5f}")


# ── Checking answers ─────────────────────────────────────────────────────

OUT_OF_STOCK_PHRASES = ("out of stock", "sold out", "not available", "unavailable", "no units")


def contains_number(text: str, number: int) -> bool:
    """True if `number` appears as a whole number in text (so 4 doesn't match 14)."""
    return re.search(rf"(?<![\d.,]){number}(?![\d.,]*\d)", text) is not None


def reports_quantity(text: str, quantity: int) -> bool:
    """True if the answer communicates the quantity: the number itself, or 'out of stock' for 0."""
    if quantity == 0:
        lowered = text.lower()
        return contains_number(text, 0) or any(phrase in lowered for phrase in OUT_OF_STOCK_PHRASES)
    return contains_number(text, quantity)


def tool_quantity(tool_results: list[dict], model: str, color: str) -> int | None:
    """The stock quantity the agent received from a database tool for one model + color."""
    for event in tool_results:
        result = event["result"]
        if event["name"] == "check_availability" and result.get("model") == model \
                and str(result.get("color", "")).lower() == color:
            return result["available_quantity"]
        if event["name"] == "get_product_info":
            for product in result.get("products", []):
                if product["model"] != model:
                    continue
                for variant in product["variants"]:
                    if variant["color"].lower() == color:
                        return variant["quantity"]
    return None


def retrieved_sources(tool_results: list[dict]) -> set[str]:
    """The documents search_docs returned chunks from."""
    return {
        r["source"]
        for e in tool_results if e["name"] == "search_docs"
        for r in e["result"].get("results", [])
    }


def spec_value(markdown_path, feature: str) -> str:
    """Value of a feature in a spec sheet's flattened table: the line after 'Display'."""
    lines = [line for line in clean_text(markdown_path.read_text(encoding="utf-8")).splitlines() if line]
    index = lines.index(feature)
    return lines[index + 1]
