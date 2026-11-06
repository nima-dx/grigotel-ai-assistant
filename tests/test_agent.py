"""
Integration tests for the GrigoTel agent (they call the real OpenAI API).

Location: tests/test_agent.py

Usage (from the project root):
    uv add --dev pytest
    uv run pytest tests/test_agent.py -v
"""

import os
import re

import pytest

from grigotel_ai_assistant.agent import GrigoTelAgent
from grigotel_ai_assistant.build_index import clean_text
from grigotel_ai_assistant.convert_docs import MARKDOWN_DIR
from grigotel_ai_assistant.tools import create_tools
from judge import assert_criteria, evaluate_agent_performance
from utils import collect_tools

pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")

DATABASE_TOOLS = {"get_product_info", "check_availability"}


# ── Fixtures & helpers ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tools():
    """One DB connection + index for all tests (DuckDB allows one connection config per file)."""
    return create_tools()


@pytest.fixture
def agent(tools):
    """A fresh agent (empty conversation) per test."""
    return GrigoTelAgent(tools)


def ask(agent: GrigoTelAgent, question: str):
    """Run one question; return the final response and the tool events."""
    events = list(agent.chat_stream(question))
    final = next(e for e in events if e["type"] == "final")
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    tool_results = [e for e in events if e["type"] == "tool_result"]
    return final["response"], tool_calls, tool_results


def contains_number(text: str, number: int) -> bool:
    """True if `number` appears as a whole number in text (so 4 doesn't match 14)."""
    return re.search(rf"(?<![\d.,]){number}(?![\d.,]*\d)", text) is not None


OUT_OF_STOCK_PHRASES = ("out of stock", "sold out", "not available", "unavailable", "no units")


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


def spec_value(markdown_path, feature: str) -> str:
    """Value of a feature in a spec sheet's flattened table: the line after 'Display'."""
    lines = [line for line in clean_text(markdown_path.read_text(encoding="utf-8")).splitlines() if line]
    index = lines.index(feature)
    return lines[index + 1]


# ── Tests ────────────────────────────────────────────────────────────────

def test_stock_count_comes_from_database(agent, tools):
    # Ground truth, read directly from DuckDB (stock changes when orders are placed)
    row = tools.con.cursor().execute(
        """
        SELECT i.quantity
        FROM Stock s
        JOIN Inventory_Stock i USING (item_id)
        WHERE s.model = 'Alexy G26' AND lower(s.color) = 'black'
        """
    ).fetchone()
    assert row is not None, "No 'Alexy G26' Black variant in the database"
    expected = row[0]

    response, tool_calls, tool_results = ask(agent, "how many Alexy G26 Black is available in the inventory?")

    # The agent looked it up in the database ...
    called = {c["name"] for c in tool_calls}
    assert called & DATABASE_TOOLS, f"No database tool called; tools used: {called or 'none'}"
    assert "GrigoTel database" in response.sources

    # ... received the correct number from the tool (exact, independent of wording) ...
    assert tool_quantity(tool_results, "Alexy G26", "black") == expected

    # ... and communicates it to the customer ("0" may be phrased as "out of stock")
    assert isinstance(response.answer, str) and response.answer.strip()
    assert reports_quantity(response.answer, expected), (
        f"Expected the answer to report {expected} "
        f"{'(or out of stock) ' if expected == 0 else ''}, got: {response.answer}"
    )


def test_fold_display_spec_comes_from_markdown(agent):
    # Ground truth, read directly from the Fold spec sheet's markdown
    fold_md = next(MARKDOWN_DIR.glob("*G26_Fold*.md"))
    display = spec_value(fold_md, "Display")
    sizes = re.findall(r"\d+(?:\.\d+)?", display)  # e.g. ['7.6'] from '7.6" Foldable AMOLED'
    assert sizes, f"No numbers found in the Fold display spec: {display!r}"

    response, tool_calls, tool_results = ask(agent, "What is the display specification of Alexy G26 Fold?")

    # The agent searched the documents ...
    assert any(c["name"] == "search_docs" for c in tool_calls), "search_docs was not called"

    # ... the Fold spec sheet was among the retrieved chunks ...
    retrieved = {
        r["source"]
        for e in tool_results if e["name"] == "search_docs"
        for r in e["result"].get("results", [])
    }
    fold_source = f"{fold_md.stem}.pdf"
    assert fold_source in retrieved, f"Fold spec sheet not retrieved; got {retrieved}"

    # ... the answer cites it ...
    assert fold_source in response.sources, f"Fold spec sheet not in sources: {response.sources}"
    assert response.found_in_sources

    # ... and repeats the display size from the markdown
    assert sizes[0] in response.answer, (
        f"Expected display size {sizes[0]} (from {display!r}) in the answer, got: {response.answer}"
    )


def test_price_question_uses_product_info_first(agent):
    agent.chat("what is the price of black Alexy G26 Fold?")

    calls = collect_tools(agent.messages)
    assert calls, "The agent made no tool calls"
    assert calls[0]["name"] == "get_product_info", f"First tool was {calls[0]['name']}; all calls: {calls}"


def test_company_question_uses_search_docs_first(agent):
    agent.chat("What are the core values of the GrigoTel company?")

    calls = collect_tools(agent.messages)
    assert calls, "The agent made no tool calls"
    assert calls[0]["name"] == "search_docs", f"First tool was {calls[0]['name']}; all calls: {calls}"


def test_order_availability_judged_by_llm(agent):
    question = "I would like to order 12 Blue Alexy G26 Pro? Is there availability"
    response, tool_calls, tool_results = ask(agent, question)

    # Deterministic safety check: nothing may be ordered without the customer's confirmation
    assert agent.last_order is None, "An order was placed without confirmation"

    # Tool calls paired with their results: the judge's ground truth
    tool_trace = [
        {"name": call["name"], "arguments": call["arguments"], "result": result["result"]}
        for call, result in zip(tool_calls, tool_results)
    ]

    criteria = [
        "The assistant checked stock for the Alexy G26 Pro in Blue using a tool before answering",
        "The answer clearly says whether 12 units are available; if fewer are in stock, "
        "it states the exact available quantity from the tool result",
        "If fewer than 12 are available, the answer offers the available quantity and/or "
        "alternative colors from the tool result, and asks how the customer wants to proceed",
        "Any price or total in the answer matches the tool results",
        "The assistant does not place the order or give a checkout link; it asks the customer to confirm first",
    ]

    result = evaluate_agent_performance(criteria, question, response.answer, tool_trace)
    print(f"\nAnswer:\n{response.answer}\n\nJudge: {result.summary}")
    for check in result.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.criterion}\n         {check.reasoning}")

    assert_criteria(result, criteria)
