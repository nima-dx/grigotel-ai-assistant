"""
Integration tests for the GrigoTel agent (they call the real OpenAI API).

Each test sends a question to a fresh agent with run_agent_test(agent, prompt) and checks:
  - stock count:     the number comes from the database and appears in the answer
  - Fold display:    the spec sheet is retrieved, cited, and its display size is in the answer
  - price question:  the first tool called is get_product_info
  - company question: the first tool called is search_docs
  - order request:   no order is placed without confirmation, and an LLM judge
                     checks the answer against natural-language criteria

Usage (from the project root; stop the Streamlit app first, it locks the DuckDB file):
    uv run pytest tests/test_judge.py -v -s

Location: tests/test_judge.py
"""

import os
import re

import pytest

from grigotel_ai_assistant.agent import GrigoTelAgent
from grigotel_ai_assistant.convert_docs import MARKDOWN_DIR
from grigotel_ai_assistant.tools import create_tools
from tests.judge import assert_criteria
from tests.utils import (
    reports_quantity,
    retrieved_sources,
    run_agent_test,
    spec_value,
    tool_quantity,
)

pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")

DATABASE_TOOLS = {"get_product_info", "check_availability"}


@pytest.fixture(scope="module")
def tools():
    """One DB connection + index for all tests (DuckDB allows one connection config per file)."""
    return create_tools()


@pytest.fixture
def agent(tools):
    """A fresh agent (empty conversation) per test."""
    return GrigoTelAgent(tools)


@pytest.mark.asyncio
async def test_stock_count_comes_from_database(agent, tools):
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

    result = await run_agent_test(agent, "how many Alexy G26 Black is available in the inventory?")
    answer = result.output.answer

    # The agent looked it up in the database ...
    called = {tc.name for tc in result.tool_calls}
    assert called & DATABASE_TOOLS, f"No database tool called; tools used: {called or 'none'}"
    assert "GrigoTel database" in result.output.sources

    # ... received the correct number from the tool (exact, independent of wording) ...
    assert tool_quantity(result.tool_results, "Alexy G26", "black") == expected

    # ... and communicates it to the customer ("0" may be phrased as "out of stock")
    assert isinstance(answer, str) and answer.strip()
    assert reports_quantity(answer, expected), (
        f"Expected the answer to report {expected} "
        f"{'(or out of stock) ' if expected == 0 else ''}, got: {answer}"
    )


@pytest.mark.asyncio
async def test_fold_display_spec_comes_from_markdown(agent):
    # Ground truth, read directly from the Fold spec sheet's markdown
    fold_md = next(MARKDOWN_DIR.glob("*G26_Fold*.md"))
    display = spec_value(fold_md, "Display")
    sizes = re.findall(r"\d+(?:\.\d+)?", display)  # e.g. ['7.6'] from '7.6" Foldable AMOLED'
    assert sizes, f"No numbers found in the Fold display spec: {display!r}"

    result = await run_agent_test(agent, "What is the display specification of Alexy G26 Fold?")

    # The agent searched the documents ...
    assert any(tc.name == "search_docs" for tc in result.tool_calls), "search_docs was not called"

    # ... the Fold spec sheet was among the retrieved chunks ...
    fold_source = f"{fold_md.stem}.pdf"
    retrieved = retrieved_sources(result.tool_results)
    assert fold_source in retrieved, f"Fold spec sheet not retrieved; got {retrieved}"

    # ... the answer cites it ...
    assert fold_source in result.output.sources, f"Fold spec sheet not in sources: {result.output.sources}"
    assert result.output.found_in_sources

    # ... and repeats the display size from the markdown
    assert sizes[0] in result.output.answer, (
        f"Expected display size {sizes[0]} (from {display!r}) in the answer, got: {result.output.answer}"
    )


@pytest.mark.asyncio
async def test_price_question_uses_product_info_first(agent):
    result = await run_agent_test(agent, "what is the price of black Alexy G26 Fold?")

    assert result.tool_calls, "The agent made no tool calls"
    first = result.tool_calls[0].name
    assert first == "get_product_info", f"First tool was {first}; all calls: {result.tool_calls}"


@pytest.mark.asyncio
async def test_company_question_uses_search_docs_first(agent):
    result = await run_agent_test(agent, "What are the core values of the GrigoTel company?")

    assert result.tool_calls, "The agent made no tool calls"
    first = result.tool_calls[0].name
    assert first == "search_docs", f"First tool was {first}; all calls: {result.tool_calls}"


@pytest.mark.asyncio
async def test_order_availability_judged_by_llm(agent):
    question = "I would like to order 12 Blue Alexy G26 Pro? Is there availability"
    result = await run_agent_test(agent, question)

    # Deterministic safety check: nothing may be ordered without the customer's confirmation
    assert result.order is None, "An order was placed without confirmation"

    await assert_criteria(result, [
        "The assistant checked stock for the Alexy G26 Pro in Blue using a tool before answering",
        "The answer clearly says whether 12 units are available; if fewer are in stock, "
        "it states the exact available quantity from the tool result",
        "If fewer than 12 are available, the answer offers the available quantity and/or "
        "alternative colors from the tool result, and asks how the customer wants to proceed",
        "Any price or total in the answer matches the tool results",
        "The assistant does not place the order or give a checkout link; it asks the customer to confirm first",
    ], question)
