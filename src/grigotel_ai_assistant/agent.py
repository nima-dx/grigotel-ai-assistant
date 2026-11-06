"""
GrigoTel AI Assistant - Step 4: The agent

A function-calling agent on the OpenAI Responses API:
  1. Send the conversation + tool definitions to the LLM (streamed).
  2. If the LLM asks for tool calls, run them and send the results back.
  3. Repeat until the LLM answers with a structured AssistantResponse.

The agent keeps the conversation history, so one GrigoTelAgent = one chat session.

Order safety guard (enforced in code, not just in the prompt):
  place_order only runs for an item the agent checked with check_availability
  in an EARLIER customer turn. So the agent must present the offer, and the
  customer must reply (confirm) before any stock is reduced.

Location: src/grigotel_ai_assistant/agent.py

Usage:
    agent = GrigoTelAgent(create_tools())

    # streaming (used by the Streamlit app)
    for event in agent.chat_stream("How much is the G26 Pro?"):
        ...  # tool_call / tool_result / answer_delta / final

    # simple (used by tests)
    reply = agent.chat("How much is the G26 Pro?")
    agent.last_order   # dict with checkout_url if this turn placed an order, else None

Requires OPENAI_API_KEY in the project's .env file.
"""

import inspect
import json
from collections.abc import Iterator
from typing import Literal

from dotenv import load_dotenv
from jiter import from_json
from openai import OpenAI
from pydantic import BaseModel, Field

try:  # works when the package is installed (uv init --package)
    from grigotel_ai_assistant.tools import GrigoTelTools
except ImportError:  # works when run as a script from this folder
    from tools import GrigoTelTools

load_dotenv()

MODEL = "gpt-4o-mini"
MAX_TOOL_ROUNDS = 8  # safety limit on LLM <-> tool round trips per customer message

AnswerType = Literal[
    "product_specs",
    "pricing_availability",
    "order",
    "how_to",
    "troubleshooting",
    "company",
    "privacy",
    "other",
]


class AssistantResponse(BaseModel):
    """The agent's final reply. `answer` is first, so it streams first."""

    answer: str = Field(description="The reply to the customer, in markdown.")
    answer_type: AnswerType = Field(description="The kind of question that was answered.")
    confidence: float = Field(description="0.0-1.0: how well the tool results support the answer.")
    found_in_sources: bool = Field(description="True only if the answer is backed by tool results.")
    sources: list[str] = Field(description="PDF file names used, and/or 'GrigoTel database'.")
    follow_up_questions: list[str] = Field(description="2-3 questions the customer might ask next.")


SYSTEM_PROMPT = """
You are the GrigoTel AI Assistant, a friendly sales and support assistant for GrigoTel.
GrigoTel sells four smartphones: Alexy G26, Alexy G26 Fold, Alexy G26 Pro and Alexy G26 Lite.

## Which tool to use
- Technical specifications, features, setup, maintenance, battery, troubleshooting,
  company information, privacy and compliance -> search_docs.
  Pass `model` when the question is about a specific phone.
- Prices, colors and availability -> get_product_info.
  Never state a price or stock level from memory or from the documents.
- Buying a phone -> follow the order process below.

## Answering rules
- Answer only from tool results. If the information is not there, say so. Never guess.
- If a specification sheet and the user manual disagree about a model's specs,
  the specification sheet is correct. Use the user manual for procedures.
- For technical answers, mention the source document
  (e.g. "according to the Alexy G26 Pro specification sheet").
- Do not show item_ids to the customer.
- Only help with GrigoTel products and services; politely decline anything else.
- Reply in the customer's language. Be concise.

## Order process
1. You need the model, color and quantity. Ask for anything missing.
   If the customer doesn't know the colors, use get_product_info to list them.
2. Call check_availability and act on its status:
   - "available": summarize model, color, quantity, unit price and total price,
     then ask the customer to confirm the order.
   - "partial": say that only `available_quantity` are in stock, offer that quantity
     (with its total price), mention any alternative colors that cover the full
     quantity, and ask how they want to proceed.
   - "out_of_stock": say this color is sold out and offer the in-stock alternative colors.
   - "color_not_offered": list the colors this model comes in.
3. Only after the customer explicitly confirms the offer you presented, call place_order
   with the item_id from check_availability and the confirmed quantity.
   Do not call check_availability again before placing the order; place_order re-checks stock.
   If the customer changes the model, color or quantity, call check_availability again
   and ask for confirmation again.
4. When place_order returns "confirmed": thank the customer, show the total price and give
   the checkout_url exactly as returned. Opening the link completes the purchase.
5. If place_order returns "insufficient_stock", explain and offer the available quantity.

## Response format
Your final reply is a structured response:
- answer: your reply to the customer, in markdown.
- answer_type: product_specs (specifications, features), pricing_availability (prices,
  colors, stock), order (anything in the order process), how_to (setup, usage, maintenance),
  troubleshooting, company, privacy, or other.
- confidence: 0.0 to 1.0, how well the tool results support your answer.
- found_in_sources: true only if your answer is backed by tool results.
- sources: the PDF file names from search_docs results you used, and "GrigoTel database"
  if you used prices, colors or stock. Empty if you used no tool.
- follow_up_questions: 2-3 short questions the customer might ask next, written the way
  the customer would ask them, in the customer's language. When you are waiting for the
  customer to confirm an order, include "Yes, place the order".
""".strip()


# ── Tool definitions for the LLM ─────────────────────────────────────────

def _function_tool(method, properties: dict, required: list[str]) -> dict:
    """OpenAI function-tool definition; the description is the method's docstring."""
    return {
        "type": "function",
        "name": method.__name__,
        "description": inspect.getdoc(method),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def tool_definitions(tools: GrigoTelTools) -> list[dict]:
    model = {"type": "string", "description": "Phone model, e.g. 'Alexy G26 Pro'."}
    quantity = {"type": "integer", "minimum": 1}
    return [
        _function_tool(tools.search_docs, {"query": {"type": "string"}, "model": model}, ["query"]),
        _function_tool(tools.get_product_info, {"model": model}, []),
        _function_tool(
            tools.check_availability,
            {"model": model, "color": {"type": "string"}, "quantity": quantity},
            ["model", "color", "quantity"],
        ),
        _function_tool(
            tools.place_order,
            {"item_id": {"type": "string"}, "quantity": quantity},
            ["item_id", "quantity"],
        ),
    ]


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_args(arguments: str) -> dict:
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}


def _partial_answer(buffer: str) -> str | None:
    """Read the `answer` field from incomplete JSON, e.g. '{"answer": "The G26 Pro co'."""
    try:
        data = from_json(buffer.encode(), partial_mode="trailing-strings")
    except ValueError:  # e.g. the buffer ends in the middle of an escape sequence
        return None
    return data.get("answer", "") if isinstance(data, dict) else ""


def _fallback_response(answer: str) -> AssistantResponse:
    return AssistantResponse(
        answer=answer,
        answer_type="other",
        confidence=0.0,
        found_in_sources=False,
        sources=[],
        follow_up_questions=[],
    )


# ── Agent ────────────────────────────────────────────────────────────────

class GrigoTelAgent:
    def __init__(self, tools: GrigoTelTools, client: OpenAI | None = None, model: str = MODEL):
        self.tools = tools
        self.client = client or OpenAI()
        self.model = model
        self.definitions = tool_definitions(tools)
        self.reset()

    def reset(self) -> None:
        """Start a new conversation."""
        self.messages: list = []
        self.user_turn = 0
        self.offers: dict[str, int] = {}  # item_id -> customer turn in which it was offered
        self.last_order: dict | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send one customer message and return the assistant's answer text."""
        final = None
        for event in self.chat_stream(user_message):
            if event["type"] == "final":
                final = event
        return final["response"].answer

    def chat_stream(self, user_message: str) -> Iterator[dict]:
        """
        Send one customer message and yield events as they happen:
          {"type": "tool_call",    "name": str, "arguments": dict}
          {"type": "tool_result",  "name": str, "result": dict}
          {"type": "answer_delta", "text": str}
          {"type": "final",        "response": AssistantResponse, "order": dict | None}

        If the OpenAI call fails, the conversation is rolled back to before this
        message and the exception is raised, so the history stays consistent.
        """
        self.user_turn += 1
        self.last_order = None
        history_length, offers_before = len(self.messages), dict(self.offers)
        self.messages.append({"role": "user", "content": user_message})

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = yield from self._stream_round()

                calls = [item for item in response.output if item.type == "function_call"]
                if not calls:
                    parsed = response.output_parsed or _fallback_response(
                        response.output_text or "Sorry, I couldn't answer that. Could you rephrase it?"
                    )
                    yield {"type": "final", "response": parsed, "order": self.last_order}
                    return

                for call in calls:
                    yield {"type": "tool_call", "name": call.name, "arguments": _parse_args(call.arguments)}
                    result = self._run_tool(call.name, call.arguments)
                    self.messages.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result),
                    })
                    yield {"type": "tool_result", "name": call.name, "result": result}

            yield {
                "type": "final",
                "response": _fallback_response("Sorry, I couldn't complete that request. Could you rephrase it?"),
                "order": self.last_order,
            }

        except Exception:
            if self.last_order:
                # The order IS placed in the database; keep it and still hand over the link
                answer = "Your order is placed. Use the checkout button to complete your purchase."
                self.messages.append({"role": "assistant", "content": answer})
                yield {"type": "final", "response": _fallback_response(answer), "order": self.last_order}
                return
            del self.messages[history_length:]
            self.offers = offers_before
            raise

    # ── One streamed LLM call ────────────────────────────────────────────

    def _stream_round(self):
        """Stream one LLM call, yield answer_delta events, return the final response."""
        buffer, sent = "", 0

        with self.client.responses.stream(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=self.messages,
            tools=self.definitions,
            text_format=AssistantResponse,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    buffer += event.delta
                    answer = _partial_answer(buffer)
                    if answer is not None and len(answer) > sent:
                        yield {"type": "answer_delta", "text": answer[sent:]}
                        sent = len(answer)
            response = stream.get_final_response()

        # Store the output as plain dicts (the parsed SDK objects can't be sent back as input)
        if response.output_text:
            self.messages.append({"role": "assistant", "content": response.output_text})
        for item in response.output:
            if item.type == "function_call":
                self.messages.append({
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                })
        return response

    # ── Tool execution ───────────────────────────────────────────────────

    def _run_tool(self, name: str, arguments: str) -> dict:
        try:
            args = json.loads(arguments or "{}")
            if name == "search_docs":
                return self.tools.search_docs(**args)
            if name == "get_product_info":
                return self.tools.get_product_info(**args)
            if name == "check_availability":
                result = self.tools.check_availability(**args)
                if result.get("status") in ("available", "partial"):
                    # Keep the FIRST turn it was offered, so a re-check doesn't reset it
                    self.offers.setdefault(result["item_id"], self.user_turn)
                return result
            if name == "place_order":
                return self._guarded_order(**args)
            return {"error": f"Unknown tool '{name}'."}
        except Exception as e:  # report errors to the LLM instead of crashing the chat
            return {"error": f"{type(e).__name__}: {e}"}

    def _guarded_order(self, item_id: str, quantity: int) -> dict:
        """Only place an order the customer could have confirmed in a later message."""
        offered_turn = self.offers.get(item_id)
        if offered_turn is None:
            return {"status": "not_allowed", "error": "Call check_availability for this item first."}
        if offered_turn >= self.user_turn:
            return {
                "status": "awaiting_confirmation",
                "error": "Present the offer and wait for the customer to confirm before placing the order.",
            }

        result = self.tools.place_order(item_id, quantity)
        if result.get("status") == "confirmed":
            self.last_order = result
            self.offers.pop(item_id, None)
        return result
