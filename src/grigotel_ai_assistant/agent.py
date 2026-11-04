"""
GrigoTel AI Assistant - Step 4: The agent

A function-calling agent on the OpenAI Responses API:
  1. Send the conversation + tool definitions to the LLM.
  2. If the LLM asks for tool calls, run them and send the results back.
  3. Repeat until the LLM answers with text.

The agent keeps the conversation history, so one GrigoTelAgent = one chat session.

Order safety guard (enforced in code, not just in the prompt):
  place_order only runs for an item the agent checked with check_availability
  in an EARLIER customer turn. So the agent must present the offer, and the
  customer must reply (confirm) before any stock is reduced.

Location: src/grigotel_ai_assistant/agent.py

Usage (in the app):
    from grigotel_ai_assistant.agent import GrigoTelAgent
    from grigotel_ai_assistant.tools import create_tools

    agent = GrigoTelAgent(create_tools())
    reply = agent.chat("How much is the G26 Pro?")
    agent.last_order   # dict with checkout_url if this turn placed an order, else None

Requires OPENAI_API_KEY in the project's .env file.
"""

import inspect
import json

from dotenv import load_dotenv
from openai import OpenAI

try:  # works when the package is installed (uv init --package)
    from grigotel_ai_assistant.tools import GrigoTelTools
except ImportError:  # works when run as a script from this folder
    from tools import GrigoTelTools

load_dotenv()

MODEL = "gpt-4o-mini"
MAX_TOOL_ROUNDS = 8  # safety limit on LLM <-> tool round trips per customer message

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

    def chat(self, user_message: str) -> str:
        """Send one customer message and return the assistant's reply."""
        self.user_turn += 1
        self.last_order = None
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=self.messages,
                tools=self.definitions,
            )
            self.messages.extend(response.output)

            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return response.output_text

            for call in calls:
                result = self._run_tool(call.name, call.arguments)
                self.messages.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                })

        return "Sorry, I couldn't complete that request. Could you rephrase it?"

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
