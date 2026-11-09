"""
GrigoTel AI Assistant - Step 5: Streamlit chat UI

Location: src/grigotel_ai_assistant/app.py

Usage (from the project root):
    uv add streamlit
    uv run streamlit run src/grigotel_ai_assistant/app.py
"""

import os

import streamlit as st

st.set_page_config(page_title="GrigoTel AI Assistant", page_icon="📱")

try:  # works when the package is installed (uv init --package)
    from grigotel_ai_assistant.agent import GrigoTelAgent
    from grigotel_ai_assistant.observability import setup_logfire
    from grigotel_ai_assistant.tools import create_tools
except ImportError:  # works when run from this folder
    from agent import GrigoTelAgent
    from observability import setup_logfire
    from tools import create_tools

CURSOR = "▌"

WELCOME = (
    "Hi! I'm the **GrigoTel AI Assistant**. Ask me about the Alexy G26, G26 Fold, "
    "G26 Pro and G26 Lite: specifications, prices, colors and availability. "
    "I can also place an order for you."
)

EXAMPLE_QUESTIONS = [
    "What colors does the G26 Pro come in?",
    "Is the G26 Lite waterproof?",
    "I want to buy 2 Alexy G26 Fold phones",
]

ANSWER_TYPE_LABELS = {
    "product_specs": "Product specs",
    "pricing_availability": "Pricing & availability",
    "order": "Order",
    "how_to": "How-to",
    "troubleshooting": "Troubleshooting",
    "company": "Company",
    "privacy": "Privacy",
    "other": "Other",
}


# ── Setup ────────────────────────────────────────────────────────────────

if not os.getenv("OPENAI_API_KEY"):  # agent.py has already loaded .env
    st.error("OPENAI_API_KEY is not set. Add it to the project's `.env` file and restart the app.")
    st.stop()


@st.cache_resource
def init_observability() -> bool:
    """Logfire tracing: configured once per process, not on every Streamlit rerun."""
    setup_logfire()
    return True


init_observability()


@st.cache_resource(show_spinner="Loading GrigoTel knowledge base...")
def get_tools():
    """DuckDB connection + search index: loaded once, shared by all sessions."""
    return create_tools()


tools = get_tools()

# One agent per browser session: it holds this customer's conversation and order state
if "agent" not in st.session_state:
    st.session_state.agent = GrigoTelAgent(tools)
    st.session_state.messages = []

agent: GrigoTelAgent = st.session_state.agent


# ── Activity panel text ──────────────────────────────────────────────────

def describe_event(event: dict) -> str | None:
    """One line for the activity panel. Never shows item_ids."""
    name = event["name"]

    if event["type"] == "tool_call":
        args = event["arguments"]
        if name == "search_docs":
            line = f"🔎 Searching documents: *{args.get('query', '')}*"
            return line + (f" (model: {args['model']})" if args.get("model") else "")
        if name == "get_product_info":
            return f"📦 Checking catalog: {args.get('model') or 'all models'}"
        if name == "check_availability":
            return f"📊 Checking stock: {args.get('quantity')} × {args.get('model')} {args.get('color')}"
        return None  # place_order is described by its result

    result = event["result"]
    status = result.get("status")

    if name == "place_order":
        if status == "confirmed":
            return f"🛒 Order placed: {result['quantity']} × {result['model']} {result['color']}"
        if status in ("awaiting_confirmation", "not_allowed"):
            return "⏳ Waiting for customer confirmation"
        if status == "insufficient_stock":
            return f"⚠️ Not enough stock: {result['available_quantity']} available"
        return "⚠️ Order could not be placed"

    if "error" in result:
        return f"⚠️ {name} could not complete the request"

    if name == "search_docs":
        sources = sorted({r["source"] for r in result.get("results", [])})
        return f"📄 Sources: {', '.join(sources)}" if sources else "📄 No matching documents"
    if name == "check_availability":
        item = f"{result.get('model')} {result.get('color', '')}".strip()
        if status == "available":
            return f"✅ Available: {item} ({result['available_quantity']} in stock)"
        if status == "partial":
            return f"⚠️ Partial: only {result['available_quantity']} × {item} available"
        if status == "out_of_stock":
            return f"❌ Out of stock: {item}"
        if status == "color_not_offered":
            return f"🎨 {result['model']} doesn't come in {result['requested_color']}"
    return None


# ── Rendering ────────────────────────────────────────────────────────────

def render_activity(lines: list[str]) -> None:
    with st.status("Done", state="complete", expanded=True):
        for line in lines:
            st.markdown(line)


def render_order(order: dict) -> None:
    with st.container(border=True):
        st.markdown(
            f"**✅ Order confirmed:** {order['quantity']} × {order['model']} ({order['color']})  \n"
            f"**Total:** {order['total_price']:,.2f} {order['currency']}"
        )
        st.link_button("Proceed to checkout", order["checkout_url"], type="primary")


def render_metadata(meta: dict) -> None:
    label = ANSWER_TYPE_LABELS.get(meta["answer_type"], meta["answer_type"])
    grounded = "Yes" if meta["found_in_sources"] else "No"
    line = f"**Type:** {label} · **Confidence:** {meta['confidence']:.0%} · **Grounded:** {grounded}"
    if meta["sources"]:
        line += f"  \n**Sources:** {', '.join(meta['sources'])}"
    st.caption(line)


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_activity(msg["activity"])
        st.markdown(msg["content"])
        if msg.get("order"):
            render_order(msg["order"])
        if msg.get("metadata"):
            render_metadata(msg["metadata"])


def question_buttons(questions: list[str], key: str) -> None:
    """A row of buttons; a click starts a new agent run with that question."""
    columns = st.columns(len(questions))
    for i, (column, question) in enumerate(zip(columns, questions)):
        if column.button(question, key=f"{key}_{i}", use_container_width=True):
            st.session_state.pending_prompt = question
            st.rerun()


# ── One agent run ────────────────────────────────────────────────────────

def run_turn(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        answer_box = st.empty()
        activity: list[str] = []
        answer = ""
        final = None

        try:
            for event in agent.chat_stream(prompt):
                if event["type"] in ("tool_call", "tool_result"):
                    line = describe_event(event)
                    if line:
                        activity.append(line)
                        status.markdown(line)
                elif event["type"] == "answer_delta":
                    answer += event["text"]
                    answer_box.markdown(answer + CURSOR)
                elif event["type"] == "final":
                    final = event
        except Exception as e:
            status.update(label="Error", state="error", expanded=True)
            st.error(f"Sorry, something went wrong: {type(e).__name__}. Please try again.")
            st.session_state.messages.pop()  # the agent rolled back this message too
            return

        if not activity:
            activity.append("💬 Answered without using tools")
            status.markdown(activity[0])
        status.update(label="Done", state="complete", expanded=True)

    response = final["response"]
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "activity": activity,
        "metadata": response.model_dump(include={"answer_type", "confidence", "found_in_sources", "sources"}),
        "follow_ups": response.follow_up_questions,
        "order": final["order"],
    })
    st.rerun()  # redraw from history, with metadata, order card and follow-up buttons


# ── Page ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📱 GrigoTel")
    st.caption("AI assistant for the Alexy G26 phone line.")
    if st.button("New conversation", use_container_width=True):
        agent.reset()
        st.session_state.messages = []
        st.session_state.pop("pending_prompt", None)
        st.rerun()

st.title("GrigoTel AI Assistant")

typed = st.chat_input("Ask about GrigoTel phones...")
prompt = typed or st.session_state.pop("pending_prompt", None)

messages = st.session_state.messages
if not messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME)

for msg in messages:
    render_message(msg)

if prompt:
    run_turn(prompt)
elif not messages:
    question_buttons(EXAMPLE_QUESTIONS, key="example")
elif messages[-1]["role"] == "assistant" and messages[-1].get("follow_ups"):
    question_buttons(messages[-1]["follow_ups"], key=f"follow_up_{len(messages)}")
