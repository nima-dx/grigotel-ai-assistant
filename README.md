# GrigoTel AI Assistant

An AI shopping and support assistant for **GrigoTel**, a fictional phone company.
Capstone project for the course *AI Engineering Buildcamp: From RAG to Agents*.

GrigoTel sells four phones: **Alexy G26**, **Alexy G26 Fold**, **Alexy G26 Pro** and **Alexy G26 Lite**.

Customers chat with the assistant in a Streamlit web page. The assistant can:

- answer technical questions (specs, setup, battery, maintenance) from the PDF documents
- answer questions about the company and the privacy policy
- give prices, colors and stock from the database
- place an order, but only after the customer confirms it

The project also includes tests, an evaluation dataset, and monitoring with Logfire.

## How it works

```
Customer ──► Streamlit app ──► Agent (gpt-4o-mini)
                                  │
                ┌─────────────────┼────────────────────┐
                ▼                 ▼                    ▼
           search_docs     get_product_info     check_availability
                │                 │              place_order
                ▼                 └──────────┬─────────┘
       PDF knowledge base                    ▼
       (minsearch index)              DuckDB database
```

The agent decides which tool to call:

| Question about | Tool | Data source |
|---|---|---|
| Specs, setup, battery, company, privacy | `search_docs` | PDF documents |
| Price, colors, stock | `get_product_info` | DuckDB |
| Buying a phone | `check_availability`, then `place_order` | DuckDB |

## Project files

```
grigotel-ai-assistant/
├── GrigoTel_DB/
│   ├── Stock.csv                      # one row per model + color, with price
│   ├── inventory_stock.csv            # current quantity per item
│   └── grigotel.duckdb                # database built from the CSVs
├── GrigoTel_Docs/
│   ├── *.pdf                          # 4 spec sheets, user manual, corporate profile, privacy policy
│   ├── markdown/                      # the 7 PDFs converted to markdown
│   └── chunks.json                    # document chunks used by the search index
├── src/grigotel_ai_assistant/
│   ├── setup_db.py                    # step 1: build the database
│   ├── convert_docs.py                # step 2: convert PDFs to markdown
│   ├── build_index.py                 # step 3: chunk the documents
│   ├── tools.py                       # step 4: the agent's tools
│   ├── agent.py                       # step 5: the agent
│   ├── app.py                         # step 6: the Streamlit chat page
│   ├── download_traces.py             # step 9: download traces from Logfire
│   ├── session_cost.py                # step 9: token count and cost of a session
│   └── trace_replay/                  # step 9: turn Logfire traces back into agent runs
│       ├── converter.py
│       ├── test_converter.py
│       └── README.md
├── tests/
│   ├── test_agent.py                  # step 7: tests
│   ├── utils.py                       # helper: collect tool calls
│   └── judge.py                       # helper: LLM judge
├── evals/
│   ├── create_eval_data.ipynb         # step 8: create the evaluation questions
│   ├── questions.csv                  # step 8: the evaluation questions
│   └── evals_run_2026_11_09.json      # step 8: results of one evaluation run
├── notebooks/
│   └── logfire_data_export.ipynb      # step 9: explore data exported from Logfire
├── data/
│   ├── sessions.json                  # step 9: saved chat sessions
│   └── logs.bin                       # step 9: saved log data
├── .env                               # API keys and tokens (not in git)
├── pyproject.toml / uv.lock           # dependencies (managed with uv)
└── README.md
```

### The data

**Database tables (DuckDB)**

- `Stock`: `item_id`, `model`, `color`, `price`, `release_date`. Each model + color has its own `item_id` and price.
- `Inventory_Stock`: `item_id`, `quantity`, `update_date`. The current stock for each item.

**Documents (7 PDFs)**

- one spec sheet per phone (4 files)
- `Alexy_G26_User_Manual.pdf`: setup, maintenance, battery, troubleshooting
- `GrigoTel_Corporate_Profile.pdf`: company overview
- `Privacy_and_Compliance_Policy.pdf`

## Setup

You need [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
uv sync
```

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=sk-...
LOGFIRE_READ_TOKEN=...     # only needed for step 9 (monitoring)
```

Build the data, then start the app:

```bash
uv run src/grigotel_ai_assistant/setup_db.py
uv run src/grigotel_ai_assistant/convert_docs.py
uv run src/grigotel_ai_assistant/build_index.py
uv run streamlit run src/grigotel_ai_assistant/app.py
```

## Steps

### Step 1: Database (`setup_db.py`)

Creates the two DuckDB tables and loads the CSV files into them.

- The tables have keys, so `Inventory_Stock` can only hold items that exist in `Stock`.
- A rule makes sure stock can never go below 0.
- Running the script again resets the stock to the CSV values. This is handy after test orders.

### Step 2: Convert the PDFs (`convert_docs.py`)

Converts each PDF to a markdown file with `markitdown`, and saves it in `GrigoTel_Docs/markdown/`.

It also tags each document:

- a spec sheet gets its model name, for example `Alexy G26 Pro`
- the manual, company profile and privacy policy get the tag `general`

The order of the tagging rules matters: "G26" appears in every file name, so the more specific names are checked first.

### Step 3: Chunk the documents (`build_index.py`)

Splits the documents into smaller pieces ("chunks") with `chunk_documents` from `gitsource`.

- Each chunk is up to 2,000 characters, and chunks overlap by 1,000 characters. That way, text cut at a chunk border still appears whole in another chunk.
- Each chunk keeps its tags: model, document type and file name.
- The chunks are saved to `chunks.json`. The app loads them into a `minsearch` index when it starts.

### Step 4: Tools (`tools.py`)

Four functions the agent can call:

| Tool | What it does |
|---|---|
| `search_docs(query, model)` | Searches the documents. With a model, it returns that model's spec sheet plus the general documents. |
| `get_product_info(model)` | Returns prices, colors and stock. |
| `check_availability(model, color, quantity)` | Checks stock only. Returns `available`, `partial`, `out_of_stock` or `color_not_offered`. |
| `place_order(item_id, quantity)` | Lowers the stock, updates `update_date` and returns a (fake) checkout link. |

Design choices:

- **The LLM never writes SQL.** All queries are fixed in the code. This is safer and easier to test.
- **Flexible model names.** "pro", "g26 pro" and "Alexy G26 Pro" all work.
- **Safe orders.** `place_order` checks the stock again in the same database statement, so stock can't go negative.

### Step 5: Agent (`agent.py`)

The agent uses the OpenAI Responses API with `gpt-4o-mini`. It works in a loop:

1. Send the conversation and the tool list to the LLM.
2. If the LLM asks for a tool, run it and send back the result.
3. Repeat until the LLM writes its answer.

**System prompt.** It tells the agent which tool to use, and these rules:

- answer only from tool results; never guess
- prices and stock always come from the database
- if a spec sheet and the user manual disagree, the spec sheet is right
- follow the order process: ask for the model, color and quantity, check stock, and ask the customer to confirm

**Order guard.** The code (not only the prompt) blocks orders without confirmation. `place_order` only works for an item that was checked in an *earlier* customer message. So the customer must see the offer and reply before any stock is reduced.

**Structured output.** The final answer is a Pydantic model, `AssistantResponse`:

| Field | Meaning |
|---|---|
| `answer` | the reply to the customer |
| `answer_type` | for example `product_specs`, `pricing_availability` or `order` |
| `confidence` | 0 to 1 |
| `found_in_sources` | whether tool results back the answer |
| `sources` | the PDF names, or "GrigoTel database" |
| `follow_up_questions` | 2–3 suggested next questions |

**Streaming.** `chat_stream()` sends events as they happen: tool calls, tool results, pieces of the answer, and the final result. `chat()` is a simple version for tests.

### Step 6: Chat app (`app.py`)

A Streamlit chat page:

- **Live answers.** The answer appears word by word as it streams in.
- **Activity panel.** Shows each tool call, for example "🔎 Searching documents" or "📊 Checking stock".
- **Metadata row.** Below each answer: answer type, confidence, whether it's backed by sources, and the sources.
- **Follow-up buttons.** Clicking one asks that question right away.
- **Order card.** After a confirmed order, shows the order and a "Proceed to checkout" button. The link comes from the tool result, not from the LLM's text.
- **New conversation.** A sidebar button that clears the chat.

The tools (database and index) load once and are shared. Each browser session gets its own agent, so customers' chats never mix.

### Step 7: Evaluation (`tests/`)

Run the tests:

```bash
uv add --dev pytest
uv run pytest tests/test_agent.py -v -s
```

These tests call the real OpenAI API. Stop the app before running them, because DuckDB lets only one process write to the database.

| Test | What it checks |
|---|---|
| Stock count | Asks how many black Alexy G26 are in stock. Checks that a database tool was used, the tool got the right number, and the answer reports it ("out of stock" counts as 0). |
| Fold display | Asks for the Fold's display spec. Checks that the Fold spec sheet was found and cited, and that the display size from the markdown is in the answer. |
| Tool choice | With `collect_tools` (`utils.py`), checks the first tool used: `get_product_info` for a price question, `search_docs` for a company question. |
| LLM judge | Asks to order 12 blue G26 Pros. A second LLM (`judge.py`) checks the answer against 5 criteria, using the tool results as ground truth. The code also checks that no order was placed. |

How we test:

- **Exact checks on the data the agent received.** For example, the stock number in the tool result. These don't depend on the LLM's wording.
- **Flexible checks on the text the agent wrote.** The LLM can phrase the same fact in different ways.
- **An LLM judge for things code can't check easily.** For example, "does it ask the customer to confirm?"

### Step 8: Evaluation dataset (`evals/`)

The tests in step 7 check a few questions. Step 8 checks the agent on many questions at once.

- `create_eval_data.ipynb` creates the evaluation questions.
- `questions.csv` holds the questions, covering the different question types (specs, prices, stock, orders, company, privacy).
- The agent answers every question, and the results are saved as a JSON file named after the run date, for example `evals_run_2026_11_09.json`.

Saving each run in its own file lets you compare runs, for example before and after a change to the system prompt.

### Step 9: Monitoring with Logfire

[Logfire](https://logfire.pydantic.dev) records what the agent does: every LLM call, tool call and token count.

| File | What it does |
|---|---|
| `download_traces.py` | Downloads traces from Logfire. |
| `trace_replay/` | Turns a downloaded trace back into an agent run (messages and token usage), so it can be analysed or replayed. |
| `session_cost.py` | Adds up the tokens of a whole chat session and calculates its cost. |
| `notebooks/logfire_data_export.ipynb` | Explores the data exported from Logfire. |
| `data/` | Saved sessions and logs. |

**Session cost.** A chat session can contain several agent runs. `trace_replay` reads every run in the session and adds up their tokens, instead of keeping only the last one. The cost uses gpt-4o-mini prices:

| | Price per 1M tokens |
|---|---|
| Input | $0.15 |
| Output | $0.60 |

Downloading traces needs a Logfire **read token** (`LOGFIRE_READ_TOKEN` in `.env`). Create one in Logfire under your project's **Settings → Read tokens**.

## Known issues

- The user manual says all phones have IP69 water resistance and a 5,500 mAh battery. This doesn't match the Lite spec sheet. The agent is told to trust the spec sheet.
- Checkout links are fake (`shop.grigotel.example`). Creating the link counts as a completed purchase.
- The tests use a real LLM, so a test can sometimes fail because of wording.
