# GrigoTel AI Assistant

An AI shopping and support assistant for **GrigoTel**, a fictional phone company that sells
four phones: **Alexy G26**, **Alexy G26 Fold**, **Alexy G26 Pro** and **Alexy G26 Lite**.
Capstone project for *AI Engineering Buildcamp: From RAG to Agents*.

## The Problem

GrigoTel's product information is spread across seven PDF documents (spec sheets, a user
manual, a company profile and a privacy policy) and a separate stock database. To answer a
single question like *"Does the G26 Fold have wireless charging, and is the black one in
stock?"*, a support agent has to search the PDFs and look up the database. Customers wait,
and answers are slow and inconsistent.

## What It Does

The GrigoTel AI Assistant is a chat agent that answers questions about the phones, the company
and the privacy policy **only from GrigoTel's own documents**, gives live prices and stock
from the database, and takes orders.

A typical interaction: the customer types a question in plain language in a web chat, for
example *"I'd like to order 2 Blue Alexy G26 Pro, are they available?"*. The assistant checks
the stock and replies with the availability, the price and the total, and asks the customer
to confirm. Each answer shows the sources it used and suggests follow-up questions. The order
is placed, and a checkout link returned, only after the customer confirms. For a question like
*"How do I fix overheating?"*, it answers from the user manual and cites it. If the
information isn't in the documents, it says so instead of guessing.

## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file from the template and add your keys:

       cp .env.example .env

   - `OPENAI_API_KEY`: needed for everything
   - `LOGFIRE_TOKEN`: optional; sends traces to Logfire (or run `uv run logfire auth` once)
   - `LOGFIRE_READ_TOKEN`: only needed to download traces (step 9)

4. Install dependencies:

       uv sync

5. Build the data (steps 1–3):

       uv run src/grigotel_ai_assistant/setup_db.py
       uv run src/grigotel_ai_assistant/convert_docs.py
       uv run src/grigotel_ai_assistant/build_index.py

6. Start the chat app:

       uv run streamlit run src/grigotel_ai_assistant/app.py

7. Start Jupyter (for the notebooks):

       uv run jupyter notebook

   In VS Code, select the project's `.venv` as the notebook kernel.

## Notebooks

- `evals/create_eval_data.ipynb` - runs the agent on every evaluation question and saves the run (step 8)
- `evals/eval_judge.ipynb` - runs the LLM judge on a labeled run and compares it with the human labels (step 8)
- `notebooks/logfire_data_export.ipynb` - downloads chat traces from Logfire and turns them into agent runs (step 9)

## Data

- `GrigoTel_DB/`: the product database (CSV files and the DuckDB file built from them)
- `GrigoTel_Docs/`: the 7 PDFs, their markdown versions and the search chunks
- `evals/`: the evaluation questions, runs, human labels and judge results
- `data/` and `logfire_data/`: chat traces downloaded from Logfire (not in git: they contain chat transcripts)

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
├── tests/
│   ├── test_judge.py                  # step 7: the tests
│   ├── utils.py                       # step 7: run the agent, collect tool calls, track cost
│   └── judge.py                       # step 7: LLM judge for criteria-based tests
├── evals/
│   ├── questions.csv                  # step 8: 20 evaluation questions
│   ├── create_eval_data.ipynb         # step 8: run the agent on every question
│   ├── evals_run_<date>.json          # step 8: one evaluation run
│   ├── label_evals.py                 # step 8: Streamlit app for human labeling
│   ├── evals_run_<date>_results.json  # step 8: the run with human labels
│   ├── llm_judge.py                   # step 8: the LLM judge
│   ├── eval_judge.ipynb               # step 8: compare the judge with the human labels
│   ├── evaluate_judge.py              # step 8: the same comparison as a script
│   └── evals_run_<date>_results_evaluated.csv  # step 8: judge vs. human, per question
├── notebooks/
│   └── logfire_data_export.ipynb      # step 9: download and explore Logfire traces
├── .env.example                       # template for .env (API keys and tokens)
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
3. Repeat until the LLM writes its answer (at most 8 rounds).

**System prompt.** It tells the agent which tool to use, and these rules:

- answer only from tool results; never guess
- always call a tool for a GrigoTel question, even a follow-up
- for troubleshooting and how-to questions, give only the steps in the documents, without extra tips
- when the customer says "fingerprint", explain that the G26 uses "Vein-Scan"
- list in `sources` only the documents that contain the information
- prices and stock always come from the database
- if a spec sheet and the user manual disagree, the spec sheet is right
- follow the order process: ask for the model, color and quantity, check stock once, and ask the customer to confirm

The rules about tools, troubleshooting, "fingerprint" and sources were added after the evaluation in step 8 found answers that broke them.

**Order guard.** The code (not only the prompt) blocks orders without confirmation. `place_order` only works for an item that was checked in an *earlier* customer message. So the customer must see the offer and reply before any stock is reduced.

**Loop guard.** gpt-4o-mini sometimes repeated the same tool call until the round limit ran out. The agent now:

- doesn't run a tool call that was already made with the same arguments in this message; it tells the LLM to answer instead
- after such a repeat, and on the last round, calls the LLM with `tool_choice="none"`, so it must write an answer
- if a message still fails, removes its tool calls from the history, so the next message doesn't copy the failed pattern

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
- **Feedback buttons.** 👍 / 👎 under each answer (sent to Logfire, see step 9).
- **Order card.** After a confirmed order, shows the order and a "Proceed to checkout" button. The link comes from the tool result, not from the LLM's text.
- **New conversation.** A sidebar button that clears the chat.

The tools (database and index) load once and are shared. Each browser session gets its own agent, so customers' chats never mix.

### Step 7: Tests (`tests/`)

Run the tests from the project root:

```bash
uv run pytest tests/test_judge.py -v -s
```

Run one test by adding its name, for example `tests/test_judge.py::test_stock_count_comes_from_database`.

These tests call the real OpenAI API. Stop the app before running them, because DuckDB lets only one process write to the database.

| Test | What it checks |
|---|---|
| Stock count | Asks how many black Alexy G26 are in stock. Checks that a database tool was used, the tool got the right number, and the answer reports it ("out of stock" counts as 0). |
| Fold display | Asks for the Fold's display spec. Checks that the Fold spec sheet was found and cited, and that the display size from the markdown is in the answer. |
| Tool choice | Checks the first tool used: `get_product_info` for a price question, `search_docs` for a company question. |
| LLM judge | Asks to order 12 blue G26 Pros. Checks that no order was placed, then a second LLM (`judge.py`) checks the answer against 5 criteria, using the tool results as ground truth. |

The helpers:

- `utils.py`: `run_agent_test(agent, question)` runs the agent and returns the answer, the tool calls (`ToolCall(name, args)`), the tool results, any order and the token usage. `capture_usage` prints the tokens and cost of each run.
- `judge.py`: `await assert_criteria(result, criteria, question)` asks a PydanticAI judge agent for a pass/fail verdict and a reason per criterion, and fails the test if any criterion fails.

How we test:

- **Exact checks on the data the agent received.** For example, the stock number in the tool result. These don't depend on the LLM's wording.
- **Flexible checks on the text the agent wrote.** The LLM can phrase the same fact in different ways.
- **An LLM judge for things code can't check easily.** For example, "does it ask the customer to confirm?"

### Step 8: Evaluation (`evals/`)

The tests in step 7 check a few fixed questions. Step 8 measures the quality of the agent on a whole set of realistic questions, and builds an **LLM judge** that can label new runs automatically, so a prompt change can be checked without labeling every answer by hand.

```
questions.csv ──► create_eval_data.ipynb ──► evals_run_<date>.json
                                                   │
                                                   ▼
                                  label_evals.py (human: good / bad)
                                                   │
                                                   ▼
                                     evals_run_<date>_results.json
                                                   │
                                                   ▼
                     llm_judge.py + evaluate_judge.py / eval_judge.ipynb
                                                   │
                                                   ▼
                              evals_run_<date>_results_evaluated.csv
                                (judge vs. human, agreement report)
```

#### 8.1 Questions (`questions.csv`)

20 usage scenarios, written from the documents and the database. Each row has a `question`, a `group` and a `type`:

| Group | Questions |
|---|---|
| `product_specs` | 5 |
| `pricing_availability` | 4 |
| `order`, `how_to`, `troubleshooting`, `company`, `privacy` | 2 each |
| `other` | 1 |

The `type` covers the hard cases, not only direct questions: `wrong_terminology` (e.g. "fingerprint" for a phone with Vein-Scan), `not_in_docs`, `conflicting_sources` (manual vs. spec sheet), `multi_source`, `follow_up`, `vague`, `stock_limit`, `other_language` (French), `broad` and `off_topic`.

#### 8.2 Run the agent (`create_eval_data.ipynb`)

Runs the agent on every question and saves the run as `evals_run_<date>.json` (a second run on the same day gets `_2`).

- Each question gets a fresh agent, except `follow_up` questions, which continue the previous question's conversation.
- The agent runs on a **temporary copy of the database**, so order questions don't change the real stock (and the running app's lock doesn't get in the way).
- Each record stores the question, the previous question (for follow-ups), the system prompt, the structured response, the tool calls, any order, the token usage and the cost.

A full run of 20 questions costs about $0.014 with gpt-4o-mini.

#### 8.3 Label the answers (`label_evals.py`)

```bash
uv run streamlit run evals/label_evals.py
```

A Streamlit app for labeling each answer as **good** or **bad**, with an optional comment. It shows the question, the answer, the tools the agent called and the cited sources (the document text, or the database table), so the answer can be checked against the data. Labels are saved to `evals_run_<date>_results.json`.

The first run (`evals_run_2026_11_09`) got **15 good and 5 bad** labels. The bad answers added troubleshooting tips that aren't in the manual, cited the wrong document, confirmed a claim about "fingerprint data" (the phones use Vein-Scan), or answered a company question without calling a tool. These findings led to the new system prompt rules in step 5.

#### 8.4 The LLM judge (`llm_judge.py`)

A PydanticAI agent (gpt-4o-mini) that labels one run as `good` or `bad`, with step-by-step reasoning.

- **It gets all the reference material.** The 7 documents and the database are in its instructions. The knowledge base is small, and without it the judge can't tell a correct answer from a made-up one.
- **Core principle: judge what the answer adds, not what it leaves out.** An answer is bad when it tells the customer something the reference material doesn't support. A cautious "I couldn't find that" is not bad.
- **Five "bad" rules:** made-up content, a general-knowledge answer without a tool call, a wrong source, a false attribution (e.g. "fingerprint data") and wrong price/color/stock data.
- **A list of "good" cases**, so the judge doesn't punish correct behaviour: short or cautious answers, follow-ups that reuse earlier tool results, the order process asking for a color, and extra tool calls.
- **It must quote** each answer sentence it criticizes and show that the reference doesn't support it.

#### 8.5 Evaluate the judge (`evaluate_judge.py`, `eval_judge.ipynb`)

Before the judge can replace human labels, it must agree with them. Both files run the judge on a labeled run and compare its labels with the human ones. The notebook is for exploring; the script is for re-running:

```bash
uv run python evals/evaluate_judge.py --data evals/evals_run_2026_11_09_results.json
uv run python evals/evaluate_judge.py --data evals/evals_run_2026_11_09_results.json --model openai:gpt-4o
```

It prints the accuracy, the precision and recall for the **bad** class, the confusion matrix and every disagreement (with the human comment and the judge's reasoning), and saves the details as `evals_run_<date>_results_evaluated.csv`. "Bad" is the positive class because the judge's job is to catch bad answers, so a missed bad answer is the more costly mistake.

**Results on `evals_run_2026_11_09`:**

| Judge prompt | Agreement with human labels |
|---|---|
| First version | 65% |
| Improved version (reference material, core principle, good cases, quoting) | 95–100% (20 of 20 in the latest run) |

gpt-4o-mini agreed with the human labels more than gpt-4.1-mini and gpt-4o, which missed some bad answers.

The judge prompt was tuned on these 20 answers, so the agreement is optimistic. Before relying on it, label a new run (`evals_run_2026_11_10.json` is ready for this) and check the agreement again.

### Step 9: Monitoring with Logfire

[Logfire](https://logfire.pydantic.dev) records what the agent does: every LLM call, tool call and token count.

**In the app:**

- **One trace per chat session.** Each browser session gets a `streamlit_session` span, and every message in it is recorded under that span. "New conversation" starts a new trace.
- **User feedback.** The 👍 / 👎 buttons send a `user_feedback` event, and clicking a suggested question sends a `followup_question_clicked` event. Both are attached to the session's trace.

**Analysing the traces:**

| File | What it does |
|---|---|
| `download_traces.py` | Downloads recent sessions to `data/logs.bin` (agent runs) and `data/sessions.json` (readable summary). Options: `--limit`, `--days`. |
| `trace_replay/` | Turns a downloaded trace back into a PydanticAI `AgentRunResult` (messages and token usage), so it can be analysed or replayed. |
| `session_cost.py` | Adds up the tokens of a whole chat session and calculates its cost. |
| `notebooks/logfire_data_export.ipynb` | The same download step by step in a notebook; saves to `logfire_data/`. |

```bash
uv run python src/grigotel_ai_assistant/download_traces.py --limit 50 --days 30
uv run python src/grigotel_ai_assistant/session_cost.py <trace_id>
```

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
- The LLM-judged order test assumes fewer than 12 Blue G26 Pro are in stock. With the current stock (16), the judge sometimes misapplies the "if fewer than 12…" criteria and fails the test.
- DuckDB lets only one process write to the database. Stop the Streamlit app (and any notebook kernel using the database) before running the tests.
