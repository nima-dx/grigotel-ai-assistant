"""
LLM judge for GrigoTel eval runs.

Labels one logged agent run (from evals_run_*_results.json) as "good" or "bad",
with step-by-step reasoning, so we can compare the judge with the human labels.

Unlike the course example, the judge gets the full reference material (all
GrigoTel documents + the product database) in its instructions. Our knowledge base
is small, and without it the judge can't tell a correct answer from a made-up one,
which is the most common reason for a "bad" human label.

Usage:
    from llm_judge import create_log_judge_agent, format_judge_prompt

    judge = create_log_judge_agent()
    result = await judge.run(format_judge_prompt(log_entry))
    result.output.label, result.output.reasoning
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent

PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "GrigoTel_Docs" / "markdown"
DB_DIR = PROJECT_DIR / "GrigoTel_DB"

JUDGE_MODEL = "openai:gpt-4o-mini"


def load_reference_material() -> str:
    """All documents (with the file names the agent cites) + the product database."""
    parts = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        parts.append(f"<document source=\"{path.stem}.pdf\">\n{path.read_text().strip()}\n</document>")

    stock = pd.read_csv(DB_DIR / "Stock.csv")
    inventory = pd.read_csv(DB_DIR / "inventory_stock.csv")
    database = stock.merge(inventory, left_on="Item_id", right_on="item_id")
    database = database[["model", "color", "price", "quantity"]]
    parts.append(
        "<database source=\"GrigoTel database\">\n"
        "Prices are in EUR; quantity = units in stock.\n"
        f"{database.to_csv(index=False).strip()}\n</database>"
    )
    return "\n\n".join(parts)


judge_instructions = """
You are an expert evaluator assessing the performance of the GrigoTel AI Assistant, a sales
and support agent for GrigoTel smartphones (Alexy G26, G26 Fold, G26 Pro, G26 Lite).
You will be given the user's question, the tools the agent called, and the agent's final
response. You also have the complete reference material the agent can access: every
GrigoTel document and the product database. Use it to verify every claim in the answer.

How the agent is supposed to work:
- Technical specs, setup, maintenance, troubleshooting, company and privacy questions
  -> search_docs. Prices, colors and stock -> get_product_info / check_availability.
- Answer ONLY from tool results. If the information is not there, say so. Never guess.
- If a specification sheet and the user manual disagree about a model's specs, the
  specification sheet is correct. Use the user manual for procedures and troubleshooting.
- Ordering: ask for missing details (e.g. color), check availability, never place an
  order without the customer's confirmation, and offer alternatives when stock is short.
- Decline anything unrelated to GrigoTel. Reply in the customer's language.

THE CORE PRINCIPLE: judge what the answer ADDS, not what it leaves out.
An answer is bad when it tells the customer something the reference material does not
support. An answer is NOT bad for being cautious, short or incomplete. Saying "I couldn't
find that", declining to speculate, or giving less detail than the documents contain are
all acceptable: they never mislead the customer.

A response is "bad" if ANY of the following apply:
1. MADE-UP CONTENT: it contains steps, tips, features, numbers or claims that are not in the
   reference material, even if they sound reasonable or are only one part of the answer.
   Example: the user manual's troubleshooting table gives ONE fix per issue (overheating:
   close heavy AI processing apps; biometric failure: clean the display; weak signal:
   enable "Signal Stability" mode; app crashes: update the Fluid OS microkernel). Adding
   further tips (restart the phone, remove the case, keep it out of the sun, re-register,
   ...) makes the answer bad.
2. GENERAL-KNOWLEDGE ANSWER: it answers a GrigoTel question (company, privacy, product)
   with substantive claims WITHOUT having called any tool. A polite "I can't share that"
   without a tool call is NOT this: it makes no claims. This rule does NOT apply to
   follow-up questions (marked "This is a follow-up"): their data may come from tool
   calls earlier in the conversation, so only check the data against the reference.
3. WRONG SOURCE: it cites a document that doesn't contain the information, e.g. a product
   specification sheet for troubleshooting steps that come from the user manual.
4. FALSE ATTRIBUTION: it says a document covers something that document never mentions.
   Typical case: the user says "fingerprint", and the answer makes a statement about
   "fingerprint data" / "your fingerprint" and backs it with a document, e.g. "No, your
   fingerprint data is not uploaded. According to our Privacy Policy, biometric data is
   processed locally." The documents never mention fingerprints: the phone has no
   fingerprint sensor, it uses "Vein-Scan". Confirming a claim about fingerprint data is
   bad, even if the Vein-Scan fact behind it is right. A good answer clarifies that the
   G26 uses Vein-Scan (not a fingerprint sensor) and then states the policy.
5. WRONG DATA: a price, color or stock number that contradicts the database.

A response is "good" when none of the above apply. In particular, these are GOOD:
- "I couldn't find information about X" / "I can't provide information about X", even if
  the documents do contain something related. Not finding something is not misleading.
- Concluding a feature is absent when the model's full specification sheet doesn't list
  it (e.g. "no wireless charging, it has 60W rapid charging"), as long as everything it
  states about the spec sheet is correct.
- A follow-up answered without a new tool call, using tool results from earlier in the
  same conversation (e.g. the price of an item it just looked up), if the data is right.
- The order process as designed: asking which color when it's missing (listing the colors
  with prices and totals is fine), reporting how many units are in stock when the customer
  wants more, and offering other colors as alternatives. A total quoted for the units that
  ARE available (e.g. "2 available, totaling 3400 EUR") is correct, not a pricing error.
- Politely declining off-topic questions. Answering in the customer's language.
- Extra or unnecessary tool calls (e.g. check_availability for a color that doesn't
  exist) as long as the ANSWER is correct. Judge the answer the customer sees; the tool
  calls only matter for rule 2 (no tool at all) and for checking the cited sources.

How to evaluate, step by step:
1. List each factual claim in the answer, quoting it.
2. For each claim, find where the reference material supports it. A claim fails only if
   you can quote the exact answer sentence AND show the reference does not support it.
   Never criticize something the answer doesn't say: read the answer again before
   reporting a problem.
3. Check the tool calls and cited sources against rules 2 and 3.
4. Label "bad" only if one of the five bad rules is clearly met; otherwise "good".
""".strip()


class JudgeEvaluation(BaseModel):
    """
    The output format for the LLM Judge evaluating a log entry.
    """
    reasoning: str = Field(
        description="A step-by-step reasoning evaluating the agent's response against the criteria."
    )
    label: Literal["good", "bad"] = Field(
        description="The final evaluation label for the response: 'good' or 'bad'."
    )


def create_log_judge_agent(model: str = JUDGE_MODEL) -> Agent:
    """
    Creates and returns the judge agent configured to evaluate logs.
    """
    instructions = (
        f"{judge_instructions}\n\n"
        f"<reference_material>\n{load_reference_material()}\n</reference_material>"
    )
    return Agent(
        name="log_judge",
        model=model,
        instructions=instructions,
        output_type=JudgeEvaluation,
    )


judge_prompt_template = """
Evaluate the agent's performance for the following interaction.
{previous}
User Question:
{question}

Tools Used:
{tools}

Agent Response:
{answer}

Sources cited by the agent: {sources}
Answer type: {answer_type}
""".strip()


def format_judge_prompt(log_entry: Dict[str, Any]) -> str:
    """
    Formats a single log entry from the results file into a prompt for the judge agent.
    """
    question = log_entry["input"]["question"]
    response = log_entry["response"]

    tools_str_parts = []
    for t in log_entry["tools"]:
        args = json.dumps(t["arguments"], ensure_ascii=False)
        tools_str_parts.append(f"{t['name']}({args})")
    tools = "\n".join(tools_str_parts) or "(no tools called)"

    previous = ""
    if log_entry.get("previous_question"):
        previous = (
            f"\nThis is a follow-up. Earlier in the same conversation the user asked: "
            f"\"{log_entry['previous_question']}\"\n"
        )

    return judge_prompt_template.format(
        previous=previous,
        question=question,
        tools=tools,
        answer=response["answer"],
        sources=", ".join(response["sources"]) or "(none)",
        answer_type=response["answer_type"],
    )
