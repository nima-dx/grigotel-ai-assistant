"""
LLM judge: evaluates the agent's output against natural-language criteria.

The judge sees the customer's question, the tools the agent called (with their
results, as ground truth) and the agent's answer. For each criterion it writes
its reasoning first, then a pass/fail verdict.

Location: tests/judge.py
"""

import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

JUDGE_MODEL = "gpt-4o-mini"

JUDGE_INSTRUCTIONS = """
You are a strict evaluator of a customer-service AI assistant for GrigoTel, a phone shop.

You receive:
- the customer's question
- the tool calls the assistant made, with the results the tools returned
- the assistant's final answer to the customer

The tool results are the ground truth: prices, colors and stock levels in them are correct.
Evaluate the answer against EACH criterion separately.
For each criterion, first explain your reasoning, then decide whether it passed.
A criterion passes only if the answer and the tool calls clearly satisfy it.
""".strip()


class CriterionResult(BaseModel):
    criterion: str = Field(description="The criterion, copied exactly.")
    reasoning: str = Field(description="Why it passed or failed, citing the answer or tool results.")
    passed: bool


class JudgeResult(BaseModel):
    checks: list[CriterionResult]
    summary: str = Field(description="One-sentence overall assessment.")


def evaluate_agent_performance(
    criteria: list[str],
    question: str,
    answer: str,
    tool_trace: list[dict],
    client: OpenAI | None = None,
) -> JudgeResult:
    """
    Ask an LLM to judge the agent's answer against each criterion.

    tool_trace: [{"name": ..., "arguments": {...}, "result": {...}}, ...]
    """
    client = client or OpenAI()
    criteria_text = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1))

    prompt = f"""
## Customer question
{question}

## Tool calls and results
{json.dumps(tool_trace, indent=2, ensure_ascii=False)}

## Assistant's answer
{answer}

## Criteria
{criteria_text}
""".strip()

    response = client.responses.parse(
        model=JUDGE_MODEL,
        instructions=JUDGE_INSTRUCTIONS,
        input=[{"role": "user", "content": prompt}],
        text_format=JudgeResult,
        temperature=0,
    )
    return response.output_parsed


def assert_criteria(result: JudgeResult, criteria: list[str]) -> None:
    """Fail the test if any criterion failed (or was skipped), with the judge's reasoning."""
    assert len(result.checks) == len(criteria), (
        f"Judge returned {len(result.checks)} checks for {len(criteria)} criteria"
    )

    failed = [check for check in result.checks if not check.passed]
    if failed:
        details = "\n".join(f"- {c.criterion}\n  Reason: {c.reasoning}" for c in failed)
        raise AssertionError(f"{len(failed)} criteria failed:\n{details}\n\nSummary: {result.summary}")
