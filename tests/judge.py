"""
LLM judge: evaluates the agent's output against natural-language criteria.

A PydanticAI judge agent receives the customer's question, the agent's output and every
tool call with its result (the ground truth for prices, colors and stock). For each
criterion it returns a judgement and a pass/fail verdict (JudgeCriterion), plus an
overall summary (JudgeFeedback). It runs at temperature 0 for repeatable verdicts.

assert_criteria(result, criteria, question) prints the verdicts and fails the test if
the judge skipped a criterion or any criterion failed, listing all failures together.

Location: tests/judge.py
"""

import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from tests.utils import AgentTestResult, capture_usage, get_model_name

load_dotenv()

JUDGE_MODEL = "openai:gpt-4o-mini"

judge_instructions = """
You are an expert judge evaluating the performance of the GrigoTel AI Assistant,
a customer-service agent for a phone shop.

You receive:
- the customer's question
- the tool calls the assistant made, with the results the tools returned
- the assistant's final output to the customer

The tool results are the ground truth: prices, colors and stock levels in them are correct.
Evaluate the output against EACH criterion separately, in the order given.
For each criterion, explain your judgement with evidence from the output or the tool calls.
A criterion passes only if the output and the tool calls clearly satisfy it.
""".strip()


class JudgeCriterion(BaseModel):
    """
    Evaluation of a single test requirement or behavioral rule.
    """
    criterion_description: str = Field(
        description="The specific requirement or rule that the agent is being evaluated against, copied exactly."
    )
    judgement: str = Field(
        description="A clear explanation of why the agent passed or failed, referencing specific evidence "
                    "from the agent's output or tool calls."
    )
    passed: bool = Field(
        description="Whether the agent's response and actions satisfied this requirement."
    )


class JudgeFeedback(BaseModel):
    """
    The complete evaluation report from the judge agent, summarizing performance across all criteria.
    """
    criteria: list[JudgeCriterion] = Field(
        description="One evaluation per criterion provided in the test, in the same order."
    )
    feedback: str = Field(
        description="A holistic summary of the agent's performance, identifying key strengths or failures."
    )


def create_judge_agent(model: str = JUDGE_MODEL) -> Agent:
    return Agent(
        name="judge",
        model=model,
        instructions=judge_instructions,
        output_type=JudgeFeedback,
        model_settings={"temperature": 0},
    )


judge_user_prompt_template = """
Evaluate the agent's performance based on the following criteria:
<CRITERIA>
{criteria}
</CRITERIA>

The customer's question was:
<QUESTION>
{question}
</QUESTION>

The agent's final output was:
<AGENT_OUTPUT>
{output}
</AGENT_OUTPUT>

Tool calls and their results:
<TOOL_CALLS>
{tool_calls}
</TOOL_CALLS>
""".strip()


async def assert_criteria(result: AgentTestResult, criteria: list[str], question: str) -> JudgeFeedback:
    """Ask the judge about each criterion; fail the test if any failed (or was skipped)."""
    judge_agent = create_judge_agent()
    judge_user_prompt = judge_user_prompt_template.format(
        criteria="\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1)),
        question=question,
        output=result.output.model_dump_json(indent=2),
        tool_calls=json.dumps(result.tool_trace(), indent=2, ensure_ascii=False),
    )

    judge_result = await judge_agent.run(judge_user_prompt)
    usage = judge_result.usage  # a property in this PydanticAI version (a method in older ones)
    usage = usage() if callable(usage) else usage
    capture_usage(get_model_name(judge_agent), usage.input_tokens, usage.output_tokens)

    feedback = judge_result.output
    print("judge feedback:")
    print(feedback.feedback)
    for criterion in feedback.criteria:
        print(f"  [{'PASS' if criterion.passed else 'FAIL'}] {criterion.criterion_description}")
        print(f"         {criterion.judgement}")

    assert len(feedback.criteria) == len(criteria), (
        f"Judge returned {len(feedback.criteria)} checks for {len(criteria)} criteria"
    )
    failed = [c for c in feedback.criteria if not c.passed]
    assert not failed, (
        f"{len(failed)} criteria failed:\n"
        + "\n".join(f"- {c.criterion_description}\n  Reason: {c.judgement}" for c in failed)
        + f"\n\nFeedback: {feedback.feedback}"
    )
    return feedback
