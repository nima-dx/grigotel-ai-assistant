"""
Evaluate the LLM judge against human labels.

Runs the judge (llm_judge.py) on every run in a labeled results file (from
label_evals.py), compares its labels with the human labels and prints:
  - accuracy, and precision / recall for the "bad" class
  - the confusion matrix
  - every disagreement, with the human comment and the judge's reasoning
The detailed results are saved next to the input as <name>_evaluated.csv.

"bad" is the positive class: the judge's job is to catch bad runs.
  precision = of the runs the judge called bad, how many really are bad
  recall    = of the really bad runs, how many the judge caught

Usage (from the project root):
    uv run python evals/evaluate_judge.py --data evals/evals_run_2026_11_09_results.json
    uv run python evals/evaluate_judge.py --data evals/evals_run_2026_11_09_results.json --model openai:gpt-4o
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))  # so llm_judge can be imported from any folder
from llm_judge import JUDGE_MODEL, create_log_judge_agent, format_judge_prompt  # noqa: E402

load_dotenv()

CONCURRENCY = 5  # judge calls running at the same time
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def resolve_path(path: str) -> Path:
    """Accept paths relative to the current folder or to evals/."""
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    if (EVALS_DIR / candidate).exists():
        return EVALS_DIR / candidate
    sys.exit(f"File not found: {path}")


async def evaluate_sample(judge, row: dict, semaphore: asyncio.Semaphore) -> dict:
    prompt = CONTROL_CHARS.sub("", format_judge_prompt(row))
    async with semaphore:
        try:
            result = await judge.run(prompt)
            return {"llm_label": result.output.label, "llm_reasoning": result.output.reasoning}
        except Exception as e:  # don't let one failure (rate limit, ...) stop the batch
            return {"llm_label": "error", "llm_reasoning": f"Evaluation failed: {type(e).__name__}: {e}"}


async def run_judge(results: list[dict], model: str) -> list[dict]:
    judge = create_log_judge_agent(model)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [evaluate_sample(judge, row, semaphore) for row in results]
    return await tqdm.gather(*tasks, desc="Evaluating")  # keeps the input order


def build_dataframe(results: list[dict], judgments: list[dict]) -> pd.DataFrame:
    rows = []
    for row, judgment in zip(results, judgments):
        rows.append({
            "index": row.get("index"),
            "question": row["input"]["question"],
            "group": row["input"]["group"],
            "type": row["input"]["type"],
            "llm_answer": row["response"]["answer"],
            "human_label": row["label"],
            "human_comment": row.get("comments", ""),
            **judgment,
        })
    df = pd.DataFrame(rows)
    df["agree"] = df["human_label"] == df["llm_label"]
    return df


def print_report(df: pd.DataFrame) -> None:
    errors = df[df["llm_label"] == "error"]
    scored = df[df["llm_label"] != "error"]

    print("\n" + "=" * 60)
    print("Judge evaluation")
    print("=" * 60)

    if not errors.empty:
        print(f"\n⚠️  {len(errors)} sample(s) failed and are left out of the metrics:")
        for _, row in errors.iterrows():
            print(f"  - {row['question']}: {row['llm_reasoning']}")

    if scored.empty:
        print("\nNo samples were evaluated.")
        return

    # "bad" is the positive class
    y_true_bad = scored["human_label"] == "bad"
    y_pred_bad = scored["llm_label"] == "bad"

    tp = int((y_true_bad & y_pred_bad).sum())
    fp = int((~y_true_bad & y_pred_bad).sum())
    fn = int((y_true_bad & ~y_pred_bad).sum())
    tn = int((~y_true_bad & ~y_pred_bad).sum())

    accuracy = (tp + tn) / len(scored)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\nSamples evaluated:        {len(scored)}")
    print(f"Accuracy:                 {accuracy:.3f} ({accuracy:.1%})")
    print(f"Precision (class='bad'):  {precision:.3f} ({precision:.1%})")
    print(f"Recall    (class='bad'):  {recall:.3f} ({recall:.1%})")

    print("\nConfusion matrix:")
    print(f"{'':18}{'judge: good':>14}{'judge: bad':>14}")
    print(f"{'human: good':18}{tn:>14}{fp:>14}")
    print(f"{'human: bad':18}{fn:>14}{tp:>14}")
    if fn:
        print(f"\n{fn} bad run(s) MISSED by the judge (human bad, judge good): the more dangerous mistake.")

    disagreements = scored[~scored["agree"]]
    if disagreements.empty:
        print("\nNo disagreements. 🎉")
        return

    print(f"\n--- Disagreements ({len(disagreements)}) ---")
    for _, row in disagreements.iterrows():
        print(f"\nQ: {row['question']}  [{row['group']} / {row['type']}]")
        print(f"  Human: {row['human_label']}  ({row['human_comment'] or 'no comment'})")
        print(f"  Judge: {row['llm_label']}")
        print("  Reasoning: " + row["llm_reasoning"].replace("\n", "\n             "))
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LLM judge against human labels")
    parser.add_argument("--data", required=True, help="labeled results file, e.g. evals/evals_run_2026_11_09_results.json")
    parser.add_argument("--model", default=JUDGE_MODEL, help=f"judge model (default: {JUDGE_MODEL})")
    args = parser.parse_args()

    results_path = resolve_path(args.data)
    print(f"Loading data from {results_path}")
    with open(results_path) as f:
        results = json.load(f)

    print(f"Running the judge ({args.model}) on {len(results)} samples...")
    judgments = asyncio.run(run_judge(results, args.model))

    df = build_dataframe(results, judgments)
    print_report(df)

    output_path = results_path.with_name(f"{results_path.stem}_evaluated.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
