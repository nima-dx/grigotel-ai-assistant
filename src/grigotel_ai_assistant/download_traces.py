"""
Download recent Streamlit session traces from Logfire into the data/ folder.

Each conversation in the Streamlit app is one `streamlit_session` trace. This script:
  1. finds the most recent sessions,
  2. converts each one to a PydanticAI AgentRunResult (messages + token usage)
     with trace_replay,
  3. collects the user events logged in that session (user_feedback,
     followup_question_clicked),
  4. saves everything to:
       data/logs.bin       - pickled list of AgentRunResult (for evaluation code)
       data/sessions.json  - readable summary: one entry per session

Usage (from the project root):
    uv run python src/grigotel_ai_assistant/download_traces.py
    uv run python src/grigotel_ai_assistant/download_traces.py --limit 100 --days 7

Requires LOGFIRE_READ_TOKEN in .env.
"""

import argparse
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from logfire.query_client import LogfireQueryClient

try:  # works when the package is installed (uv init --package)
    from grigotel_ai_assistant.trace_replay import fetch_traces, trace_to_run_result
except ImportError:  # works when run as a script from this folder
    from trace_replay import fetch_traces, trace_to_run_result

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # <project root>/data
SESSION_SPAN = "streamlit_session"
EVENT_NAMES = ("user_feedback", "followup_question_clicked")


def fetch_session_ids(client: LogfireQueryClient, since: datetime, limit: int) -> list[dict]:
    """The most recent Streamlit sessions: trace_id + start time, newest first."""
    result = client.query_json_rows(
        min_timestamp=since,
        sql=f"""
        SELECT trace_id, start_timestamp
        FROM records
        WHERE span_name = '{SESSION_SPAN}'
        ORDER BY start_timestamp DESC
        LIMIT {limit}
        """,
    )
    return result["rows"]


def fetch_events(client: LogfireQueryClient, trace_ids: list[str], since: datetime) -> dict[str, list[dict]]:
    """User events (feedback, follow-up clicks) per trace_id, oldest first."""
    if not trace_ids:
        return {}
    ids_csv = ", ".join(f"'{tid}'" for tid in trace_ids)
    names_csv = ", ".join(f"'{name}'" for name in EVENT_NAMES)
    result = client.query_json_rows(
        min_timestamp=since,
        sql=f"""
        SELECT
            trace_id,
            start_timestamp,
            span_name AS event,
            attributes->>'feedback' AS feedback,
            attributes->>'question' AS question
        FROM records
        WHERE trace_id IN ({ids_csv})
          AND span_name IN ({names_csv})
        ORDER BY start_timestamp ASC
        """,
    )
    events: dict[str, list[dict]] = {}
    for row in result["rows"]:
        event = {"event": row["event"], "timestamp": row["start_timestamp"]}
        if row["feedback"] is not None:
            event["feedback"] = int(row["feedback"])
        if row["question"] is not None:
            event["question"] = row["question"]
        events.setdefault(row["trace_id"], []).append(event)
    return events


def main(limit: int, days: int) -> None:
    since = datetime.now(timezone.utc) - timedelta(days=days)

    with LogfireQueryClient(read_token=os.environ["LOGFIRE_READ_TOKEN"]) as client:
        sessions = fetch_session_ids(client, since, limit)
        trace_ids = [s["trace_id"] for s in sessions]
        traces = fetch_traces(trace_ids, client, since=since)  # sessions without LLM calls are skipped
        events = fetch_events(client, trace_ids, since)

    runs = []
    summary = []
    for session in sessions:
        trace = traces.get(session["trace_id"])
        if trace is None:
            continue
        run = trace_to_run_result(trace)
        runs.append(run)

        session_events = events.get(session["trace_id"], [])
        feedback = [e["feedback"] for e in session_events if e["event"] == "user_feedback"]
        summary.append({
            "trace_id": session["trace_id"],
            "start_timestamp": session["start_timestamp"],
            "messages": len(run.all_messages()),
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "feedback": feedback[-1] if feedback else None,  # the user's final rating
            "events": session_events,
        })

    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "logs.bin", "wb") as f_out:
        pickle.dump(runs, f_out)
    with open(DATA_DIR / "sessions.json", "w") as f_out:
        json.dump(summary, f_out, indent=2, ensure_ascii=False)

    rated = [s["feedback"] for s in summary if s["feedback"] is not None]
    print(f"Sessions found:      {len(sessions)} (last {days} days)")
    print(f"Sessions with runs:  {len(runs)}")
    print(f"Feedback:            {rated.count(1)} 👍  {rated.count(-1)} 👎")
    print(f"Saved:               {DATA_DIR / 'logs.bin'}")
    print(f"                     {DATA_DIR / 'sessions.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--limit", type=int, default=50, help="max sessions to download (default: 50)")
    parser.add_argument("--days", type=int, default=30, help="look back this many days (default: 30)")
    args = parser.parse_args()
    main(args.limit, args.days)
