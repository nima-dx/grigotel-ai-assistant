"""
Labeling app for eval runs.

Go through an evals_run_*.json file one run at a time: read the question, the
answer, its sources and the tool calls, then label the run good / not good with
a comment. Labels are saved next to the input file as <name>_results.json.

Usage (from the project root):
    uv run streamlit run evals/label_evals.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from grigotel_ai_assistant.agent import SYSTEM_PROMPT

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALS_DIR.parent
DOCS_DIR = PROJECT_DIR / "GrigoTel_Docs" / "markdown"
DB_DIR = PROJECT_DIR / "GrigoTel_DB"
DATABASE_SOURCE = "GrigoTel database"

LABELS = {"good": "👍 Good", "bad": "👎 Not good"}

st.set_page_config(page_title="Label evals", page_icon="🏷️", layout="wide")


# ── Files ────────────────────────────────────────────────────────────────

def results_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_results.json")


@st.cache_data
def load_runs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_labels(input_path: Path) -> dict[int, dict]:
    """Saved labels, keyed by the run's index in the input file."""
    path = results_path(input_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return {item["index"]: item for item in json.load(f)}


def save_labels(input_path: Path, labels: dict[int, dict]) -> None:
    items = [labels[i] for i in sorted(labels)]
    with open(results_path(input_path), "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


# ── Sources ──────────────────────────────────────────────────────────────

@st.cache_data
def load_document(source: str) -> str | None:
    """The markdown text of a cited document ('X.pdf' -> GrigoTel_Docs/markdown/X.md)."""
    path = DOCS_DIR / f"{Path(source).stem}.md"
    return path.read_text() if path.exists() else None


@st.cache_data
def load_database() -> pd.DataFrame:
    stock = pd.read_csv(DB_DIR / "Stock.csv")
    inventory = pd.read_csv(DB_DIR / "inventory_stock.csv")
    return stock.merge(inventory, left_on="Item_id", right_on="item_id").drop(columns="item_id")


def render_source(source: str) -> None:
    if source == DATABASE_SOURCE:
        with st.expander(f"🗄️ {source}"):
            st.dataframe(load_database(), hide_index=True)
        return
    text = load_document(source)
    if text is None:
        st.markdown(f"⚠️ **{source}**: not a known document")
        return
    with st.expander(f"📄 {source}"):
        st.text(text)


# ── One run ──────────────────────────────────────────────────────────────

def render_run(run: dict) -> None:
    row, response = run["input"], run["response"]

    st.caption(f"Group: **{row['group']}** · Type: **{row['type']}**")
    if run.get("previous_question"):
        st.caption(f"↪️ Follow-up, asked after: *{run['previous_question']}*")

    st.subheader("User question")
    st.markdown(row["question"])

    with st.expander("System instructions"):
        if run.get("system_prompt"):
            st.text(run["system_prompt"])
        else:
            st.caption("Not stored in this run: showing the CURRENT system prompt from agent.py.")
            st.text(SYSTEM_PROMPT)

    st.subheader("Answer")
    with st.container(border=True):
        st.markdown(response["answer"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Answer type", response["answer_type"])
    c2.metric("Confidence", f"{response['confidence']:.0%}")
    c3.metric("Found in sources", "Yes" if response["found_in_sources"] else "No")
    c4.metric("Cost", f"${run['cost']:.4f}")

    st.subheader("Sources")
    if response["sources"]:
        for source in response["sources"]:
            render_source(source)
    else:
        st.caption("No sources cited.")

    if response.get("follow_up_questions"):
        st.markdown("**Suggested follow-ups:** " + " · ".join(response["follow_up_questions"]))

    if run.get("order"):
        st.success(f"Order placed: {json.dumps(run['order'])}")

    with st.expander(f"Tool calls ({len(run['tools'])})"):
        if not run["tools"]:
            st.caption("The agent answered without calling any tool.")
        for i, call in enumerate(run["tools"], 1):
            st.markdown(f"**{i}. `{call['name']}`**")
            st.json(call["arguments"])


def label_form(run: dict, index: int, saved: dict | None) -> dict | None:
    """Good / not good + comment, at the bottom. Returns the new label when saved."""
    st.divider()
    st.subheader("Your label")
    with st.form(key=f"label_{index}"):
        options = list(LABELS)
        label = st.radio(
            "Is this a good run?",
            options,
            format_func=LABELS.get,
            index=options.index(saved["label"]) if saved else None,
            horizontal=True,
        )
        comments = st.text_area("Comments", value=saved["comments"] if saved else "")
        if not st.form_submit_button("Save and go to next unlabeled", type="primary"):
            return None
    if label is None:
        st.error("Choose good or not good first.")
        return None
    return {
        "index": index,
        **run,
        "label": label,
        "comments": comments.strip(),
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def next_unlabeled(total: int, labels: dict, after: int) -> int | None:
    """The first unlabeled index after `after`, wrapping around to the start."""
    for i in list(range(after + 1, total)) + list(range(0, after + 1)):
        if i not in labels:
            return i
    return None


# ── Page ─────────────────────────────────────────────────────────────────

input_files = sorted(
    (p for p in EVALS_DIR.glob("evals_run_*.json") if not p.stem.endswith("_results")),
    reverse=True,  # newest run first
)
if not input_files:
    st.error(f"No evals_run_*.json files in {EVALS_DIR}. Run create_eval_data.ipynb first.")
    st.stop()

with st.sidebar:
    st.title("🏷️ Label evals")
    input_path = st.selectbox("Eval run", input_files, format_func=lambda p: p.name)

runs = load_runs(str(input_path))
labels = load_labels(input_path)

# Start each file at its first unlabeled run
if st.session_state.get("file") != input_path.name:
    st.session_state.file = input_path.name
    st.session_state.index = next_unlabeled(len(runs), labels, -1) or 0

# After a save: jump to the next run. Set here, before the "Go to run" widget
# is drawn, because Streamlit doesn't allow changing a widget's value after that.
if "pending_index" in st.session_state:
    st.session_state.index = st.session_state.pop("pending_index")
if toast := st.session_state.pop("pending_toast", None):
    st.toast(toast)

with st.sidebar:
    st.progress(len(labels) / len(runs), text=f"Labeled {len(labels)} of {len(runs)}")
    good = sum(1 for item in labels.values() if item["label"] == "good")
    st.caption(f"👍 {good} · 👎 {len(labels) - good}")
    st.caption(f"Saving to `{results_path(input_path).name}`")

    st.selectbox(
        "Go to run",
        range(len(runs)),
        key="index",
        format_func=lambda i: f"{'✅' if i in labels else '⬜'} {i + 1}. {runs[i]['input']['question'][:45]}",
    )

index = st.session_state.index
st.title(f"Run {index + 1} of {len(runs)}")

render_run(runs[index])

new_label = label_form(runs[index], index, labels.get(index))
if new_label:
    labels[index] = new_label
    save_labels(input_path, labels)
    following = next_unlabeled(len(runs), labels, index)
    if following is None:
        st.session_state.pending_toast = "All runs labeled! 🎉"
    else:
        st.session_state.pending_toast = f"Saved run {index + 1}"
        st.session_state.pending_index = following
    st.rerun()
