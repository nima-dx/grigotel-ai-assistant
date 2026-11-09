import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TypeVar, overload

from pydantic import BaseModel
from pydantic_ai._agent_graph import GraphAgentState
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class TraceData:
    """Raw data fetched from Logfire for a single agent run trace."""
    all_messages: list[dict]
    input_tokens: int = 0
    output_tokens: int = 0


AGENT_SPAN_FILTER = "(span_name = 'agent run' OR span_name LIKE 'invoke_agent %')"

# Newer Logfire versions require min_timestamp on every query: how far back to search.
DEFAULT_LOOKBACK = timedelta(days=30)


def _min_timestamp(since: datetime | None) -> datetime:
    return since or datetime.now(timezone.utc) - DEFAULT_LOOKBACK


def _parse_messages(value) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value)
    return value or []


def _is_agent_span(row: dict) -> bool:
    name = row.get('span_name') or ''
    return name == 'agent run' or name.startswith('invoke_agent ')


def _rows_to_trace_data(rows: list[dict]) -> TraceData:
    """Combine the spans of one trace (oldest first) into one TraceData.

    Two layouts are supported:

    1. PydanticAI agents: each agent-run span holds `pydantic_ai.all_messages` and
       the usage of its own run. A grouped session trace has one span per run, so
       tokens are SUMMED over these spans and messages come from the latest one
       (with message_history, its all_messages already includes earlier turns).

    2. Hand-written agents on the OpenAI SDK (logfire.instrument_openai): the agent
       span (if any) is empty, and every LLM call is a `chat` span holding
       `gen_ai.input.messages` / `gen_ai.output.messages` and its own usage. Tokens
       are summed over the chat spans; the conversation is the latest chat span's
       system instructions + input + output.
    """
    agent_rows = [r for r in rows if _is_agent_span(r) and r.get('all_messages')]
    if agent_rows:
        return TraceData(
            all_messages=_parse_messages(agent_rows[-1]['all_messages']),
            input_tokens=sum(int(r.get('input_tokens') or 0) for r in agent_rows),
            output_tokens=sum(int(r.get('output_tokens') or 0) for r in agent_rows),
        )

    chat_rows = [r for r in rows if not _is_agent_span(r) and r.get('input_tokens') is not None]
    if not chat_rows:
        return TraceData(all_messages=[])

    last = chat_rows[-1]
    system = _parse_messages(last.get('system_instructions'))
    messages = [{'role': 'system', 'parts': system}] if system else []
    messages += _parse_messages(last.get('input_messages'))
    messages += _parse_messages(last.get('output_messages'))
    return TraceData(
        all_messages=messages,
        input_tokens=sum(int(r.get('input_tokens') or 0) for r in chat_rows),
        output_tokens=sum(int(r.get('output_tokens') or 0) for r in chat_rows),
    )


TRACE_COLUMNS = """
            trace_id,
            span_name,
            attributes->'pydantic_ai.all_messages' as all_messages,
            attributes->'gen_ai.system_instructions' as system_instructions,
            attributes->'gen_ai.input.messages' as input_messages,
            attributes->'gen_ai.output.messages' as output_messages,
            attributes->>'gen_ai.usage.input_tokens' as input_tokens,
            attributes->>'gen_ai.usage.output_tokens' as output_tokens"""

# Agent-run spans, plus LLM-call spans that carry token usage
TRACE_SPAN_FILTER = f"({AGENT_SPAN_FILTER} OR attributes->>'gen_ai.usage.input_tokens' IS NOT NULL)"


def fetch_trace(trace_id: str, client: object, since: datetime | None = None) -> TraceData:
    """Fetch raw OTel data for a trace from Logfire (tokens summed over the whole trace).

    Args:
        trace_id: The trace ID to query.
        client: LogfireQueryClient instance.
        since: Only search spans newer than this (default: last 30 days).
    """
    result = client.query_json_rows(
        min_timestamp=_min_timestamp(since),
        sql=f"""
        SELECT {TRACE_COLUMNS}
        FROM records
        WHERE trace_id = '{trace_id}'
          AND {TRACE_SPAN_FILTER}
        ORDER BY start_timestamp ASC
        """
    )
    trace = _rows_to_trace_data(result['rows'])
    if not trace.all_messages:
        raise ValueError(f'No agent run or LLM call spans found for trace_id={trace_id}')
    return trace


def fetch_traces(
    trace_ids: list[str], client: object, since: datetime | None = None
) -> dict[str, TraceData]:
    """Fetch raw OTel data for multiple traces in a single query (tokens summed per trace).

    Args:
        trace_ids: List of trace IDs to query.
        client: LogfireQueryClient instance.
        since: Only search spans newer than this (default: last 30 days).

    Returns:
        Dict mapping trace_id to TraceData. Missing traces are omitted.
    """
    if not trace_ids:
        return {}

    ids_csv = ', '.join(f"'{tid}'" for tid in trace_ids)
    result = client.query_json_rows(
        min_timestamp=_min_timestamp(since),
        sql=f"""
        SELECT {TRACE_COLUMNS}
        FROM records
        WHERE trace_id IN ({ids_csv})
          AND {TRACE_SPAN_FILTER}
        ORDER BY start_timestamp ASC
        """
    )

    rows_by_trace: dict[str, list[dict]] = {}
    for row in result['rows']:
        rows_by_trace.setdefault(row['trace_id'], []).append(row)

    traces = {tid: _rows_to_trace_data(rows) for tid, rows in rows_by_trace.items()}
    return {tid: t for tid, t in traces.items() if t.all_messages}


def _extract_output(messages: list[ModelMessage], output_type: type[T] | None = None) -> str | dict | T:
    """Extract the final output from a message list.

    For text output: returns the concatenated text from the last ModelResponse.
    For structured output: the last ModelResponse is a tool call (e.g. final_result),
    so we return its args dict — or validate it against output_type if provided.
    """
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            text_parts = [p.content for p in msg.parts if isinstance(p, TextPart)]
            if text_parts:
                return ''.join(text_parts)
            tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
            if tool_calls:
                args = tool_calls[-1].args
                if isinstance(args, str):
                    args = json.loads(args)
                if output_type is not None:
                    return output_type.model_validate(args)
                return args or {}
            break
    return ''


def otel_to_model_messages(otel_messages: list[dict]) -> list[ModelMessage]:
    """Convert OTel-format messages (from Logfire) back to PydanticAI ModelMessage objects.

    OTel format uses role/parts/type, PydanticAI uses kind/parts/part_kind.
    Consecutive system/user messages are merged into a single ModelRequest.
    """
    messages: list[ModelMessage] = []
    pending_request_parts = []
    tool_names: dict[str, str] = {}  # tool_call_id -> tool name, for results without a name

    for msg in otel_messages:
        role = msg['role']
        parts = msg.get('parts', [])

        if role == 'assistant':
            if pending_request_parts:
                messages.append(ModelRequest(parts=pending_request_parts))
                pending_request_parts = []

            response_parts = []
            for part in parts:
                ptype = part['type']
                if ptype == 'text':
                    response_parts.append(TextPart(content=part.get('content', '')))
                elif ptype == 'tool_call':
                    args = part.get('arguments', {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_names[part.get('id', '')] = part['name']
                    response_parts.append(ToolCallPart(
                        tool_name=part['name'],
                        args=args,
                        tool_call_id=part.get('id', ''),
                    ))
                elif ptype == 'thinking':
                    response_parts.append(ThinkingPart(content=part.get('content', '')))

            messages.append(ModelResponse(
                parts=response_parts,
                finish_reason=msg.get('finish_reason'),
            ))

        elif role == 'system':
            for part in parts:
                if part['type'] == 'text':
                    pending_request_parts.append(
                        SystemPromptPart(content=part.get('content', ''))
                    )

        elif role in ('user', 'tool'):
            for part in parts:
                ptype = part['type']
                if ptype == 'text':
                    pending_request_parts.append(
                        UserPromptPart(content=part.get('content', ''))
                    )
                elif ptype == 'tool_call_response':
                    pending_request_parts.append(ToolReturnPart(
                        tool_name=part.get('name') or tool_names.get(part.get('id', ''), ''),
                        content=part.get('result', part.get('response', '')),
                        tool_call_id=part.get('id', ''),
                    ))

    if pending_request_parts:
        messages.append(ModelRequest(parts=pending_request_parts))

    return messages


@overload
def trace_to_run_result(trace: TraceData) -> AgentRunResult[str | dict]: ...
@overload
def trace_to_run_result(trace: TraceData, output_type: type[T]) -> AgentRunResult[T]: ...

def trace_to_run_result(trace: TraceData, output_type: type[T] | None = None) -> AgentRunResult:
    """Convert fetched TraceData into an AgentRunResult.

    Args:
        trace: TraceData from fetch_trace().
        output_type: Optional Pydantic BaseModel class to validate the output against.
    """
    messages = otel_to_model_messages(trace.all_messages)
    output = _extract_output(messages, output_type)

    state = GraphAgentState(
        message_history=messages,
        usage=RunUsage(
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
        ),
    )

    return AgentRunResult(output=output, _state=state)
