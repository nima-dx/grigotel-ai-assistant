"""
GrigoTel AI Assistant - Observability (Pydantic Logfire)

Call setup_logfire() once per process, before the agent makes any OpenAI calls.
Locally the SDK reads the write token from .logfire/logfire_credentials.json
(created by the Logfire CLI, git-ignored). In production set LOGFIRE_TOKEN instead.
Without a token nothing is sent, so the app and tests still work offline.

Location: src/grigotel_ai_assistant/observability.py
"""

import logfire

SERVICE_NAME = "grigotel-ai-assistant"

_configured = False


def setup_logfire() -> None:
    """Configure Logfire and instrument the OpenAI SDK (safe to call more than once)."""
    global _configured
    if _configured:
        return
    logfire.configure(service_name=SERVICE_NAME, send_to_logfire="if-token-present")
    logfire.instrument_openai()  # one span per Responses API call: model, tokens, messages
    _configured = True
