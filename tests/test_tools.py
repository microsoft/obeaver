"""
Unit tests for ofoundry tool-calling utilities (no model required).

Run with:
  pytest tests/test_tools.py -v
"""

from __future__ import annotations

import json

import pytest

from ofoundry.tools import (
    ChatResponse,
    ToolCall,
    build_tool_result_message,
    inject_tools_into_messages,
    parse_tool_call,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------


def test_parse_tagged_block() -> None:
    text = """<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
</tool_call>"""
    tc = parse_tool_call(text)
    assert tc is not None
    assert tc.name == "get_weather"
    assert tc.arguments["city"] == "Paris"
    assert tc.arguments["unit"] == "celsius"
    assert tc.id.startswith("call_")


def test_parse_tagged_block_with_surrounding_text() -> None:
    text = "Sure, I'll check the weather!\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>\n"
    tc = parse_tool_call(text)
    assert tc is not None
    assert tc.name == "get_weather"
    assert tc.arguments["city"] == "Tokyo"


def test_parse_bare_json() -> None:
    text = '{"name": "calculator", "arguments": {"expression": "2 + 2"}}'
    tc = parse_tool_call(text)
    assert tc is not None
    assert tc.name == "calculator"
    assert tc.arguments["expression"] == "2 + 2"


def test_parse_arguments_as_string() -> None:
    """Some models serialise arguments as a JSON string."""
    inner = json.dumps({"city": "Berlin"})
    text = f'<tool_call>{{"name": "get_weather", "arguments": {json.dumps(inner)}}}</tool_call>'
    tc = parse_tool_call(text)
    assert tc is not None
    assert tc.name == "get_weather"
    assert tc.arguments["city"] == "Berlin"


def test_parse_returns_none_for_plain_text() -> None:
    assert parse_tool_call("Hello, how can I help you?") is None


def test_parse_returns_none_for_partial_json() -> None:
    assert parse_tool_call('{"name": "foo"}') is None   # missing "arguments"


# ---------------------------------------------------------------------------
# ToolCall.to_openai_dict
# ---------------------------------------------------------------------------


def test_to_openai_dict() -> None:
    tc = ToolCall(id="call_abc123", name="get_weather", arguments={"city": "Rome"})
    d = tc.to_openai_dict()
    assert d["id"] == "call_abc123"
    assert d["type"] == "function"
    assert d["function"]["name"] == "get_weather"
    args = json.loads(d["function"]["arguments"])
    assert args["city"] == "Rome"


# ---------------------------------------------------------------------------
# inject_tools_into_messages
# ---------------------------------------------------------------------------


def test_inject_into_existing_system_message() -> None:
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    result = inject_tools_into_messages(msgs, SAMPLE_TOOLS)

    assert result[0]["role"] == "system"
    assert "You are helpful." in result[0]["content"]
    assert "get_weather" in result[0]["content"]
    assert result[1]["role"] == "user"
    # Original untouched
    assert msgs[0]["content"] == "You are helpful."


def test_inject_prepends_system_when_absent() -> None:
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    result = inject_tools_into_messages(msgs, SAMPLE_TOOLS)

    assert result[0]["role"] == "system"
    assert "calculator" in result[0]["content"]
    assert result[1]["role"] == "user"


def test_inject_does_not_mutate_original() -> None:
    original = [{"role": "user", "content": "Hi"}]
    inject_tools_into_messages(original, SAMPLE_TOOLS)
    assert len(original) == 1   # unchanged


# ---------------------------------------------------------------------------
# build_tool_result_message
# ---------------------------------------------------------------------------


def test_build_tool_result_message_string() -> None:
    tc = ToolCall(id="call_xyz", name="get_weather", arguments={})
    msg = build_tool_result_message(tc, "Sunny, 22°C")
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_xyz"
    assert msg["content"] == "Sunny, 22°C"


def test_build_tool_result_message_dict() -> None:
    tc = ToolCall(id="call_xyz", name="get_weather", arguments={})
    msg = build_tool_result_message(tc, {"temperature": 22, "condition": "sunny"})
    assert msg["role"] == "tool"
    parsed = json.loads(msg["content"])
    assert parsed["temperature"] == 22


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------


def test_chat_response_defaults() -> None:
    resp = ChatResponse(content="Hello!")
    assert resp.content == "Hello!"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"


def test_chat_response_with_tool_calls() -> None:
    tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    resp = ChatResponse(tool_calls=[tc], finish_reason="tool_calls")
    assert resp.content is None
    assert len(resp.tool_calls) == 1
    assert resp.finish_reason == "tool_calls"
