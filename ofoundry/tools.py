"""
Tool-calling utilities shared by OrtEngine and FoundryEngine.

Strategy
--------
ORT engine  (onnxruntime-genai)
    Tools are serialised as JSON Schema inside the system prompt.
    The model is expected to reply with a ``<tool_call>`` block when it
    decides to call a function::

        <tool_call>
        {"name": "get_weather", "arguments": {"city": "Paris"}}
        </tool_call>

    ``parse_tool_call()`` extracts and validates that JSON.
    Reference: https://github.com/microsoft/onnxruntime-genai/blob/main/docs/ConstrainedDecoding.md

Foundry Local engine
    Tools are forwarded natively to the Foundry Local daemon via the
    standard OpenAI ``tools`` parameter; the daemon handles formatting and
    parsing.  ``parse_tool_call()`` is not needed on this path.
    Reference: https://github.com/microsoft/Foundry-Local/blob/main/samples/python/functioncalling/fl_tools.ipynb
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool / function call emitted by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_dict(self) -> dict:
        """Serialise to the OpenAI wire format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class ChatResponse:
    """Unified structured response returned by ``engine.chat()``."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# System-prompt injection
# ---------------------------------------------------------------------------

_TOOL_SYSTEM_ADDENDUM = """\
# Tool Calling

You have access to external tools listed below. When the user's request
requires calling a tool, you MUST output ONLY a JSON object in EXACTLY
this format and nothing else:

<tool_call>
{{"name": "<function_name>", "arguments": {{<key>: <value>}}}}
</tool_call>

Rules:
- Output the <tool_call> block alone. Do NOT add any explanation before or after.
- Use valid JSON inside the block.
- "arguments" must be a JSON object (dict), not a string.
- If no tool is needed, answer normally without a <tool_call> block.

Example — if the user asks "What is the weather in London?" and a
`get_weather(city: str)` tool is available, respond with:

<tool_call>
{{"name": "get_weather", "arguments": {{"city": "London"}}}}
</tool_call>

Available tools (JSON Schema):
{tools_json}"""


def format_tools_for_system(tools: list[dict]) -> str:
    """Return the tool-calling addendum to append to the system message."""
    return _TOOL_SYSTEM_ADDENDUM.format(
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2)
    )


def inject_tools_into_messages(
    messages: list[dict],
    tools: list[dict],
) -> list[dict]:
    """
    Return a **new** message list with the tool-calling instruction block
    either appended to the existing system message or prepended as a new
    system message.  The original list is never mutated.
    """
    addendum = format_tools_for_system(tools)
    msgs = [dict(m) for m in messages]          # shallow-copy each message
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = msgs[0]["content"] + "\n\n" + addendum
    else:
        msgs.insert(0, {"role": "system", "content": addendum})
    return msgs


# ---------------------------------------------------------------------------
# Tool-call parser
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
# Phi-3 native format
_PHI3_TOOL_RE = re.compile(
    r"<\|function_calls\|>\s*(\{.*?\})\s*<\|/function_calls\|>",
    re.DOTALL,
)
# Markdown code block: ```json ... ```
_CODE_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)
# <functioncall> ... </functioncall> (some Mistral-style models)
_FUNCTIONCALL_RE = re.compile(
    r"<functioncall>\s*(\{.*?\})\s*</functioncall>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> ToolCall | None:
    """
    Extract a :class:`ToolCall` from raw model output.

    Tries these formats in order:
      1. ``<tool_call>{...}</tool_call>``
      2. ``<|function_calls|>{...}<|/function_calls|>`` (Phi-3 native)
      3. ``<functioncall>{...}</functioncall>`` (Mistral-style)
      4. Markdown code block  ` ```json {...} ``` `
      5. ``{"function_call": {"name": ..., "arguments": ...}}``
      6. Bare JSON with ``name`` + ``arguments`` keys

    Returns ``None`` when no tool call is detected.
    """
    for pattern in (_TOOL_CALL_RE, _PHI3_TOOL_RE, _FUNCTIONCALL_RE, _CODE_BLOCK_RE):
        m = pattern.search(text)
        if m:
            try:
                parsed = json.loads(m.group(1))
                result = _try_make_tool_call(parsed)
                if result:
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

    # Bare JSON anywhere in the text — scan each '{' position and try to
    # parse a complete JSON object (handles nested braces correctly).
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        for end in range(len(text), i, -1):
            if text[end - 1] != "}":
                continue
            try:
                parsed = json.loads(text[i:end])
                result = _try_make_tool_call(parsed)
                if result:
                    return result
                break  # valid JSON but not a tool call — stop extending
            except json.JSONDecodeError:
                continue

    return None


def _try_make_tool_call(obj: dict) -> ToolCall | None:
    """Return a ToolCall if *obj* looks like a function call, else None."""
    # Format: {"name": ..., "arguments": ...}
    if "name" in obj and "arguments" in obj:
        return _make_tool_call(obj)

    # Format: {"function_call": {"name": ..., "arguments": ...}}
    if "function_call" in obj:
        inner = obj["function_call"]
        if isinstance(inner, dict) and "name" in inner:
            return _make_tool_call(inner)

    # Format: {"tool": "name", "parameters": {...}} (some models)
    if "tool" in obj and isinstance(obj["tool"], str):
        return _make_tool_call({"name": obj["tool"], "arguments": obj.get("parameters", {})})

    return None


def _make_tool_call(obj: dict) -> ToolCall:
    args = obj.get("arguments", {})
    # Some models serialise arguments as a JSON string
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        name=str(obj["name"]),
        arguments=args,
    )


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def build_tool_result_message(tool_call: ToolCall, result: Any) -> dict:
    """
    Build the ``tool`` role message to append to the conversation after
    executing a tool call.
    """
    content = (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False)
    )
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": content,
    }
