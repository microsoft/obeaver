"""
FoundryEngine — wraps the Microsoft Foundry Local SDK for on-device inference.

Uses the OpenAI-compatible endpoint exposed by the foundry-local daemon, so
generation is fully delegated to the managed runtime (auto hardware acceleration:
NPU › GPU › CPU).

Install requirements:
    pip install foundry-local-sdk openai
    # macOS: brew install microsoft/foundrylocal/foundrylocal
    # Windows: winget install Microsoft.FoundryLocal
"""

from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING, Iterator

from obeaver.engine_ort import GenerationConfig

if TYPE_CHECKING:
    from obeaver.tools import ChatResponse


class FoundryEngine:
    """
    Engine backed by Microsoft Foundry Local (foundry-local-sdk).

    The foundry daemon is started automatically on first use; the model is
    downloaded from the catalog and loaded into the runtime.  All inference
    goes through the OpenAI-compatible `/v1/chat/completions` endpoint the
    daemon exposes locally.

    This engine shares the same duck-typed interface as ``OrtEngine``
    (``model_name`` property + ``stream()`` method) so the two are
    interchangeable everywhere in obeaver.
    """

    def __init__(self, model_alias: str, device: str | None = None) -> None:
        try:
            from foundry_local import FoundryLocalManager
        except ImportError as exc:
            raise ImportError(
                "foundry-local-sdk is required for the 'foundry' engine.\n"
                "Install it with: pip install foundry-local-sdk\n"
                "Also install the Foundry Local daemon:\n"
                "  macOS:   brew install microsoft/foundrylocal/foundrylocal\n"
                "  Windows: winget install Microsoft.FoundryLocal"
            ) from exc

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the 'foundry' engine.\n"
                "Install it with: pip install openai"
            ) from exc

        self._alias = model_alias
        _device = self._map_device(device) or "CPU"

        manager = FoundryLocalManager(bootstrap=True)

        # Check if the model is already cached; if not, notify the user
        # and enable INFO logging so the SDK's tqdm progress bar is visible.
        import logging
        cached_aliases = {m.alias for m in manager.list_cached_models()}
        need_download = model_alias not in cached_aliases

        if need_download:
            # Resolve model info for a user-friendly message
            _info = manager.get_model_info(model_alias, device=_device)
            _size = f" ({_info.file_size_mb} MB)" if _info and _info.file_size_mb else ""
            print(
                f"Model '{model_alias}' is not cached locally. "
                f"Downloading{_size} for the first time, please wait..."
            )
            # Ensure the SDK logger is at INFO level so tqdm progress is shown
            _sdk_logger = logging.getLogger("foundry_local")
            _prev_level = _sdk_logger.level
            if _prev_level > logging.INFO or _prev_level == logging.NOTSET:
                _sdk_logger.setLevel(logging.INFO)
                if not _sdk_logger.handlers:
                    _sdk_logger.addHandler(logging.StreamHandler())

        model_info = manager.download_model(model_alias, device=_device)

        if need_download:
            print(f"Download complete. Loading model '{model_alias}'...")
            # Restore the previous log level
            _sdk_logger.setLevel(_prev_level)

        # Use a very long TTL (24 hours) to prevent auto-unload.
        # The default SDK TTL is only 600s (10 min) which causes cold-reload
        # penalties after short idle periods.
        model_info = manager.load_model(model_alias, device=_device, ttl=86400)

        self._model_id: str = model_info.id
        self._manager = manager  # keep reference for model listing

        # Use httpx with keep-alive connection pooling for lower-latency
        # streaming. The default openai client creates a new connection per
        # request; using a persistent transport reduces TTFT significantly.
        import httpx
        _transport = httpx.HTTPTransport(
            retries=0,
            http2=False,  # Foundry Local doesn't support h2
        )
        _http_client = httpx.Client(
            transport=_transport,
            timeout=httpx.Timeout(timeout=300.0, connect=10.0),
        )
        self._client = OpenAI(
            base_url=manager.endpoint,
            api_key=manager.api_key or "local",
            http_client=_http_client,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._alias

    def stream(
        self,
        messages: list[dict],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Yield decoded text fragments via the Foundry Local streaming API."""
        if config is None:
            config = GenerationConfig()

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def complete(
        self,
        messages: list[dict],
        config: GenerationConfig | None = None,
    ) -> str:
        """Non-streaming completion — returns the full text at once.

        Uses ``stream=False`` against the Foundry daemon which is significantly
        faster than consuming a streaming iterator and joining the fragments.
        """
        if config is None:
            config = GenerationConfig()

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        config: GenerationConfig | None = None,
    ) -> "ChatResponse":
        """
        Non-streaming generation with optional tool-calling support.

        Strategy:
        1. Try native OpenAI tools parameter (Foundry Local supports it).
        2. Parse functools[...] response format used by some FL models.
        3. Fall back to prompt injection + <tool_call> parsing (OrtEngine strategy).

        Reference:
        https://github.com/microsoft/Foundry-Local/blob/main/samples/python/functioncalling/fl_tools.ipynb
        """
        from obeaver.tools import (
            ChatResponse,
            ToolCall,
            inject_tools_into_messages,
            parse_tool_call,
        )

        if config is None:
            config = GenerationConfig()

        if tools:
            # Use near-zero temperature for deterministic tool-calling output
            # (fl_tools.ipynb uses temperature=0.00001)
            clone = GenerationConfig()
            clone.__dict__.update(config.__dict__)
            clone.temperature = 0.00001
            clone.top_p = 1.0

            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=clone.max_new_tokens,
                temperature=clone.temperature,
                top_p=clone.top_p,
                stream=False,
                tools=tools,
                tool_choice="auto",
            )
            choice = response.choices[0]

            # Path 1: native tool_calls field
            if choice.message.tool_calls:
                tcs = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in choice.message.tool_calls
                ]
                return ChatResponse(tool_calls=tcs, finish_reason="tool_calls")

            # Path 2: functools[...] inline format
            content = choice.message.content or ""
            if "functools" in content:
                try:
                    match = re.search(r"functools\[(.*)\]", content, re.DOTALL)
                    if match:
                        funcs = json.loads("[" + match.group(1) + "]")
                        tcs = [
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:12]}",
                                name=f["name"],
                                arguments=f.get("arguments", {}),
                            )
                            for f in funcs
                        ]
                        return ChatResponse(tool_calls=tcs, finish_reason="tool_calls")
                except (json.JSONDecodeError, KeyError, re.error):
                    pass

        # Path 3: prompt injection fallback (same as OrtEngine)
        effective_messages = inject_tools_into_messages(messages, tools) if tools else messages

        if tools:
            clone = GenerationConfig()
            clone.__dict__.update(config.__dict__)
            clone.temperature = 0.0
            config = clone

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=effective_messages,  # type: ignore[arg-type]
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stream=False,
        )
        choice = response.choices[0]
        full_text = choice.message.content or ""

        if tools:
            tc = parse_tool_call(full_text)
            if tc:
                return ChatResponse(tool_calls=[tc], finish_reason="tool_calls")

        return ChatResponse(
            content=full_text,
            finish_reason=choice.finish_reason or "stop",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_device(device: str | None) -> str | None:
        """Normalise execution-provider / device strings for Foundry.

        Returns uppercase values that match ``foundry_local.models.DeviceType``
        (e.g. ``'CPU'``, ``'GPU'``, ``'NPU'``).
        """
        if device is None:
            return None
        mapping = {
            "cpu": "CPU",
            "gpu": "GPU",
            "cuda": "GPU",
            "npu": "NPU",
            "dml": "GPU",
        }
        return mapping.get(device.lower(), None)
