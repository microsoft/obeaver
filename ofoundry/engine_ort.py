"""
Engine — wraps onnxruntime-genai for streaming text generation.

Core pattern follows:
  https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import onnxruntime_genai as og  # type: ignore

if TYPE_CHECKING:
    from ofoundry.tools import ChatResponse


@dataclass
class GenerationConfig:
    """Sampling / search options forwarded to onnxruntime-genai."""

    max_length: int = 4096
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True


class OrtEngine:
    """
    Thin wrapper around onnxruntime-genai that supports:
      - Loading a model from a local ONNX GenAI directory
      - Stateless token streaming (one full turn at a time)
      - Multi-turn chat via a persistent generator

    Thread-safety: each call to `stream()` holds a lock so that
    concurrent requests are serialised rather than corrupted.
    """

    def __init__(self, model_path: str | Path, execution_provider: str = "cpu") -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Auto-detect the subdirectory that actually contains genai_config.json
        model_path = self._resolve_model_dir(model_path)

        # Patch unsupported model types so onnxruntime-genai can load them.
        # e.g. Qwen3-VL uses "qwen3vl" but ORT GenAI only knows "qwen2_5_vl";
        # the two architectures are ONNX-compatible (Qwen3 adds QK-norm which
        # is already baked into the exported ONNX graph).
        self._patch_model_type(model_path)

        self._lock = threading.Lock()
        self._model_path = model_path
        self._ep = execution_provider.lower()

        # Load model & tokenizer (expensive — done once at startup)
        self._model: og.Model = og.Model(str(model_path))
        self._tokenizer: og.Tokenizer = og.Tokenizer(self._model)
        self._token_stream: og.TokenizerStream = self._tokenizer.create_stream()
        self._mm_processor = self._init_multimodal_processor()
        self._model_type: str = self._detect_model_type(model_path)
        self._eos_token_ids: set[int] = self._load_eos_token_ids(model_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Model type aliases that onnxruntime-genai does not yet recognise.
    # Map unsupported type → compatible supported type.
    _MODEL_TYPE_ALIASES: dict[str, str] = {
        "qwen3vl": "qwen2_5_vl",
        "qwen3_vl": "qwen2_5_vl",
    }

    @staticmethod
    def _patch_model_type(model_dir: Path) -> None:
        """Rewrite ``model.type`` in genai_config.json when the value is not
        supported by the installed onnxruntime-genai.

        This is non-destructive: we only patch types we have an explicit
        alias for, and we leave all other fields untouched.
        """
        config_path = model_dir / "genai_config.json"
        if not config_path.exists():
            return
        with open(config_path) as f:
            data = json.load(f)
        current_type = data.get("model", {}).get("type", "")
        replacement = OrtEngine._MODEL_TYPE_ALIASES.get(current_type)
        if replacement and current_type != replacement:
            data["model"]["type"] = replacement
            with open(config_path, "w") as f:
                json.dump(data, f, indent=4)

    @staticmethod
    def _resolve_model_dir(path: Path) -> Path:
        """Return the directory that contains genai_config.json.

        If *path* itself has it, return *path*.  Otherwise do a breadth-first
        search one level at a time so that tarballs like
          models/phi3-mini-int4/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/
        are found automatically without requiring the user to know the exact
        sub-path.
        """
        if (path / "genai_config.json").exists():
            return path
        # BFS up to 4 directory levels deep
        candidates: list[Path] = [path]
        for _ in range(4):
            next_level: list[Path] = []
            for parent in candidates:
                for child in sorted(parent.iterdir()):
                    if child.is_dir():
                        if (child / "genai_config.json").exists():
                            return child
                        next_level.append(child)
            candidates = next_level
        raise FileNotFoundError(
            f"Could not find genai_config.json under {path}. "
            "Make sure the path points to a valid ONNX GenAI model directory."
        )

    @staticmethod
    def _detect_model_type(model_dir: Path) -> str:
        """Read model_type from config.json (e.g. 'phi3', 'qwen3')."""
        config_json = model_dir / "config.json"
        if config_json.exists():
            try:
                with open(config_json) as f:
                    data = json.load(f)
                return data.get("model_type", "").lower()
            except (json.JSONDecodeError, OSError):
                pass
        # Fallback: check genai_config.json for model.type
        genai_json = model_dir / "genai_config.json"
        if genai_json.exists():
            try:
                with open(genai_json) as f:
                    data = json.load(f)
                return data.get("model", {}).get("type", "").lower()
            except (json.JSONDecodeError, OSError):
                pass
        return ""

    @staticmethod
    def _load_eos_token_ids(model_dir: Path) -> set[int]:
        """Load EOS token IDs from genai_config.json."""
        genai_json = model_dir / "genai_config.json"
        if genai_json.exists():
            try:
                with open(genai_json) as f:
                    data = json.load(f)
                eos = data.get("model", {}).get("eos_token_id", [])
                if isinstance(eos, int):
                    return {eos}
                if isinstance(eos, list):
                    return set(eos)
            except (json.JSONDecodeError, OSError):
                pass
        return set()

    def _init_multimodal_processor(self):
        """Initialize multimodal processor if the model supports it.

        Qwen VL models expose ``create_multimodal_processor()``. Text-only
        models generally do not, so we return None in that case.
        """
        try:
            return self._model.create_multimodal_processor()
        except (AttributeError, TypeError, Exception):
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_path.name

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def supports_multimodal(self) -> bool:
        return self._mm_processor is not None

    def stream(
        self,
        messages: list[dict],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """
        Yield decoded text fragments for *messages* (OpenAI message list).

        The generator is stateless — a fresh og.Generator is created for
        each call so that concurrent calls do not share state.

        For VL (multimodal) models this delegates to :meth:`stream_multimodal`
        because VL models require the multimodal processor path even for
        text-only inputs.
        """
        # VL models must always go through the processor path
        if self._mm_processor is not None:
            yield from self.stream_multimodal(
                messages=messages, image_path=None, config=config,
            )
            return

        if config is None:
            config = GenerationConfig()

        prompt = self._apply_chat_template(messages)
        tokens = self._tokenizer.encode(prompt)

        with self._lock:
            params = og.GeneratorParams(self._model)
            params.set_search_options(**self._build_search_options(config))

            generator = og.Generator(self._model, params)
            generator.append_tokens(tokens)
            stream = self._tokenizer.create_stream()

            try:
                while not generator.is_done():
                    generator.generate_next_token()
                    new_token = generator.get_next_tokens()[0]
                    # Stop early if we hit an EOS token
                    if new_token in self._eos_token_ids:
                        break
                    fragment = stream.decode(new_token)
                    if not fragment:
                        continue
                    # Filter out leaked special tokens
                    if fragment.strip() in _SPECIAL_TOKEN_STRINGS:
                        break
                    yield fragment
            except KeyboardInterrupt:
                pass
            finally:
                # Explicitly delete to free native resources promptly
                del generator

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        config: GenerationConfig | None = None,
    ) -> "ChatResponse":
        """
        Non-streaming generation with optional tool-calling support.

        For the ORT engine, tool calling uses prompt injection:

        1. Tool definitions are serialised as JSON Schema and appended to the
           system message (see :func:`~ofoundry.tools.inject_tools_into_messages`).
        2. The model's full response is collected.
        3. The output is scanned for a ``<tool_call>`` block.

        When a tool call is detected a :class:`~ofoundry.tools.ChatResponse`
        with ``tool_calls`` is returned; otherwise ``content`` is populated.

        References
        ----------
        * https://github.com/microsoft/onnxruntime-genai/blob/main/docs/ConstrainedDecoding.md
        * https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python
        """
        from ofoundry.tools import (
            ChatResponse,
            inject_tools_into_messages,
            parse_tool_call,
        )

        effective = inject_tools_into_messages(messages, tools) if tools else messages

        # Force greedy decoding when tools are present so the model
        # outputs structured JSON reliably rather than sampling noise.
        if tools:
            clone = GenerationConfig()
            if config is not None:
                clone.__dict__.update(config.__dict__)
            clone.temperature = 0.0
            clone.do_sample = False
            config = clone

        full_text = "".join(self.stream(effective, config))

        if tools:
            tc = parse_tool_call(full_text)
            if tc:
                return ChatResponse(tool_calls=[tc], finish_reason="tool_calls")

        return ChatResponse(content=full_text, finish_reason="stop")

    def stream_multimodal(
        self,
        messages: list[dict],
        image_path: str | None,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Yield decoded text fragments for VL models (text or image+text).

        This follows the olive-recipes Qwen3-VL inference pattern:
        - apply_chat_template(JSON, add_generation_prompt=True)
        - inputs = processor(prompt, images=...)
        - generator.set_inputs(inputs)
        """
        if self._mm_processor is None:
            raise RuntimeError(
                "Current model does not expose a multimodal processor. "
                "Use a VL model converted with vision/embedding components."
            )

        if config is None:
            config = GenerationConfig()

        images = None
        if image_path:
            images = og.Images.open(image_path)

        full_prompt = self._tokenizer.apply_chat_template(
            json.dumps(messages),
            add_generation_prompt=True,
        )

        inputs = self._mm_processor(full_prompt, images=images)

        with self._lock:
            params = og.GeneratorParams(self._model)
            params.set_search_options(**self._build_search_options(config))

            generator = og.Generator(self._model, params)
            generator.set_inputs(inputs)
            stream = self._mm_processor.create_stream()

            try:
                while not generator.is_done():
                    generator.generate_next_token()
                    new_token = generator.get_next_tokens()[0]
                    if new_token in self._eos_token_ids:
                        break
                    fragment = stream.decode(new_token)
                    if not fragment:
                        continue
                    if fragment.strip() in _SPECIAL_TOKEN_STRINGS:
                        break
                    yield fragment
            except KeyboardInterrupt:
                pass
            finally:
                del generator

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """
        Apply the model's chat template if available, otherwise fall back
        to a model-specific or generic format.

        Priority:
        1. Model-specific formatter (Qwen3, etc.) — known to produce
           correct results for the model architecture.
        2. og.Tokenizer.apply_chat_template (requires JSON string input
           + chat_template.jinja in model dir).
        3. Phi-3 / generic instruct fallback.
        """
        # 1. Model-specific formatters take priority
        #    (but NOT for VL models — they use the tokenizer's built-in
        #    chat template via apply_chat_template to handle multimodal
        #    content correctly)
        if self._model_type.startswith("qwen") and self._mm_processor is None:
            return _format_messages_qwen3(messages)

        # 2. Try the tokenizer's built-in chat template (JSON string API)
        try:
            prompt = self._tokenizer.apply_chat_template(
                json.dumps(messages)
            )
            return prompt
        except (AttributeError, TypeError, Exception):
            pass

        # 3. Fallback: Phi-3 / generic instruct formatting
        return _format_messages_fallback(messages)

    @staticmethod
    def _build_search_options(config: GenerationConfig) -> dict:
        opts: dict = {
            "max_length": config.max_length,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "do_sample": config.do_sample,
        }
        return opts


# ---------------------------------------------------------------------------
# Special tokens that should stop or be filtered from streamed output
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_STRINGS: set[str] = {
    "<|im_end|>", "<|im_start|>", "<|endoftext|>",      # Qwen / ChatML
    "<|end|>", "<|user|>", "<|assistant|>", "<|system|>",  # Phi-3
    "</s>", "<s>",                                          # Generic
}


# ---------------------------------------------------------------------------
# Fallback chat-template formatters
# ---------------------------------------------------------------------------

def _format_messages_qwen3(messages: list[dict]) -> str:
    """Qwen3 ChatML format with /no_think prefix for user messages."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n/no_think {content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _format_messages_fallback(messages: list[dict]) -> str:
    """Phi-3 instruct format used when the tokenizer has no apply_chat_template."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}<|end|>")
        elif role == "user":
            parts.append(f"<|user|>\n{content}<|end|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}<|end|>")
    parts.append("<|assistant|>")
    return "\n".join(parts)
