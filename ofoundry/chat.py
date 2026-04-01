"""
Interactive CLI chat session — mirrors the loop in:
  https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py

Usage (standalone, without the CLI wrapper):
    python -m ofoundry.chat -m ./path/to/model
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from ofoundry.brand import (
    ARG_STYLE,
    COMMAND_STYLE,
    ERROR_STYLE,
    MUTED_STYLE,
    PATH_STYLE,
    PROMPT_ASSISTANT_STYLE,
    TEXT_STYLE,
    VALUE_STYLE,
    markup,
)
from ofoundry.engine_ort import GenerationConfig

console = Console()


def run_chat(
    model_path: str | Path,
    engine_type: str = "foundry",
    system_prompt: str = "You are a helpful AI assistant.",
    max_length: int = 4096,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    show_timings: bool = False,
    execution_provider: str = "cpu",
) -> None:
    """
    Start an interactive multi-turn chat session in the terminal.

    Parameters
    ----------
    engine_type:
        ``"foundry"`` (default) — uses Foundry Local daemon, ``model_path``
        is treated as a catalog alias (e.g. ``"phi-3.5-mini"``).
        ``"ort"`` — uses onnxruntime-genai, ``model_path`` must be a local
        directory containing ``genai_config.json``.
    """
    if engine_type == "foundry":
        from ofoundry.engine_foundrylocal import FoundryEngine
        console.print(
            f"\n{markup('ofoundry', COMMAND_STYLE)} "
            f"{markup('Foundry Local engine', VALUE_STYLE)} · "
            f"{markup('model alias', TEXT_STYLE)} {markup(model_path, PATH_STYLE)}..."
        )
        engine: object = FoundryEngine(
            model_alias=str(model_path),
            device=execution_provider or "cpu",
        )
    else:
        from ofoundry.engine_ort import OrtEngine
        console.print(
            f"\n{markup('ofoundry', COMMAND_STYLE)} "
            f"{markup('ORT engine', VALUE_STYLE)} · "
            f"{markup('loading model from', TEXT_STYLE)} {markup(model_path, PATH_STYLE)}..."
        )
        engine = OrtEngine(model_path=model_path, execution_provider=execution_provider)
    console.print(f"{markup('✓ Model ready:', VALUE_STYLE)} {markup(engine.model_name, TEXT_STYLE)}\n")
    console.print(
        f"{markup('Type your message and press ', MUTED_STYLE)}{markup('Enter', COMMAND_STYLE)}"
        f"{markup('. Type ', MUTED_STYLE)}{markup('/bye', COMMAND_STYLE)}"
        f"{markup(' or press ', MUTED_STYLE)}{markup('Ctrl+Q', COMMAND_STYLE)}"
        f"{markup(' to exit.', MUTED_STYLE)}\n"
    )
    if engine_type == "ort" and getattr(engine, "supports_multimodal", False):
        console.print(
            f"{markup('VL input supported: use ', MUTED_STYLE)}"
            f"{markup('image:/path/to/image.jpg your prompt', COMMAND_STYLE)}"
            f"{markup(' (or only ', MUTED_STYLE)}{markup('image:/path/to/image.jpg', COMMAND_STYLE)}"
            f"{markup(').', MUTED_STYLE)}\n"
        )

    config = GenerationConfig(
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    # Conversation history (keeps multi-turn context)
    history: list[dict] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    while True:
        # ---- Prompt user ----
        try:
            user_input = Prompt.ask(f"\n[{ARG_STYLE}]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n{markup('Goodbye.', MUTED_STYLE)}")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in {"/bye", "/exit", "/quit"}:
            console.print(markup("Goodbye.", MUTED_STYLE))
            break

        image_path: str | None = None
        message_content: str | list[dict]
        message_content = user_input
        prompt_text = user_input

        # VL syntax (compatible with olive-recipes interactive example):
        #   image:/path/to/image.jpg Describe this image
        #   image:/path/to/image.jpg
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_path = parts[0][6:]
            prompt_text = parts[1].strip() if len(parts) > 1 else "Describe this image"

            if engine_type != "ort" or not getattr(engine, "supports_multimodal", False):
                console.print(
                    f"{markup('Error:', ERROR_STYLE)} Current engine/model does not support image input."
                )
                continue

            message_content = [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]

        history.append({"role": "user", "content": message_content})

        # ---- Stream assistant reply ----
        console.print(f"\n[{PROMPT_ASSISTANT_STYLE}]Assistant[/]: ", end="")

        t0 = time.perf_counter()
        first_token_time: float | None = None
        assistant_reply_parts: list[str] = []

        try:
            if engine_type == "ort" and getattr(engine, "supports_multimodal", False):
                # VL models: always use multimodal path (even for text-only)
                for fragment in engine.stream_multimodal(
                    messages=history,
                    image_path=image_path,
                    config=config,
                ):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    print(fragment, end="", flush=True)
                    assistant_reply_parts.append(fragment)
            else:
                for fragment in engine.stream(messages=history, config=config):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    print(fragment, end="", flush=True)
                    assistant_reply_parts.append(fragment)
        except KeyboardInterrupt:
            console.print(f"\n{markup('--generation interrupted--', MUTED_STYLE)}")

        elapsed = time.perf_counter() - t0
        assistant_reply = "".join(assistant_reply_parts)
        history.append({"role": "assistant", "content": assistant_reply})

        print()  # newline after streamed output

        if show_timings and first_token_time is not None:
            ttft = first_token_time - t0
            gen_time = elapsed - ttft
            n_tokens = len(assistant_reply_parts)
            tps = n_tokens / gen_time if gen_time > 0 else 0.0
            console.print(
                markup(f"  TTFT {ttft:.2f}s | {n_tokens} tokens | {tps:.1f} tok/s", MUTED_STYLE)
            )
