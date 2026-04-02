"""
CLI entry point for obeaver.

Commands:
  obeaver run     — interactive terminal chat
  obeaver serve   — OpenAI-compatible HTTP server
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console


def _default_engine() -> str:
    """Return the default engine for the current platform.

    Linux: only ORT is supported → 'ort'
    macOS / Windows: Foundry Local is available → 'foundry'
    """
    return "ort" if sys.platform.startswith("linux") else "foundry"

app = typer.Typer(
    name="obeaver",
    help=(
        "LLM inference powered by Foundry Local (macOS/Windows default) "
        "or ONNX Runtime GenAI (Linux default)."
    ),
    add_completion=False,
    invoke_without_command=True,
)

def _print_banner() -> None:
    try:
        import pyfiglet
        # Render 'o' and 'Beaver' separately so the lowercase o is
        # vertically padded to match the taller uppercase B, keeping
        # both letters clearly readable.
        o_lines = pyfiglet.figlet_format("o", font="big").rstrip("\n").splitlines()
        f_lines = pyfiglet.figlet_format("Beaver", font="big").rstrip("\n").splitlines()
        max_h = max(len(o_lines), len(f_lines))
        o_w = max(len(l) for l in o_lines)
        while len(o_lines) < max_h:
            o_lines.insert(0, " " * o_w)
        f_w = max(len(l) for l in f_lines)
        while len(f_lines) < max_h:
            f_lines.insert(0, " " * f_w)
        combined = [ol.ljust(o_w) + " " + fl for ol, fl in zip(o_lines, f_lines)]
        big = "\n".join(combined)
    except Exception:
        big = "oBeaver"
    # Pad all lines to the same width so the block stays aligned when centered
    lines = big.rstrip("\n").splitlines()
    max_w = max(len(l) for l in lines) if lines else 0
    padded = "\n".join(l.ljust(max_w) for l in lines)
    console.rule(style="cyan")
    console.print()
    console.print(padded, style="bold cyan")
    console.print()
    console.print("Welcome to use obeaver !", style="bold cyan")
    console.print("ONNX \u2764\ufe0f  \U0001f9ab = oBeaver", style="bold magenta")
    console.print("\U0001f4bb\U0001f4bb\U0001f4bb\U0001f4bb\U0001f4bb\U0001f4bb\U0001f4bb\U0001f4bb  run, embedding, tools, convert, fine-tuning", style="bold yellow")
    console.print("\u2601\ufe0f\u2601\ufe0f\u2601\ufe0f\u2601\ufe0f\u2601\ufe0f\u2601\ufe0f\u2601\ufe0f\u2601\ufe0f\u2601\ufe0f  Docker\uff0cAzure Container Apps , AKS", style="bold blue")
    console.print()

    # Show configured model directories
    from obeaver.config import get_models_dir, get_ort_models_dir, get_foundrylocal_models_dir, get_ort_cache_dir
    console.print(f"  [bold]Models dir:[/]         [green]{get_models_dir()}[/]")
    console.print(f"  [bold]ORT models:[/]         [green]{get_ort_models_dir()}[/]")
    console.print(f"  [bold]FoundryLocal models:[/] [green]{get_foundrylocal_models_dir()}[/]")
    console.print(f"  [bold]HF cache:[/]           [green]{get_ort_cache_dir()}[/]")
    console.print()
    console.print("  [dim]Run [bold]obeaver init[/][dim] to change the model save location.[/]")
    console.print()
    console.rule(style="cyan")


@app.callback()
def _main(ctx: typer.Context) -> None:
    """LLM inference powered by Foundry Local (default) or ONNX Runtime GenAI."""
    if ctx.invoked_subcommand is None:
        _print_banner()


ENGINE_HELP = (
    "Inference engine: 'foundry' (Foundry Local, default on macOS/Windows) "
    "or 'ort' (onnxruntime-genai, default on Linux). "
    "Automatically set to 'ort' for VL models."
)


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


def _sync_foundrylocal_cache_dir(cache_dir: Path) -> tuple[bool, str]:
    """Point Foundry Local's cache root at *cache_dir* when the CLI is available."""
    foundry_exe = shutil.which("foundry")
    if foundry_exe is None:
        return False, "Foundry Local CLI not found on PATH; skipped cache path update."

    try:
        subprocess.run(
            [foundry_exe, "cache", "cd", str(cache_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "Foundry Local cache path update timed out."
    except OSError as exc:
        return False, f"Failed to run Foundry Local CLI: {exc}"
    except subprocess.CalledProcessError as exc:
        detail = str(exc).strip()
        return False, f"Foundry Local cache path update failed: {detail}"

    try:
        result = subprocess.run(
            [foundry_exe, "cache", "location"],
            check=True,
            capture_output=True,
            text=True,
        )
        confirmed = result.stdout.strip() or str(cache_dir)
        prefix = "Cache directory path:"
        if prefix in confirmed:
            confirmed = confirmed.split(prefix, 1)[1].strip() or str(cache_dir)
    except (OSError, subprocess.CalledProcessError):
        confirmed = str(cache_dir)

    return True, confirmed


@app.command()
def init(
    models_dir: Optional[str] = typer.Argument(
        default=None,
        help="Root directory for storing models. "
             "ORT models → <dir>/ort/, Foundry Local models → <dir>/foundrylocal/.",
    ),
) -> None:
    """
    Initialise obeaver settings (model save location).

    If no path is given, you will be prompted to enter one interactively.
    The configuration is stored in ~/.obeaver/config.json.
    """
    from obeaver.config import (
        get_foundrylocal_models_dir,
        get_models_dir,
        get_ort_cache_dir,
        get_ort_models_dir,
        set_models_dir,
    )

    current = get_models_dir()
    if models_dir is None:
        console.print(f"\n[bold]Current model save location:[/] [green]{current}[/]")
        console.print(f"  ORT models:         [green]{get_ort_models_dir()}[/]")
        console.print(f"  FoundryLocal models: [green]{get_foundrylocal_models_dir()}[/]")
        console.print(f"  HF cache:           [green]{get_ort_cache_dir()}[/]")
        console.print()
        models_dir = typer.prompt(
            "Enter new model save location (press Enter to keep current)",
            default=str(current),
        )

    resolved = set_models_dir(models_dir)

    # Create the sub-directories
    ort_dir = resolved / "ort"
    fl_dir = resolved / "foundrylocal"
    cache_d = resolved / "cache_dir"
    ort_dir.mkdir(parents=True, exist_ok=True)
    fl_dir.mkdir(parents=True, exist_ok=True)
    cache_d.mkdir(parents=True, exist_ok=True)
    cache_synced, cache_message = _sync_foundrylocal_cache_dir(fl_dir)

    console.print()
    console.print(f"[bold green]\u2713  Model save location set to:[/] [green]{resolved}[/]")
    console.print(f"   ORT models:         [green]{ort_dir}[/]")
    console.print(f"   FoundryLocal models: [green]{fl_dir}[/]")
    if cache_synced:
        console.print(f"   FoundryLocal cache:  [green]{cache_message}[/] [dim](configured)[/]")
    else:
        console.print(f"   FoundryLocal cache:  [yellow]{cache_message}[/]")
    console.print(f"   HF cache:           [green]{cache_d}[/]")
    console.print(f"\n   [dim]Config saved to ~/.obeaver/config.json[/]")


def _is_vl_model(model_path: str) -> bool:
    """Return True if *model_path* points to a VL (vision-language) model.

    Detection is lightweight — checks for ``vision.onnx`` or the ``vision``
    key in ``genai_config.json`` without loading the model.
    """
    import json as _json

    p = Path(model_path)
    if not p.is_dir():
        return False
    # Quick check: vision.onnx in the directory
    if (p / "vision.onnx").exists():
        return True
    # Fallback: check genai_config.json
    config_path = p / "genai_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = _json.load(f)
            return "vision" in cfg.get("model", {})
        except Exception:
            pass
    # BFS one level for nested model dirs (e.g. cpu_and_mobile/...)
    for child in p.iterdir():
        if child.is_dir():
            if (child / "vision.onnx").exists():
                return True
    return False

import io, os
if sys.platform == "win32":
    # Force UTF-8 output on Windows to avoid cp1252 encoding errors
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

console = Console()


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


@app.command()
def run(
    model_path: str = typer.Argument(
        ...,
        help="Foundry model alias (e.g. 'phi-3.5-mini') or ORT model directory path.",
    ),
    engine_type: str = typer.Option(None, "-E", "--engine", help=ENGINE_HELP),
    system_prompt: str = typer.Option(
        "You are a helpful AI assistant.",
        "-sp", "--system-prompt",
        help="System prompt prepended to every conversation",
    ),
    max_length: int = typer.Option(4096, "--max-length", help="Max total sequence length (input + output tokens)"),
    max_new_tokens: int = typer.Option(1024, "--max-new-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature"),
    top_p: float = typer.Option(0.95, "--top-p", help="Nucleus sampling probability"),
    top_k: int = typer.Option(50, "--top-k", help="Top-k sampling"),
    repetition_penalty: float = typer.Option(1.0, "--repetition-penalty", help="Repetition penalty (1.0 = no penalty)"),
    timings: bool = typer.Option(False, "--timings", "-g", help="Show per-turn timing stats"),
    execution_provider: str = typer.Option(
        "cpu", "-e", "--ep",
        help="Execution provider for ORT engine (cpu). Ignored for foundry engine.",
    ),
) -> None:
    """Start an interactive multi-turn chat session in the terminal."""
    if sys.platform.startswith("linux") and engine_type and engine_type != "ort":
        console.print(
            "[bold red]Error:[/] On Linux, only the 'ort' engine is supported. "
            "Foundry Local is not available on Linux.",
        )
        raise typer.Exit(code=1)

    # VL models only work with the ORT engine — auto-switch silently.
    if _is_vl_model(model_path):
        if engine_type and engine_type != "ort":
            console.print(
                "[bold yellow]Warning:[/] VL models require the 'ort' engine. "
                "Switching from '{}' to 'ort' automatically.".format(engine_type),
            )
        engine_type = "ort"

    from obeaver.chat import run_chat

    run_chat(
        model_path=model_path,
        engine_type=engine_type or _default_engine(),
        system_prompt=system_prompt,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        show_timings=timings,
        execution_provider=execution_provider,
    )


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


@app.command()
def serve(
    model_path: str = typer.Argument(
        ...,
        help="Foundry model alias (e.g. 'phi-3.5-mini') or ORT model directory path.",
    ),
    engine_type: str = typer.Option(None, "-E", "--engine", help=ENGINE_HELP),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(18000, "--port", "-p", help="Bind port"),
    execution_provider: str = typer.Option(
        "cpu", "-e", "--ep",
        help="Execution provider for ORT engine (cpu). Ignored for foundry engine.",
    ),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)"),
) -> None:
    """
    Launch the OpenAI-compatible HTTP API server (no dashboard UI).

    Compatible with any OpenAI client pointing to http://<host>:<port>/v1
    """
    if sys.platform.startswith("linux") and engine_type and engine_type != "ort":
        console.print(
            "[bold red]Error:[/] On Linux, only the 'ort' engine is supported. "
            "Foundry Local is not available on Linux.",
        )
        raise typer.Exit(code=1)

    # VL models only work with the ORT engine — auto-switch silently.
    if _is_vl_model(model_path):
        if engine_type and engine_type != "ort":
            console.print(
                "[bold yellow]Warning:[/] VL models require the 'ort' engine. "
                "Switching from '{}' to 'ort' automatically.".format(engine_type),
            )
        engine_type = "ort"

    import uvicorn

    from obeaver.server import build_app

    _engine = engine_type or _default_engine()
    console.print(
        f"\n[bold cyan]obeaver serve[/] — engine=[yellow]{_engine}[/]  "
        f"model=[green]{model_path}[/]  "
        f"addr=[blue]http://{host}:{port}[/]\n"
    )

    application = build_app(
        model_path=model_path,
        engine_type=_engine,
        execution_provider=execution_provider,
    )

    uvicorn.run(
        application,
        host=host,
        port=port,
        reload=False,
        log_level="info",
        timeout_keep_alive=30,
    )


# ---------------------------------------------------------------------------
# dashboard command
# ---------------------------------------------------------------------------


@app.command()
def dashboard(
    engine_type: str = typer.Option(None, "-e", "--engine", help=ENGINE_HELP),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(1573, "--port", "-p", help="Bind port"),
) -> None:
    """
    Launch the obeaver dashboard (benchmark & monitoring UI).

    By default uses Foundry Local and lists cached models.
    Use --engine ort to list local ONNX models from ./models instead.
    """
    import uvicorn

    from obeaver.config import get_models_dir
    from obeaver.server import build_dashboard_app

    _engine = engine_type or _default_engine()

    # Check if models exist for the selected engine before launching
    models_root = get_models_dir()
    engine_folder = "ort" if _engine == "ort" else "foundrylocal"
    engine_dir = models_root / engine_folder
    rows: list[tuple[str, str, str]] = []
    if engine_dir.is_dir():
        rows.extend(_collect_models(engine_dir, engine_folder))
    if not rows:
        if _engine == "ort":
            console.print(
                f"\n[bold red]No models found in {engine_dir}.[/] "
                "Please add an ORT model before launching the dashboard.\n\n"
                "  [dim]Examples:[/]\n"
                "    obeaver convert Qwen/Qwen3-0.6B                  "
                "[dim]# convert a HF model to ONNX[/]\n"
                "    hf download onnx-community/Qwen3-Embedding-0.6B "
                f"--local-dir {engine_dir}/Qwen3-Embedding-0.6B  "
                "[dim]# download a pre-converted model[/]\n"
            )
        else:
            console.print(
                f"\n[bold red]No models found in {engine_dir}.[/] "
                "Please load a Foundry Local model before launching the dashboard.\n\n"
                "  [dim]Example:[/]\n"
                "    obeaver run phi-4-mini          "
                "[dim]# downloads & caches a Foundry Local model[/]\n"
            )
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold cyan]obeaver dashboard[/] — engine=[yellow]{_engine}[/]  "
        f"addr=[blue]http://{host}:{port}[/]\n"
    )

    application = build_dashboard_app(engine_type=_engine)
    uvicorn.run(
        application,
        host=host,
        port=port,
        reload=False,
        log_level="info",
        timeout_keep_alive=30,
    )


# ---------------------------------------------------------------------------
# embed command  (ONNX-only, no engine selection)
# ---------------------------------------------------------------------------


@app.command()
def embed(
    model_path: str = typer.Argument(
        ...,
        help="Path to a local ONNX embedding model directory "
             "(e.g. onnx-community/embeddinggemma-300m-ONNX).",
    ),
    text: Optional[str] = typer.Argument(
        default=None,
        help="Text to embed.  If omitted, enter an interactive loop.",
    ),
    execution_provider: str = typer.Option(
        "cpu", "-e", "--ep",
        help="ONNX execution provider (cpu, cuda, directml).",
    ),
) -> None:
    """
    Compute text embeddings with an ONNX embedding model.

    NOTE: Embeddings are ONNX-only — no alternative engine is supported.
    """
    from obeaver.engine_embedding import EmbeddingEngine

    console.print(
        f"\n[bold cyan]obeaver embed[/] — model=[green]{model_path}[/]  "
        f"ep=[yellow]{execution_provider}[/]\n"
    )
    engine = EmbeddingEngine(model_path=model_path, execution_provider=execution_provider)

    if text:
        vectors = engine.embed(text)
        console.print(f"[bold]Embedding ([dim]{len(vectors[0])}[/] dims):[/]")
        console.print(vectors[0])
        return

    # Interactive loop
    console.print("Type text and press [bold]Enter[/] to get its embedding.  "
                  "Press [bold]Ctrl-C[/] to quit.\n")
    while True:
        try:
            query = console.input("[bold green]text>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/]")
            break
        if not query:
            continue
        vectors = engine.embed(query)
        dims = len(vectors[0])
        preview = vectors[0][:8]
        console.print(
            f"[dim]embedding ({dims} dims, first 8): "
            f"{[round(v, 6) for v in preview]} …[/]\n"
        )


# ---------------------------------------------------------------------------
# serve-embed command  (ONNX-only, no engine selection)
# ---------------------------------------------------------------------------


@app.command(name="serve-embed")
def serve_embed(
    model_path: str = typer.Argument(
        ...,
        help="Path to a local ONNX embedding model directory.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(18001, "--port", "-p", help="Bind port"),
    execution_provider: str = typer.Option(
        "cpu", "-e", "--ep",
        help="ONNX execution provider (cpu, cuda, directml).",
    ),
) -> None:
    """
    Launch an OpenAI-compatible embedding server (POST /v1/embeddings).

    NOTE: Embeddings are ONNX-only — no alternative engine is supported.

    Compatible with any OpenAI client:
        client.embeddings.create(model=\"...\", input=\"text\")
    """
    import uvicorn

    from obeaver.server import build_embed_app

    console.print(
        f"\n[bold cyan]obeaver serve-embed[/] — model=[green]{model_path}[/]  "
        f"ep=[yellow]{execution_provider}[/]  "
        f"addr=[blue]http://{host}:{port}[/]\n"
    )

    application = build_embed_app(
        model_path=model_path,
        execution_provider=execution_provider,
    )

    uvicorn.run(
        application,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# check command
# ---------------------------------------------------------------------------


@app.command()
def check() -> None:
    """Check the runtime environment and Foundry Local installation status."""
    import shutil
    import subprocess

    console.rule("[bold cyan]obeaver environment check[/]", style="cyan")
    console.print()

    # ── Platform ──────────────────────────────────────────────────────────
    platform_name = sys.platform
    if platform_name.startswith("linux"):
        console.print(f"[bold]Platform:[/] [yellow]{platform_name}[/]")
        console.print()
        console.print(
            "[bold red]✗  Foundry Local is not supported on Linux.[/]\n"
            "   Only the [bold]ORT[/] engine (onnxruntime-genai) is available.\n"
            "   Use [bold]obeaver run --engine ort[/] or [bold]obeaver serve --engine ort[/]."
        )
        console.print()
        console.rule(style="cyan")
        return

    if platform_name == "darwin":
        platform_label = "macOS"
    elif platform_name == "win32":
        platform_label = "Windows"
    else:
        platform_label = platform_name

    console.print(f"[bold]Platform:[/] [green]{platform_label}[/]")
    console.print()

    # ── Foundry Local daemon ───────────────────────────────────────────────
    # The CLI binary is named 'foundry' on macOS/Windows
    foundry_bin = shutil.which("foundry")

    if foundry_bin is None:
        console.print("[bold red]✗  Foundry Local CLI not found.[/]")
        console.print()
        console.print("[bold]Install instructions:[/]")
        if platform_name == "darwin":
            console.print(
                "    [cyan]brew install microsoft/foundrylocal/foundrylocal[/]"
            )
        else:  # win32
            console.print(
                "    [cyan]winget install Microsoft.FoundryLocal[/]"
            )
        console.print()
        console.print(
            "  For more information, see: "
            "[link=https://github.com/microsoft/foundry-local]"
            "https://github.com/microsoft/foundry-local[/link]"
        )
    else:
        console.print(f"[bold green]✓  Foundry Local CLI found:[/] [dim]{foundry_bin}[/]")

        # Try to get the version
        try:
            result = subprocess.run(
                ["foundry", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version_output = (result.stdout or result.stderr).strip()
            if version_output:
                console.print(f"   [bold]Version:[/] [yellow]{version_output}[/]")
            else:
                console.print("   [dim]Version information not available.[/]")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            console.print("   [dim]Could not retrieve version information.[/]")

        # Check if the SDK is importable
        console.print()
        try:
            import foundry_local  # noqa: F401
            console.print("[bold green]✓  foundry-local-sdk (Python) is installed.[/]")
            try:
                from importlib.metadata import version as pkg_version
                sdk_ver = pkg_version("foundry-local-sdk")
                console.print(f"   [bold]SDK version:[/] [yellow]{sdk_ver}[/]")
            except Exception:
                pass
        except ImportError:
            console.print(
                "[bold yellow]⚠  foundry-local-sdk (Python) not found.[/]\n"
                "   Install with: [cyan]pip install foundry-local-sdk[/]"
            )

    console.print()

    # ── Hugging Face authentication ────────────────────────────────────────
    try:
        from huggingface_hub import get_token
        hf_token = get_token()
        if hf_token:
            console.print("[bold green]✓  Hugging Face authentication found.[/]")
            # Show a shortened version of the token for confirmation
            token_preview = hf_token[:20] + "..." if len(hf_token) > 20 else hf_token
            console.print(f"   [dim]Token:[/] [yellow]{token_preview}[/]")
        else:
            console.print("[bold yellow]⚠  Hugging Face not authenticated.[/]")
            console.print()
            console.print("[bold]Why authenticate with Hugging Face?[/]")
            console.print("   • Download gated models (Llama 2, Mistral, etc.)")
            console.print("   • Higher rate limits on model downloads")
            console.print("   • Access to private model repositories")
            console.print()
            console.print("[bold]How to authenticate:[/]")
            console.print("   [bold]Option 1 — CLI:[/]")
            console.print("       [cyan]huggingface-cli login[/]")
            console.print()
            console.print("   [bold]Option 2 — Python:[/]")
            console.print("       [cyan]python -c \"from huggingface_hub import login; login()\"[/]")
            console.print()
            console.print("   [bold]Option 3 — Environment variable:[/]")
            console.print("       [cyan]export HF_TOKEN='your_token_here'[/]")
            console.print()
            console.print("[bold]Get your token:[/]")
            console.print("   Visit: [link=https://huggingface.co/settings/tokens]"
                          "https://huggingface.co/settings/tokens[/link]")
    except ImportError:
        console.print(
            "[bold yellow]⚠  huggingface-hub not installed.[/]\n"
            "   Install with: [cyan]pip install huggingface-hub[/]"
        )

    console.print()
    console.rule(style="cyan")


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# convert command  (model conversion via onnxruntime-genai model builder)
# ---------------------------------------------------------------------------


@app.command()
def convert(
    model_name: str = typer.Argument(
        ...,
        help="Hugging Face model name or local path (e.g. Qwen/Qwen3-0.6B).",
    ),
    model_type: str = typer.Option(
        "text", "-t", "--type",
        help="Model type: 'text' for text-only models (default), "
             "'vl' for vision-language models (uses Olive pipeline).",
    ),
    output: Optional[str] = typer.Option(
        None, "-o", "--output",
        help="Output directory. Default: <models_dir>/ort/<model>_ONNX_<PRECISION>_<EP>",
    ),
    precision: str = typer.Option(
        "int4", "-p", "--precision",
        help="Quantization precision: fp32, fp16, or int4.",
    ),
    execution_provider: str = typer.Option(
        "cpu", "-e", "--ep",
        help="Execution provider: cpu or cuda (gpu).",
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "-c", "--cache-dir",
        help="HF download cache directory (default: <models_dir>/cache_dir).",
    ),
    extra_options: Optional[str] = typer.Option(
        None, "--extra-options",
        help="Space-separated KEY=VALUE pairs passed to the model builder "
             "(e.g. --extra-options 'shared_embeddings=true int4_block_size=64'). "
             "Only used for text models.",
    ),
    build_from_source: bool = typer.Option(
        False, "--build-from-source",
        help="Build onnxruntime-genai from source before VL conversion. "
             "Automatically detects OS and device to select the correct build "
             "configuration. Only used for VL models.",
    ),
) -> None:
    """
    Convert a Hugging Face model to optimised ONNX format.

    For text models (default): uses onnxruntime_genai.models.builder.
    For VL models (--type vl): uses Olive pipeline from olive-recipes.

    Examples:

        obeaver convert Qwen/Qwen3-0.6B
        → text model, saves to <models_dir>/ort/Qwen3-0.6B_ONNX_INT4_CPU

        obeaver convert Qwen/Qwen3-0.6B -o ./my_custom_dir

        obeaver convert Qwen/Qwen3-0.6B -p fp16 -e cuda
        → saves to <models_dir>/ort/Qwen3-0.6B_ONNX_FP16_CUDA

        obeaver convert Qwen/Qwen3-VL-2B-Instruct --type vl
        → VL model via Olive, saves to <models_dir>/ort/Qwen3-VL-2B-Instruct_VL_ONNX_INT4_CPU
    """
    import subprocess

    # Normalise execution provider aliases
    ep = execution_provider.strip().lower()
    if ep in ("gpu", "cuda"):
        ep = "cuda"
    elif ep == "cpu":
        ep = "cpu"
    else:
        console.print(
            f"[bold red]Error:[/] unsupported execution provider '{execution_provider}'. "
            "Only 'cpu' and 'cuda' (gpu) are supported."
        )
        raise typer.Exit(code=1)

    # Validate precision
    valid_precisions = ("fp32", "fp16", "int4")
    if precision.lower() not in valid_precisions:
        console.print(
            f"[bold red]Error:[/] unsupported precision '{precision}'. "
            f"Choose from: {', '.join(valid_precisions)}."
        )
        raise typer.Exit(code=1)

    # Validate model type
    mt = model_type.strip().lower()
    if mt not in ("text", "vl"):
        console.print(
            f"[bold red]Error:[/] unsupported model type '{model_type}'. "
            "Choose from: text, vl."
        )
        raise typer.Exit(code=1)

    # Build default output path under <models_dir>/ort/
    from obeaver.config import get_ort_cache_dir, get_ort_models_dir
    if output is None:
        ort_dir = get_ort_models_dir()
        short_name = model_name.split("/")[-1]
        if mt == "vl":
            output = str(ort_dir / f"{short_name}_VL_ONNX_{precision.upper()}_{ep.upper()}")
        else:
            output = str(ort_dir / f"{short_name}_ONNX_{precision.upper()}_{ep.upper()}")

    output_path = Path(output).resolve()

    # Default cache_dir under <models_dir>/ort/cache_dir
    if cache_dir is None:
        cache_dir = str(get_ort_cache_dir())

    console.print()
    console.rule("[bold cyan]obeaver convert[/]", style="cyan")
    console.print(f"  [bold]Model:[/]      [green]{model_name}[/]")
    console.print(f"  [bold]Type:[/]       [yellow]{mt}[/]")
    console.print(f"  [bold]Precision:[/]  [yellow]{precision.lower()}[/]")
    console.print(f"  [bold]EP:[/]         [yellow]{ep}[/]")
    console.print(f"  [bold]Output:[/]     [blue]{output_path}[/]")
    console.rule(style="cyan")
    console.print()

    if mt == "vl":
        # VL model: use Olive pipeline from olive-recipes
        from obeaver.convert_vl import convert_vl_model

        try:
            convert_vl_model(
                model_name=model_name,
                output_dir=str(output_path),
                device=ep,
                build_from_source=build_from_source,
                cache_dir=cache_dir,
            )
        except ValueError as e:
            console.print(f"[bold red]Error:[/] {e}")
            raise typer.Exit(code=1)
        except subprocess.CalledProcessError as e:
            console.print(
                f"\n[bold red]VL conversion failed[/] (exit code {e.returncode})."
            )
            raise typer.Exit(code=e.returncode)
    else:
        # Text model: use onnxruntime-genai model builder
        cache_path = Path(cache_dir).resolve()
        console.print(f"  [bold]Cache:[/]      [dim]{cache_path}[/]")
        console.print()

        cmd = [
            sys.executable, "-m", "onnxruntime_genai.models.builder",
            "-m", model_name,
            "-o", str(output_path),
            "-p", precision.lower(),
            "-e", ep,
            "-c", str(cache_path),
        ]

        if extra_options:
            cmd.append("--extra_options")
            cmd.extend(extra_options.split())

        console.print(f"[dim]Running: {' '.join(cmd)}[/]\n")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            console.print(
                f"\n[bold red]Conversion failed[/] (exit code {result.returncode})."
            )
            raise typer.Exit(code=result.returncode)

    console.print(f"\n[bold green]✓  Model saved to {output_path}[/]")


# ---------------------------------------------------------------------------
# models command
# ---------------------------------------------------------------------------


def _detect_model_type(model_dir: Path) -> str:
    """Detect whether a model directory is Text, Vision + Text, or Embeddings.

    Checks the directory itself and one level of sub-directories (e.g. the
    ``v<N>/`` version folders used by Foundry Local).

    • Vision + Text: contains ``vision.onnx`` or genai_config.json with a
      ``vision`` component.
    • Embeddings: folder name contains 'embedding' (case-insensitive).
    • Text: everything else.
    """
    import json as _json

    # Collect candidate dirs: the model dir itself + immediate sub-dirs
    candidates = [model_dir]
    for sub in model_dir.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            candidates.append(sub)

    for d in candidates:
        if (d / "vision.onnx").exists():
            return "Vision + Text"
        genai_cfg = d / "genai_config.json"
        if genai_cfg.exists():
            try:
                with open(genai_cfg) as f:
                    cfg = _json.load(f)
                if "vision" in cfg.get("model", {}):
                    return "Vision + Text"
            except Exception:
                pass

    # --- Embeddings ---
    if "embedding" in model_dir.name.lower():
        return "Embeddings"

    # --- Default: Text ---
    return "Text"


_MODEL_MARKERS = ("genai_config.json", "config.json", "inference_model.json")


def _has_model_artifacts(d: Path) -> bool:
    """Return True if directory *d* directly contains model artefact files."""
    for m in _MODEL_MARKERS:
        if (d / m).exists():
            return True
    if any(d.glob("*.onnx")):
        return True
    return False


def _is_model_dir(d: Path) -> bool:
    """Return True if *d* is an actual model directory (not a publisher/category folder).

    A model directory either:
    • contains model artefact files directly, or
    • contains version sub-folders (``v<N>/``) that hold artefact files
      (Foundry Local layout).
    """
    if _has_model_artifacts(d):
        return True
    # Check for version sub-folders like v1/, v5/ etc.
    import re
    for sub in d.iterdir():
        if sub.is_dir() and re.match(r"^v\d+$", sub.name):
            if _has_model_artifacts(sub):
                return True
    return False


def _collect_models(engine_dir: Path, engine_name: str) -> list[tuple[str, str, str]]:
    """Recursively collect model entries from *engine_dir*.

    For ORT the models sit directly under ``ort/<model>/``.
    For Foundry Local the layout is ``foundrylocal/<publisher>/<model>/v<N>/``,
    so publisher folders are traversed transparently.
    """
    rows: list[tuple[str, str, str]] = []
    for child in sorted(engine_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if _is_model_dir(child):
            rows.append((child.name, engine_name, _detect_model_type(child)))
        else:
            # Likely a publisher/category folder — look one level deeper
            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir() or grandchild.name.startswith("."):
                    continue
                if _is_model_dir(grandchild):
                    rows.append((grandchild.name, engine_name, _detect_model_type(grandchild)))
    return rows


@app.command()
def models() -> None:
    """
    List local models discovered under the configured models directory.

    Scans the ``foundrylocal`` and ``ort`` sub-folders (skips ``cache_dir``)
    and displays each model's name, engine type, and model category
    (Text, Vision + Text, or Embeddings).
    """
    from obeaver.config import get_models_dir

    models_root = get_models_dir()

    console.print()
    console.rule("[bold cyan]obeaver models[/]", style="cyan")
    console.print(f"  [bold]Models dir:[/] [green]{models_root}[/]")
    console.print()

    rows: list[tuple[str, str, str]] = []  # (name, engine, type)

    for engine_name in ("foundrylocal", "ort"):
        engine_dir = models_root / engine_name
        if not engine_dir.is_dir():
            continue
        rows.extend(_collect_models(engine_dir, engine_name))

    if not rows:
        console.print("  [dim]No models found. Use [bold]obeaver convert[/] to add models.[/]")
        console.print()
        console.rule(style="cyan")
        return

    # Pretty-print as a Rich table
    from rich.table import Table

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="green", min_width=20)
    table.add_column("Engine", style="yellow", min_width=14)
    table.add_column("Type", style="cyan", min_width=14)

    type_icons = {
        "Text": "💬 Text",
        "Vision + Text": "👁️ Vision + Text",
        "Embeddings": "📐 Embeddings",
    }

    for name, engine, mtype in rows:
        table.add_row(name, engine, type_icons.get(mtype, mtype))

    console.print(table)
    console.print(f"\n  [dim]Total: {len(rows)} model(s)[/]")
    console.print()
    console.rule(style="cyan")


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the obeaver version."""
    from obeaver._version import __version__

    console.print(f"[bold cyan]obeaver[/] v{__version__}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
