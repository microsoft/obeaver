# Changelog

All notable changes to the **oFoundry** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.0] — 2026-03-24

### Added

#### Dashboard Enhancements
- **Model selector dropdown** — switch between cached Foundry Local models at runtime without restarting the server. NPU/GPU-accelerated models display a ⚡ badge.
- **Inference parameters panel** — tune temperature, top-p, top-k, max tokens, and repetition penalty from the UI. Includes Creative / Balanced / Precise presets.
- **Performance metrics** — each chat response displays TTFT (time to first token), tok/s (throughput), and total token count.
- **Conversation sidebar** — persistent chat history with new/rename/delete and one-click switching.
- **System prompt configuration** — editable system prompt in the sidebar panel.
- **Server logs panel** — live request log showing method, path, status, and timing.
- **Export** — download conversations as JSON or Markdown.
- **Dark / light theme toggle**.
- **Regenerate and copy** buttons on individual messages.

#### Server & Engine Performance
- **Server-side model warmup** — on startup, a short inference primes the model's KV cache, eliminating cold-start latency for the first real request.
- **Non-blocking async pipeline** — `get_all_memory()`, `list_cached_models()`, and non-streaming inference all run in thread pools so the asyncio event loop is never blocked.
- **GPU/NPU info caching** — slow PowerShell/WMI queries for GPU and NPU detection run once and are cached, removing a major source of event-loop stalls.
- **Queue-based streaming** — replaced per-token `run_in_executor` with a producer/consumer queue pattern, reducing per-token dispatch overhead.
- **Non-streaming `complete()` method** — the Foundry engine now uses `stream=False` against the daemon for non-streaming requests, which is significantly faster than joining a streaming iterator.
- **Thread pool increase** — `ThreadPoolExecutor` raised from 2 to 4 workers to prevent deadlocks between warmup and concurrent requests.
- **Model TTL set to 24 hours** — `load_model(ttl=86400)` prevents the Foundry daemon from auto-unloading the model after 10 minutes of idle.

#### New API Endpoints
- **`GET /api/models/available`** — list all cached Foundry Local models with alias, device type, and size.
- **`POST /api/models/load`** — hot-swap the active model at runtime (Foundry engine only).

### Changed

- **README.md** — updated Web Dashboard section with new HD screenshots, expanded feature list, and two new API endpoints in the reference table.
- **README.zh-cn.md** — matching updates in Chinese with new screenshots and API endpoints.
- **blog_post.md** — comprehensive update with new screenshots, model selector, inference parameters, performance metrics, and updated roadmap.
- **Screenshots** — replaced old dashboard screenshots with 7 new full-screen HD (1920×1080) captures taken in Chrome.
- **Uvicorn tuning** — HTTP/1.1 protocol, `timeout_keep_alive=30`, `limit_max_requests=0`.
- **httpx transport** — OpenAI client uses `httpx.HTTPTransport` with keep-alive for lower-latency streaming.
- **SSE headers** — `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no` on streaming responses.

### Fixed
- **Event loop blocking** — `get_all_memory()` (which spawns PowerShell subprocesses for GPU/NPU detection) was called synchronously on the async event loop, causing the server to stall for 10-20 seconds per chat request when browser tabs were polling. Now runs in a thread pool with cached results.
- **Model auto-unload** — Foundry Local's default 600-second TTL caused cold-reload penalties after short idle periods. Now set to 86400s.

---

## [0.1.1] — 2026-03-23

### Added

#### Web Dashboard
- **Built-in web dashboard** (`static/index.html`) — full-featured single-page UI served at `/` when the server is running:
  - Real-time CPU, GPU, and NPU memory utilisation with circular progress indicators.
  - Device detection labels (processor name, GPU model, NPU model).
  - Active / Idle / Unavailable status badges per device.
  - Model info bar showing loaded model name, engine type, platform, and Python version.
  - oFoundry process stats (PID, resident memory, virtual memory).
  - Interactive "Chat with Model" panel for sending messages directly from the browser.
  - Memory statistics auto-refresh every 3 seconds.
  - Dark theme UI with colour-coded device panels (blue = CPU, purple = GPU, green = NPU).

#### System Monitoring
- **Resource monitor module** (`monitor.py`) — new 345-line module providing:
  - `get_cpu_memory()` — system RAM usage via `psutil` with fallback for environments without it.
  - `get_gpu_memory()` — NVIDIA GPU memory via `nvidia-smi`, AMD via `rocm-smi`, Qualcomm Adreno detection.
  - `get_npu_memory()` — NPU detection for Qualcomm Hexagon, Intel Meteor Lake, and Apple Neural Engine.
  - `get_all_memory()` — aggregated JSON-serialisable dict for the dashboard API.
  - Cross-platform support (Windows, macOS, Linux).

#### Server Enhancements
- **`GET /`** — automatic redirect to the dashboard UI on both chat and embedding apps.
- **`GET /api/system/memory`** — new REST endpoint returning real-time CPU/GPU/NPU memory stats as JSON.
- **Static file serving** — FastAPI `StaticFiles` mount at `/static` for the dashboard assets.
- **Engine label** in health endpoint — `GET /health` now includes `"engine"` field.

#### Screenshots
- Added dashboard screenshots: `dashboard-full.png`, `dashboard-header.png`, `dashboard-chat.png`, `dashboard-chat-response.png`, `dashboard-gpu-npu.png`.

#### Testing
- **`test_server.py`** — expanded with 375 lines of new tests covering chat completions, streaming, model listing, health check, and the new memory API endpoint.
- **`test_monitor.py`** — 160 lines of unit tests for all monitor functions (CPU, GPU, NPU, fallbacks).
- **`test_cli.py`** — extended with additional CLI command and option validation tests.
- **`test_config.py`** — extended with configuration validation tests.

### Changed

#### Documentation
- **`README.md`** — major rewrite with a "Why ofoundry?" value-proposition table, streamlined installation instructions, and dashboard documentation with screenshots.
- **`README.zh-cn.md`** — updated Chinese documentation to match.
- **`README.md.bak`** — backup of the previous README preserved.
- **`blog_post.md`** — new introductory blog post: "Introducing oFoundry: Lightweight Local LLM Inference with a Built-in Dashboard".

#### Dependencies
- Added `psutil>=5.9.0` to `pyproject.toml` dependencies (required by the monitor module).

---

## [0.1.0] — 2026-03-22

Initial public release of oFoundry — a CPU-first, cross-platform LLM inference server powered by ONNX Runtime GenAI with an OpenAI-compatible API.

### Added

#### Core Engines
- **ORT Engine** (`engine_ort.py`) — streaming text generation via `onnxruntime-genai`, with configurable sampling parameters (`max_length`, `max_new_tokens`, `temperature`, `top_p`, `top_k`).
- **Foundry Local Engine** (`engine_foundrylocal.py`) — on-device inference through the Microsoft Foundry Local SDK with auto hardware acceleration (NPU > GPU > CPU). Default engine on macOS and Windows.
- **Embedding Engine** (`engine_embedding.py`) — ONNX-only text embedding engine supporting models such as `embeddinggemma-300m-ONNX`, with HuggingFace `AutoTokenizer` integration.
- **OS-aware default engine selection** — Linux defaults to ORT; macOS/Windows default to Foundry Local.

#### Tool Calling
- **Tool-calling framework** (`tools.py`) — shared utilities for both engines:
  - ORT engine: tools serialised as JSON Schema in the system prompt; model replies with `<tool_call>` blocks parsed and validated by `parse_tool_call()`.
  - Foundry Local engine: tools forwarded natively via the standard OpenAI `tools` parameter.

#### Server
- **OpenAI-compatible HTTP server** (`server.py`) — FastAPI/Uvicorn server exposing:
  - `GET /v1/models` — list loaded models.
  - `POST /v1/chat/completions` — chat completions (streaming and non-streaming).
  - `GET /health` — health check endpoint.
  - Embedding endpoint support.
- **Dashboard UI** (`static/index.html`) — built-in web dashboard served by the FastAPI app.

#### CLI
- **`ofoundry run`** — interactive terminal chat with model/engine selection (`-m`, `-E` flags).
- **`ofoundry serve`** — launch the HTTP server with configurable `--host` and `--port`.
- **`ofoundry serve-embed`** — serve embedding models via HTTP.
- **`ofoundry embed`** — run embedding from the command line.
- **`ofoundry check`** — verify runtime dependencies and environment.
- **`ofoundry version`** — print the current version.
- **`ofoundry convert`** — model conversion utilities.
- **ASCII banner** on startup using `pyfiglet`.
- **Rich console output** for improved terminal UX.
- **Cloud CLI support** — commands for cloud deployment workflows.

#### System Monitoring
- **Resource monitor** (`monitor.py`) — real-time CPU, GPU, and NPU memory usage reporting for the dashboard UI.

#### Docker
- **CPU-only Docker image** (`docker/Dockerfile.cpu`):
  - Multi-platform support: `linux/amd64` (PyPI install) and `linux/arm64` (source build).
  - Configurable `PYTHON_VERSION` and `ORT_GENAI_REF` build arguments.
  - Supports both interactive chat and HTTP server modes.

#### Configuration
- **`config.py`** — centralised configuration with sensible defaults (host, port, execution provider, generation parameters, system prompt).

#### Testing
- `test_cli.py` — CLI command registration, help output, and option validation.
- `test_config.py` — default configuration values and path validation.
- `test_engine.py` — ORT engine unit tests.
- `test_tools.py` — tool-calling parse/validation tests.
- `test_embedding_api.py` — embedding API integration tests.
- `test_server.py` — server endpoint tests.
- `test_monitor.py` — resource monitor tests.
- `test_foundrylocal_api.py` — Foundry Local engine tests.
- `test_ort_api.py` — ORT API tests.
- `run_tool_test.py` — end-to-end tool-calling test runner with sample request fixtures.

#### Documentation
- Comprehensive `README.md` (English) and `README.zh-cn.md` (Chinese).
- Updated project logo (`img/logo.png`).

#### Project Setup
- `pyproject.toml` with full dependency list (onnxruntime-genai, foundry-local-sdk, openai, fastapi, uvicorn, typer, rich, pyfiglet, huggingface-hub, numpy, torch, psutil).
- Dev dependencies: `pytest`, `httpx`.
- Entry point: `ofoundry = ofoundry.cli:app`.
- Apache-2.0 license.
- Python >= 3.10 required.
