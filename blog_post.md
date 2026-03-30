# Introducing oFoundry: Lightweight Local LLM Inference with a Built-in Dashboard

Running large language models locally has never been more accessible. With **oFoundry**, you can load, serve, and interact with ONNX and Foundry Local models on your own hardware, all through a unified CLI, an OpenAI-compatible API, and a real-time web dashboard that visualises memory usage across CPU, GPU, and NPU — plus a full-featured chat interface with live performance metrics.

## What is oFoundry?

oFoundry is a lightweight, open-source framework for local LLM inference. It supports two backend engines:

- **Foundry Local** (default on macOS and Windows): powered by [Microsoft Foundry Local](https://github.com/microsoft/foundry-local), with automatic model downloading and hardware acceleration across NPU, GPU, and CPU.
- **ORT** (default on Linux, available everywhere): powered by [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai), loading models directly from a local directory with zero cloud dependency.

Both engines expose the same interface: a Typer CLI for interactive chat, a FastAPI server compatible with any OpenAI client, and now a built-in web dashboard.

## Why Local Inference Matters

Cloud-based LLM APIs are convenient, but they come with trade-offs: latency, cost, data privacy concerns, and a dependency on network connectivity. Local inference sidesteps all of these.

With oFoundry, your data never leaves your machine. Inference runs entirely on local hardware, whether that is a CPU, a discrete GPU, or the growing family of Neural Processing Units (NPUs) found in modern laptops and desktops.

## The New Web Dashboard

The headline feature in this release is the **built-in web dashboard**. When you start the oFoundry server, simply open your browser and navigate to the server address (e.g. `http://127.0.0.1:1573/`) to access it.

### What the Dashboard Shows

<p align="center">
  <img src="Screenshots/01_fullpage_hd.png" alt="oFoundry Dashboard: full view showing model selector, memory cards, inference parameters, and chat" width="800"/>
</p>

**Model Selector**
A dropdown at the top lets you switch between cached models at runtime. NPU-accelerated models are marked with a ⚡ badge, making it easy to choose faster hardware-optimised variants. Switching models is instant — no server restart required.

**System Info Bar**
At the top of the page, you will see the currently loaded model name, the active engine (Foundry Local or ORT), the platform (Windows, macOS, or Linux), and the Python version. A health indicator shows whether the server is online and responding.

**CPU Memory**
A real-time gauge displays system RAM utilisation. The card shows total, used, and available memory, updated every three seconds. This is particularly useful when loading large models that consume significant amounts of system memory.

<p align="center">
  <img src="Screenshots/02_top_viewport_hd.png" alt="Dashboard top: model selector, system info bar, and CPU memory gauge" width="800"/>
</p>

**GPU Memory**
For systems with NVIDIA GPUs, the dashboard queries `nvidia-smi` to display dedicated video memory usage. On Windows, it also detects GPUs via WMI. AMD GPUs with ROCm are supported as well. If no GPU is detected, the card clearly indicates that the device is unavailable.

**NPU Memory**
The dashboard checks for Intel NPUs (including those marketed as "AI Boost" on Intel Core Ultra processors) and Qualcomm NPUs (such as those in Snapdragon X Elite devices). As NPU tooling matures, this card will provide increasingly detailed memory and utilisation data.

<p align="center">
  <img src="Screenshots/03_memory_cards_hd.png" alt="CPU, GPU, and NPU memory cards with real-time gauges" width="800"/>
</p>

**Inference Parameters**
A dedicated panel on the right side lets you tune inference parameters in real time: temperature, top-p, top-k, max tokens, and repetition penalty. Quick presets (Creative, Balanced, Precise) are available for common use cases. A system prompt field lets you customise the model's behaviour without writing code.

**Process Memory**
A separate bar shows the memory footprint of the oFoundry process itself: resident set size (RSS) and virtual memory size (VMS). This helps you understand exactly how much memory the inference runtime is consuming.

**Chat Interface with Performance Metrics**
The chat interface allows you to send messages to the loaded model directly from your browser. Responses stream in token by token with real-time rendering. Each response displays key performance metrics:

- **TTFT** (Time to First Token) — how quickly the model starts responding
- **tok/s** — sustained token generation speed
- **Token count** — total tokens generated

<p align="center">
  <img src="Screenshots/05_chat_response_hd.png" alt="Chat with model showing streaming response, TTFT, tok/s stats" width="800"/>
</p>

**Additional Features**
- **Conversation sidebar** — saved chat history with one-click switching
- **Server logs** — live request log showing method, path, status code, and timing
- **Export** — download conversations as JSON or Markdown
- **Dark/light theme** — toggle between visual modes

### Why Monitor Memory?

When running LLMs locally, memory is the primary constraint. A 7-billion-parameter model quantised to INT4 still requires several gigabytes of RAM. Understanding how memory is distributed across CPU, GPU, and NPU helps you:

- **Choose the right model size** for your hardware.
- **Detect memory pressure** before the system starts swapping to disc.
- **Verify hardware acceleration** is actually being used (for example, confirming that inference is running on the GPU rather than falling back to CPU).
- **Optimise execution provider** selection (CPU, CUDA, DirectML) based on real usage data.

## Getting Started

### Installation

```bash
pip install -e .
```

For the Foundry Local engine, install the system daemon:

```bash
# macOS
brew install microsoft/foundrylocal/foundrylocal

# Windows
winget install Microsoft.FoundryLocal
```

### Start the Server

```bash
# Foundry Local engine (macOS/Windows)
ofoundry serve phi-3.5-mini

# ORT engine (all platforms)
ofoundry serve --engine ort ./models/phi3-mini-int4
```

### Open the Dashboard

Navigate to `http://127.0.0.1:1573/` in your browser. The dashboard loads automatically and begins polling for memory statistics.

### Interactive Chat (Terminal)

If you prefer the command line:

```bash
ofoundry run phi-3.5-mini
ofoundry run --engine ort ./models/phi3-mini-int4
```

## Embedding Support

oFoundry also supports text embeddings via a dedicated ONNX-only engine. Supported models include Qwen3-Embedding (0.6B, 4B, 8B) and EmbeddingGemma (300M).

```bash
# One-shot embedding
ofoundry embed ./models/Qwen3-Embedding-0.6B "Hello, world!"

# Embedding server
ofoundry serve-embed ./models/Qwen3-Embedding-0.6B
```

The embedding server exposes an OpenAI-compatible `/v1/embeddings` endpoint and includes the same dashboard with memory monitoring.

## Tool Calling

Both engines support the OpenAI function-calling interface. The Foundry Local engine forwards tools natively to the daemon, whilst the ORT engine serialises tool definitions into the system prompt and parses structured `<tool_call>` blocks from the model output.

This enables building agent workflows where the model can request external function calls (for example, fetching weather data or querying a database), with the results fed back into the conversation.

## Docker Support

A CPU-only Docker image is provided for containerised deployments on both `linux/amd64` and `linux/arm64`:

```bash
docker buildx build --platform=linux/amd64 \
  -f docker/Dockerfile.cpu -t ofoundry-cpu .

docker run -d --rm \
  -p 1573:1573 \
  -v /path/to/models/phi3-mini-int4:/models \
  ofoundry-cpu serve -m /models -E ort --host 0.0.0.0 --port 1573
```

## Architecture Overview

```
ofoundry/
├── engine_foundrylocal.py  # FoundryEngine: wraps foundry-local-sdk
├── engine_ort.py           # OrtEngine: wraps onnxruntime-genai
├── engine_embedding.py     # EmbeddingEngine: ONNX-only
├── chat.py                 # Interactive multi-turn CLI chat loop
├── server.py               # FastAPI server (chat, embeddings, dashboard)
├── monitor.py              # System resource monitoring (CPU, GPU, NPU)
├── cli.py                  # Typer CLI
├── config.py               # ServerConfig dataclass
├── tools.py                # Tool-calling utilities
└── static/
    └── index.html          # Web dashboard UI
```

The monitoring module (`monitor.py`) detects available hardware using platform-native tools: `psutil` for CPU memory, `nvidia-smi` for NVIDIA GPUs, WMI for Windows GPUs, and PnP device enumeration for NPUs. All data is exposed via a `/api/system/memory` endpoint that the dashboard polls.

## What is Next

oFoundry is under active development. Recent additions and planned features include:

- ✅ Runtime model switching via dashboard dropdown
- ✅ Inference parameter tuning (temperature, top-p, top-k) with presets
- ✅ Performance metrics (TTFT, tok/s) displayed per response
- ✅ NPU-accelerated model badges (⚡) in model selector
- ✅ Conversation history with export (JSON/Markdown)
- ✅ Server-side model warmup for faster first responses
- ✅ Optimised streaming pipeline following [Foundry Local best practices](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/reference/reference-best-practice)
- Model conversion via `ofoundry convert` (wrapping the onnxruntime-genai model builder)
- Additional execution providers (CUDA, DirectML)
- Enhanced NPU memory reporting as driver tooling improves
- Historical memory usage charts in the dashboard

## Contributing

oFoundry is licensed under Apache 2.0. Contributions are welcome. Whether it is a bug fix, a new feature, or improved documentation, please open an issue or submit a pull request.

---

*oFoundry: local LLM inference, made simple.*
