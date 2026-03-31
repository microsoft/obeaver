"""
OpenAI-compatible FastAPI server for obeaver.

Endpoints:
  GET  /v1/models                — list loaded model
  POST /v1/chat/completions      — chat (streaming + non-streaming)
  GET  /health                   — health check

Launch via:
  obeaver serve ./path/to/model
  # or directly:
  uvicorn obeaver.server:build_app --factory --port 18000
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json as _json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from obeaver.engine_ort import GenerationConfig
from obeaver.monitor import get_all_memory

# ---------------------------------------------------------------------------
# Pydantic models (OpenAI schema subset)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    # content can be a plain string OR a list of content parts (OpenAI
    # multimodal format: [{"type": "text", ...}, {"type": "image_url", ...}])
    content: Optional[str | list[Any]] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


# ---------------------------------------------------------------------------
# Image helpers for VL (Vision-Language) models
# ---------------------------------------------------------------------------


def _extract_image_from_messages(messages: list[dict]) -> str | None:
    """Scan messages for an image reference and return its local path.

    Supports OpenAI multimodal content format::

        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://..."}},
            {"type": "text", "text": "Describe this image"}
        ]}

    Also accepts the olive-recipes compact format::

        {"type": "image", "image": "/local/path.jpg"}

    For HTTP(S) URLs the image is downloaded to a temporary file.
    For ``file://`` or plain local paths, the path is returned directly.

    Returns None if no image is found.
    """
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            # OpenAI format: {"type": "image_url", "image_url": {"url": ...}}
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                return _resolve_image_url(url)
            # Compact format: {"type": "image", "image": "/path"}
            if part.get("type") == "image" and part.get("image"):
                return _resolve_image_url(part["image"])
    return None


def _resolve_image_url(url: str) -> str:
    """Convert an image URL or local path into a local file path.

    - ``file:///path/to/img.jpg`` → ``/path/to/img.jpg``
    - ``/absolute/path.jpg`` or ``./relative/path.jpg`` → resolved to absolute
    - ``https://...`` → downloaded to a temp file, path returned
    """
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return str(Path(parsed.path).resolve())
    if parsed.scheme in ("", "."):
        # Local path — resolve to absolute so og.Images.open() can find it
        return str(Path(url).resolve())
    if parsed.scheme in ("http", "https"):
        return _download_image(url)
    return url


def _download_image(url: str) -> str:
    """Download an image URL to a temporary file and return the path."""
    import urllib.request

    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="obeaver_img_")
    os.close(fd)
    urllib.request.urlretrieve(url, tmp_path)  # noqa: S310 — user-controlled URL
    return tmp_path


def _messages_have_images(messages: list[dict]) -> bool:
    """Return True if any message contains image content parts."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                return True
    return False


def _build_vl_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI multimodal messages to the format expected by
    onnxruntime-genai's ``apply_chat_template``.

    OpenAI sends ``{"type": "image_url", "image_url": {"url": ...}}``;
    the GenAI tokenizer expects ``{"type": "image"}`` (the actual pixel
    data is passed separately via ``og.Images``).
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_parts: list[dict] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                new_parts.append({"type": "image"})
            elif isinstance(part, dict) and part.get("type") == "image" and "image" in part:
                new_parts.append({"type": "image"})
            else:
                new_parts.append(part)
        out.append({**msg, "content": new_parts})
    return out


class EmbeddingRequest(BaseModel):
    model: str = ""
    input: str | list[str]


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = Field(default=1024)
    temperature: Optional[float] = Field(default=1.0)
    top_p: Optional[float] = Field(default=1.0)
    top_k: Optional[int] = Field(default=50)
    repetition_penalty: Optional[float] = Field(default=1.0)
    tools: Optional[list[dict]] = Field(default=None)
    tool_choice: Optional[str | dict] = Field(default=None)


class LoadModelRequest(BaseModel):
    alias: str
    model_id: str | None = None
    device: str | None = None


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

_engine: object | None = None
_embed_engine: object | None = None
_foundry_manager: object | None = None   # FoundryLocalManager instance (if foundry engine)
_engine_type: str = "foundry"

# Reuse a single thread pool across all streaming requests to avoid
# the overhead of creating/destroying threads on every chat turn.
# Use 4 workers so warmup + concurrent requests don't deadlock.
_stream_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def build_embed_app(
    model_path: str | Path,
    execution_provider: str = "cpu",
) -> FastAPI:
    """
    Create a FastAPI application that exposes an OpenAI-compatible
    ``/v1/embeddings`` endpoint powered by the ONNX embedding engine.

    NOTE: Embeddings are ONNX-only — no engine selection is offered.
    """
    global _embed_engine

    from obeaver.engine_embedding import EmbeddingEngine

    _embed_engine = EmbeddingEngine(
        model_path=model_path,
        execution_provider=execution_provider,
    )

    app = FastAPI(title="obeaver-embed", version="0.1.0")

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "obeaver embedding API server", "docs": "/docs"}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": _embed_engine.model_name}  # type: ignore[union-attr]

    @app.get("/api/system/memory")
    async def system_memory() -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_all_memory)

    @app.get("/v1/models")
    async def list_models() -> dict:
        model_id = _embed_engine.model_name  # type: ignore[union-attr]
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "obeaver",
                }
            ],
        }

    @app.post("/v1/embeddings")
    async def create_embeddings(request: EmbeddingRequest) -> JSONResponse:
        assert _embed_engine is not None

        loop = asyncio.get_event_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            vectors: list[list[float]] = await loop.run_in_executor(
                pool, _embed_engine.embed, request.input  # type: ignore[union-attr]
            )

        model_name = request.model or _embed_engine.model_name  # type: ignore[union-attr]
        inputs = request.input if isinstance(request.input, list) else [request.input]
        total_tokens = sum(len(t.split()) for t in inputs)

        data = [
            {"object": "embedding", "embedding": vec, "index": i}
            for i, vec in enumerate(vectors)
        ]
        return JSONResponse({
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        })

    return app


# ---------------------------------------------------------------------------
# Dashboard-only app (no model loaded)
# ---------------------------------------------------------------------------


def build_dashboard_app(engine_type: str = "foundry") -> FastAPI:
    """Create a dashboard FastAPI app with model listing, loading, and chat.

    ``engine_type='foundry'``  — lists Foundry Local cached models;
    loads via FoundryEngine using the real model ID.

    ``engine_type='ort'``      — scans ``./models`` for local ONNX dirs
    (ignoring cache_dir and embedding models); loads via OrtEngine.
    """
    import sys as _sys

    app = FastAPI(title="obeaver-dashboard", version="0.1.0")

    _dash_state: dict = {
        "engine": None,          # loaded engine instance
        "engine_type": engine_type,
        "current_model": None,   # display name of loaded model
        "is_vl": False,          # whether current model is VL
    }

    # ---- Foundry Local manager (only for foundry engine) ----
    _dashboard_manager = None
    if engine_type == "foundry" and not _sys.platform.startswith("linux"):
        try:
            from foundry_local import FoundryLocalManager
            _dashboard_manager = FoundryLocalManager(bootstrap=True)
        except Exception:
            pass

    # ---- Static UI ----
    _static_dir = Path(__file__).parent / "static"
    if _static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/")
    async def ui_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/index.html")

    @app.get("/health")
    async def health() -> dict:
        engine_label = "Foundry Local" if engine_type == "foundry" else "ORT GenAI"
        resp: dict = {"status": "ok", "mode": "dashboard", "engine": engine_label}
        if _dash_state["current_model"]:
            resp["model"] = _dash_state["current_model"]
        return resp

    @app.get("/api/image-proxy")
    async def image_proxy(path: str) -> "FileResponse":
        """Serve a local image file so the browser can display it.

        Only allows image file extensions to prevent arbitrary file reads.
        """
        from fastapi.responses import FileResponse

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        allowed_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
        if resolved.suffix.lower() not in allowed_ext:
            raise HTTPException(status_code=400, detail="Not an image file")
        return FileResponse(str(resolved))

    @app.get("/api/system/memory")
    async def system_memory() -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_all_memory)

    def _scan_ort_models() -> list[dict]:
        """Scan the configured ORT models directory for local ONNX model directories."""
        from obeaver.config import get_ort_models_dir

        models_dir = get_ort_models_dir()
        if not models_dir.is_dir():
            return []
        skip_names = {"cache_dir", ".ds_store"}
        skip_keywords = {"embedding", "embed"}
        results = []
        for child in sorted(models_dir.iterdir()):
            if not child.is_dir():
                continue
            name_lower = child.name.lower()
            if name_lower in skip_names:
                continue
            if any(kw in name_lower for kw in skip_keywords):
                continue
            results.append({
                "alias": child.name,
                "id": str(child),
                "device": "cpu",
                "size_mb": 0,
            })
        return results

    def _scan_foundrylocal_models() -> list[dict]:
        """Scan the configured Foundry Local models directory for local model directories."""
        from obeaver.config import get_foundrylocal_models_dir

        models_dir = get_foundrylocal_models_dir()
        if not models_dir.is_dir():
            return []
        results = []
        # Foundry Local models are stored as <vendor>/<model_name> (e.g. Microsoft/Phi-4-mini...)
        for vendor in sorted(models_dir.iterdir()):
            if not vendor.is_dir() or vendor.name.startswith("."):
                continue
            for child in sorted(vendor.iterdir()):
                if not child.is_dir():
                    continue
                results.append({
                    "alias": child.name,
                    "id": str(child),
                    "device": "cpu",
                    "size_mb": 0,
                })
        return results

    @app.get("/api/models/available")
    async def available_models() -> dict:
        current = _dash_state["current_model"]
        engine_label = "foundry" if engine_type == "foundry" else "ort"

        if engine_type == "ort":
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, _scan_ort_models)
            return {"current": current, "engine": engine_label, "models": models}

        # Foundry mode
        if _dashboard_manager is None:
            # No SDK available — scan foundrylocal subfolder directly
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, _scan_foundrylocal_models)
            return {"current": current, "engine": engine_label, "models": models}

        def _list() -> dict:
            try:
                cached = _dashboard_manager.list_cached_models()
                seen: set[str] = set()
                models = []
                for m in cached:
                    alias_lower = m.alias.lower()
                    if any(skip in alias_lower for skip in ('whisper', 'speech', 'embed')):
                        continue
                    key = f"{m.alias}:{m.device_type}"
                    if key in seen:
                        continue
                    seen.add(key)
                    models.append({
                        "alias": m.alias,
                        "id": m.id,
                        "device": m.device_type,
                        "size_mb": m.file_size if hasattr(m, 'file_size') else 0,
                    })
                return {"current": current, "engine": engine_label, "models": models}
            except Exception as exc:
                return {"current": current, "engine": engine_label, "models": [], "error": str(exc)}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    @app.get("/v1/models")
    async def list_models() -> dict:
        if _dash_state["engine"] is None:
            return {"object": "list", "data": []}
        model_id = _dash_state["engine"].model_name
        return {
            "object": "list",
            "data": [{
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "obeaver",
            }],
        }

    @app.post("/api/models/load")
    async def load_model(req: LoadModelRequest) -> dict:
        """Load a model in-process. VL models always use OrtEngine."""
        import concurrent.futures

        model_identifier = req.model_id or req.alias

        # Detect VL model to force ORT engine
        _is_vl = False
        model_path = Path(model_identifier)
        if model_path.is_dir():
            _is_vl = (
                (model_path / "vision.onnx").exists()
                or any((c / "vision.onnx").exists() for c in model_path.iterdir() if c.is_dir())
            )

        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                if _is_vl or engine_type == "ort":
                    # VL models always use ORT engine
                    from obeaver.engine_ort import OrtEngine
                    new_engine = await loop.run_in_executor(
                        pool,
                        lambda: OrtEngine(
                            model_path=model_identifier,
                            execution_provider="cpu",
                        ),
                    )
                else:
                    from obeaver.engine_foundrylocal import FoundryEngine
                    new_engine = await loop.run_in_executor(
                        pool,
                        lambda: FoundryEngine(
                            model_alias=model_identifier,
                            device=req.device if req.device and req.device.upper() != "CPU" else None,
                        ),
                    )
            _dash_state["engine"] = new_engine
            _dash_state["current_model"] = req.alias
            _dash_state["is_vl"] = _is_vl
            return {"status": "ok", "model": new_engine.model_name, "is_vl": _is_vl}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse | StreamingResponse:
        if _dash_state["engine"] is None:
            raise HTTPException(status_code=400, detail="No model loaded. Select a model first.")

        eng = _dash_state["engine"]
        messages = [m.to_dict() for m in request.messages]
        config = GenerationConfig(
            max_new_tokens=request.max_tokens or 1024,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            top_k=request.top_k or 50,
            repetition_penalty=request.repetition_penalty or 1.0,
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = request.model or eng.model_name

        # ---- VL (Vision-Language) multimodal path ---------------------
        if _messages_have_images(messages):
            if not _dash_state.get("is_vl") or not getattr(eng, "supports_multimodal", False):
                raise HTTPException(
                    status_code=400,
                    detail="Current model does not support image input. Load a VL model first.",
                )
            image_path = _extract_image_from_messages(messages)
            vl_messages = _build_vl_messages(messages)

            if request.stream:
                async def _vl_stream() -> AsyncIterator[bytes]:
                    created = int(time.time())
                    import queue as _queue
                    token_q: _queue.Queue[str | None] = _queue.Queue(maxsize=64)
                    loop = asyncio.get_event_loop()

                    def _produce() -> None:
                        try:
                            for frag in eng.stream_multimodal(
                                messages=vl_messages,
                                image_path=image_path,
                                config=config,
                            ):
                                token_q.put(frag)
                        finally:
                            token_q.put(None)

                    loop.run_in_executor(_stream_pool, _produce)
                    while True:
                        fragment = await loop.run_in_executor(
                            None, lambda: token_q.get(timeout=300)
                        )
                        if fragment is None:
                            break
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": fragment},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {_json.dumps(data, separators=(',',':'))}\n\n".encode()
                    done_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {_json.dumps(done_data, separators=(',',':'))}\n\n".encode()
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    _vl_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )

            # Non-streaming VL
            loop = asyncio.get_event_loop()
            full_text = await loop.run_in_executor(
                _stream_pool,
                lambda: "".join(eng.stream_multimodal(
                    messages=vl_messages,
                    image_path=image_path,
                    config=config,
                )),
            )
            return JSONResponse({
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        # ---- Normal text chat path -----------------------------------
        if request.stream:
            async def _stream() -> AsyncIterator[bytes]:
                created = int(time.time())
                import queue as _queue
                token_q: _queue.Queue[str | None] = _queue.Queue(maxsize=64)
                loop = asyncio.get_event_loop()

                def _produce() -> None:
                    try:
                        for frag in eng.stream(messages=messages, config=config):
                            token_q.put(frag)
                    finally:
                        token_q.put(None)

                loop.run_in_executor(_stream_pool, _produce)
                while True:
                    fragment = await loop.run_in_executor(
                        None, lambda: token_q.get(timeout=300)
                    )
                    if fragment is None:
                        break
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": fragment},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {_json.dumps(data, separators=(',',':'))}\n\n".encode()
                done_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {_json.dumps(done_data, separators=(',',':'))}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming
        loop = asyncio.get_event_loop()
        full_text = await loop.run_in_executor(
            _stream_pool,
            lambda: eng.complete(messages=messages, config=config)
            if hasattr(eng, "complete")
            else "".join(eng.stream(messages=messages, config=config)),
        )
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    return app


def build_app(
    model_path: str | Path,
    engine_type: str = "foundry",
    execution_provider: str = "cpu",
) -> FastAPI:
    """Create the FastAPI application and load the model via the chosen engine.

    This is the API-only mode — no dashboard UI is served.
    """
    global _engine

    global _foundry_manager, _engine_type
    _engine_type = engine_type

    if engine_type == "foundry":
        from obeaver.engine_foundrylocal import FoundryEngine
        _engine = FoundryEngine(
            model_alias=str(model_path),
            device=None if execution_provider == "cpu" else execution_provider,
        )
        # Reuse the manager reference from the engine to avoid a second bootstrap
        try:
            _foundry_manager = getattr(_engine, '_manager', None)
            if _foundry_manager is None:
                from foundry_local import FoundryLocalManager
                _foundry_manager = FoundryLocalManager(bootstrap=True)
        except Exception:
            _foundry_manager = None
    else:
        from obeaver.engine_ort import OrtEngine
        _engine = OrtEngine(model_path=model_path, execution_provider=execution_provider)

    app = FastAPI(title="obeaver", version="0.1.0")

    # ---- Warm up the model on startup ----
    @app.on_event("startup")
    async def _warmup_model() -> None:
        import logging
        log = logging.getLogger("obeaver")
        log.info("Warming up model with a short inference...")
        try:
            loop = asyncio.get_event_loop()
            warmup_msgs = [{"role": "user", "content": "Hi"}]
            warmup_cfg = GenerationConfig(max_new_tokens=1, temperature=0.0)
            await loop.run_in_executor(
                _stream_pool,
                lambda: "".join(_engine.stream(messages=warmup_msgs, config=warmup_cfg)),  # type: ignore
            )
            log.info("Model warmup complete.")
        except Exception as exc:
            log.warning("Model warmup failed (non-fatal): %s", exc)

    _engine_label = engine_type

    # ---- Routes ----

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "obeaver API server", "docs": "/docs"}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": _engine.model_name, "engine": _engine_label}  # type: ignore[union-attr]

    @app.get("/api/system/memory")
    async def system_memory() -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_all_memory)

    @app.get("/v1/models")
    async def list_models() -> dict:
        model_id = _engine.model_name  # type: ignore[union-attr]
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "obeaver",
                }
            ],
        }

    # ---- Model switching (Foundry engine only) ----

    @app.get("/api/models/available")
    async def available_models() -> dict:
        """List cached models the user can switch to."""
        current = _engine.model_name  # type: ignore[union-attr]
        if _foundry_manager is None:
            return {"current": current, "models": [{"alias": current, "id": current, "device": "cpu", "size_mb": 0}]}

        def _list() -> dict:
            try:
                cached = _foundry_manager.list_cached_models()  # type: ignore[union-attr]
                seen: set[str] = set()
                models = []
                for m in cached:
                    alias_lower = m.alias.lower()
                    if any(skip in alias_lower for skip in ('whisper', 'speech', 'embed')):
                        continue
                    key = f"{m.alias}:{m.device_type}"
                    if key in seen:
                        continue
                    seen.add(key)
                    models.append({
                        "alias": m.alias,
                        "id": m.id,
                        "device": m.device_type,
                        "size_mb": m.file_size if hasattr(m, 'file_size') else 0,
                    })
                return {"current": current, "engine": _engine_type, "models": models}
            except Exception as exc:
                return {"current": current, "engine": _engine_type, "models": [], "error": str(exc)}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    @app.post("/api/models/load")
    async def load_model(req: LoadModelRequest) -> dict:
        """Switch the active model (Foundry engine only)."""
        global _engine
        if _foundry_manager is None:
            raise HTTPException(status_code=400, detail="Model switching is only supported with the Foundry engine.")
        try:
            from obeaver.engine_foundrylocal import FoundryEngine
            loop = asyncio.get_event_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                new_engine = await loop.run_in_executor(
                    pool,
                    lambda: FoundryEngine(
                        model_alias=req.alias,
                        device=req.device if req.device and req.device.upper() != "CPU" else None,
                    ),
                )
            _engine = new_engine
            return {"status": "ok", "model": _engine.model_name}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse | StreamingResponse:
        assert _engine is not None

        messages = [m.to_dict() for m in request.messages]
        config = GenerationConfig(
            max_new_tokens=request.max_tokens or 1024,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            top_k=request.top_k or 50,
            repetition_penalty=request.repetition_penalty or 1.0,
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = request.model or _engine.model_name

        # ---- VL (Vision-Language) multimodal path ---------------------
        if _messages_have_images(messages):
            if _engine_type != "ort" or not getattr(_engine, "supports_multimodal", False):
                raise HTTPException(
                    status_code=400,
                    detail="Current engine/model does not support image input. "
                           "Use a VL model loaded with the ORT engine.",
                )
            image_path = _extract_image_from_messages(messages)
            vl_messages = _build_vl_messages(messages)

            if request.stream:
                return StreamingResponse(
                    _stream_multimodal_response(
                        vl_messages, image_path, config, completion_id, model_name,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )

            loop = asyncio.get_event_loop()
            full_text = await loop.run_in_executor(
                _stream_pool,
                lambda: "".join(
                    _engine.stream_multimodal(  # type: ignore[union-attr]
                        messages=vl_messages,
                        image_path=image_path,
                        config=config,
                    )
                ),
            )
            return JSONResponse(
                _build_full_response(full_text, completion_id, model_name)
            )

        # ---- Tool calling path ----------------------------------------
        # When tools are provided we always use engine.chat() which returns
        # a structured ChatResponse.  Streaming is emulated when requested.
        if request.tools:
            loop = asyncio.get_event_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                chat_resp = await loop.run_in_executor(
                    pool,
                    lambda: _engine.chat(  # type: ignore[union-attr]
                        messages=messages,
                        tools=request.tools,
                        config=config,
                    ),
                )

            if chat_resp.tool_calls:
                resp_dict = _build_tool_call_response(
                    chat_resp.tool_calls, completion_id, model_name
                )
            else:
                resp_dict = _build_full_response(
                    chat_resp.content or "", completion_id, model_name
                )

            if request.stream:
                # Emit as a single SSE chunk so streaming clients are happy
                return StreamingResponse(
                    _single_chunk_stream(resp_dict, completion_id, model_name),
                    media_type="text/event-stream",
                )
            return JSONResponse(resp_dict)

        # ---- Normal (no tools) path -----------------------------------
        if request.stream:
            return StreamingResponse(
                _stream_response(messages, config, completion_id, model_name),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        # Run non-streaming inference in thread pool to avoid blocking
        # the async event loop (engine calls are synchronous I/O).
        # Use engine.complete() (stream=False against the daemon) which is
        # much faster than consuming a streaming iterator and joining.
        loop = asyncio.get_event_loop()
        _eng = _engine
        full_text = await loop.run_in_executor(
            _stream_pool,
            lambda: _eng.complete(messages=messages, config=config)  # type: ignore[union-attr]
            if hasattr(_eng, "complete")
            else "".join(_eng.stream(messages=messages, config=config)),  # type: ignore[union-attr]
        )

        return JSONResponse(
            _build_full_response(full_text, completion_id, model_name)
        )

    return app


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


async def _stream_response(
    messages: list[dict],
    config: GenerationConfig,
    completion_id: str,
    model_name: str,
) -> AsyncIterator[bytes]:
    assert _engine is not None
    loop = asyncio.get_event_loop()

    created = int(time.time())

    # Run the entire streaming generator in a single thread using a Queue.
    # This avoids per-token run_in_executor overhead and keeps the generator
    # alive in one thread rather than hopping back and forth.
    import queue as _queue
    token_q: _queue.Queue[str | None] = _queue.Queue(maxsize=64)
    _sentinel = None

    def _produce() -> None:
        try:
            for frag in _engine.stream(messages=messages, config=config):
                token_q.put(frag)
        finally:
            token_q.put(_sentinel)  # signal end

    loop.run_in_executor(_stream_pool, _produce)

    while True:
        # Use run_in_executor for the blocking queue.get so we don't block
        # the event loop while waiting for the next token.
        fragment = await loop.run_in_executor(
            None, lambda: token_q.get(timeout=300)
        )
        if fragment is _sentinel:
            break
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": fragment},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {_json.dumps(data, separators=(',',':'))}\n\n".encode()

    yield b"data: [DONE]\n\n"


def _build_stream_chunk(fragment: str, completion_id: str, model_name: str) -> bytes:
    data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": fragment},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {_json.dumps(data, separators=(',',':'))}\n\n".encode()


def _build_full_response(full_text: str, completion_id: str, model_name: str) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,   # accurate counts not tracked in prototype
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


def _build_tool_call_response(
    tool_calls: list,
    completion_id: str,
    model_name: str,
) -> dict:
    import json as _json

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc.to_openai_dict() for tc in tool_calls],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


async def _single_chunk_stream(
    resp_dict: dict,
    completion_id: str,
    model_name: str,
) -> AsyncIterator[bytes]:
    """Emit a single non-streaming response as SSE so stream=True clients work."""
    import json as _json

    choice = resp_dict["choices"][0]["message"]
    # Emit a delta chunk with the full content / tool_calls
    delta: dict = {"role": "assistant"}
    if choice.get("content") is not None:
        delta["content"] = choice["content"]
    if choice.get("tool_calls"):
        delta["tool_calls"] = choice["tool_calls"]

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    yield f"data: {_json.dumps(chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _stream_multimodal_response(
    messages: list[dict],
    image_path: str | None,
    config: GenerationConfig,
    completion_id: str,
    model_name: str,
) -> AsyncIterator[bytes]:
    """SSE streaming for VL (multimodal) inference via OrtEngine.stream_multimodal."""
    assert _engine is not None
    loop = asyncio.get_event_loop()
    created = int(time.time())

    import queue as _queue
    token_q: _queue.Queue[str | None] = _queue.Queue(maxsize=64)
    _sentinel = None

    def _produce() -> None:
        try:
            for frag in _engine.stream_multimodal(  # type: ignore[union-attr]
                messages=messages,
                image_path=image_path,
                config=config,
            ):
                token_q.put(frag)
        finally:
            token_q.put(_sentinel)

    loop.run_in_executor(_stream_pool, _produce)

    while True:
        fragment = await loop.run_in_executor(
            None, lambda: token_q.get(timeout=300)
        )
        if fragment is _sentinel:
            break
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": fragment},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {_json.dumps(data, separators=(',',':'))}\n\n".encode()

    yield b"data: [DONE]\n\n"
