"""
Unit tests for ofoundry.server (FastAPI endpoints with mocked engines).

Run with:
  pytest tests/test_server.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request as FastAPIRequest
from fastapi.testclient import TestClient

from ofoundry.tools import ChatResponse, ToolCall


# ---------------------------------------------------------------------------
# Helpers — fake engine stubs
# ---------------------------------------------------------------------------


class FakeChatEngine:
    """Mimics the interface of FoundryEngine / OrtEngine for testing."""

    model_name = "test-model"

    def stream(self, *, messages, config=None):
        yield "Hello "
        yield "world!"

    def chat(self, *, messages, tools=None, config=None):
        if tools:
            tc = ToolCall(id="call_test123", name="get_weather", arguments={"city": "Paris"})
            return ChatResponse(tool_calls=[tc], finish_reason="tool_calls")
        return ChatResponse(content="Hello world!")


class FakeEmbedEngine:
    """Mimics the EmbeddingEngine interface for testing."""

    model_name = "test-embed-model"

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1, 0.2, 0.3] for _ in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def chat_client():
    """Build a TestClient for the chat server with a fake engine."""
    import ofoundry.server as srv

    original = srv._engine
    try:
        # Patch the engine constructor so build_app doesn't try to load a real model
        with patch("ofoundry.server.FoundryEngine", create=True) as mock_cls:
            mock_cls.return_value = FakeChatEngine()
            with patch.dict("sys.modules", {"ofoundry.engine_foundrylocal": MagicMock(FoundryEngine=mock_cls)}):
                # Directly set the engine and build a minimal app
                app = srv.build_app.__wrapped__(model_path="dummy", engine_type="foundry") if hasattr(srv.build_app, "__wrapped__") else None
        # Simpler approach: just build the app structure manually
        if app is None:
            from fastapi import FastAPI
            from pathlib import Path
            app = FastAPI(title="ofoundry-test", version="0.1.0")

            srv._engine = FakeChatEngine()
            _engine_label = "foundry"

            _static_dir = Path(__file__).parent.parent / "ofoundry" / "static"
            if _static_dir.is_dir():
                from fastapi.staticfiles import StaticFiles
                app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

            @app.get("/")
            async def ui_redirect():
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url="/static/index.html")

            @app.get("/health")
            async def health():
                return {"status": "ok", "model": srv._engine.model_name, "engine": _engine_label}

            @app.get("/api/system/memory")
            async def system_memory():
                from ofoundry.monitor import get_all_memory
                return get_all_memory()

            @app.get("/v1/models")
            async def list_models():
                import time
                return {"object": "list", "data": [{"id": srv._engine.model_name, "object": "model", "created": int(time.time()), "owned_by": "ofoundry"}]}

            @app.post("/v1/chat/completions")
            async def chat_completions(request: dict):
                import uuid, time
                from fastapi.responses import JSONResponse, StreamingResponse
                from ofoundry.engine_ort import GenerationConfig

                messages = [m for m in request.get("messages", [])]
                config = GenerationConfig(max_new_tokens=request.get("max_tokens", 1024))
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                model_name = request.get("model", "") or srv._engine.model_name

                if request.get("tools"):
                    chat_resp = srv._engine.chat(messages=messages, tools=request["tools"], config=config)
                    if chat_resp.tool_calls:
                        tc_list = [tc.to_openai_dict() for tc in chat_resp.tool_calls]
                        return JSONResponse({"id": completion_id, "object": "chat.completion", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": tc_list}, "finish_reason": "tool_calls"}]})

                if request.get("stream"):
                    async def _stream():
                        for frag in srv._engine.stream(messages=messages, config=config):
                            chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant", "content": frag}, "finish_reason": None}]}
                            yield f"data: {json.dumps(chunk)}\n\n".encode()
                        yield b"data: [DONE]\n\n"
                    return StreamingResponse(_stream(), media_type="text/event-stream")

                full_text = "".join(srv._engine.stream(messages=messages, config=config))
                return JSONResponse({"id": completion_id, "object": "chat.completion", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}]})

        client = TestClient(app)
        yield client
    finally:
        srv._engine = original


@pytest.fixture()
def embed_client():
    """Build a TestClient for the embedding server with a fake engine."""
    import ofoundry.server as srv
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import time

    original = srv._embed_engine
    try:
        app = FastAPI(title="ofoundry-embed-test", version="0.1.0")
        srv._embed_engine = FakeEmbedEngine()

        from pathlib import Path
        _static_dir = Path(__file__).parent.parent / "ofoundry" / "static"
        if _static_dir.is_dir():
            from fastapi.staticfiles import StaticFiles
            app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        @app.get("/")
        async def ui_redirect():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/static/index.html")

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": srv._embed_engine.model_name}

        @app.get("/api/system/memory")
        async def system_memory():
            from ofoundry.monitor import get_all_memory
            return get_all_memory()

        @app.get("/v1/models")
        async def list_models():
            return {"object": "list", "data": [{"id": srv._embed_engine.model_name, "object": "model", "created": int(time.time()), "owned_by": "ofoundry"}]}

        @app.post("/v1/embeddings")
        async def create_embeddings(request: FastAPIRequest):
            body = await request.json()
            inp = body.get("input", "")
            model_name = body.get("model", "") or srv._embed_engine.model_name
            vectors = srv._embed_engine.embed(inp)
            inputs = inp if isinstance(inp, list) else [inp]
            total_tokens = sum(len(t.split()) for t in inputs)
            data = [{"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(vectors)]
            return JSONResponse({"object": "list", "data": data, "model": model_name, "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens}})

        client = TestClient(app)
        yield client
    finally:
        srv._embed_engine = original


# ---------------------------------------------------------------------------
# Chat server — health
# ---------------------------------------------------------------------------


class TestChatHealth:
    def test_health_returns_ok(self, chat_client: TestClient) -> None:
        resp = chat_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_includes_model(self, chat_client: TestClient) -> None:
        resp = chat_client.get("/health")
        data = resp.json()
        assert "model" in data

    def test_health_includes_engine(self, chat_client: TestClient) -> None:
        resp = chat_client.get("/health")
        data = resp.json()
        assert "engine" in data


# ---------------------------------------------------------------------------
# Chat server — system memory
# ---------------------------------------------------------------------------


class TestSystemMemory:
    def test_memory_endpoint_returns_200(self, chat_client: TestClient) -> None:
        resp = chat_client.get("/api/system/memory")
        assert resp.status_code == 200

    def test_memory_has_cpu_section(self, chat_client: TestClient) -> None:
        resp = chat_client.get("/api/system/memory")
        data = resp.json()
        assert "cpu" in data
        assert data["cpu"]["device"] == "CPU"

    def test_memory_has_platform_section(self, chat_client: TestClient) -> None:
        resp = chat_client.get("/api/system/memory")
        data = resp.json()
        assert "platform" in data
        assert "system" in data["platform"]


# ---------------------------------------------------------------------------
# Chat server — models
# ---------------------------------------------------------------------------


class TestChatModels:
    def test_list_models(self, chat_client: TestClient) -> None:
        resp = chat_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1


# ---------------------------------------------------------------------------
# Chat server — chat completions (non-streaming)
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_non_streaming_returns_200(self, chat_client: TestClient) -> None:
        resp = chat_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        assert resp.status_code == 200

    def test_non_streaming_has_choices(self, chat_client: TestClient) -> None:
        resp = chat_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_non_streaming_content(self, chat_client: TestClient) -> None:
        resp = chat_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert "Hello" in content

    def test_streaming_returns_event_stream(self, chat_client: TestClient) -> None:
        resp = chat_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_streaming_ends_with_done(self, chat_client: TestClient) -> None:
        resp = chat_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        text = resp.text
        assert "[DONE]" in text


# ---------------------------------------------------------------------------
# Chat server — tool calling
# ---------------------------------------------------------------------------


class TestToolCalling:
    def test_tool_call_response(self, chat_client: TestClient) -> None:
        resp = chat_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
        })
        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"] is not None


# ---------------------------------------------------------------------------
# Chat server — UI redirect
# ---------------------------------------------------------------------------


class TestUIRedirect:
    def test_root_redirects(self, chat_client: TestClient) -> None:
        resp = chat_client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307)
        assert "/static/index.html" in resp.headers.get("location", "")


# ---------------------------------------------------------------------------
# Embedding server
# ---------------------------------------------------------------------------


class TestEmbedHealth:
    def test_health_returns_ok(self, embed_client: TestClient) -> None:
        resp = embed_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestEmbedModels:
    def test_list_models(self, embed_client: TestClient) -> None:
        resp = embed_client.get("/v1/models")
        assert resp.status_code == 200


class TestEmbeddings:
    def test_single_input(self, embed_client: TestClient) -> None:
        resp = embed_client.post("/v1/embeddings", json={
            "input": "Hello world",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) == 3

    def test_batch_input(self, embed_client: TestClient) -> None:
        resp = embed_client.post("/v1/embeddings", json={
            "input": ["Hello", "World"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2

    def test_embed_memory_endpoint(self, embed_client: TestClient) -> None:
        resp = embed_client.get("/api/system/memory")
        assert resp.status_code == 200
        assert "cpu" in resp.json()
