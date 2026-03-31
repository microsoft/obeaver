"""
Minimal smoke-test for the obeaver package (no real model required).
"""

import sys
import types

import pytest


def _make_mock_og():
    """Return a mock onnxruntime_genai module so tests run without the real library."""
    og = types.ModuleType("onnxruntime_genai")

    class _FakeStream:
        def decode(self, token):
            return chr(65 + (token % 26))  # A–Z cycling

    class _FakeTokenizer:
        def encode(self, text):
            return [ord(c) % 128 for c in text[:8]]

        def create_stream(self):
            return _FakeStream()

    class _FakeParams:
        input_ids = []

        def set_search_options(self, **kwargs):
            pass

    class _FakeGenerator:
        def __init__(self):
            self._count = 0

        def is_done(self):
            return self._count >= 5

        def generate_next_token(self):
            self._count += 1

        def get_next_tokens(self):
            return [self._count]

        def append_tokens(self, tokens):
            pass

        def __del__(self):
            pass

    class _FakeModel:
        pass

    og.Model = lambda path: _FakeModel()
    og.Tokenizer = lambda model: _FakeTokenizer()
    og.GeneratorParams = lambda model: _FakeParams()
    og.Generator = lambda model, params: _FakeGenerator()

    return og


# Patch before importing engine
sys.modules.setdefault("onnxruntime_genai", _make_mock_og())

import tempfile, pathlib


def _mock_model_dir():
    """Create a temp directory that looks like an ONNX GenAI model dir."""
    d = tempfile.mkdtemp()
    pathlib.Path(d, "genai_config.json").write_text("{}")
    pathlib.Path(d, "model.onnx").write_text("")
    return d


def test_engine_stream():
    from obeaver.engine_ort import GenerationConfig, OrtEngine

    model_dir = _mock_model_dir()
    engine = OrtEngine(model_path=model_dir)

    messages = [{"role": "user", "content": "Hello"}]
    tokens = list(engine.stream(messages=messages, config=GenerationConfig(max_new_tokens=5)))
    assert len(tokens) == 5
    assert all(isinstance(t, str) for t in tokens)


def test_format_messages_fallback():
    from obeaver.engine_ort import _format_messages_fallback

    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    result = _format_messages_fallback(msgs)
    assert "Be helpful." in result
    assert "Hi" in result
    assert result.endswith("<|assistant|>")


def test_version():
    from obeaver._version import __version__

    assert isinstance(__version__, str)
    assert "." in __version__
