"""
obeaver — ONNX Runtime GenAI inference server (CPU-first)
Inspired by oMLX, powered by onnxruntime-genai instead of MLX.

Chat engines:  OrtEngine (ONNX), FoundryEngine (Foundry Local)
Embedding:     EmbeddingEngine (ONNX-only, no alternative backend)
"""

from obeaver._version import __version__

__all__ = ["__version__"]
