"""
Embedding engine — ONNX Runtime only.

Wraps onnxruntime inference for text embedding ONNX models such as
  onnx-community/embeddinggemma-300m-ONNX
Tokenization is handled with HuggingFace transformers' AutoTokenizer.

NOTE: Embedding is ONNX-only.  Unlike the chat engines (OrtEngine /
FoundryEngine), there is no alternative engine backend for this task.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Union

import numpy as np


class EmbeddingEngine:
    """
    ONNX-only embedding engine.

    Loads a text-embedding ONNX model and its tokenizer from a local
    directory (or a path resolved by the caller from the HuggingFace Hub).

    Workflow:
      1. Tokenize input text(s) with AutoTokenizer.
      2. Run the ONNX session.
      3. Mean-pool the last hidden state over non-padding tokens.
      4. L2-normalise the resulting vectors.

    Thread-safety: the ONNX session is stateless; run() calls are
    protected by a lock to be safe across provider implementations.
    """

    def __init__(self, model_path: str | Path, execution_provider: str = "cpu") -> None:
        try:
            import onnxruntime as ort  # noqa: F401  (import check only)
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for the embedding engine.  "
                "Install it with:  pip install onnxruntime"
            ) from e

        try:
            from transformers import AutoTokenizer  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "transformers is required for the embedding engine.  "
                "Install it with:  pip install transformers"
            ) from e

        # Suppress the "PyTorch was not found" advisory — only the tokenizer
        # is used here; the ONNX model is loaded by onnxruntime, not PyTorch.
        import transformers.utils.logging as _hf_logging
        _hf_logging.set_verbosity_error()

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        onnx_file = self._find_onnx_model(model_path)

        ep_map: dict[str, str] = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
        }
        provider = ep_map.get(execution_provider.lower(), "CPUExecutionProvider")

        import onnxruntime as ort

        self._session = ort.InferenceSession(str(onnx_file), providers=[provider])
        self._lock = threading.Lock()
        self._model_path = model_path
        self._model_name = model_path.name

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Cache which inputs the model actually accepts
        self._input_names: set[str] = {inp.name for inp in self._session.get_inputs()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_onnx_model(model_path: Path) -> Path:
        """Return the ONNX model file within *model_path*."""
        # Prefer a file literally named model.onnx
        for candidate in sorted(model_path.glob("*.onnx")):
            if candidate.name == "model.onnx":
                return candidate

        # Any .onnx at the top level
        top_level = sorted(model_path.glob("*.onnx"))
        if top_level:
            return top_level[0]

        # Recurse one extra level (some repos put it in onnx/ subdirectory)
        deep = sorted(model_path.glob("**/*.onnx"))
        if deep:
            return deep[0]

        raise FileNotFoundError(
            f"No .onnx file found under {model_path}.  "
            "Make sure the directory contains a valid ONNX embedding model."
        )

    @staticmethod
    def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean-pool *last_hidden_state* over the non-padding token positions."""
        # attention_mask: [batch, seq_len]  -> [batch, seq_len, 1]
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(last_hidden_state * mask, axis=1)
        count = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / count

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return vectors / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """
        Compute L2-normalised embeddings for one or more texts.

        Parameters
        ----------
        texts:
            A single string or a list of strings.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        # Build the inputs the model actually requires
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        ort_inputs: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            ort_inputs["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            ort_inputs["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names and "token_type_ids" in encoded:
            ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
        if "position_ids" in self._input_names:
            # Standard HuggingFace approach: cumulative sum of the attention mask
            # gives correct positions even for padded sequences.
            # Padding positions (mask == 0) are set to 1 (a safe dummy value).
            position_ids = np.cumsum(attention_mask, axis=1) - 1
            position_ids = np.where(attention_mask == 0, 1, position_ids)
            ort_inputs["position_ids"] = position_ids.astype(np.int64)

        with self._lock:
            outputs = self._session.run(None, ort_inputs)

        output: np.ndarray = outputs[0]

        if output.ndim == 3:
            # last_hidden_state: [batch, seq_len, hidden_dim]  -> mean-pool
            embeddings = self._mean_pool(output, attention_mask)
        else:
            # Already pooled: [batch, hidden_dim]
            embeddings = output

        embeddings = self._l2_normalize(embeddings)
        return embeddings.tolist()
