"""
Integration tests for the obeaver embedding server.

Assumes the server is already running:
  obeaver serve-embed <model_path> --port 1574

Run with:
  python tests/test_embedding_api.py
"""

from __future__ import annotations

import math
import sys

from openai import OpenAI

BASE_URL = "http://127.0.0.1:1574/v1"
client = OpenAI(base_url=BASE_URL, api_key="unused")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_embedding() -> None:
    """A single string should return one embedding vector."""
    response = client.embeddings.create(
        model="embedding",
        input="Hello, world!",
    )
    assert response.object == "list"
    assert len(response.data) == 1

    emb = response.data[0]
    assert emb.object == "embedding"
    assert emb.index == 0
    assert isinstance(emb.embedding, list)
    assert len(emb.embedding) > 0
    assert all(isinstance(v, float) for v in emb.embedding)

    print(f"[PASS] test_single_embedding  — dim={len(emb.embedding)}")


def test_batch_embeddings() -> None:
    """Multiple inputs should return one vector per input, in order."""
    texts = ["apple", "banana", "cherry"]
    response = client.embeddings.create(
        model="embedding",
        input=texts,
    )
    assert len(response.data) == len(texts)
    for i, item in enumerate(response.data):
        assert item.index == i
        assert len(item.embedding) > 0

    print(f"[PASS] test_batch_embeddings  — {len(texts)} vectors returned")


def test_embedding_is_normalised() -> None:
    """Vectors should be L2-normalised (norm ≈ 1.0)."""
    response = client.embeddings.create(
        model="embedding",
        input="Normalisation check",
    )
    vec = response.data[0].embedding
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-3, f"Expected norm ≈ 1.0, got {norm:.6f}"

    print(f"[PASS] test_embedding_is_normalised  — norm={norm:.6f}")


def test_semantic_similarity() -> None:
    """Semantically similar sentences should score higher than unrelated ones."""
    response = client.embeddings.create(
        model="embedding",
        input=[
            "The cat sat on the mat.",
            "A cat is resting on a rug.",
            "The stock market closed higher today.",
        ],
    )
    vecs = [item.embedding for item in response.data]
    sim_related = cosine_similarity(vecs[0], vecs[1])
    sim_unrelated = cosine_similarity(vecs[0], vecs[2])

    assert sim_related > sim_unrelated, (
        f"Expected similar pair ({sim_related:.4f}) > unrelated pair ({sim_unrelated:.4f})"
    )
    print(
        f"[PASS] test_semantic_similarity  — "
        f"related={sim_related:.4f}  unrelated={sim_unrelated:.4f}"
    )


def test_usage_fields() -> None:
    """Response should include usage token counts."""
    response = client.embeddings.create(
        model="embedding",
        input="Usage field check",
    )
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens > 0

    print(
        f"[PASS] test_usage_fields  — "
        f"prompt_tokens={response.usage.prompt_tokens}  "
        f"total_tokens={response.usage.total_tokens}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_single_embedding,
        test_batch_embeddings,
        test_embedding_is_normalised,
        test_semantic_similarity,
        test_usage_fields,
    ]

    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"\n{failed}/{len(tests)} test(s) FAILED.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed.")
