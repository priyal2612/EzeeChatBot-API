"""
Embedding service.

Uses OpenAI text-embedding-3-small:
  - 1536-dim vectors, strong multilingual support
  - ~10x cheaper than text-embedding-ada-002 with better performance
  - Supports Matryoshka dims — can be reduced to 256-dim at query time
    to trade a tiny quality hit for faster dot products (not used here
    for clarity, but worth doing with >100k chunks).

Retrieval:
  - Cosine similarity computed in pure Python (numpy).
  - For a production system with >10k chunks, replace with an ANN index
    (FAISS, Qdrant, or pgvector) for sub-millisecond retrieval.
"""

import os
import math
from typing import List, Dict, Any

from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5  # number of chunks to retrieve per query


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a list of embedding vectors for the given texts."""
    client = _get_client()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach embedding vectors to a list of chunk dicts in-place."""
    texts = [c["text"] for c in chunks]
    vectors = await embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def retrieve_top_k(
    query: str, chunks: List[Dict[str, Any]], k: int = TOP_K
) -> List[Dict[str, Any]]:
    """
    Embed the query, then return the top-k most similar chunks.
    Chunks with no embedding are silently skipped.
    """
    if not chunks:
        return []

    (query_vec,) = await embed_texts([query])

    scored = [
        (chunk, _cosine_similarity(query_vec, chunk["embedding"]))
        for chunk in chunks
        if chunk.get("embedding") is not None
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored[:k]]
