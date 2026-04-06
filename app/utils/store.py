"""
In-memory store for bot metadata, embeddings, and stats.

In production this would be backed by:
  - A vector DB (Pinecone, Qdrant, pgvector) for chunk embeddings
  - Redis or Postgres for stats

For this assessment we use in-process dicts so the entire project runs
with zero external infrastructure.
"""

from typing import Dict, Any
import threading

# bot_id -> {
#   "name": str,
#   "chunks": [{"text": str, "embedding": List[float], "metadata": dict}],
#   "stats": {
#       "total_messages": int,
#       "latencies_ms": List[float],
#       "total_tokens": int,
#       "unanswered": int,
#   }
# }
_store: Dict[str, Any] = {}
_lock = threading.Lock()


def init_store():
    """Called at startup — nothing to do for in-memory store."""
    pass


def create_bot(bot_id: str, name: str | None = None):
    with _lock:
        _store[bot_id] = {
            "name": name or bot_id,
            "chunks": [],
            "stats": {
                "total_messages": 0,
                "latencies_ms": [],
                "total_tokens": 0,
                "unanswered": 0,
            },
        }


def bot_exists(bot_id: str) -> bool:
    return bot_id in _store


def get_bot(bot_id: str) -> dict:
    if bot_id not in _store:
        raise KeyError(f"Bot '{bot_id}' not found")
    return _store[bot_id]


def add_chunks(bot_id: str, chunks: list):
    with _lock:
        _store[bot_id]["chunks"].extend(chunks)


def get_chunks(bot_id: str) -> list:
    return _store[bot_id]["chunks"]


def record_message(bot_id: str, latency_ms: float, tokens_used: int, unanswered: bool):
    with _lock:
        stats = _store[bot_id]["stats"]
        stats["total_messages"] += 1
        stats["latencies_ms"].append(latency_ms)
        stats["total_tokens"] += tokens_used
        if unanswered:
            stats["unanswered"] += 1


def get_stats(bot_id: str) -> dict:
    bot = get_bot(bot_id)
    stats = bot["stats"]
    latencies = stats["latencies_ms"]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # GPT-4o pricing (as of 2025): ~$5/1M input tokens, ~$15/1M output tokens
    # We track total tokens and use a blended rate of $10/1M as a conservative estimate.
    estimated_cost = (stats["total_tokens"] / 1_000_000) * 10.0

    return {
        "bot_id": bot_id,
        "bot_name": bot["name"],
        "total_messages": stats["total_messages"],
        "avg_latency_ms": round(avg_latency, 2),
        "estimated_cost_usd": round(estimated_cost, 6),
        "unanswered_questions": stats["unanswered"],
        "chunks_in_kb": len(bot["chunks"]),
    }
