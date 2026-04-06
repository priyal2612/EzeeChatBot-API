"""
LLM service — grounded RAG answer generation.

Design decisions:
  - Model: GPT-4o (default). Can be swapped to gpt-4o-mini via env var
    LLM_MODEL for lower cost in high-volume scenarios.
  - Grounding: The system prompt explicitly instructs the model to answer
    ONLY from the provided context and to say a specific fallback phrase
    if the answer is not there. We detect that phrase in the caller to
    increment the "unanswered" counter.
  - Streaming: yields text tokens as they arrive so the HTTP layer can
    stream them to the client.
  - Token tracking: we accumulate usage from the stream's final chunk
    for cost estimation.
"""

import os
from typing import AsyncIterator, List, Dict, Any

from openai import AsyncOpenAI

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# The exact phrase we instruct the model to use when it cannot answer.
# We match on this to flag unanswered questions — keep it distinctive.
CANNOT_ANSWER_PHRASE = "I cannot find the answer in the provided knowledge base."

SYSTEM_TEMPLATE = """\
You are EzeeChatBot, a helpful assistant that answers questions strictly based on \
the context excerpts provided below. Do not use any external knowledge.

RULES:
1. Answer only from the CONTEXT. If the answer is not there, respond with exactly:
   "{cannot_answer}"
2. Be concise and factual.
3. If quoting, cite the relevant passage briefly.

CONTEXT:
{context}
""".strip()

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['text']}")
    return "\n\n".join(parts)


async def stream_answer(
    user_message: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
) -> AsyncIterator[tuple[str, int | None]]:
    """
    Yields (token_text, total_tokens_or_None) tuples.
    total_tokens is only non-None on the last yield (from the usage chunk).
    """
    client = _get_client()
    context = build_context_block(context_chunks)
    system_prompt = SYSTEM_TEMPLATE.format(
        context=context,
        cannot_answer=CANNOT_ANSWER_PHRASE,
    )

    messages = [{"role": "system", "content": system_prompt}]
    # Inject conversation history (last 10 turns max to control context size)
    for turn in conversation_history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    total_tokens = None
    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.2,  # low temp — we want faithful retrieval, not creativity
        max_tokens=1024,
    )

    async for chunk in stream:
        if chunk.usage:
            total_tokens = chunk.usage.total_tokens

        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content, None

    yield "", total_tokens  # final sentinel with token count
