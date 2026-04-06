"""
Chunking Strategy — Sentence-aware sliding window with overlap.

WHY NOT naive character splitting?
  - Splitting at arbitrary character positions severs sentences mid-thought,
    breaking the semantic unit that an embedding model needs to produce a
    meaningful vector.

WHAT WE DO INSTEAD:
  1. Tokenise into sentences using a simple regex (no NLTK dependency).
  2. Build chunks by accumulating sentences until we reach TARGET_TOKENS.
  3. Carry OVERLAP_SENTENCES from the previous chunk into the next so that
     answers that span a chunk boundary can still be retrieved.
  4. Preserve positional metadata (chunk index, approx char offset) so we
     could highlight the source passage in a UI later.

TRADE-OFFS:
  - Regex sentence splitting is good enough for English prose but will
    mis-split on abbreviations like "Dr. Smith" or decimal numbers.
    With more time I'd swap in spaCy's sentencizer.
  - TARGET_TOKENS = 300 is a heuristic. For technical docs with long
    code blocks a larger window (512) works better. For FAQ-style content
    a smaller window (128) gives more precise retrieval.
"""

import re
from typing import List, Dict, Any

TARGET_TOKENS = 300       # ~300 words per chunk — fits in one embedding call
OVERLAP_SENTENCES = 2     # sentences carried forward from previous chunk
APPROX_TOKENS_PER_WORD = 1.33  # rough token estimate (no tiktoken dep)


def _rough_token_count(text: str) -> int:
    return int(len(text.split()) * APPROX_TOKENS_PER_WORD)


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a regex that handles common cases:
    - Period / ! / ? followed by whitespace and a capital letter
    - Newlines that act as paragraph breaks
    """
    # Normalise whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Split on sentence-ending punctuation followed by space+capital,
    # or on blank lines (paragraph breaks).
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?:\n\n+)", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, source_metadata: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Returns a list of chunk dicts:
    {
        "text": str,
        "token_estimate": int,
        "chunk_index": int,
        "metadata": { "source": ..., "char_offset": int }
    }
    """
    source_metadata = source_metadata or {}
    sentences = _split_sentences(text)
    chunks = []
    current_sentences: List[str] = []
    current_tokens = 0
    char_offset = 0
    chunk_index = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = _rough_token_count(sentence)

        # If a single sentence is already over the target (e.g. a very long
        # table row), emit it as its own chunk rather than skipping it.
        if sentence_tokens >= TARGET_TOKENS:
            if current_sentences:
                chunks.append(_build_chunk(current_sentences, chunk_index, char_offset, source_metadata))
                char_offset += sum(len(s) for s in current_sentences)
                chunk_index += 1
                current_sentences = current_sentences[-OVERLAP_SENTENCES:]
                current_tokens = sum(_rough_token_count(s) for s in current_sentences)

            chunks.append(_build_chunk([sentence], chunk_index, char_offset, source_metadata))
            char_offset += len(sentence)
            chunk_index += 1
            current_sentences = []
            current_tokens = 0
            i += 1
            continue

        if current_tokens + sentence_tokens > TARGET_TOKENS and current_sentences:
            # Flush the current chunk
            chunks.append(_build_chunk(current_sentences, chunk_index, char_offset, source_metadata))
            char_offset += sum(len(s) for s in current_sentences)
            chunk_index += 1

            # Keep the last N sentences as overlap for the next chunk
            overlap = current_sentences[-OVERLAP_SENTENCES:]
            current_sentences = overlap
            current_tokens = sum(_rough_token_count(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_tokens += sentence_tokens
        i += 1

    # Flush any remaining sentences
    if current_sentences:
        chunks.append(_build_chunk(current_sentences, chunk_index, char_offset, source_metadata))

    return chunks


def _build_chunk(sentences: List[str], index: int, char_offset: int, metadata: dict) -> Dict[str, Any]:
    text = " ".join(sentences)
    return {
        "text": text,
        "token_estimate": _rough_token_count(text),
        "chunk_index": index,
        "metadata": {**metadata, "char_offset": char_offset},
        "embedding": None,  # populated by embedding service
    }
