# EzeeChatBot API

A minimal **Retrieval-Augmented Generation (RAG)** chatbot backend.  
Upload a knowledge base (text or URL), get back a `bot_id`, and chat — grounded only in your content.

---

## Quick Start

```bash
# 1. Clone & enter
git clone <repo>
cd EzeeChatBot-API

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run
uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## API Reference

### `POST /upload`
Ingest a knowledge base.

```json
// Text payload
{
  "type": "text",
  "content": "Your document content here...",
  "bot_name": "My Product FAQ"
}

// URL payload
{
  "type": "url",
  "content": "https://example.com/docs/faq"
}
```

Response:
```json
{
  "bot_id": "a3f1c9b2",
  "bot_name": "My Product FAQ",
  "chunks_stored": 14,
  "message": "Knowledge base created. Use bot_id 'a3f1c9b2' to chat."
}
```

---

### `POST /chat`
Ask a question. Returns a **Server-Sent Events** stream.

```json
{
  "bot_id": "a3f1c9b2",
  "user_message": "What is your refund policy?",
  "conversation_history": [
    { "role": "user",      "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help?" }
  ]
}
```

Each SSE event:
```
data: The refund window is

data:  30 days from purchase.

data: [DONE]
```

If the answer is not in the knowledge base:
```
data: I cannot find the answer in the provided knowledge base.

data: [DONE]
```

---

### `GET /stats/{bot_id}`

```json
{
  "bot_id": "a3f1c9b2",
  "bot_name": "My Product FAQ",
  "total_messages": 42,
  "avg_latency_ms": 1340.5,
  "estimated_cost_usd": 0.000312,
  "unanswered_questions": 3,
  "chunks_in_kb": 14
}
```

---

## Chunking Strategy

### What we do
**Sentence-aware sliding window with overlap** (`app/services/chunker.py`):

1. **Sentence tokenisation** — regex split on `.!?` followed by a capital letter, or on blank lines (paragraph breaks). No external NLP dependency.
2. **Accumulate sentences** until the chunk reaches ~300 tokens (≈ 225 words).
3. **Carry 2 sentences of overlap** from the previous chunk into the next. This preserves context at chunk boundaries so a question whose answer spans two chunks can still be retrieved.
4. **Metadata preservation** — each chunk records its `chunk_index` and approximate `char_offset` so the source passage can be highlighted in a UI.

### Why not naive character splitting?
Splitting at every N characters severs sentences mid-thought. The embedding model then encodes a fragment rather than a complete semantic unit, producing a lower-quality vector. Retrieval quality degrades — especially for longer, complex questions.

### Why 300 tokens / 2-sentence overlap?
- **300 tokens** fits comfortably within `text-embedding-3-small`'s 8k-token limit while keeping chunks focused enough for precise retrieval.
- **2-sentence overlap** is a pragmatic minimum. With zero overlap, answers that happen to start at a chunk boundary are invisible to retrieval. With large overlap (e.g. 50%) the index grows and redundant chunks fill the context window.
- For FAQ / short-paragraph content, 128-token chunks with 1-sentence overlap would sharpen retrieval further.

---

## What I Would Do Differently With More Time

1. **Replace regex sentence splitting with spaCy's sentencizer.** Regex mis-splits on abbreviations ("Dr. Smith"), decimals, and bullet points. spaCy handles these correctly with negligible extra latency.

2. **Swap the in-memory cosine search for Qdrant (self-hosted) or pgvector.** Pure-Python cosine scan is O(n) and will become the bottleneck beyond ~5,000 chunks. An ANN index gives sub-millisecond retrieval with ~95% recall at any scale.

3. **Add PDF ingestion via `pdfplumber`.** Most real-world knowledge bases live in PDFs. `pdfplumber` preserves heading hierarchy and tables — both are important for chunking strategy because a table row should not be split across chunks.

4. **Persistent storage for stats.** The in-memory stats dict is lost on restart. A single Postgres table (or even SQLite) would suffice. Token cost estimates would also become more accurate with per-model pricing tables.

5. **Re-ranking pass.** After the top-5 cosine retrieval, a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the candidates against the actual query. This consistently improves answer quality at the cost of one extra local inference call (~50 ms).

---

## Architecture Notes

```
POST /upload
  └─ fetch_url / validate text
      └─ chunk_text()          ← sentence-aware, ~300 tokens, 2-sent overlap
          └─ embed_chunks()    ← text-embedding-3-small via OpenAI
              └─ store → bot_id namespace

POST /chat
  └─ retrieve_top_k()          ← cosine similarity, top-5 chunks
      └─ (relevance guard)     ← if no chunks above 0.30 → fallback
          └─ stream_answer()   ← GPT-4o, system prompt with context
              └─ SSE stream → client
                  └─ record_message() ← latency, tokens, unanswered flag

GET /stats/{bot_id}
  └─ aggregate from in-memory stats dict
```

### Multi-bot isolation
Every `bot_id` is a UUID-derived key. Chunks are stored in a per-bot list. The retrieval function is always scoped to a single bot's chunk list — there is no shared vector namespace and no risk of cross-bot bleed.

---

## Running Tests (example curl)

```bash
# Upload text
curl -s -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"type":"text","content":"Our refund policy is 30 days. Contact support@example.com.","bot_name":"Demo"}' | jq

# Chat (streaming)
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"bot_id":"<bot_id>","user_message":"What is the refund window?","conversation_history":[]}'

# Stats
curl -s http://localhost:8000/stats/<bot_id> | jq
```
