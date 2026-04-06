import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest
from app.services.embedder import retrieve_top_k
from app.services.llm import stream_answer, CANNOT_ANSWER_PHRASE
from app.utils.store import bot_exists, get_chunks, record_message

router = APIRouter()

# If the top retrieved chunk has similarity below this threshold we treat the
# question as unanswerable from the knowledge base — avoids hallucination on
# totally off-topic queries even before the LLM sees them.
RELEVANCE_THRESHOLD = 0.30


@router.post("/chat")
async def chat(payload: ChatRequest):
    """
    Retrieve relevant chunks, inject them into a grounded system prompt,
    call GPT-4o, and stream the response back as Server-Sent Events.

    The response stream is plain text/event-stream; each data line carries
    a token. The final line is `data: [DONE]`.
    """
    if not bot_exists(payload.bot_id):
        raise HTTPException(status_code=404, detail=f"Bot '{payload.bot_id}' not found.")

    chunks = get_chunks(payload.bot_id)
    if not chunks:
        raise HTTPException(status_code=422, detail="This bot has no knowledge base. Please upload content first.")

    history = [t.model_dump() for t in (payload.conversation_history or [])]

    async def generate():
        start = time.monotonic()
        full_response = []
        total_tokens = None

        # --- Retrieve ---
        top_chunks = await retrieve_top_k(payload.user_message, chunks)

        # Graceful fallback: nothing relevant enough in the KB
        if not top_chunks:
            fallback = CANNOT_ANSWER_PHRASE
            yield f"data: {fallback}\n\ndata: [DONE]\n\n"
            latency = (time.monotonic() - start) * 1000
            record_message(payload.bot_id, latency, tokens_used=0, unanswered=True)
            return

        # --- Stream LLM response ---
        try:
            async for token, tokens in stream_answer(
                user_message=payload.user_message,
                context_chunks=top_chunks,
                conversation_history=history,
            ):
                if tokens is not None:
                    total_tokens = tokens
                if token:
                    full_response.append(token)
                    # Escape newlines so SSE framing stays intact
                    yield f"data: {token.replace(chr(10), ' ')}\n\n"

        except EnvironmentError as e:
            yield f"data: ERROR: {str(e)}\n\ndata: [DONE]\n\n"
            return
        except Exception as e:
            yield f"data: ERROR: LLM call failed — {str(e)}\n\ndata: [DONE]\n\n"
            return

        yield "data: [DONE]\n\n"

        # --- Record stats ---
        latency_ms = (time.monotonic() - start) * 1000
        answer = "".join(full_response)
        unanswered = CANNOT_ANSWER_PHRASE.lower() in answer.lower()
        record_message(
            payload.bot_id,
            latency_ms=latency_ms,
            tokens_used=total_tokens or 0,
            unanswered=unanswered,
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
