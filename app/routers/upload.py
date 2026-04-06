import uuid
from fastapi import APIRouter, HTTPException

from app.models.schemas import UploadRequest, UploadResponse, UploadType
from app.services.chunker import chunk_text
from app.services.embedder import embed_chunks
from app.utils.fetcher import fetch_url
from app.utils.store import create_bot, add_chunks

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload(payload: UploadRequest):
    """
    Ingest a knowledge base — either raw text or a URL.

    Steps:
      1. Fetch / validate content.
      2. Chunk with semantic sentence-aware splitting.
      3. Embed each chunk via OpenAI text-embedding-3-small.
      4. Store in the bot's isolated namespace.
      5. Return a bot_id for subsequent /chat requests.
    """
    # --- 1. Acquire content ---
    if payload.type == UploadType.url:
        try:
            text, source = fetch_url(payload.content)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        source_metadata = {"source_type": "url", "url": payload.content}
    else:
        text = payload.content.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Content must not be empty.")
        if len(text) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Text payload exceeds 5 MB limit.")
        source_metadata = {"source_type": "text"}

    # --- 2. Chunk ---
    chunks = chunk_text(text, source_metadata=source_metadata)
    if not chunks:
        raise HTTPException(status_code=422, detail="No meaningful content could be extracted.")

    # --- 3. Embed ---
    try:
        chunks = await embed_chunks(chunks)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {str(e)}")

    # --- 4. Store ---
    bot_id = str(uuid.uuid4())[:8]  # short, human-readable
    create_bot(bot_id, name=payload.bot_name)
    add_chunks(bot_id, chunks)

    return UploadResponse(
        bot_id=bot_id,
        bot_name=payload.bot_name,
        chunks_stored=len(chunks),
        message=f"Knowledge base created. Use bot_id '{bot_id}' to chat.",
    )
