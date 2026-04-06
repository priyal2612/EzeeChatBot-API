from fastapi import APIRouter, HTTPException
from app.models.schemas import BotStats
from app.utils.store import bot_exists, get_stats

router = APIRouter()


@router.get("/stats/{bot_id}", response_model=BotStats)
def stats(bot_id: str):
    """
    Returns usage statistics for a bot:
      - total_messages: total /chat calls served
      - avg_latency_ms: mean end-to-end response time (wall clock, ms)
      - estimated_cost_usd: blended token cost at $10/1M tokens
      - unanswered_questions: calls where the bot could not find the answer
      - chunks_in_kb: number of vector chunks in the knowledge base
    """
    if not bot_exists(bot_id):
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found.")
    return get_stats(bot_id)
