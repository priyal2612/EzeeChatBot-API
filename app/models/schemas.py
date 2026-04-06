from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from enum import Enum


class UploadType(str, Enum):
    text = "text"
    url = "url"


class UploadRequest(BaseModel):
    type: UploadType
    content: str  # raw text OR a URL string
    bot_name: Optional[str] = None

    class Config:
        json_schema_extra = {
            "examples": [
                {"type": "text", "content": "Your knowledge base text here..."},
                {"type": "url", "content": "https://example.com/docs"},
            ]
        }


class UploadResponse(BaseModel):
    bot_id: str
    bot_name: Optional[str]
    chunks_stored: int
    message: str


class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    bot_id: str
    user_message: str
    conversation_history: Optional[List[ConversationTurn]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "abc123",
                "user_message": "What is the refund policy?",
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help?"},
                ],
            }
        }


class BotStats(BaseModel):
    bot_id: str
    bot_name: Optional[str]
    total_messages: int
    avg_latency_ms: float
    estimated_cost_usd: float
    unanswered_questions: int
    chunks_in_kb: int
