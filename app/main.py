from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, chat, stats
from app.utils.store import init_store

app = FastAPI(
    title="EzeeChatBot API",
    description="A minimal RAG chatbot API — upload a knowledge base, get a grounded chatbot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_store()

app.include_router(upload.router, tags=["Upload"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(stats.router, tags=["Stats"])

@app.get("/health")
def health():
    return {"status": "ok"}
