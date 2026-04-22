from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
import asyncio
import traceback

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBALS
model = None
index = None
texts = None
ready = False

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
async def load_all():
    global model, index, texts, ready

    try:
        print("🔄 Loading embedding model...")
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        print("🔄 Loading FAISS...")
        index = faiss.read_index("index.faiss")

        print("🔄 Loading texts...")
        with open("texts.json", "r", encoding="utf-8") as f:
            texts = json.load(f)

        ready = True
        print("✅ SYSTEM READY (RAG ONLY)")

    except Exception as e:
        print("❌ LOAD ERROR:", str(e))
        traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_all())

# ─────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ready" if ready else "loading",
        "faiss_loaded": index is not None
    }

# ─────────────────────────────────────────
# RETRIEVE
# ─────────────────────────────────────────
def retrieve(query, k=3):
    try:
        q_embed = model.encode([query]).astype("float32")
        D, I = index.search(q_embed, k)
        return [texts[i] for i in I[0] if 0 <= i < len(texts)]
    except:
        traceback.print_exc()
        return []

# ─────────────────────────────────────────
# CHAT (RAG ONLY)
# ─────────────────────────────────────────
@app.get("/chat")
async def chat(q: str):
    if not q.strip():
        return {"response": "Ask something valid."}

    if not ready:
        return {"response": "Server loading... try again."}

    try:
        results = await asyncio.to_thread(retrieve, q)

        if not results:
            return {"response": "No relevant information found."}

        # 🔥 Clean formatted answer
        answer = "Here’s what I found:\n\n"

        for i, res in enumerate(results[:3], 1):
            answer += f"{i}. {res}\n\n"

        return {"response": answer.strip()}

    except Exception as e:
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return {"response": f"ERROR: {str(e)}"}

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)