from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import faiss
import traceback
import os
import asyncio
from openai import AsyncOpenAI

# ═══════════════════════════════════════════════════════════
#  Globals — loaded ONCE at startup, reused every request
# ═══════════════════════════════════════════════════════════
model  = None
index  = None
texts  = None
client = None


# ═══════════════════════════════════════════════════════════
#  Lifespan — loads everything BEFORE the first request
#  Render's health check won't pass until this completes,
#  so no request ever hits a cold / unloaded state.
# ═══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, index, texts, client

    # 1. OpenAI async client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set!")
    client = AsyncOpenAI(api_key=api_key)
    print("✅ OpenAI async client ready")

    # 2. Embedding model (CPU-bound, run in thread so loop stays free)
    print("🔄 Loading embedding model...")
    model = await asyncio.to_thread(SentenceTransformer, "paraphrase-MiniLM-L3-v2")
    print("✅ Embedding model loaded")

    # 3. FAISS index
    print("🔄 Loading FAISS index...")
    index = faiss.read_index("index.faiss")
    print(f"✅ FAISS index loaded — {index.ntotal} vectors")

    # 4. Texts
    print("🔄 Loading texts.json...")
    with open("texts.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    print(f"✅ Texts loaded — {len(texts)} entries")

    print("🚀 Backend fully ready — accepting requests")
    yield
    print("🛑 Shutting down")


# ═══════════════════════════════════════════════════════════
#  App
# ═══════════════════════════════════════════════════════════
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────────────────────────────────────
#  Health check — Render pings GET / to confirm service is up
# ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    loaded = all(x is not None for x in [model, index, texts])
    return {
        "status":  "ready" if loaded else "loading",
        "vectors": index.ntotal if index else 0,
        "texts":   len(texts)   if texts  else 0,
    }


# ───────────────────────────────────────────────────────────
#  FAISS retrieval — CPU-bound, isolated so it can be
#  offloaded to a thread without blocking the event loop
# ───────────────────────────────────────────────────────────
def _retrieve(query: str, k: int = 3) -> list:
    q_embed = model.encode([query])
    D, I    = index.search(np.array(q_embed).astype("float32"), k)
    return [texts[i] for i in I[0] if 0 <= i < len(texts)]


# ───────────────────────────────────────────────────────────
#  /chat  — fully async end-to-end
#  • embedding offloaded to thread    → event loop never blocks
#  • OpenAI call is native async      → no thread waste
#  • 25 s hard timeout on OpenAI      → no hanging requests
# ───────────────────────────────────────────────────────────
@app.get("/chat")
async def chat(q: str):
    if not q or not q.strip():
        return {"response": "Please ask a question."}

    q = q.strip()

    # Step 1: embed + retrieve (runs in threadpool)
    try:
        results = await asyncio.to_thread(_retrieve, q)
    except Exception:
        traceback.print_exc()
        return {"response": "Error searching the knowledge base. Please retry."}

    if not results:
        return {"response": "No relevant information found in the knowledge base."}

    context = "\n\n".join(results)

    # Step 2: build prompt
    prompt = (
        "You are an expert in drug discovery and SAR (Structure-Activity Relationship) analysis.\n\n"
        "Use ONLY the context below to answer the question. "
        "Be concise and precise. "
        "If the context does not contain enough information, say so honestly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {q}\n\n"
        "Answer:"
    )

    # Step 3: OpenAI async call with hard timeout
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,   # lower = faster + more factual
                max_tokens=512,    # cap length → faster replies
            ),
            timeout=25.0,          # abandon after 25 s
        )
        answer = response.choices[0].message.content.strip()
        return {"response": answer}

    except asyncio.TimeoutError:
        print("⏰ OpenAI call timed out after 25 s")
        return {"response": "The AI took too long to respond. Please try again."}

    except Exception:
        traceback.print_exc()
        return {"response": "Error generating AI response. Please retry."}


# ═══════════════════════════════════════════════════════════
#  Local entrypoint — Render uses start.sh instead
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)