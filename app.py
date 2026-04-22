from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import json
import faiss
import traceback
import os
import asyncio
from openai import AsyncOpenAI

EMBED_MODEL = "text-embedding-3-small"   # must match what embed.py used

index  = None
texts  = None
client = None
_ready = False


async def _load_assets():
    global index, texts, client, _ready

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set!")
    client = AsyncOpenAI(api_key=api_key)
    print("OpenAI async client ready")

    print("Loading FAISS index...")
    index = await asyncio.to_thread(faiss.read_index, "index.faiss")
    print(f"FAISS index loaded — {index.ntotal} vectors")

    print("Loading texts.json...")
    with open("texts.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    print(f"Texts loaded — {len(texts)} entries")

    _ready = True
    print("Backend fully ready — accepting requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_assets())   # port binds immediately
    yield
    print("Shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status":  "ready" if _ready else "loading",
        "vectors": index.ntotal if index else 0,
        "texts":   len(texts)   if texts  else 0,
    }


async def _retrieve(query: str, k: int = 3) -> list:
    """Embed the query with OpenAI, then search FAISS — no local model needed."""
    resp    = await client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_embed = np.array([resp.data[0].embedding], dtype="float32")
    D, I    = index.search(q_embed, k)
    return [texts[i] for i in I[0] if 0 <= i < len(texts)]


@app.get("/chat")
async def chat(q: str):
    if not _ready:
        return {"response": "The assistant is still warming up. Please try again in a few seconds."}

    if not q or not q.strip():
        return {"response": "Please ask a question."}

    q = q.strip()

    try:
        results = await _retrieve(q)
    except Exception:
        traceback.print_exc()
        return {"response": "Error searching the knowledge base. Please retry."}

    if not results:
        return {"response": "No relevant information found in the knowledge base."}

    context = "\n\n".join(results)

    prompt = (
        "You are an expert in drug discovery and SAR (Structure-Activity Relationship) analysis.\n\n"
        "Use ONLY the context below to answer the question. "
        "Be concise and precise. "
        "If the context does not contain enough information, say so honestly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {q}\n\n"
        "Answer:"
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            ),
            timeout=25.0,
        )
        return {"response": response.choices[0].message.content.strip()}

    except asyncio.TimeoutError:
        return {"response": "The AI took too long to respond. Please try again."}

    except Exception:
        traceback.print_exc()
        return {"response": "Error generating AI response. Please retry."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)