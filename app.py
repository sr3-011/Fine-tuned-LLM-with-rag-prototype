from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import faiss

app = FastAPI()

# ✅ CORS (VERY IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Lazy-loaded globals (prevents crash on Render)
model = None
index = None
texts = None


# ✅ Load resources only when needed (avoids memory crash at startup)
def load_resources():
    global model, index, texts

    if model is None:
        from sentence_transformers import SentenceTransformer

        print("🔄 Loading model...")
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # lighter model

        print("🔄 Loading FAISS index...")
        index = faiss.read_index("index.faiss")

        print("🔄 Loading texts...")
        with open("texts.json", "r", encoding="utf-8") as f:
            texts = json.load(f)

        print("✅ All resources loaded!")


# ✅ Health check route (important for Render)
@app.get("/")
def home():
    return {"status": "API is running 🚀"}


# ✅ Chat endpoint
@app.get("/chat")
def chat(q: str):
    try:
        load_resources()

        # Encode query
        q_embed = model.encode([q])

        # Search in FAISS
        D, I = index.search(np.array(q_embed).astype("float32"), 3)

        results = [texts[i] for i in I[0]]

        if not results:
            return {"response": "No relevant information found."}

        best = results[0]

        # Extract answer
        if "Answer:" in best:
            answer = best.split("Answer:")[1].strip()
        else:
            answer = best

        return {"response": answer}

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"response": f"Error: {str(e)}"}