from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import faiss
import os
app = FastAPI()

# CORS (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route (VERY IMPORTANT for Render health check)
@app.get("/")
def root():
    return {"status": "ok"}

# Chat route (LIGHT VERSION for now)
@app.get("/chat")
def chat(q: str):
    return {
        "response": f"Demo response: {q}"
    }


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

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/chat")
def chat(q: str):
    try:
        load_resources()

        # Embed query
        q_embed = model.encode([q])

        # FAISS search
        D, I = index.search(np.array(q_embed).astype("float32"), 3)
        results = [texts[i] for i in I[0]]

        if not results:
            return {"response": "No relevant info found."}

        context = "\n".join(results)

        # 🔥 OpenAI generation
        prompt = f"""
You are a drug discovery expert.

Context:
{context}

Question:
{q}

Answer clearly:
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}