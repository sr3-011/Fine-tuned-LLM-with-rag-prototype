from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import faiss
import traceback
import os
from openai import OpenAI

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Globals (lazy loading)
model = None
index = None
texts = None


# ✅ Root route (Render health check)
@app.get("/")
def root():
    return {"status": "API running 🚀"}


# ✅ Load resources safely
def load_resources():
    global model, index, texts

    try:
        if model is None:
            from sentence_transformers import SentenceTransformer
            print("🔄 Loading model...")
            model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        if index is None:
            print("🔄 Loading FAISS index...")
            index = faiss.read_index("index.faiss")

        if texts is None:
            print("🔄 Loading texts...")
            with open("texts.json", "r", encoding="utf-8") as f:
                texts = json.load(f)

        print("✅ Resources ready!")

    except Exception:
        print("❌ RESOURCE LOAD ERROR")
        traceback.print_exc()
        raise RuntimeError("Failed to load backend resources")


# ✅ Chat endpoint (HYBRID RAG)
@app.get("/chat")
def chat(q: str):
    try:
        load_resources()

        # 🔹 Step 1: Embed query
        try:
            q_embed = model.encode([q])
        except Exception:
            print("❌ EMBEDDING ERROR")
            traceback.print_exc()
            return {"response": "Error generating embedding."}

        # 🔹 Step 2: FAISS search
        try:
            D, I = index.search(np.array(q_embed).astype("float32"), 3)
            results = [texts[i] for i in I[0]]
        except Exception:
            print("❌ FAISS ERROR")
            traceback.print_exc()
            return {"response": "Error searching knowledge base."}

        if not results:
            return {"response": "No relevant information found."}

        context = "\n\n".join(results)

        # 🔹 Step 3: OpenAI generation
        try:
            prompt = f"""
You are an expert in drug discovery and SAR analysis.

Use the context below to answer the question clearly and concisely.

Context:
{context}

Question:
{q}

Answer:
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )

            answer = response.choices[0].message.content
            return {"response": answer}

        except Exception:
            print("❌ OPENAI ERROR")
            traceback.print_exc()
            return {"response": "Error generating AI response."}

    except RuntimeError:
        return {"response": "Server failed to initialize. Please retry."}

    except Exception:
        print("❌ UNKNOWN ERROR")
        traceback.print_exc()
        return {"response": "Unexpected server error."}