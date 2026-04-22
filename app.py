from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

app = FastAPI()

# ✅ CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Lazy load model (VERY IMPORTANT for Render)
model = None

def get_model():
    global model
    if model is None:
        print("🔄 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded")
    return model

# ✅ Load FAISS index + texts
index = faiss.read_index("index.faiss")

with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# 🔍 Search function
def search(query, k=3):
    model = get_model()
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed).astype("float32"), k)
    return [texts[i] for i in I[0]]

# 🤖 Answer function
def ask(query):
    try:
        results = search(query)

        if not results:
            return "No relevant information found."

        best = results[0]

        # Extract clean answer
        if "Answer:" in best:
            return best.split("Answer:")[1].strip()
        else:
            return best

    except Exception as e:
        print("❌ Error:", e)
        return "Something went wrong."

# 🌐 API endpoint
@app.get("/chat")
def chat(q: str):
    return {"response": ask(q)}

# ✅ Required for Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)