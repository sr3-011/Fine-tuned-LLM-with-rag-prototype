from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index("index.faiss")

with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# 🔍 Search
def search(query, k=3):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed).astype("float32"), k)
    return [texts[i] for i in I[0]]

# 🤖 Answer
def ask(query):
    try:
        results = search(query)

        if not results:
            return "No relevant information found."

        best = results[0]

        if "Answer:" in best:
            answer = best.split("Answer:")[1].strip()
        else:
            answer = best

        return answer

    except Exception as e:
        return "Error processing request."

@app.get("/chat")
def chat(q: str):
    return {"response": ask(q)}