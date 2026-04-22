"""
embed.py  —  rebuild index.faiss + texts.json using OpenAI embeddings
Run locally: python embed.py
Requires:    OPENAI_API_KEY env var set
"""
from openai import OpenAI
import faiss
import numpy as np
import json
import os

EMBED_MODEL = "text-embedding-3-small"   # 1536-dim, cheap, fast
BATCH_SIZE  = 100                        # max items per API call

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("Loading dataset...")
texts = []
with open("data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        msgs = data.get("messages", [])
        if len(msgs) < 2:
            continue
        user      = msgs[0]["content"]
        assistant = msgs[1]["content"]
        texts.append(f"Question: {user}\nAnswer: {assistant}")

print(f"Loaded {len(texts)} entries")

# ── 2. Embed in batches ──────────────────────────────────────────────────────
print("Creating embeddings via OpenAI...")
all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
    vecs  = [item.embedding for item in resp.data]
    all_embeddings.extend(vecs)
    print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)} embedded")

embeddings = np.array(all_embeddings, dtype="float32")

# ── 3. Build & save FAISS index ──────────────────────────────────────────────
dim   = embeddings.shape[1]           # 1536 for text-embedding-3-small
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, "index.faiss")

with open("texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f)

print(f"Done — {index.ntotal} vectors saved to index.faiss")