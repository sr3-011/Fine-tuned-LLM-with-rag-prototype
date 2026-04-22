"""
embed.py — FINAL SAFE VERSION
Run: python embed.py
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

MODEL_NAME = "paraphrase-MiniLM-L3-v2"
MAX_TEXTS = 3000   # keep safe for disk

print("🔄 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ── Load dataset ─────────────────────────
print("📂 Loading dataset...")
texts = []

with open("data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        msgs = data.get("messages", [])

        if len(msgs) < 2:
            continue

        user = msgs[0]["content"]
        assistant = msgs[1]["content"]

        texts.append(f"Question: {user}\nAnswer: {assistant}")

# limit size (prevents disk error)
texts = texts[:MAX_TEXTS]

print(f"✅ Loaded {len(texts)} entries")

# ── Embeddings ─────────────────────────
print("🧠 Creating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# ── FAISS ─────────────────────────
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# ── Safe save ─────────────────────────
tmp_file = "index_temp.faiss"
final_file = "index.faiss"

print("💾 Saving index...")
faiss.write_index(index, tmp_file)
os.replace(tmp_file, final_file)

# ── Save texts ─────────────────────────
with open("texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False)

print("✅ DONE — index + texts saved")