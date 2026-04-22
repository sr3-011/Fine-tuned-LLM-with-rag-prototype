from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = []

print("📂 Loading dataset...")

with open("data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        messages = data["messages"]

        if len(messages) < 2:
            continue

        user = messages[0]["content"]
        assistant = messages[1]["content"]

        texts.append(f"Question: {user}\nAnswer: {assistant}")

print(f"✅ Loaded {len(texts)} entries")

print("🧠 Creating embeddings locally...")

embeddings = model.encode(texts)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, "index.faiss")

with open("texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f)

print("🎉 Done (FREE embeddings)")