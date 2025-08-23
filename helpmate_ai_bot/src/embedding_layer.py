# src/embedding_layer.py

# Import the libraries
import os
import re
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer


# === CLEANING FUNCTIONS ===

def clean_email_body(body: str) -> str:
    body = re.sub(r'[\r\n]+', ' ', body)  # remove newlines
    body = re.sub(r'\s+', ' ', body)  # normalize spaces
    body = re.sub(r'On .* wrote:', '', body)  # remove quoted replies
    return body.strip()


# === CHUNKING STRATEGY ===

def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunk = ' '.join(words[i:i + max_tokens])
        if len(chunk.split()) > 10:
            chunks.append(chunk)
    return chunks


# === EMBEDDING + CHROMA ===

class EmbeddingProcessor:
    def __init__(self, client, chroma_path: str = "chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.client = client
        self.chroma_collection = self.client.get_or_create_collection(name="email_chunks")

    def process_emails(self, df: pd.DataFrame):
        all_chunks = []
        metadatas = []

        for idx, row in df.iterrows():
            cleaned_body = clean_email_body(row['body'])
            chunks = chunk_text(cleaned_body)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{row['thread_id']}_{idx}_{i}"
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "thread_id": row["thread_id"],
                        "subject": row["subject"],
                        "from": row["from"],
                        "timestamp": row["timestamp"]
                    }
                })

        print(f"Embedding {len(all_chunks)} chunks...")
        embeddings = self.model.encode([c['text'] for c in all_chunks], show_progress_bar=True).tolist()

        self.chroma_collection.add(
            documents=[c['text'] for c in all_chunks],
            embeddings=embeddings,
            metadatas=[c['metadata'] for c in all_chunks],
            ids=[c['id'] for c in all_chunks]
        )
        

    def get_collection(self):
        return self.chroma_collection
