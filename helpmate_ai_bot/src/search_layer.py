# src/search_layer.py

import hashlib
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from src.utils import convert_np_types

from .cache import Cache

# === CONFIGURATION ===
CACHE_PATH = "cache/search_cache.json"
CHROMA_PATH = "chroma_db"

class SearchEngine:
    def __init__(
        self,
        client,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.reranker = CrossEncoder(cross_encoder_model_name)
        self.cache = Cache(CACHE_PATH)

        self.client = client
        self.collection = self.client.get_or_create_collection(name="email_chunks")

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.encode(query).tolist()

    def search(self, query: str, top_k: int = 5, filter_thread_id: int = None) -> List[Dict]:
        query_hash = hashlib.md5(query.encode()).hexdigest()

        if self.cache.contains(query_hash):
            return self.cache.get(query_hash)

        query_embedding = self.embed_query(query)

        search_args = {
                            "query_embeddings": [query_embedding],
                            "n_results": top_k * 2,  # get more to allow for reranking
                      }
        
        if filter_thread_id is not None:
            search_args["where"] = {"thread_id": int(filter_thread_id)}

        results = self.collection.query(**search_args)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # === Re-ranking ===
        pairs = [(query, doc) for doc in documents]
        if not pairs:
            return []
        scores = self.reranker.predict(pairs)

        reranked = sorted(zip(documents, metadatas, scores), key=lambda x: x[2], reverse=True)

        top_chunks = [
            {"chunk": doc, "metadata": meta, "score": score}
            for doc, meta, score in reranked[:top_k]
        ]

        scored_chunks = convert_np_types(top_chunks)

        if not scored_chunks:
            return []

        self.cache.set(query_hash, scored_chunks)
        return top_chunks
