# main.py

import pandas as pd
from src.embedding_layer import EmbeddingProcessor
from src.search_layer import SearchEngine
from src.generation_layer import generate_answer

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# === CONFIG ===
DATA_PATH = "email_dataset/email_threads.csv"
TOP_K = 3

# === LOAD DATA ===
print("Loading dataset...")
df = pd.read_csv(DATA_PATH).dropna(subset=["body"])

chroma_client = chromadb.Client(Settings(persist_directory="chroma_db"))

# === EMBEDDING PHASE ===
print("Embedding data...")
embedder = EmbeddingProcessor(client=chroma_client)
embedder.process_emails(df)

# === SEARCH PHASE ===
search_engine = SearchEngine(client=chroma_client)

# === QUERIES TO TEST ===
queries = [
    "What summary does the thread provide about delays in project delivery?",
    "What decision was made about budget increase in email thread about resource allocation?",
    "What strategy was proposed in thread_id 100 regarding risk management?"
]

# === RUN PIPELINE ===
for query in queries:
    print(f"\n=== QUERY: {query} ===")

    # Search
    top_chunks = search_engine.search(query, top_k=TOP_K)

    # Print top chunks
    print("\nTop Retrieved Chunks:")
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"{chunk['chunk']}")
        print(f"Metadata: {chunk['metadata']}")

    # Generate Answer
    answer = generate_answer(query, top_chunks)
    print("\nGenerated Answer:")
    print(answer)
