# scripts/query_test.py

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# === 1. Load API Key ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# === 2. Load FAISS index ===
FAISS_INDEX_DIR = "storage/faiss_index"

print("📦 Loading FAISS index from:", FAISS_INDEX_DIR)
vector_store = FaissVectorStore.from_persist_dir(FAISS_INDEX_DIR)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
query_engine = index.as_query_engine()

# === 3. Ask a Question ===
question = "What risks are mentioned in the filing?"
print(f"\n❓ Query: {question}\n")

response = query_engine.query(question)
print("🧠 Answer:\n", response)