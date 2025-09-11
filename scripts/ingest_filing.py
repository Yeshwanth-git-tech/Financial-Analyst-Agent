# import os
# import shutil
# from dotenv import load_dotenv


# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI
# import faiss

# # === 1. Load OpenAI Key ===
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY not found in environment variables")

# # === 2. Configure LlamaIndex Settings (New API) ===
# Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-4o")
# Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-ada-002")

# # === 3. Paths ===
# # FILING_DIR = "data/sec_filings"
# # FAISS_INDEX_DIR = "storage/faiss_index"

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# FILING_DIR = os.path.join(SCRIPT_DIR, "..", "data", "sec_filings")
# FAISS_INDEX_DIR = os.path.join(SCRIPT_DIR, "..", "storage", "faiss_index")


# # === 4. Clean Old Index ===
# if os.path.exists(FAISS_INDEX_DIR):
#     print("🗑️  Cleaning old index...")
#     shutil.rmtree(FAISS_INDEX_DIR)
# os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# # === 5. Load Documents ===
# print("📂 Loading filings from:", FILING_DIR)
# if not os.path.exists(FILING_DIR):
#     raise FileNotFoundError(f"Filing directory not found: {FILING_DIR}")

# try:
#     documents = SimpleDirectoryReader(
#         input_dir=FILING_DIR, 
#         recursive=True,
#         filename_as_id=True  # Use filename as document ID
#     ).load_data()
#     print(f"✅ Loaded {len(documents)} documents")
# except Exception as e:
#     print(f"❌ Error loading documents: {e}")
#     raise

# # === 6. Build FAISS Vector Store ===
# print("🧠 Creating FAISS vector store...")

# # Create FAISS index with correct embedding dimension (1536 for text-embedding-ada-002)
# faiss_index = faiss.IndexFlatL2(1536)
# vector_store = FaissVectorStore(faiss_index=faiss_index)

# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# # === 7. Create Index ===
# print("⚙️  Indexing with OpenAI embeddings...")
# try:
#     index = VectorStoreIndex.from_documents(
#         documents, 
#         storage_context=storage_context,
#         show_progress=True
#     )
#     print("✅ Index created successfully")
# except Exception as e:
#     print(f"❌ Error creating index: {e}")
#     raise

# # # === 8. Persist to Disk ===
# # print("💾 Saving index to disk...")
# # try:
# #     index.storage_context.persist(persist_dir=FAISS_INDEX_DIR)
# #     print(f"✅ Vector index saved to: {FAISS_INDEX_DIR}")
# # except Exception as e:
# #     print(f"❌ Error saving index: {e}")
# #     raise

# # print("🎉 Ingestion complete!")

# # # === 8. Persist to Disk ===
# # print("💾 Saving index to disk...")
# # try:
# #     # ✅ Save the .faiss binary file
# #     vector_store.save(FAISS_INDEX_DIR)

# #     # ✅ Save metadata (index_store.json, docstore.json, etc.)
# #     index.storage_context.persist(persist_dir=FAISS_INDEX_DIR)

# #     print(f"✅ Vector index and metadata saved to: {FAISS_INDEX_DIR}")
# # except Exception as e:
# #     print(f"❌ Error saving index: {e}")
# #     raise

# # === 8. Persist to Disk ===
# print("💾 Saving index to disk...")
# try:
#     # ✅ Save everything via LlamaIndex persist method
#     index.storage_context.persist(persist_dir=FAISS_INDEX_DIR)

#     print(f"✅ Vector index and metadata saved to: {FAISS_INDEX_DIR}")
# except Exception as e:
#     print(f"❌ Error saving index: {e}")
#     raise

# print("🎉 Ingestion complete!")

# # === Optional: Quick test of the index ===
# print("\n🔍 Testing query engine...")
# try:
#     query_engine = index.as_query_engine()
#     response = query_engine.query("What is this document about?")
#     print(f"Test query response: {response}")
# except Exception as e:
#     print(f"❌ Error testing query engine: {e}")



# from load_documents import extract_documents


# # Get the directory of the current script
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Define input/output directories using absolute paths
# INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "sec_filings")
# SAVE_TXT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "extracted_txt")

# # Extract documents with absolute paths
# documents = extract_documents(
#     input_dir=INPUT_DIR,
#     save_txt_dir=SAVE_TXT_DIR
# )

import os
import shutil
from dotenv import load_dotenv

from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import faiss

# === 1. Load OpenAI Key ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# === 2. Configure LlamaIndex Settings (New API) ===
Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-ada-002")
ROOT_DIR = Path(__file__).resolve().parent.parent
# === 3. Paths ===
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# FILING_DIR = os.path.join(SCRIPT_DIR, "..", "data", "sec_filings")
# # FAISS_INDEX_DIR = os.path.join(SCRIPT_DIR, "..", "storage", "faiss_index")

FAISS_INDEX_DIR = ROOT_DIR / "storage" / "faiss_index"
FILING_DIR = ROOT_DIR / "data" / "sec_filings"
print("✅ ROOT_DIR:", ROOT_DIR)
print("FL", FILING_DIR)
print("✅ ROOT_DIR:", ROOT_DIR)
print("✅ FAISS_INDEX_DIR:", FAISS_INDEX_DIR)


# === 4. Clean Old Index ===
if os.path.exists(FAISS_INDEX_DIR):
    print("🗑️  Cleaning old index...")
    shutil.rmtree(FAISS_INDEX_DIR)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# === 5. Load Documents ===
print("📂 Loading filings from:", FILING_DIR)
if not os.path.exists(FILING_DIR):
    raise FileNotFoundError(f"Filing directory not found: {FILING_DIR}")

try:
    documents = SimpleDirectoryReader(
        input_dir=FILING_DIR, 
        recursive=True,
        filename_as_id=True
    ).load_data()
    print(f"✅ Loaded {len(documents)} documents")
except Exception as e:
    print(f"❌ Error loading documents: {e}")
    raise

# === 6. Build FAISS Vector Store ===
print("🧠 Creating FAISS vector store...")
faiss_index = faiss.IndexFlatL2(1536)
vector_store = FaissVectorStore(faiss_index=faiss_index)  # ✅ FIXED: required param

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# === 7. Create Index ===
print("⚙️  Indexing with OpenAI embeddings...")
try:
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ Index created successfully")
except Exception as e:
    print(f"❌ Error creating index: {e}")
    raise

# === 8. Persist to Disk ===
print("💾 Saving index to disk...")
try:
    index.storage_context.persist(persist_dir=FAISS_INDEX_DIR)
    print(f"✅ JSON vector store saved to: {FAISS_INDEX_DIR}")
except Exception as e:
    print(f"❌ Error saving index: {e}")
    raise

# # === Save FAISS index binary to disk ===
# try:
#     index.storage_context.persist(persist_dir=str(FAISS_INDEX_DIR ))
#     print(f"✅ FAISS binary index saved to: {FAISS_INDEX_DIR /'index.faiss'}")
# except Exception as e:
#     print(f"❌ Error saving FAISS binary: {e}")

# # === Save FAISS metadata ===
# index.storage_context.persist(persist_dir=FAISS_INDEX_DIR)

# === Save raw FAISS index ===
# try:
#     faiss_index = index.vector_store._faiss_index  # ✅ Must use _faiss_index
#     faiss.write_index(faiss_index, os.path.join(FAISS_INDEX_DIR, "index.faiss"))
#     print("✅ True FAISS binary index saved to:", os.path.join(FAISS_INDEX_DIR, "index.faiss"))
# except AttributeError as e:
#     print("❌ Could not access FAISS index — likely missing _faiss_index")
#     raise
# except Exception as e:
#     print(f"❌ Failed to write FAISS binary index: {e}")
#     raise

# === Save raw FAISS binary (Optional but useful for debugging or backup) ===
try:
    faiss_index = index.vector_store._faiss_index
    faiss.write_index(faiss_index, str(FAISS_INDEX_DIR / "index.faiss"))
    print(f"✅ FAISS binary index saved to: {FAISS_INDEX_DIR / 'index.faiss'}")
except AttributeError:
    print("❌ Could not access FAISS binary index (check _faiss_index)")

print("🎉 Ingestion complete!")

# === Optional: Quick test of the index ===
print("\n🔍 Testing query engine...")
try:
    query_engine = index.as_query_engine()
    response = query_engine.query("What is this document about?")
    print(f"Test query response: {response}")
except Exception as e:
    print(f"❌ Error testing query engine: {e}")

# === 9. Extract raw documents from SEC ===
from scripts.load_documents import extract_documents

INPUT_DIR = ROOT_DIR /  "data" / "sec_filings"
SAVE_TXT_DIR = ROOT_DIR /  "data" / "extracted_txt"

documents = extract_documents(
    input_dir=INPUT_DIR,
    save_txt_dir=SAVE_TXT_DIR
)