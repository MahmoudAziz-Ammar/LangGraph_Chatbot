import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
LLM_MODEL       = "llama-3.3-70b-versatile"  # gratuit via Groq

# ── Embeddings ─────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, 0€

# ── Chemins ────────────────────────────────────
DATA_DIR         = "data/docs"
VECTORSTORE_PATH = "vectorstore/faiss_index"

# ── Chunking ───────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50