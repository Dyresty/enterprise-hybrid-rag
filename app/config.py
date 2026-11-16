"""Central settings, read from environment variables (.env supported)."""
import os

from dotenv import load_dotenv

load_dotenv()

# Vector store: QDRANT_URL set -> Qdrant server (docker compose); unset -> embedded local storage.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_local")
COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# Any OpenAI-compatible endpoint: OpenAI, vLLM (http://host:8000/v1) or Ollama (http://localhost:11434/v1).
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"

# Retrieval
CANDIDATES = int(os.getenv("CANDIDATES", "30"))  # per retriever, before fusion
TOP_K = int(os.getenv("TOP_K", "5"))  # chunks after reranking
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
