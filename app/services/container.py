from app.retrieval.retriever import Retriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker

from app.services.rag_service import RAGService
from app.services.context_builder import ContextBuilder
from app.llm.llm_service import LLMService

from app.vectorstore.qdrant_service import VectorStore



# -----------------------------
# Dense Retriever
# -----------------------------

dense_retriever = Retriever()



# -----------------------------
# BM25 Retriever
# -----------------------------

vector_store = VectorStore()


try:

    bm25_documents = vector_store.get_all_documents()


except Exception as e:

    print(
        "Could not load documents from Qdrant:",
        e
    )

    bm25_documents = []


print(
    "BM25 DOCUMENT COUNT:",
    len(bm25_documents)
)


bm25_retriever = BM25Retriever(
    bm25_documents
)



# -----------------------------
# Hybrid Retriever
# -----------------------------

hybrid_retriever = HybridRetriever(
    dense_retriever,
    bm25_retriever
)



# -----------------------------
# Reranker
# -----------------------------

reranker = Reranker()



# -----------------------------
# Services
# -----------------------------

context_builder = ContextBuilder()

llm = LLMService()


rag_service = RAGService(
    hybrid_retriever,
    reranker,
    context_builder,
    llm
)