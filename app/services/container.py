"""Builds every service once (models are loaded a single time) and wires them together."""
from functools import lru_cache

from app.embeddings.embedding_service import EmbeddingModel
from app.llm.llm_service import LLMService
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.services.context_builder import ContextBuilder
from app.services.feedback_store import FeedbackStore
from app.services.ingestion_service import DocumentIngestionService
from app.services.rag_service import RAGService
from app.vectorstore.qdrant_service import VectorStore


class Container:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(dim=self.embedding_model.dim)
        self.dense = Retriever(self.embedding_model, self.vector_store)
        self.bm25 = BM25Retriever(self.vector_store.get_all_documents())
        self.hybrid = HybridRetriever(self.dense, self.bm25)
        self.reranker = Reranker()
        self.ingestion = DocumentIngestionService(self.embedding_model, self.vector_store, self.bm25)
        self.feedback = FeedbackStore()
        self.rag = RAGService(self.hybrid, self.reranker, ContextBuilder(), LLMService(), self.feedback)


@lru_cache
def get_container():
    return Container()
