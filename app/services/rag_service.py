from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker

from app.llm.prompt_builder import create_rag_prompt
from app.services.context_builder import ContextBuilder
from app.llm.llm_service import LLMService
from app.llm.prompt_builder import create_rag_prompt

class RAGService:
    def __init__(
        self,
        hybrid_retriever,
        reranker,
        context_builder,
        llm
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm = llm
    def retrieve(
        self,
        query,
        top_k=20
    ):
        documents = (
            self.hybrid_retriever
            .search(
                query,
                top_k=20
            )
        )
        reranked_documents = (
            self.reranker
            .rerank(
                query,
                documents,
                top_k=top_k
            )
        )
        return reranked_documents

    def generate_answer(
        self,
        query
    ):
        documents = self.retrieve(query)
        context = (
            self.context_builder
            .build_context(
                documents
            )
        )
        prompt = create_rag_prompt(
            query,
            context
        )
        answer = (
            self.llm
            .generate(
                prompt
            )
        )
        return {
            "query": query,
            "answer": answer,
            "sources": documents
        }