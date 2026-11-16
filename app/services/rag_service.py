from app import config
from app.llm.prompt_builder import NOT_FOUND, create_messages

# Cross-encoder scores are sigmoid probabilities in [0, 1]. On the eval set, valid and out-of-scope
# questions overlap above ~0.03, so this only drops clear junk; the prompt handles the rest.
MIN_RERANK_SCORE = 0.01


class RAGService:
    def __init__(self, hybrid_retriever, reranker, context_builder, llm):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm = llm

    def retrieve(self, query, top_k=None):
        candidates = self.hybrid_retriever.search(query, top_k=config.CANDIDATES)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k or config.TOP_K)
        return [d for d in reranked if d["rerank_score"] >= MIN_RERANK_SCORE]

    def generate_answer(self, question, top_k=None):
        documents = self.retrieve(question, top_k)
        if not documents:
            # Nothing relevant: refuse without calling the LLM, so it cannot hallucinate.
            return {"question": question, "answer": NOT_FOUND, "sources": []}

        context, sources = self.context_builder.build(documents)
        answer = self.llm.chat(create_messages(question, context))
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "n": s["n"],
                    "document_id": s["document_id"],
                    "filename": s["filename"],
                    "page": s["page"],
                    "title": s.get("title"),
                    "url": s.get("url"),
                    "score": s["rerank_score"],
                }
                for s in sources
            ],
        }
