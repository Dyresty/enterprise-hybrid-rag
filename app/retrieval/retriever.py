from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vectorstore import VectorStore

class Retriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
    def search(
        self,
        query,
        top_k=5
    ):
        query_embedding = (
            self.embedding_model
            .generate_embeddings(
                [query]
            )[0]
        )
        results = (
            self.vector_store
            .search(
                query_embedding,
                limit=top_k
            )
        )
        documents = []
        for result in results:
            documents.append({
                "score": result.score,
                "document_id":
                    result.payload["document_id"],
                "filename":
                    result.payload["filename"],
                "page":
                    result.payload["page"],
                "text":
                    result.payload["text"]
            })
        return documents