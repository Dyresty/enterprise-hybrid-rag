from app.embeddings.embedding_service import EmbeddingModel
from app.vectorstore.qdrant_service import VectorStore


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


        documents=[]


        for result in results:

            payload=result.payload

            documents.append(
                {
                    "dense_score": result.score,

                    "document_id":
                        payload["document_id"],

                    "parent_id":
                        payload["parent_id"],

                    "filename":
                        payload["filename"],

                    "chunk_id":
                        payload["chunk_id"],

                    "page":
                        payload["page"],

                    "text":
                        payload["text"]
                }
            )


        print("DENSE RETRIEVER OUTPUT:")
        print(documents[0])


        return documents