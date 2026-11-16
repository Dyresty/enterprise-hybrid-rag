class Retriever:
    """Dense retrieval: embed the query, nearest-neighbour search in Qdrant."""

    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def search(self, query, top_k=5):
        vector = self.embedding_model.embed_query(query)
        return [
            {**r.payload, "dense_score": r.score}
            for r in self.vector_store.search(vector, limit=top_k)
        ]
