from rank_bm25 import BM25Okapi
class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        tokenized_docs = [
            doc["text"].lower().split()
            for doc in documents
        ]
        self.bm25 = BM25Okapi(
            tokenized_docs
        )

    def search(
        self,
        query,
        top_k=5
    ):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(
            tokens
        )
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        results = []
        for index, score in ranked[:top_k]:
            results.append(
            {
                "score": float(score),
                "document_id": self.documents[index]["document_id"],
                "filename": self.documents[index]["filename"],
                "chunk_id": self.documents[index]["chunk_id"],
                "page": self.documents[index]["page"],
                "text": self.documents[index]["text"]
            }
        )
        return results