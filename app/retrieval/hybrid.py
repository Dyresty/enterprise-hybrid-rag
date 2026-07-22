class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever

    def min_max_normalize(results):

        scores = [
            r["score"]
            for r in results
        ]

        min_score=min(scores)
        max_score=max(scores)

        for r in results:
            r["normalized_score"] = (
                (r["score"]-min_score)
                /
                (max_score-min_score+1e-8)
            )

        return results
    
    def search(
        self,
        query,
        top_k=20
    ):
        dense_results = self.dense.search(
            query,
            top_k
        )
        sparse_results = self.bm25.search(
            query,
            top_k
        )
        combined = {}
        for rank,item in enumerate(dense_results):
            key = (
                item["document_id"],
                item["chunk_id"]
            )
            combined[key] = {
                "metadata":item,
                "score":0
            }
            combined[key]["score"] += (
                1/(60+rank+1)
            )
        for rank,item in enumerate(sparse_results):
            key = (
                item["document_id"],
                item["chunk_id"]
            )
            if key not in combined:
                combined[key]={
                    "metadata":item,
                    "score":0
                }
            combined[key]["score"] += (
                1/(60+rank+1)
            )
        results = sorted(
            combined.values(),
            key=lambda x:x["score"],
            reverse=True
        )
        return [
            {
                **r["metadata"],
                "hybrid_score":r["score"]
            }
            for r in results[:top_k]
        ]