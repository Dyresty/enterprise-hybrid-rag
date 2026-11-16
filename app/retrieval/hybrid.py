RRF_K = 60


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Fuse ranked lists by rank, not score: score(d) = sum over lists of 1 / (k + rank).

    Rank-based fusion needs no normalisation between cosine similarity and BM25 scores,
    which live on completely different scales.
    """
    fused = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = (doc["document_id"], doc["chunk_id"])
            entry = fused.setdefault(key, {**doc, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever

    def search(self, query, top_k=20):
        dense = self.dense.search(query, top_k)
        sparse = self.bm25.search(query, top_k)
        return reciprocal_rank_fusion([dense, sparse])[:top_k]
