from sentence_transformers import CrossEncoder

from app import config


class Reranker:
    """Cross-encoder reads query and chunk together: slower than a bi-encoder, much more precise.
    So it only re-scores the small candidate set from hybrid retrieval."""

    def __init__(self):
        self.model = CrossEncoder(config.RERANK_MODEL)

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
        scores = self.model.predict([[query, f"{d.get('title') or ''}\n{d['text']}"] for d in documents])
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        return sorted(documents, key=lambda d: d["rerank_score"], reverse=True)[:top_k]
