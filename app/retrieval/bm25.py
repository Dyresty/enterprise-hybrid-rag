import re

from rank_bm25 import BM25Okapi

STOPWORDS = set(
    "a an the and or of to in on for with by is are was were be been this that these those it its as at from "
    "which what who whom how why when where does do did can could should would will about into than then".split()
)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    """Sparse lexical retrieval. Catches exact terms (acronyms, grant IDs, names) that embeddings blur."""

    def __init__(self, documents):
        self.rebuild(documents)

    def rebuild(self, documents):
        self.documents = documents
        # Index title + chunk text so project names/acronyms are matchable.
        corpus = [tokenize(f"{d.get('title') or ''} {d['text']}") for d in documents]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query, top_k=5):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**self.documents[i], "sparse_score": float(scores[i])} for i in ranked if scores[i] > 0]
