"""Retrieval metrics, computed at document level (several chunks of one document count once)."""


def unique_docs(chunks):
    seen, ranked = set(), []
    for c in chunks:
        if c["document_id"] not in seen:
            seen.add(c["document_id"])
            ranked.append(c["document_id"])
    return ranked


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / k


def hit_at_k(ranked, relevant, k):
    return float(bool(set(ranked[:k]) & set(relevant)))


def reciprocal_rank(ranked, relevant):
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0
