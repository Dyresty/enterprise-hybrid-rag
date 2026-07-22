from app.retrieval.retriever import Retriever
from app.retrieval.document_store import DocumentStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker


query = "What happened in the Luminaris laboratory fire?"


# Dense retrieval
dense = Retriever()


# Load documents for BM25
store = DocumentStore()
documents = store.load_documents()


# Sparse retrieval
bm25 = BM25Retriever(
    documents
)


# Hybrid retrieval
hybrid = HybridRetriever(
    dense,
    bm25
)


# Get candidates
results = hybrid.search(
    query,
    top_k=20
)


print("\nBEFORE RERANKING")

for r in results[:5]:
    print("----------------")
    print("Page:", r["page"])
    print(r["text"][:300])


# Reranking
reranker = Reranker()


reranked_results = reranker.rerank(
    query,
    results,
    top_k=5
)


print("\nAFTER RERANKING")

for r in reranked_results:

    print("----------------")

    print(r)