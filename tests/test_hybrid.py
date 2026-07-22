from app.retrieval.retriever import Retriever
from app.retrieval.document_store import DocumentStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever

dense = Retriever()
store = DocumentStore()
documents = store.load_documents()
bm25 = BM25Retriever(
    documents
)

hybrid = HybridRetriever(
    dense,
    bm25
)

results = hybrid.search(
    "What happened in the Luminaris laboratory fire?",
    top_k=20
)

for r in results:
    print("----------------")
    print(r)