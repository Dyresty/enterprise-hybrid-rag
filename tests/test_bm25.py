from app.retrieval.document_store import DocumentStore
from app.retrieval.bm25 import BM25Retriever



store = DocumentStore()

documents = store.load_documents()


bm25 = BM25Retriever(
    documents
)


results = bm25.search(
    "Luminaris Industries laboratory",
    top_k=5
)


for r in results:

    print("----------------")

    print(r)

    