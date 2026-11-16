from app.ingestion.chunker import chunk_documents
from app.ingestion.parser import extract_pdf_text


class DocumentIngestionService:
    def __init__(self, embedding_model, vector_store, bm25_retriever):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25 = bm25_retriever

    def ingest_pdf(self, file_path, document_id, filename):
        pages = extract_pdf_text(file_path)
        chunks = self.ingest_pages(pages, document_id, filename)
        return {"pages": len(pages), "total_chunks": len(chunks)}

    def ingest_pages(self, pages, document_id, filename, source_type="pdf", title=None, url=None, refresh=True):
        chunks = chunk_documents(pages, document_id, filename, source_type, title, url)
        if not chunks:
            return []
        # Title in the embedded text lets the vector match on the document name too.
        texts = [f"{title or filename}\n{c['text']}" for c in chunks]
        self.vector_store.insert_chunks(chunks, self.embedding_model.embed_documents(texts))
        if refresh:
            self.refresh_bm25()
        return chunks

    def refresh_bm25(self):
        # BM25 is an in-memory index over all chunks; rebuild it so new documents are searchable at once.
        self.bm25.rebuild(self.vector_store.get_all_documents())
