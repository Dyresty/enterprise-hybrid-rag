from app.ingestion.parser import extract_pdf_text
from app.ingestion.chunker import chunk_documents

from app.embeddings.embedding_service import EmbeddingModel
from app.vectorstore.qdrant_service import VectorStore


class DocumentIngestionService:

    def __init__(self):

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore()

        self.vector_store.create_collection()


    def ingest_document(
        self,
        file_path,
        document_id,
        filename
    ):

        # 1. Extract PDF text

        pages = extract_pdf_text(
            file_path
        )


        # 2. Chunk document

        chunks = chunk_documents(
            pages,
            document_id,
            filename
        )

        print(chunks[0])
        print(chunks[1])
        print(chunks[2])
        print(
            chunks[0]["metadata"]["parent_id"]
            ==
            chunks[1]["metadata"]["parent_id"]
        )
        # 3. Prepare embedding text

        texts = [
            f"""
            Document: {chunk['metadata']['filename']}

            Page: {chunk['metadata']['page']}

            Content:
            {chunk['text']}
            """
            for chunk in chunks
        ]


        # 4. Generate embeddings

        embeddings = (
            self.embedding_model
            .generate_embeddings(texts)
        )


        # 5. Store in Qdrant

        self.vector_store.insert_chunks(
            chunks,
            embeddings
        )


        return {
            "pages": len(pages),
            "total_chunks": len(chunks),
            "first_chunk":
                chunks[0]["text"]
                if chunks
                else ""
        }