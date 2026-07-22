from app.retrieval.vectorstore import VectorStore
class DocumentStore:
    def __init__(self):
        self.vector_store = VectorStore()
    def load_documents(self):
        points = []
        offset = None
        while True:
            result = (
                self.vector_store.client.scroll(
                    collection_name="documents",
                    limit=100,
                    offset=offset
                )
            )
            batch, offset = result
            points.extend(batch)
            if offset is None:
                break
        documents = []
        for point in points:
            documents.append(
                {
                    "document_id":
                        point.payload["document_id"],
                    "filename":
                        point.payload["filename"],
                    "chunk_id":
                        point.payload["chunk_id"],
                    "page":
                        point.payload["page"],
                    "text":
                        point.payload["text"]
                    
                }
            )
        return documents