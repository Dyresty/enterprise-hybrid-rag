import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host="localhost",
            port=6333
        )
        self.collection_name = "documents"

    def create_collection(self):
        collections = (
            self.client
            .get_collections()
            .collections
        )
        existing = [
            collection.name
            for collection in collections
        ]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )

    def insert_chunks(
        self,
        chunks,
        embeddings
    ):
        points = []
        for chunk, vector in zip(
            chunks,
            embeddings
        ):
            points.append(
                PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{chunk['document_id']}_{chunk['chunk_id']}"
                        )
                    ),
                    vector=vector,
                    payload={
                        "document_id":
                            chunk["document_id"],
                        "filename":
                            chunk["filename"],
                        "chunk_id":
                            chunk["chunk_id"],
                        "page":
                            chunk["page"],
                        "text":
                            chunk["text"]
                    }
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_vector,
        limit=5
    ):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        return results.points