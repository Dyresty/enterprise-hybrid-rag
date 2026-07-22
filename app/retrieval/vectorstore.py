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
        for idx, (chunk, vector) in enumerate(
            zip(chunks, embeddings)
        ):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=chunk
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )