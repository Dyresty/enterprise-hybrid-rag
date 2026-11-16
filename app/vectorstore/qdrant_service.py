import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config


class VectorStore:
    def __init__(self, dim=768):
        if config.QDRANT_URL:
            self.client = QdrantClient(url=config.QDRANT_URL)
        else:
            self.client = QdrantClient(path=config.QDRANT_PATH)
        self.collection_name = config.COLLECTION
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def insert_chunks(self, chunks, embeddings, batch_size=256):
        points = [
            PointStruct(
                # Deterministic id: re-ingesting the same document overwrites instead of duplicating.
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c['metadata']['document_id']}_{c['metadata']['chunk_id']}")),
                vector=vector,
                payload={**c["metadata"], "text": c["text"], "parent_text": c["parent_text"]},
            )
            for c, vector in zip(chunks, embeddings)
        ]
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.collection_name, points=points[i : i + batch_size])

    def search(self, query_vector, limit=5):
        return self.client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=limit
        ).points

    def get_all_documents(self):
        """All chunk payloads, paginated (used to build the BM25 index)."""
        documents, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name, limit=1000, offset=offset, with_vectors=False
            )
            documents.extend(p.payload for p in points)
            if offset is None:
                return documents

    def close(self):
        self.client.close()

    def count(self):
        return self.client.count(self.collection_name).count
