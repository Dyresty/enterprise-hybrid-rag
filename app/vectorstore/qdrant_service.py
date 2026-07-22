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
            metadata = chunk["metadata"]
            points.append(
                PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{metadata['document_id']}_{metadata['chunk_id']}"
                        )
                    ),
                    vector=vector,
                    payload={
                        "document_id": metadata["document_id"],
                        "parent_id": metadata["parent_id"],
                        "filename": metadata["filename"],
                        "chunk_id": metadata["chunk_id"],
                        "page": metadata["page"],
                        "section": metadata["section"],
                        "created_at": metadata["created_at"],
                        "source_type": metadata["source_type"],
                        "text": chunk["text"]
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


    def get_all_documents(self):

        documents=[]


        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000
        )


        for point in points:

            documents.append(

                {
                    "text":
                        point.payload["text"],

                    "metadata":
                    {
                        "document_id":
                            point.payload["document_id"],

                        "parent_id":
                            point.payload["parent_id"],

                        "filename":
                            point.payload["filename"],

                        "chunk_id":
                            point.payload["chunk_id"],

                        "page":
                            point.payload["page"]
                    }
                }

            )


        return documents