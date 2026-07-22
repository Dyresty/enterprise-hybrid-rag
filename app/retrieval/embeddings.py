from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    
    def __init__(self):
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )
        return embeddings.tolist()