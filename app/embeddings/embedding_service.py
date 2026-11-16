from sentence_transformers import SentenceTransformer

from app import config

# BGE models expect this instruction on queries (not on passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=len(texts) > 500
        ).tolist()

    def embed_query(self, query):
        prefix = QUERY_INSTRUCTION if "bge" in config.EMBED_MODEL.lower() else ""
        return self.model.encode(prefix + query, normalize_embeddings=True).tolist()
