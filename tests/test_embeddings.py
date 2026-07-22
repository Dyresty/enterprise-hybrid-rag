from app.retrieval.embeddings import EmbeddingModel


model = EmbeddingModel()


texts = [
    "Documents are written with care and precision.",
    "This file is very important."
]


vectors = model.generate_embeddings(texts)


print(len(vectors))

print(len(vectors[0]))