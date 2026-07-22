from app.retrieval.retriever import Retriever


retriever = Retriever()


query = "What happened in the laboratory fire?"


results = retriever.search(
    query,
    top_k=5
)


for result in results:

    print("----------------")

    print(result)