from rank_bm25 import BM25Okapi
import re


class BM25Retriever:


    def __init__(self, documents):

        self.documents = documents


        if not documents:
            self.bm25 = None
            return


        corpus = [
            self._tokenize(doc["text"])
            for doc in documents
        ]


        self.bm25 = BM25Okapi(
            corpus
        )


    def search(
        self,
        query,
        top_k=5
    ):

        if self.bm25 is None:
            return []


        tokenized_query = self._tokenize(
            query
        )


        scores = self.bm25.get_scores(
            tokenized_query
        )


        ranked = sorted(
            range(len(scores)),
            key=lambda i:scores[i],
            reverse=True
        )


        results=[]


        for idx in ranked[:top_k]:

            doc=self.documents[idx]


            results.append(
                {
                    "sparse_score":
                        float(scores[idx]),


                    "document_id":
                        doc["metadata"]["document_id"],


                    "parent_id":
                        doc["metadata"]["parent_id"],


                    "filename":
                        doc["metadata"]["filename"],


                    "chunk_id":
                        doc["metadata"]["chunk_id"],


                    "page":
                        doc["metadata"]["page"],


                    "text":
                        doc["text"]
                }
            )


        return results



    def _tokenize(
        self,
        text
    ):

        text=text.lower()


        # remove punctuation
        text=re.sub(
            r"[^a-z0-9\s]",
            "",
            text
        )


        tokens=text.split()


        return tokens