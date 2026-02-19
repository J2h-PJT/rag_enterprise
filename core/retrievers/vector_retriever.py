

from core.interfaces.retriever_base import BaseRetriever


class VectorRetriever(BaseRetriever):

    def __init__(self, vectorstore, k=20):
        self.vectorstore = vectorstore
        self.k = k

    def retrieve(self, query: str, selected_ids=None):

        filter_dict = {}

        if selected_ids:
            filter_dict["file_id"] = {"$in": selected_ids}

        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=self.k,
            filter=filter_dict if filter_dict else None,
        )

        results = []

        for doc, score in docs_with_scores:
            doc.metadata["vector_score"] = float(score)
            results.append(doc)

        return results
