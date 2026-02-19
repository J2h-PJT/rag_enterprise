class BaseRetriever:
    def __init__(self, vectorstore, top_k=20):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def get_relevant_documents(self, query, filters=None):
        return self.vectorstore.similarity_search(
            query,
            k=self.top_k,
            filter=filters
        )
