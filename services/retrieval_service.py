class RetrievalService:
    def __init__(self, retriever, reranker, config):
        self.retriever = retriever
        self.reranker = reranker
        self.config = config

    def retrieve(self, query, filters=None):
        docs = self.retriever.get_relevant_documents(query, filters)
        docs = self.reranker.rerank(query, docs, self.config.RERANK_TOP_K)
        return docs
