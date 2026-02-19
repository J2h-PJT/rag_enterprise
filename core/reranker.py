class BaseReranker:
    def rerank(self, query, docs, top_k=8):
        return docs[:top_k]

class RerankerFactory:
    @staticmethod
    def create(config):
        return BaseReranker()
