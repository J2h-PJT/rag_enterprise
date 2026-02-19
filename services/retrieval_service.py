# ✔ 여기까지는 pure retrieval pipeline
# ✔ filter는 QAService가 적용
# 🔥 왜 filter를 RetrievalService에 안 넣었나?
# 이유:
# Filter는 “답변 정책”이다.
# 예:
# QA는 strict filter
# Summary는 loose filter
# Admin 모드는 필터 없음
# RetrievalService에 넣으면 정책 고정됨
# RetrievalService는 전략이 vector인지 hybrid인지 모른다. 이게 DI의 힘이다.

class RetrievalService:

    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker

    def retrieve(self, query, k=10):

        docs = self.retriever.get_relevant_documents(query)

        reranked = self.reranker.rerank(query, docs)

        return reranked
    
    # def retrieve(self, query, selected_ids=None):

    #     # 1. Vector Search
    #     docs = self.retriever.retrieve(query, selected_ids)

    #     if not docs:
    #         return []

    #     # 2. Rerank
    #     docs = self.reranker.rerank(query, docs)

    #     return docs
