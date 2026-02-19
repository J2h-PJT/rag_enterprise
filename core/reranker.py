# 3️⃣ Reranker 설계 (중요)
# Reranker는 향후 가장 많이 바뀔 부분.
# BGE reranker
# Cohere rerank
# Custom cross encoder
# 💎 중요 설계 포인트
# Reranker는 상태를 가지면 안 된다
# → 캐시 제외하면 stateless 유지
# LLM은 직접 호출하지 말고 항상 체인에서 사용
# → prompt | llm.get_model() | parser
# Embedding normalize는 반드시 유지
# → cosine 기반 vector search 안정화

from sentence_transformers import CrossEncoder
from config import RERANK_MODEL
from core.interfaces.reranker_base import BaseReranker


class BGEReranker(BaseReranker):

    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL)

    def rerank(self, query, docs):

        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc.metadata["score"] = float(score)

        return sorted(
            docs,
            key=lambda x: x.metadata["score"],
            reverse=True
        )


class RerankerFactory:

    @staticmethod
    def create(provider: str = "bge") -> BaseReranker:

        if provider == "bge":
            return BGEReranker()

        raise ValueError(f"Unsupported reranker provider: {provider}")

