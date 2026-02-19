# 3️⃣ Reranker 설계 (중요)
# Reranker는 향후 가장 많이 바뀔 부분.
# BGE reranker
# Cohere rerank
# Custom cross encoder

from sentence_transformers import CrossEncoder
from config import RERANK_MODEL


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL)

    def rerank(self, query, docs):

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc.metadata["score"] = float(score)

        return sorted(
            docs,
            key=lambda x: x.metadata["score"],
            reverse=True
        )
