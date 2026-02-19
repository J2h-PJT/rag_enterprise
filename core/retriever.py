# 2️⃣ Retriever 분리 (core)
# Retriever는 “검색 전략”이다.
# 비즈니스 정책은 몰라도 된다. > core
# ✔ Retriever는 vector 점수만 책임
# ✔ rerank 모름
# ✔ filter 모름


class VectorRetriever:

    def __init__(self, vectorstore, k=20):
        self.vectorstore = vectorstore
        self.k = k

    def retrieve(self, query: str, selected_ids=None):

        filter_dict = {}

        if selected_ids:
            filter_dict["file_id"] = {"$in": selected_ids}

        docs = self.vectorstore.similarity_search_with_score(
            query,
            k=self.k,
            filter=filter_dict if filter_dict else None,
        )

        # LangChain returns (doc, score)
        results = []
        for doc, score in docs:
            doc.metadata["vector_score"] = float(score)
            results.append(doc)

        return results
