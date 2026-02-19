
# retrieval 확장 시 유연하게 서비스 코드는 절대 수정하지 않아도 된다.
# retriever = RetrieverFactory.create("hybrid", vectorstore)
# retriever = RetrieverFactory.create("multi_stage", vectorstore)

from core.retrievers.vector_retriever import VectorRetriever


class RetrieverFactory:

    @staticmethod
    def create(strategy: str, vectorstore):

        if strategy == "vector":
            return VectorRetriever(vectorstore)

        raise ValueError(f"Unsupported retriever strategy: {strategy}")
