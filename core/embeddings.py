# 1️⃣ Embeddings 설계
# 목표
# HuggingFace
# OpenAI
# 향후 다른 모델 교체 가능하게.
# ✔ normalize_embeddings=True → cosine 기반 안정화
# ✔ 나중에 OpenAIEmbedding 추가 가능

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
from core.interfaces.embedding_base import BaseEmbedding


class HFEmbedding(BaseEmbedding):

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)


class EmbeddingFactory:

    @staticmethod
    def create(provider: str = "hf") -> BaseEmbedding:

        if provider == "hf":
            return HFEmbedding()

        raise ValueError(f"Unsupported embedding provider: {provider}")

