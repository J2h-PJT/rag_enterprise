# ✔ embedding 교체 가능
# ✔ DB engine 외부 주입

from langchain_postgres import PGVector
from config import COLLECTION_NAME
from core.embeddings import EmbeddingFactory


class VectorStoreFactory:

    @staticmethod
    def create(engine, embedding_provider="hf"):

        embeddings = EmbeddingFactory.create(embedding_provider)

        return PGVector(
            connection=engine,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
            use_jsonb=True,
        )

