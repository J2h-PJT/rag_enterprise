from langchain_community.vectorstores import PGVector
from sqlalchemy import text

class VectorStoreFactory:
    @staticmethod
    def create(config, embeddings, engine):
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        return PGVector(
            connection_string=config.DB_URL,
            embedding_function=embeddings,
            collection_name="documents"
        )
