

from core.embeddings import EmbeddingFactory
from core.llm import LLMFactory
from core.reranker import RerankerFactory
from core.vectorstore import VectorStoreFactory
from sqlalchemy import create_engine


class CoreEngine:

    def __init__(self, config):

        # Infra
        self.db_engine = create_engine(config.DB_URL)

        # AI Components
        self.embeddings = EmbeddingFactory.create(config)
        self.llm = LLMFactory.create(config)
        self.reranker = RerankerFactory.create(config)

        # Storage
        self.vectorstore = VectorStoreFactory.create(
            config,
            self.embeddings
        )
