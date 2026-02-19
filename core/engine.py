from sqlalchemy import create_engine
from core.embeddings import EmbeddingFactory
from core.llm import LLMFactory
from core.reranker import RerankerFactory
from core.vectorstore import VectorStoreFactory

class CoreEngine:
    def __init__(self, config):
        self.config = config
        self.db_engine = create_engine(config.DB_URL)
        self.embeddings = EmbeddingFactory.create(config)
        self.llm = LLMFactory.create(config)
        self.reranker = RerankerFactory.create(config)
        self.vectorstore = VectorStoreFactory.create(
            config,
            self.embeddings,
            self.db_engine
        )
