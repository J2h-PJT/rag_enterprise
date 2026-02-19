import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_URL = os.getenv("DB_URL")
    MAX_CONTEXT_TOKENS = 3000
    RETRIEVAL_TOP_K = 20
    RERANK_TOP_K = 8
