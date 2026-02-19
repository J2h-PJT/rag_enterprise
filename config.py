import os

PG_CONN_STR = os.getenv(
    "PG_CONN_STR",
    "postgresql+psycopg://postgres:postgres@localhost:5432/rag_db"
)

COLLECTION_NAME = "multi_pdf_collection"
UPLOAD_DIR = "uploaded_pdfs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "EEVE-Korean-Chat-10.8B"
