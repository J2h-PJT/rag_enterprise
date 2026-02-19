# 4️⃣ 최종 객체 조립 (Composition Root)

from core.db import create_db_engine
from core.vectorstore import VectorStoreFactory
from core.retrievers.retriever_factory import RetrieverFactory
from core.reranker import RerankerFactory
from core.llm import LLMFactory
from services.retrieval_service import RetrievalService
from services.filter_service import HybridFilterService
from services.qa_service import QAService


# ------------------------
# Core 구성
# ------------------------

engine = create_db_engine()

vectorstore = VectorStoreFactory.create(engine)

retriever = RetrieverFactory.create(
    strategy="vector",
    vectorstore=vectorstore
)

reranker = RerankerFactory.create("bge")

llm = LLMFactory.create("local")

# ------------------------
# Service 구성
# ------------------------

retrieval_service = RetrievalService(
    retriever=retriever,
    reranker=reranker
)

filter_service = HybridFilterService(
    z_threshold=0.5,
    ratio_threshold=0.6
)

qa_service = QAService(
    retrieval_service=retrieval_service,
    filter_service=filter_service,
    llm=llm
)

# ------------------------
# 실행
# ------------------------

stream = qa_service.answer(
    query="이 문서의 핵심 내용은?",
    history="",
    selected_ids=None
)

for chunk in stream:
    print(chunk, end="", flush=True)
