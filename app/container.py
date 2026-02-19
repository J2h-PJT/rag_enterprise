from core.engine import CoreEngine
from retrievers.base_retriever import BaseRetriever
from services.retrieval_service import RetrievalService
from services.filter_service import FilterService
from services.compression_service import CompressionService
from services.context_manager import ContextManager
from services.qa_service import QAService
from services.pdf_processor import PDFProcessor
from worker.document_worker import DocumentWorker
from utils.chunker import TextChunker

def build_container(config):
    core = CoreEngine(config)

    retriever = BaseRetriever(
        vectorstore=core.vectorstore,
        top_k=config.RETRIEVAL_TOP_K
    )

    retrieval_service = RetrievalService(
        retriever=retriever,
        reranker=core.reranker,
        config=config
    )

    filter_service = FilterService()
    compression_service = CompressionService()
    context_manager = ContextManager(config.MAX_CONTEXT_TOKENS)
    chunker = TextChunker()

    qa_service = QAService(
        retrieval_service=retrieval_service,
        filter_service=filter_service,
        context_manager=context_manager,
        compression_service=compression_service,
        llm=core.llm
    )

    pdf_processor = PDFProcessor(
        db_engine=core.db_engine,
        vectorstore=core.vectorstore,
        chunker=chunker
    )

    worker = DocumentWorker(core, pdf_processor)

    return {
        "core": core,
        "qa_service": qa_service,
        "worker": worker
    }
