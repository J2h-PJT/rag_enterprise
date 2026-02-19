# 🔥 이제 PDF Processor 설계
# PDFProcessor는 단순 텍스트 추출기가 아니다.
# 역할:
# PDF 로딩
# text_cleaner 적용
# semantic chunking
# metadata 추가
# vector 저장
# DB 저장
# PDF → Clean → Split → Metadata Attach → Dedup → Save(Vector+DB)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.text_cleaner import clean_text
from utils.chunk_utils import deduplicate_chunks


class PDFProcessor:

    def __init__(self, engine, vectorstore):
        self.engine = engine
        self.vectorstore = vectorstore

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

    # ------------------------
    # Main Pipeline
    # ------------------------
    def process(self, file_path: str, file_id: str):

        pages = self._load_pdf(file_path)

        documents = self._clean_pages(pages)

        chunks = self._split_documents(documents)

        chunks = self._attach_metadata(chunks, file_id)

        chunks = deduplicate_chunks(chunks)

        self._save_to_vectorstore(chunks)

        self._save_to_db(chunks, file_id)

    # ------------------------
    # Steps
    # ------------------------

    def _load_pdf(self, file_path):

        loader = PyPDFLoader(file_path)
        return loader.load()

    def _clean_pages(self, pages):

        for doc in pages:
            doc.page_content = clean_text(doc.page_content)

        return pages

    def _split_documents(self, documents):

        return self.splitter.split_documents(documents)

    def _attach_metadata(self, chunks, file_id):

        for idx, chunk in enumerate(chunks):
            chunk.metadata["file_id"] = file_id
            chunk.metadata["chunk_id"] = f"{file_id}_{idx}"

        return chunks

    def _save_to_vectorstore(self, chunks):

        self.vectorstore.add_documents(chunks)

    def _save_to_db(self, chunks, file_id):

        from sqlalchemy import text

        with self.engine.begin() as conn:
            for chunk in chunks:
                conn.execute(text("""
                    INSERT INTO pdf_documents
                    (file_id, chunk_id, content)
                    VALUES (:file_id, :chunk_id, :content)
                """), {
                    "file_id": file_id,
                    "chunk_id": chunk.metadata["chunk_id"],
                    "content": chunk.page_content
                })

