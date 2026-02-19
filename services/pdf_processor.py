from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from utils.text_cleaner import clean_text

class PDFProcessor:
    def __init__(self, db_engine, vectorstore, chunker):
        self.db_engine = db_engine
        self.vectorstore = vectorstore
        self.chunker = chunker

    def process(self, file_path, file_id):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        cleaned = []
        for d in docs:
            cleaned.append(
                Document(
                    page_content=clean_text(d.page_content),
                    metadata={
                        "file_id": file_id,
                        "source": file_path
                    }
                )
            )
        chunks = self.chunker.split(cleaned)
        self.vectorstore.add_documents(chunks)
