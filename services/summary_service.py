from sqlalchemy import text
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class SummaryService:

    def __init__(self, engine, llm):
        self.engine = engine
        self.llm = llm

    def summarize(self, file_id):

        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT page, content
                FROM pdf_documents
                WHERE file_id = :fid
                ORDER BY page
            """), {"fid": file_id}).fetchall()

        template = """
            다음은 문서 한 페이지입니다:

            {content}

            핵심만 요약하세요.
            """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        for page, content in rows:
            yield page, chain.stream({"content": content})
