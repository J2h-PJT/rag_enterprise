# 2️⃣ Compression Service 설계
# 압축은 두 방식이 있다:
# ① Extractive compression
# 문장 중 중요 문장만 추출
# ② Abstractive compression
# LLM으로 요약
# 우리는 LLM 기반 compression을 설계한다.

# ✔ query-aware compression
# ✔ 문서 relevance 유지

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class CompressionService:

    def __init__(self, llm):
        self.llm = llm

    def compress(self, query, docs):

        template = """
            다음 문서를 질문과 관련된 내용만 남기고 압축하세요.

            질문:
            {question}

            문서:
            {context}

            불필요한 설명은 제거하세요.
            """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            prompt
            | self.llm.get_model()
            | StrOutputParser()
        )

        compressed_docs = []

        for doc in docs:
            compressed_text = chain.invoke({
                "question": query,
                "context": doc.page_content
            })

            doc.page_content = compressed_text
            compressed_docs.append(doc)

        return compressed_docs
