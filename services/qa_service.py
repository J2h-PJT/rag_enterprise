from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QAService:

    def __init__(
        self,
        retrieval_service,
        filter_service,
        context_manager,
        compression_service,
        llm
    ):
        self.retrieval_service = retrieval_service
        self.filter_service = filter_service
        self.context_manager = context_manager
        self.compression_service = compression_service
        self.llm = llm

        self.chain = self._build_chain()

    # ------------------------
    # Prompt Builder
    # ------------------------
    def _build_chain(self):

        template = """
        당신은 문서 기반 QA 시스템입니다.

        이전 대화:
        {history}

        참고 문서:
        {context}

        사용자 질문:
        {question}

        규칙:
        - 반드시 문서 내용 기반으로 답하세요.
        - 추측하지 마세요.
        - 문서에 없으면 "문서에서 찾을 수 없습니다."라고 답하세요.
        """

        prompt = ChatPromptTemplate.from_template(template)

        return (
            prompt
            | self.llm.get_model()
            | StrOutputParser()
        )

    # ------------------------
    # Main QA Method
    # ------------------------
    def answer(self, query, history="", selected_ids=None):

        docs = self.retrieval_service.retrieve(query, selected_ids)

        docs = self.filter_service.apply(docs)

        if not docs:
            return iter(["관련 문서를 찾지 못했습니다."])

        # 1️⃣ 1차 토큰 제한
        docs = self.context_manager.trim(docs)

        # 2️⃣ 여전히 길면 압축
        docs = self.compression_service.compress(query, docs)

        # 3️⃣ 다시 trim
        docs = self.context_manager.trim(docs)

        context = "\n\n".join(d.page_content for d in docs)

        return self.chain.stream({
            "history": history,
            "context": context,
            "question": query
        })
