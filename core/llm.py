# ✔ get_model()로 LangChain 객체 반환
# ✔ Streaming은 LangChain에서 chain.stream 사용

from langchain_community.chat_models import ChatOllama
from config import LLM_MODEL
from core.interfaces.llm_base import BaseLLM


class LocalLLM(BaseLLM):

    def __init__(self):
        self.model = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1,
        )

    def get_model(self):
        return self.model


class LLMFactory:

    @staticmethod
    def create(provider: str = "local") -> BaseLLM:

        if provider == "local":
            return LocalLLM()

        raise ValueError(f"Unsupported LLM provider: {provider}")
