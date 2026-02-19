"""
services/context_manager.py

Chat History + Context 관리 모듈
OpenAI 의존성 제거 (tiktoken 사용 안함)
"""

from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ContextManager:
    """
    대화 히스토리 관리 클래스
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[BaseMessage] = []

    def add_user_message(self, content: str):
        self.history.append(HumanMessage(content=content))
        self._trim_history()

    def add_ai_message(self, content: str):
        self.history.append(AIMessage(content=content))
        self._trim_history()

    def get_history(self) -> List[BaseMessage]:
        return self.history

    def clear(self):
        self.history = []

    def _trim_history(self):
        """
        최근 max_history 개수만 유지
        """
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

