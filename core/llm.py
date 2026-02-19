"""
core/llm.py

Ollama 기반 LLM 설정 모듈
LangChain 0.2.x 전용
"""

from langchain_ollama import OllamaLLM


def get_llm(
    model: str = "llama3",
    temperature: float = 0.1,
):
    """
    Ollama LLM 반환

    Args:
        model (str): Ollama에 pull 되어있는 모델명
        temperature (float): 생성 다양성

    Returns:
        OllamaLLM
    """
    return OllamaLLM(
        model=model,
        temperature=temperature,
    )

