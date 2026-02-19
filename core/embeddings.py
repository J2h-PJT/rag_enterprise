"""
core/embeddings.py

Ollama Embedding 모델 설정
pgvector와 함께 사용
"""

from langchain_ollama import OllamaEmbeddings


def get_embeddings(
    model: str = "nomic-embed-text",
):
    """
    Ollama Embeddings 반환

    ⚠ 사전 실행 필요:
        ollama pull nomic-embed-text

    Args:
        model (str): embedding 모델명

    Returns:
        OllamaEmbeddings
    """
    return OllamaEmbeddings(
        model=model
    )
