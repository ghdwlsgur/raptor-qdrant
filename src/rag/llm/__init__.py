from typing import Optional

from src.core.config import settings
from src.rag.llm.base import BaseChatbotModel

__all__ = ["BaseChatbotModel", "create_chatbot"]


def create_chatbot(provider: Optional[str] = None) -> BaseChatbotModel:
    name = (provider or settings.LLM_PROVIDER).strip().lower()

    if name == "ollama":
        from src.rag.llm.ollama import Ollama

        return Ollama()

    if name == "bedrock":
        from src.rag.llm.bedrock import AmazonBedrock

        return AmazonBedrock()

    raise ValueError(
        f"unknown LLM_PROVIDER '{name}'. supported: ollama, bedrock"
    )
