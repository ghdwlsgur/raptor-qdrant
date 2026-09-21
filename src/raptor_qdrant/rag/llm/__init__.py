from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.llm.base import BaseChatbotModel

__all__ = ["BaseChatbotModel", "create_chatbot"]


def create_chatbot(provider: str | None = None) -> BaseChatbotModel:
    name = (provider or settings.LLM_PROVIDER).strip().lower()

    if name == "ollama":
        from raptor_qdrant.rag.llm.ollama import Ollama

        return Ollama()

    if name == "bedrock":
        from raptor_qdrant.rag.llm.bedrock import AmazonBedrock

        return AmazonBedrock()

    raise ValueError(
        f"unknown LLM_PROVIDER '{name}'. supported: ollama, bedrock"
    )
