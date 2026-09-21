from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.llm.base import BaseChatbotModel

__all__ = ["BaseChatbotModel", "create_chatbot"]


def create_chatbot(
    provider: str | None = None, model: str | None = None
) -> BaseChatbotModel:
    """공급자에 맞는 LLM 을 만든다.

    model 은 Ollama 에서만 뜻이 있다. 요약 전용으로 더 작은 모델을 따로 띄울
    때 쓴다. Bedrock 은 BEDROCK_MODEL_ID 설정을 따른다.
    """
    name = (provider or settings.LLM_PROVIDER).strip().lower()

    if name == "ollama":
        from raptor_qdrant.rag.llm.ollama import Ollama

        return Ollama(model=model) if model else Ollama()

    if name == "bedrock":
        from raptor_qdrant.rag.llm.bedrock import AmazonBedrock

        return AmazonBedrock()

    raise ValueError(
        f"unknown LLM_PROVIDER '{name}'. supported: ollama, bedrock"
    )
