import logging
from abc import ABC, abstractmethod

from src.rag.llm.bedrock import AmazonBedrock

logger = logging.getLogger(__name__)


class BaseSummarizationModel(ABC):
    """모든 요약 모델이 상속받아야 할 기본 클래스"""

    @abstractmethod
    def summarize(self, context: str, max_tokens: int) -> str:
        pass


class BedrockSummarizer(BaseSummarizationModel):
    def __init__(self):
        try:
            self.bedrock_client = AmazonBedrock()
            logger.info("bedrock summarizer initialized")
        except Exception as e:
            raise ValueError(f"failed to initialize Bedrock client: {e}")

    def summarize(self, context: str, max_tokens: int = 500) -> str:
        if not context.strip():
            logger.warning(
                "summarize called with empty context. returning empty string."
            )
            return ""

        try:
            prompt = (
                f"Please provide a concise summary of the following text. "
                f"Focus on the key facts, entities, and relationships.\n\n"
                f"Text to summarize:\n---\n{context}\n---\n\n"
                f"Concise Summary:"
            )

            summary = self.bedrock_client.generate_response(
                prompt, max_tokens=max_tokens
            )
            return summary.strip() if summary else ""

        except Exception as e:
            logger.error(f"error during summarization with Bedrock: {e}")
            return f"error during summarization: {e}"
