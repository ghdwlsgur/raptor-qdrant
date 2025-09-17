import logging
from abc import ABC, abstractmethod

from src.rag.llm.bedrock import AmazonBedrock
from src.rag.utils import load_prompt

logger = logging.getLogger(__name__)


class BaseSummarizationModel(ABC):
    """모든 요약 모델이 상속받아야 할 기본 클래스"""

    @abstractmethod
    def summarize(self, text: str) -> str:
        pass


class BedrockSummarizer(BaseSummarizationModel):
    def __init__(self):
        try:
            self.bedrock_client = AmazonBedrock()
            logger.info("bedrock summarizer initialized")
        except Exception as e:
            raise ValueError(f"failed to initialize Bedrock client: {e}")

    def summarize(self, text: str) -> str:
        if not text.strip():
            logger.warning(
                "summarize called with empty context. returning empty string."
            )
            return ""

        try:
            prompt_template = load_prompt("prompt/summarizer.md")
            prompt = prompt_template.format(text=text)

            summary = self.bedrock_client.answer(context="", question=prompt)
            return summary.strip() if summary else ""

        except Exception as e:
            logger.error(f"error during summarization with Bedrock: {e}")
            return f"error during summarization: {e}"
