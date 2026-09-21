import logging
from abc import ABC, abstractmethod

from raptor_qdrant.rag.llm import BaseChatbotModel, create_chatbot
from raptor_qdrant.rag.utils import load_prompt

logger = logging.getLogger(__name__)

NO_SUMMARY = "NO_SUMMARY"
MIN_USABLE_SUMMARY_CHARS = 20
SENTINEL_DECORATION = ":：-–—*·.,\"'`[]()<>{}# \t\n"  # noqa: RUF001


def is_unusable_summary(text: str) -> bool:
    if not text or not text.strip():
        return True

    undecorated = text.strip().strip(SENTINEL_DECORATION).upper()
    if not undecorated or undecorated == NO_SUMMARY:
        return True

    return len(text.strip()) < MIN_USABLE_SUMMARY_CHARS


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, text: str) -> str:
        pass


class LLMSummarizer(BaseSummarizationModel):
    def __init__(self, llm: BaseChatbotModel | None = None):
        try:
            self.llm = llm or create_chatbot()
        except Exception as e:
            raise ValueError(f"failed to initialize summarizer: {e}") from e

        logger.info(f"summarizer initialized with {self.llm.describe}")

    def summarize(self, text: str) -> str:
        if not text.strip():
            logger.warning("summarize called with empty context")
            return NO_SUMMARY

        try:
            prompt = load_prompt("prompt/summarizer.md").format(text=text)
            return self.llm.answer(context="", question=prompt).strip()
        except Exception as e:
            logger.error(f"error during summarization: {e}")
            return NO_SUMMARY
