# 파일 경로: src/rag/summarization_models.py

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)


class BaseSummarizationModel(ABC):
    """모든 요약 모델이 상속받아야 할 기본 클래스"""

    @abstractmethod
    def summarize(self, context: str, max_tokens: int) -> str:
        pass


class OpenAISummarizationModel(BaseSummarizationModel):
    """
    OpenAI의 Chat Completion API를 사용하여 텍스트를 요약하는 클래스.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        try:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info(
                f"OpenAISummarizationModel initialized with model: {self.model}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize OpenAI client. Is OPENAI_API_KEY set? Error: {e}"
            )

    @retry(
        wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6)
    )
    def summarize(self, context: str, max_tokens: int = 500) -> str:
        if not context.strip():
            logger.warning(
                "Summarize called with empty context. Returning empty string."
            )
            return ""

        try:
            prompt = (
                f"Please provide a concise summary of the following text. "
                f"Focus on the key facts, entities, and relationships.\n\n"
                f"Text to summarize:\n---\n{context}\n---\n\n"
                f"Concise Summary:"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )

            summary = response.choices[0].message.content
            return summary.strip() if summary else ""

        except Exception as e:
            logger.error(f"Error during summarization with {self.model}: {e}")
            return f"Error during summarization: {e}"
