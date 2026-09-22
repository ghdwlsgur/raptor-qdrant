import logging

import openai
from openai.types.chat import ChatCompletion

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.constants import (
    OPENAI_MAX_RETRIES,
    OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE,
)
from raptor_qdrant.rag.llm.base import BaseChatbotModel

logger = logging.getLogger(__name__)

# 추론 모델은 temperature 를 고정값으로만 받는다. 보내면 400 이다
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

HEALTH_CHECK_MAX_TOKENS = 16


# https://platform.openai.com/docs/api-reference/chat
class ChatGPT(BaseChatbotModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = OPENAI_MAX_TOKENS,
        temperature: float = OPENAI_TEMPERATURE,
    ):
        self.model = model or settings.OPENAI_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            self.client = openai.OpenAI(
                api_key=api_key or settings.OPENAI_API_KEY or None,
                max_retries=OPENAI_MAX_RETRIES,
            )
        except Exception as e:
            raise ValueError(f"failed to initialize openai client: {e}") from e

        logger.info(f"openai client created for model {self.model}")

    @property
    def describe(self) -> str:
        return f"ChatGPT({self.model})"

    def supports_temperature(self) -> bool:
        return not self.model.startswith(REASONING_MODEL_PREFIXES)

    def health_check(self) -> None:
        """키·모델 접근을 본 호출과 같은 모양으로 한 번 확인한다."""
        self._create("ping", HEALTH_CHECK_MAX_TOKENS)
        logger.info(f"openai health check passed: {self.model}")

    def complete(self, prompt: str) -> str:
        completion = self._create(prompt, self.max_tokens)
        choice = completion.choices[0]

        if choice.finish_reason == "length":
            logger.warning(
                f"openai response hit max_tokens ({self.max_tokens}), "
                "the text is cut off"
            )

        return (choice.message.content or "").strip()

    def _create(self, prompt: str, max_tokens: int) -> ChatCompletion:
        temperature = (
            self.temperature
            if self.supports_temperature()
            else openai.NOT_GIVEN
        )

        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
        except openai.NotFoundError as e:
            raise RuntimeError(
                f"openai model '{self.model}' is not available to this "
                f"account: {e}"
            ) from e
        except openai.AuthenticationError as e:
            raise RuntimeError(
                f"openai rejected the credentials. set OPENAI_API_KEY: {e}"
            ) from e
        except openai.PermissionDeniedError as e:
            raise RuntimeError(
                f"openai denied access to '{self.model}': {e}"
            ) from e
        except openai.RateLimitError as e:
            raise RuntimeError(f"openai rate limit hit: {e}") from e
        except openai.APIConnectionError as e:
            raise RuntimeError(f"cannot reach openai: {e}") from e
        except openai.APIStatusError as e:
            raise RuntimeError(
                f"openai api call failed: {e.status_code} {e.message}"
            ) from e
