import json
import logging
import urllib.error
import urllib.request

from tenacity import retry, stop_after_attempt, wait_random_exponential

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.constants import (
    OLLAMA_MAX_RETRIES,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
)
from raptor_qdrant.rag.llm.base import BaseChatbotModel
from raptor_qdrant.rag.utils import load_prompt

logger = logging.getLogger(__name__)


# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
class Ollama(BaseChatbotModel):
    def __init__(
        self,
        model: str = settings.OLLAMA_MODEL,
        host: str = settings.OLLAMA_HOST,
        max_tokens: int = OLLAMA_MAX_TOKENS,
        temperature: float = OLLAMA_TEMPERATURE,
        timeout: int = OLLAMA_TIMEOUT_SECONDS,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        logger.info(
            f"ollama client created for model {self.model} at {self.host}"
        )

    @property
    def describe(self) -> str:
        return f"Ollama({self.model})"

    def health_check(self) -> None:
        """서버와 모델이 준비됐는지 확인한다. 아니면 RuntimeError."""
        try:
            with urllib.request.urlopen(
                f"{self.host}/api/tags", timeout=10
            ) as response:
                tags = json.loads(response.read())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach ollama at {self.host}: {e}. "
                "start it with `ollama serve`"
            ) from e

        available = [m.get("name", "") for m in tags.get("models", [])]
        wanted = self._fully_qualified_model_tag()
        if wanted not in available:
            raise RuntimeError(
                f"model '{self.model}' is not available in ollama. "
                f"pull it with `ollama pull {self.model}`. "
                f"currently available: {available or '(none)'}"
            )

        logger.info(f"ollama health check passed: {wanted}")

    def _fully_qualified_model_tag(self) -> str:
        return self.model if ":" in self.model else f"{self.model}:latest"

    def _create_prompt(self, context: str, question: str) -> list:
        prompt_template = load_prompt("prompt/chatbot.md")
        formatted_prompt = prompt_template.format(
            context=context, question=question
        )
        return [{"role": "user", "content": formatted_prompt}]

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(OLLAMA_MAX_RETRIES),
    )
    def answer(self, context: str, question: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": self._create_prompt(context, question),
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                body = json.loads(response.read())
            return body["message"]["content"].strip()

        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            logger.error(f"ollama returned HTTP {e.code}: {detail}")
            raise RuntimeError(
                f"ollama api call failed: {e.code} {detail}"
            ) from e
        except (urllib.error.URLError, TimeoutError) as e:
            logger.error(f"error calling ollama: {e}")
            raise RuntimeError(f"ollama api call failed: {e}") from e
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"unexpected ollama response shape: {e}")
            raise RuntimeError(
                f"ollama returned an unreadable response: {e}"
            ) from e
