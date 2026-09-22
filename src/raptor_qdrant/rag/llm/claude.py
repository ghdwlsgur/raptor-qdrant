import logging
from pathlib import Path
from typing import Literal

import anthropic
from anthropic import Omit
from anthropic.types.beta import BetaMessage, BetaOutputConfigParam

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.constants import (
    ANTHROPIC_FALLBACK_BETA,
    ANTHROPIC_MAX_RETRIES,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_OAUTH_BETA,
)
from raptor_qdrant.rag.llm.base import BaseChatbotModel

logger = logging.getLogger(__name__)

Effort = Literal["", "low", "medium", "high", "xhigh", "max"]

HEALTH_CHECK_MAX_TOKENS = 16


def read_oauth_token(path: str) -> str | None:
    """OAuth 액세스 토큰을 파일에서 읽는다.

    토큰은 이 함수와 SDK 클라이언트 사이에서만 오간다. 설정값도 로그도 명령줄도
    경로만 본다. 값을 환경변수로 옮기면 프로세스 목록과 자식 프로세스에 그대로
    흘러간다.
    """
    target = path.strip()
    if not target:
        return None

    try:
        token = Path(target).expanduser().read_text(encoding="utf-8").strip()
    except OSError as e:
        raise ValueError(
            f"cannot read the anthropic oauth token file at {target}: {e}"
        ) from e

    if not token:
        raise ValueError(f"the oauth token file at {target} is empty")
    return token


# https://docs.anthropic.com/en/api/messages
class Claude(BaseChatbotModel):
    """Anthropic Messages API 로 답변과 요약을 받는다.

    Opus 5 계열은 temperature 를 받지 않고(400) 사고가 기본으로 켜져 있다.
    깊이와 지출은 output_config.effort 로 조절한다. effort 를 받지 않는 구형
    모델(claude-haiku-4-5 등)을 쓸 거면 ANTHROPIC_EFFORT 를 비운다.

    자격증명은 셋 중 하나다. ANTHROPIC_OAUTH_TOKEN_FILE 이 가리키는 OAuth
    토큰, ANTHROPIC_API_KEY, 아니면 SDK 가 환경에서 알아서 찾는 것.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        effort: Effort | None = None,
        max_tokens: int = ANTHROPIC_MAX_TOKENS,
        oauth_token_file: str | None = None,
    ):
        self.model = model or settings.ANTHROPIC_MODEL
        self.effort: Effort = (
            effort if effort is not None else settings.ANTHROPIC_EFFORT
        )
        self.max_tokens = max_tokens

        token = read_oauth_token(
            oauth_token_file
            if oauth_token_file is not None
            else settings.ANTHROPIC_OAUTH_TOKEN_FILE
        )
        self.uses_oauth = token is not None

        try:
            self.client = anthropic.Anthropic(
                # 둘을 함께 넘기면 X-Api-Key 와 Authorization 이 같이 나간다
                api_key=None
                if token
                else (api_key or settings.ANTHROPIC_API_KEY or None),
                auth_token=token,
                max_retries=ANTHROPIC_MAX_RETRIES,
            )
        except Exception as e:
            raise ValueError(
                f"failed to initialize anthropic client: {e}"
            ) from e

        logger.info(
            f"anthropic client created for model {self.model} "
            f"({'oauth token file' if self.uses_oauth else 'api key'})"
        )

    @property
    def describe(self) -> str:
        return f"Claude({self.model})"

    def health_check(self) -> None:
        """키·모델 접근과 요청 파라미터를 실제 호출 한 번으로 확인한다.

        models.retrieve 로는 effort 가 이 모델에서 받아들여지는지 알 수 없다.
        그 400 은 요약을 돌리기 시작해야 드러나고, 그때는 임베딩에 쓴 시간이
        이미 지나간 뒤다. 본 호출과 같은 모양을 열여섯 토큰으로 미리 태운다.
        """
        self._create("ping", HEALTH_CHECK_MAX_TOKENS)
        logger.info(f"anthropic health check passed: {self.model}")

    def complete(self, prompt: str) -> str:
        message = self._create(prompt, self.max_tokens)

        if message.stop_reason == "refusal":
            category = getattr(message.stop_details, "category", None)
            raise RuntimeError(
                f"anthropic declined to answer (category {category})"
            )
        if message.stop_reason == "max_tokens":
            logger.warning(
                f"anthropic response hit max_tokens ({self.max_tokens}), "
                "the text is cut off"
            )

        return "".join(
            block.text for block in message.content if block.type == "text"
        ).strip()

    def _betas(self) -> list[str]:
        betas = [ANTHROPIC_FALLBACK_BETA]
        if self.uses_oauth:
            # 토큰을 직접 넘긴 경로에서는 SDK 가 이 플래그를 붙여주지 않는다.
            # 없으면 Bearer 인증 자체가 열리지 않는다
            betas.append(ANTHROPIC_OAUTH_BETA)
        return betas

    def _create(self, prompt: str, max_tokens: int) -> BetaMessage:
        output_config: BetaOutputConfigParam | Omit = anthropic.omit
        if self.effort:
            output_config = {"effort": self.effort}

        try:
            return self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
                betas=self._betas(),
                fallbacks="default",
            )
        except anthropic.NotFoundError as e:
            raise RuntimeError(
                f"anthropic model '{self.model}' is not available to this "
                f"account: {e}"
            ) from e
        except anthropic.AuthenticationError as e:
            raise RuntimeError(
                "anthropic rejected the credentials. set ANTHROPIC_API_KEY "
                f"or run `ant auth login`: {e}"
            ) from e
        except anthropic.PermissionDeniedError as e:
            raise RuntimeError(
                f"anthropic denied access to '{self.model}': {e}"
            ) from e
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"anthropic rate limit hit: {e}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"cannot reach anthropic: {e}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(
                f"anthropic api call failed: {e.status_code} {e.message}"
            ) from e
