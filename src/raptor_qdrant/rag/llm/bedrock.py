import json
import logging

import boto3
from botocore.config import Config
from tenacity import retry, stop_after_attempt, wait_random_exponential

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.constants import (
    BEDROCK_ANTHROPIC_VERSION,
    BEDROCK_MAX_POOL_CONNECTIONS,
    BEDROCK_MAX_RETRIES,
    BEDROCK_MAX_TOKENS,
    BEDROCK_TEMPERATURE,
)
from raptor_qdrant.rag.llm.base import BaseChatbotModel

logger = logging.getLogger(__name__)


# https://aws.amazon.com/ko/blogs/tech/stream-chatbot-for-amazon-bedrock/
# https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AnthropicClaude_section.html
class AmazonBedrock(BaseChatbotModel):
    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        max_tokens: int = BEDROCK_MAX_TOKENS,
        temperature: float = BEDROCK_TEMPERATURE,
    ):
        """Bedrock 클라이언트를 초기화하고 설정을 구성

        Args:
            model_id (str, optional): 사용할 Bedrock 모델의 ID
            region (str, optional): AWS 리전
            max_tokens (int, optional): 모델이 생성할 수 있는 최대 토큰 수
            temperature (float, optional): 답변의 창의성 파라미터
        """
        self.model_id = model_id or settings.BEDROCK_MODEL_ID
        self.region = region or settings.AWS_REGION
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            config = Config(
                region_name=self.region,
                retries={
                    "max_attempts": BEDROCK_MAX_RETRIES,
                    "mode": "adaptive",
                },
                max_pool_connections=BEDROCK_MAX_POOL_CONNECTIONS,
            )

            self.bedrock_runtime = boto3.client(
                "bedrock-runtime",
                config=config,
            )
            logger.info(f"bedrock client created for model {self.model_id}")
        except Exception as e:
            logger.error(f"failed to initialize bedrock client: {e}")
            raise ValueError(
                f"failed to initialize bedrock client: {e}"
            ) from e

    @property
    def describe(self) -> str:
        return f"AmazonBedrock({self.model_id})"

    def _create_prompt(self, prompt: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

    @retry(
        wait=wait_random_exponential(
            min=1, max=10
        ),  # 실패 시 1초에서 10초 사이의 랜덤한 시간(지수 분포)을 기다린 후 재시도
        stop=stop_after_attempt(3),  # 최대 3번까지 재시도
    )
    def complete(self, prompt: str) -> str:
        messages = self._create_prompt(prompt)

        body = json.dumps(
            {
                "anthropic_version": BEDROCK_ANTHROPIC_VERSION,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": messages,
            }
        )

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"].strip()

        except Exception as e:
            logger.error(f"error calling Bedrock: {e}")
            raise RuntimeError(f"bedrock api call failed: {e}") from e
