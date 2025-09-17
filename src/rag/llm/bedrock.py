import json
import logging
import boto3

from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_random_exponential
from botocore.config import Config
from src.core.config import settings
from src.rag.constants import (
    BEDROCK_MAX_TOKENS,
    BEDROCK_ANTHROPIC_VERSION,
    BEDROCK_TEMPERATURE,
    BEDROCK_MAX_POOL_CONNECTIONS,
    BEDROCK_MAX_RETRIES,
)
from src.rag.utils import load_prompt

logger = logging.getLogger(__name__)


class BaseChatbotModel(ABC):
    @abstractmethod
    def answer(self, context: str, question: str) -> str:
        """주어진 컨텍스트와 질문을 바탕으로 답변을 생성하는 추상 메서드

        Args:
            context (str): 질문에 답변하기 위해 참고할 컨텍스트
            question (str): 사용자의 질문

        Returns:
            str: 모델이 생성한 답변
        """
        pass


# https://aws.amazon.com/ko/blogs/tech/stream-chatbot-for-amazon-bedrock/
# https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AnthropicClaude_section.html
class AmazonBedrock(BaseChatbotModel):
    def __init__(
        self,
        model_id: str = settings.BEDROCK_MODEL_ID,
        region: str = settings.AWS_REGION,
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
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            config = Config(
                region_name=self.region,
                retries={
                    'max_attempts': BEDROCK_MAX_RETRIES,  # 최대 재시도 횟수
                    'mode': 'adaptive',  # 재시도 간격 등을 동적으로 조절
                },
                max_pool_connections=BEDROCK_MAX_POOL_CONNECTIONS,  # 동시 연결 개수
            )

            self.bedrock_runtime = boto3.client(
                "bedrock-runtime",
                config=config,
            )
            logger.info(f"bedrock client created for model {self.model_id}")
        except Exception as e:
            logger.error(f"failed to initialize bedrock client: {e}")
            raise ValueError(f"failed to initialize bedrock client: {e}")

    def _create_prompt(self, context: str, question: str) -> list:
        prompt_template = load_prompt("prompt/chatbot.md")
        formatted_prompt = prompt_template.format(
            context=context, question=question
        )

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": formatted_prompt,
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
    def answer(self, context: str, question: str) -> str:
        messages = self._create_prompt(context, question)

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

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text'].strip()

        except Exception as e:
            logger.error(f"error calling Bedrock: {e}")
            raise RuntimeError(f"bedrock api call failed: {e}")
