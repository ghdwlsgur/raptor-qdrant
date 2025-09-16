import json
import logging
import boto3
from botocore.config import Config

from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_random_exponential
from src.core.config import settings

logger = logging.getLogger(__name__)


class BaseChatbotModel(ABC):
    @abstractmethod
    def answer(self, context: str, question: str) -> str:
        """일반적인 질답 메서드"""
        pass


# https://aws.amazon.com/ko/blogs/tech/stream-chatbot-for-amazon-bedrock/
# https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AnthropicClaude_section.html
class AmazonBedrock(BaseChatbotModel):
    def __init__(
        self,
        model_id: str = settings.BEDROCK_MODEL_ID,
        region: str = settings.AWS_REGION,
        max_tokens: int = 2048,
    ):
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens

        try:
            # Connection pool 설정을 통해 동시 연결 수 증가
            config = Config(
                region_name=self.region,
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                },
                max_pool_connections=50,  # 기본값 10에서 50으로 증가
            )

            self.bedrock_runtime = boto3.client(
                "bedrock-runtime",
                config=config,
            )
            logger.info(
                f"amazon bedrock client initialized with model: {self.model_id}"
            )
        except Exception as e:
            logger.error(f"failed to initialize amazon bedrock client: {e}")
            raise ValueError(f"failed to initialize amazon bedrock client: {e}")

    def _create_prompt(self, context: str, question: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"다음 컨텍스트를 바탕으로 질문에 답해주세요.\n\n컨텍스트:\n{context}\n\n질문: {question}\n\n질문과 같은 언어로 정확하고 완전한 답변을 제공해주세요.",
                    }
                ],
            }
        ]

    @retry(
        wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3)
    )
    def answer(self, context: str, question: str) -> str:
        messages = self._create_prompt(context, question)

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
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
