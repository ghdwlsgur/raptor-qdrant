from pydantic_settings import BaseSettings
from typing import Literal
from dotenv import load_dotenv

load_dotenv(override=True)


class Settings(BaseSettings):
    LOG_LEVEL: Literal["debug", "info", "warning", "error", "critical"] = "info"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    EMBEDDING_MODEL: str = "nlpai-lab/KURE-v1"

    # AWS Bedrock 설정
    AWS_BEARER_TOKEN_BEDROCK: str = ""
    AWS_REGION: str = "ap-northeast-2"
    BEDROCK_MODEL_ID: str = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"

    @property
    def EMBEDDING_MODEL_STRING(self) -> str:
        return self.EMBEDDING_MODEL.split("/")[-1]


settings = Settings()
