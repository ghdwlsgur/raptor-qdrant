from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    LOG_LEVEL: Literal["debug", "info", "warning", "error", "critical"] = (
        "info"
    )
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    VAULT_PATH: str = "~/Documents/Obsidian Vault"
    COLLECTION_NAME: str = "obsidian"
    # 빌드 락 같은 실행 상태를 두는 곳
    STATE_DIR: str = "~/.local/state/raptor-qdrant"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    EMBEDDING_MODEL: str = "nlpai-lab/KURE-v1"

    LLM_PROVIDER: Literal["ollama", "bedrock"] = "ollama"

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"

    AWS_REGION: str = "ap-northeast-2"
    BEDROCK_MODEL_ID: str = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"

    @property
    def EMBEDDING_MODEL_STRING(self) -> str:
        return self.EMBEDDING_MODEL.split("/")[-1]


settings = Settings()
