from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    LOG_LEVEL: Literal["debug", "info", "warning", "error", "critical"] = (
        "info"
    )
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    # 진행 로그를 남길 파일. 회전 10MB, 5개 보관. 비우면 stdout 만 쓴다
    LOG_FILE: str = "~/.local/state/raptor-qdrant/raptor-qdrant.log"

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
    # 요약 전용 Ollama 모델. 비우면 OLLAMA_MODEL 을 쓴다. 중간 요약은 검색
    # 앵커 역할이라 답변 모델보다 작은 것으로도 충분한 경우가 많다
    OLLAMA_SUMMARY_MODEL: str = ""
    # 클러스터 요약 동시 실행 수. 0 이면 공급자 기본값(ollama 2, bedrock 10).
    # Ollama 는 서버의 OLLAMA_NUM_PARALLEL 이상으로 올려도 직렬화된다
    SUMMARY_WORKERS: int = 0

    AWS_REGION: str = "ap-northeast-2"
    BEDROCK_MODEL_ID: str = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"

    @property
    def EMBEDDING_MODEL_STRING(self) -> str:
        return self.EMBEDDING_MODEL.split("/")[-1]


settings = Settings()
