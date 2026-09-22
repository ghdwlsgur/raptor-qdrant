from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# 실제 환경변수가 .env 를 이긴다. 한 번 적어둔 .env 가 그때그때 주는
# `COLLECTION_NAME=other raptor-qdrant ...` 를 덮어버리면 안 된다
load_dotenv(override=False)


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
    # 임베딩을 올릴 장치. 비우면 자동으로 고른다 (mac 에서는 cpu, 이유는
    # embedding.py 의 embedding_device 주석 참고). cuda·mps·cpu 를 직접 줄 수 있다
    EMBEDDING_DEVICE: str = ""
    # 후보를 다시 줄 세우는 크로스 인코더. 비우면 재순위화를 하지 않는다.
    # 질의마다 후보 전체를 채점하므로 CPU 에서는 몇 초가 붙는다
    RERANKER_MODEL: str = ""
    # 다시 세울 후보 수. 0 이면 받아온 후보 전부. 채점이 비싸서 여기를 줄이면
    # 그만큼 빨라진다
    RERANK_CANDIDATES: int = 0

    LLM_PROVIDER: Literal[
        "ollama",
        "bedrock",
        "anthropic",
        "claude",
        "openai",
        "chatgpt",
        "gpt",
    ] = "ollama"
    # 요약 전용 모델. 비우면 답변 모델을 그대로 쓴다. 중간 요약은 검색
    # 앵커 역할이라 답변 모델보다 작은 것으로도 충분한 경우가 많다
    SUMMARY_MODEL: str = ""
    # 클러스터 요약 동시 실행 수. 0 이면 공급자 기본값(constants.py).
    # Ollama 는 서버의 OLLAMA_NUM_PARALLEL 이상으로 올려도 직렬화된다
    SUMMARY_WORKERS: int = 0

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"

    AWS_REGION: str = "ap-northeast-2"
    BEDROCK_MODEL_ID: str = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"

    # 비우면 SDK 가 ANTHROPIC_API_KEY 환경변수와 ant 로그인 프로필을 차례로 본다
    ANTHROPIC_API_KEY: str = ""
    # OAuth 액세스 토큰이 든 파일 경로. 토큰 값이 아니라 경로만 설정에 둔다.
    # 값을 설정에 넣으면 .env 와 프로세스 환경에 그대로 남는다
    ANTHROPIC_OAUTH_TOKEN_FILE: str = ""
    ANTHROPIC_MODEL: str = "claude-opus-5"
    # 사고 깊이와 토큰 지출을 함께 정한다. 요약만 돌릴 거면 low 로 내려도 된다.
    # effort 를 받지 않는 모델(claude-haiku-4-5 등)에는 비워서 보내지 않는다
    ANTHROPIC_EFFORT: Literal["", "low", "medium", "high", "xhigh", "max"] = (
        "medium"
    )

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4.1-mini"

    @property
    def EMBEDDING_MODEL_STRING(self) -> str:
        return self.EMBEDDING_MODEL.split("/")[-1]


settings = Settings()
