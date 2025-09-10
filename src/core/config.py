from pydantic_settings import BaseSettings
from typing import Literal
from dotenv import load_dotenv

load_dotenv(override=True)


class Settings(BaseSettings):
    LOG_LEVEL: Literal["debug", "info", "warning", "error", "critical"] = "info"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333


settings = Settings()
