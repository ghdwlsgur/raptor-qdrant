import logging
import os
from typing import List
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

from src.core.config import settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        pass


# https://huggingface.co/nlpai-lab/KURE-v1
# https://huggingface.co/nlpai-lab/KoE5
class KoreanEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        try:
            logger.info(f"loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"embedding model loaded successfully")
        except Exception as e:
            logger.error(f"failed to load embedding model {model_name}: {e}")
            raise ValueError(f"failed to initialize embedding model: {e}")

    def create_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        if not text.strip():
            logger.warning("empty text provided for embedding")
            return []

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"failed to create embedding: {e}")
            raise RuntimeError(f"embedding creation failed: {e}")
