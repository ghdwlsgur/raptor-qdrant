import logging
import os
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer

from raptor_qdrant.core.config import settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> list[float]:
        """텍스트를 임베딩 벡터로 변환"""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """임베딩 벡터의 차원 수를 반환"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """LlamaIndex 쪽에 같은 모델을 지정할 때 쓰는 식별자"""


# https://huggingface.co/nlpai-lab/KURE-v1
# https://huggingface.co/nlpai-lab/KoE5
class KoreanEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """KoreanEmbeddingModel 초기화

        Args:
            model_name (str, optional): 사용할 모델의 이름
        """
        try:
            self.model = SentenceTransformer(model_name)
            self._model_name = model_name
            logger.info(f"{model_name} loaded successfully")
        except Exception as e:
            logger.error(f"failed to load embedding model {model_name}: {e}")
            raise ValueError(
                f"failed to initialize embedding model: {e}"
            ) from e

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        """임베딩 벡터의 차원 수를 반환"""
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(
                f"{self._model_name} did not report an embedding dimension"
            )
        return dimension

    def create_embedding(self, text: str) -> list[float]:
        """텍스트를 임베딩 벡터로 변환한다. 빈 텍스트는 ValueError."""
        if not text or not text.strip():
            raise ValueError("cannot create an embedding for empty text")

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"failed to create embedding: {e}")
            raise RuntimeError(f"embedding creation failed: {e}") from e
