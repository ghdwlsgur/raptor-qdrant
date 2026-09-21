import logging
import os
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.constants import EMBEDDING_BATCH_SIZE

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> list[float]:
        """텍스트를 임베딩 벡터로 변환"""

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 한 번에 임베딩한다.

        기본 구현은 하나씩 돈다. 배치를 지원하는 모델은 이 메서드를 덮어써서
        한 번의 호출로 처리하는 것이 훨씬 빠르다.
        """
        return [self.create_embedding(text) for text in texts]

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
        return self.create_embeddings([text])[0]

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """텍스트 묶음을 배치로 임베딩한다.

        하나씩 encode 를 부르면 호출마다 GPU 왕복이 생긴다. 리스트로 넘기면
        모델이 알아서 배치로 묶어 처리한다.
        """
        if not texts:
            return []
        if any(not text or not text.strip() for text in texts):
            raise ValueError("cannot create an embedding for empty text")

        try:
            embeddings = self.model.encode(
                texts, batch_size=EMBEDDING_BATCH_SIZE, convert_to_numpy=True
            )
            return [vector.tolist() for vector in embeddings]
        except Exception as e:
            logger.error(f"failed to create embeddings: {e}")
            raise RuntimeError(f"embedding creation failed: {e}") from e
