import logging
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.constants import (
    CPU_EMBEDDING_BATCH_SIZE,
    EMBEDDING_BATCH_SIZE,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 모델을 올리는 동안은 한 스레드만 들어간다. functools.cache 는 결과를
# 기억할 뿐 첫 호출을 직렬화하지 않아, 스레드 둘이 같은 가중치를 동시에
# 올리다 torch 가 meta tensor 를 복사하지 못하고 터진다. 로딩은 몇 초지만
# 한 번뿐이므로 그동안 잡고 있어도 된다. 재진입 가능해야 하는 것은
# sentence_transformer 가 llama_embedding 을 거쳐 가기 때문이다
_LOAD_LOCK = threading.RLock()


def shared_model[T](
    store: dict[str, Any], key: str, build: Callable[[], T]
) -> T:
    """같은 키로는 한 번만 만들고 그것을 나눠 쓴다."""
    with _LOAD_LOCK:
        if key not in store:
            store[key] = build()
        return store[key]


_LLAMA_HANDLES: dict[str, Any] = {}
_SENTENCE_TRANSFORMERS: dict[str, Any] = {}


def embedding_batch_size() -> int:
    """장치에 맞는 배치 크기."""
    return (
        CPU_EMBEDDING_BATCH_SIZE
        if embedding_device() == "cpu"
        else EMBEDDING_BATCH_SIZE
    )


def embedding_device() -> str | None:
    """임베딩을 올릴 장치. None 이면 라이브러리가 고른다.

    비워 두면 MPS 가 있는 mac 에서 cpu 를 고른다. MPS 할당자는 노트를 이어
    자르는 동안 드라이버 풀을 계속 키우고 empty_cache 로도 돌려주지 않는다.
    노트 여섯 개에 12.7GB 까지 불어나는 것을 확인했고, 그래서 볼트가 조금만
    커지면 인덱싱이 중간에 메모리 부족으로 죽는다. cpu 는 2.6 배 느리지만
    끝까지 간다. 여유가 있으면 EMBEDDING_DEVICE=mps 로 되돌리면 된다.
    """
    configured = settings.EMBEDDING_DEVICE.strip()
    if configured:
        return configured

    import torch

    if torch.backends.mps.is_available():
        return "cpu"
    return None


def llama_embedding(model_name: str) -> "HuggingFaceEmbedding":
    """LlamaIndex 가 쓰는 임베딩 핸들. 모델 이름당 하나만 만든다.

    청커와 리트리버가 각자 만들면 같은 가중치가 메모리에 여러 벌 뜬다.
    KURE-v1 한 벌이 기가바이트 단위라 벌수가 그대로 메모리다.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    def build() -> HuggingFaceEmbedding:
        device = embedding_device()
        logger.info(
            f"loading llama-index embedding handle for {model_name} "
            f"on {device or 'the default device'}"
        )
        return HuggingFaceEmbedding(model_name=model_name, device=device)

    return shared_model(_LLAMA_HANDLES, model_name, build)


def sentence_transformer(model_name: str) -> SentenceTransformer:
    """가중치 한 벌. 직접 쓰는 쪽과 LlamaIndex 쪽이 같은 것을 본다.

    llama-index 핸들은 안에 SentenceTransformer 를 하나 들고 있다. 그것을
    같이 쓰면 프로세스에 가중치가 한 벌만 남는다. 따로 올리면 MPS 에서만
    5GB 가까이 잡아먹고, 긴 노트를 의미 분할하다 메모리가 모자라 죽는다.
    핸들 쪽 구현이 바뀌어 못 꺼내면 그때는 따로 올린다.
    """

    def build() -> SentenceTransformer:
        shared = getattr(llama_embedding(model_name), "_model", None)
        if isinstance(shared, SentenceTransformer):
            return shared

        logger.info(
            f"loading a separate sentence-transformer for {model_name}"
        )
        return SentenceTransformer(model_name, device=embedding_device())

    return shared_model(_SENTENCE_TRANSFORMERS, model_name, build)


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
    def __init__(self, model_name: str | None = None):
        """모델 이름만 잡아둔다. 가중치는 처음 쓸 때 올린다.

        status 나 sync --dry-run 은 임베딩을 한 번도 부르지 않는다. 엔진을
        세울 때 기가바이트짜리 가중치부터 올리면 그 명령들이 이유 없이 느리다.
        """
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = sentence_transformer(self._model_name)
                logger.info(f"{self._model_name} loaded successfully")
            except Exception as e:
                logger.error(
                    f"failed to load embedding model {self._model_name}: {e}"
                )
                raise ValueError(
                    f"failed to initialize embedding model: {e}"
                ) from e
        return self._model

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
                texts, batch_size=embedding_batch_size(), convert_to_numpy=True
            )
            return [vector.tolist() for vector in embeddings]
        except Exception as e:
            logger.error(f"failed to create embeddings: {e}")
            raise RuntimeError(f"embedding creation failed: {e}") from e
