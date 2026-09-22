"""후보를 다시 줄 세우는 크로스 인코더.

하이브리드 검색은 질문과 청크를 따로 벡터로 만들어 견준다. 크로스 인코더는
둘을 함께 읽고 점수를 매긴다. 그만큼 잘 맞히고 그만큼 느리다. 후보 마흔여덟
개를 채점하는 값이라 검색 전체에 쓰지 않고 마지막 줄 세우기에만 쓴다.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.constants import CHUNK_MAX_TOKENS
from raptor_qdrant.rag.embedding import embedding_device

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    @abstractmethod
    def scores(self, query: str, documents: Sequence[str]) -> list[float]:
        """질문과 각 문서를 함께 읽고 관련도를 매긴다."""

    @property
    def describe(self) -> str:
        return self.__class__.__name__


@cache
def cross_encoder(model_name: str) -> "CrossEncoder":
    """모델 이름당 하나. 처음 쓸 때 올린다.

    max_length 를 청크 상한에 맞춘다. 두지 않으면 토크나이저 기본값(8192)을
    따라가는데, 512 토큰짜리 청크를 채점하면서 그 길이로 패딩하느라 같은
    일이 두 배 넘게 걸린다.
    """
    from sentence_transformers import CrossEncoder

    device = embedding_device()
    logger.info(
        f"loading reranker {model_name} on {device or 'the default device'}"
    )
    return CrossEncoder(model_name, device=device, max_length=CHUNK_MAX_TOKENS)


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL

    @property
    def describe(self) -> str:
        return f"CrossEncoder({self.model_name})"

    def scores(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        pairs = [[query, document] for document in documents]
        return [
            float(score)
            for score in cross_encoder(self.model_name).predict(pairs)
        ]


def default_reranker() -> BaseReranker | None:
    """설정이 비어 있으면 재순위화를 하지 않는다."""
    name = settings.RERANKER_MODEL.strip()
    return CrossEncoderReranker(name) if name else None


def rerank(
    query: str,
    nodes: Sequence[Any],
    reranker: BaseReranker,
    limit: int = 0,
) -> list[Any]:
    """후보를 다시 줄 세운다.

    채점은 비싸다. limit 을 주면 상위 그만큼만 다시 세우고 나머지는 원래
    순서로 뒤에 붙인다. 아래쪽 후보는 어차피 최종 선택에 들 일이 드물다.

    노드의 score 를 재순위 점수로 바꾼다. 무엇이 순서를 정했는지와 보고되는
    점수가 어긋나면 로그를 읽는 사람이 헤맨다.
    """
    if not nodes:
        return []

    head = list(nodes[:limit]) if limit else list(nodes)
    tail = list(nodes[limit:]) if limit else []

    ranked = sorted(
        zip(head, reranker.scores(query, [n.text for n in head]), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    for node, score in ranked:
        node.score = score
    return [node for node, _ in ranked] + tail
