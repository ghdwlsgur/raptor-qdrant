import logging
import re
from abc import ABC, abstractmethod
from typing import cast

from llama_index.core import Document
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import TextNode

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.chunker.models.chunk_metadata import (
    ChunkingMethod,
    ChunkMetadata,
)
from raptor_qdrant.rag.constants import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_CHUNK_BUFFER_SIZE,
)
from raptor_qdrant.rag.embedding import BaseEmbeddingModel, llama_embedding
from raptor_qdrant.rag.utils import (
    TOKEN_COUNT_KEY,
    count_tokens,
    resolve_token_count,
)

logger = logging.getLogger(__name__)


def tag_chunk(
    node: TextNode,
    inherited_metadata: dict,
    method: ChunkingMethod,
    token_count: int | None = None,
) -> TextNode:
    if token_count is None:
        token_count = count_tokens(node.get_content())

    node.metadata = {
        **inherited_metadata,
        **ChunkMetadata(chunked_by=method, token_count=token_count).to_dict(),
    }
    return node


def _retag(node: TextNode, parts: list[str]) -> TextNode:
    node.set_content("\n\n".join(part for part in parts if part.strip()))
    node.metadata = {
        **(node.metadata or {}),
        TOKEN_COUNT_KEY: count_tokens(node.get_content()),
    }
    return node


# 전각 문장부호는 한국어 문서에서 실제로 쓰인다
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。？！])\s+")  # noqa: RUF001


def _segments(text: str, max_tokens: int) -> list[str]:
    """줄 단위로 나누되, 한 줄이 상한을 넘으면 문장으로 더 나눈다.

    줄을 먼저 보는 것은 마크다운이라서다. 목록과 표를 문장 기준으로 자르면
    행 중간이 끊긴다.
    """
    parts: list[str] = []
    for line in text.splitlines():
        if count_tokens(line) <= max_tokens:
            parts.append(line)
        else:
            parts.extend(SENTENCE_BOUNDARY.split(line))
    return parts


def _pack(parts: list[str], max_tokens: int) -> list[str]:
    packed: list[str] = []
    current: list[str] = []

    for part in parts:
        trial = [*current, part]
        if current and count_tokens("\n".join(trial)) > max_tokens:
            packed.append("\n".join(current))
            current = [part]
        else:
            current = trial

    if current:
        packed.append("\n".join(current))
    return [piece for piece in packed if piece.strip()]


def split_oversized(
    nodes: list[TextNode], max_tokens: int = CHUNK_MAX_TOKENS
) -> list[TextNode]:
    """상한을 넘는 조각을 더 자른다.

    512 는 의미 분할을 시작하는 기준일 뿐이고, 분할 결과의 길이는 아무도
    보장하지 않는다. 실측으로 잎의 8%가 상한을 넘었고 가장 큰 것이 2,031
    토큰이었다. 그런 조각 하나가 4096 예산의 절반을 먹고 다른 근거를 밀어낸다.
    """
    out: list[TextNode] = []

    for node in nodes:
        text = node.get_content()
        if count_tokens(text) <= max_tokens:
            out.append(node)
            continue

        pieces = _pack(_segments(text, max_tokens), max_tokens)
        logger.debug(
            f"split a {count_tokens(text)} token chunk into {len(pieces)}"
        )
        out.append(_retag(node, [pieces[0]]))
        for piece in pieces[1:]:
            out.append(
                _retag(
                    TextNode(text=piece, metadata=dict(node.metadata or {})),
                    [piece],
                )
            )

    return out


def merge_small_chunks(
    nodes: list[TextNode],
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> list[TextNode]:
    """토큰이 모자란 조각을 뒤따르는 조각에 붙인다.

    마크다운 파서는 본문 없는 제목도 노드 하나로 내놓는다. 그 여덟 토큰짜리
    조각이 짧고 일반적인 질의와 가까워 상위권을 차지하면서 정작 컨텍스트에는
    아무것도 보태지 못한다. 실측으로 개괄 질문마다 상위 다섯 중 한둘이
    그런 조각이었다. 제목은 버리지 않고 뒤 본문의 머리말로 붙인다.

    붙여서 상한을 넘기면 붙이지 않는다. 작은 조각을 없애자고 큰 조각을
    만들면 이번엔 그게 컨텍스트 예산을 통째로 먹는다.
    """
    if not nodes:
        return nodes

    merged: list[TextNode] = []
    carried: list[TextNode] = []

    def texts_of(held: list[TextNode]) -> list[str]:
        return [node.get_content() for node in held]

    def fits(parts: list[str]) -> bool:
        return count_tokens("\n\n".join(parts)) <= max_tokens

    for node in nodes:
        content = node.get_content()
        if resolve_token_count(node.metadata, content) < min_tokens:
            carried.append(node)
            continue

        held = texts_of(carried)
        if carried and fits([*held, content]):
            merged.append(_retag(node, [*held, content]))
        else:
            if carried:
                merged.append(_retag(carried[0], held))
            merged.append(node)
        carried = []

    if carried:
        held = texts_of(carried)
        if merged and fits([merged[-1].get_content(), *held]):
            _retag(merged[-1], [merged[-1].get_content(), *held])
        else:
            merged.append(_retag(carried[0], held))

    return merged


class BaseChunker(ABC):
    """청킹 전략의 기본 클래스"""

    @abstractmethod
    def chunk(self, text: str) -> list[TextNode]:
        """텍스트를 TextNode 리스트로 분할"""
        pass


class HybridChunker(BaseChunker):
    """하이브리드 청킹 클래스
    1차로 Markdown 구조를 기준으로 텍스트를 분할
    2차로 청크가 큰 경우에 의미론적 분할 적용
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        max_tokens: int = CHUNK_MAX_TOKENS,
        buffer_size: int = SEMANTIC_CHUNK_BUFFER_SIZE,
        breakpoint_percentile_threshold: int = SEMANTIC_BREAKPOINT_PERCENTILE,
        min_tokens: int = CHUNK_MIN_TOKENS,
        semantic: bool | None = None,
    ):
        """초기화 및 파서 설정

        Args:
            embedding_model (BaseEmbeddingModel): 문장의 의미를 벡터로 변환하는 임베딩 모델
            max_tokens (int, optional): 해당 값을 초과할 경우 의미론적 분할
            buffer_size (int, optional): SemanticSplitter가 분할 지점 주변의 문맥을 유지하기 위해 사용하는 버퍼 크기
            breakpoint_percentile_threshold (int, optional): 문장 간 의미적 거리의 분포에서 분할점으로 판단할 임계값
        """
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.semantic = (
            semantic if semantic is not None else settings.SEMANTIC_CHUNKING
        )
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

        # 마크다운 구조 파서 생성
        self.markdown_parser = MarkdownNodeParser()
        self._semantic_parser: SemanticSplitterNodeParser | None = None

    @property
    def semantic_parser(self) -> SemanticSplitterNodeParser:
        """의미 기반 파서. 임베딩 모델을 물고 있어 처음 쓸 때 만든다.

        긴 섹션이 있어야 부른다. 청킹을 한 번도 하지 않는 status·sync 가
        청커를 세운다는 이유만으로 가중치를 올릴 이유는 없다.
        """
        if self._semantic_parser is None:
            self._semantic_parser = SemanticSplitterNodeParser(
                buffer_size=self.buffer_size,
                breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
                embed_model=llama_embedding(self.embedding_model.model_name),
            )
        return self._semantic_parser

    def chunk(self, text: str) -> list[TextNode]:
        """텍스트를 하이브리드 방식으로 청킹, 1차로 마크다운 구조, 2차로 의미론적 분할 적용

        Args:
            text (str): 청킹할 원본 텍스트

        Returns:
            List[TextNode]: 청킹된 텍스트 노드 리스트
        """
        # LlamaIndex의 파서들은 Document 객체 리스트를 입력으로 받음
        documents = [Document(text=text)]

        # 마크다운 파서로 구조적 노드 추출
        structural_nodes = cast(
            list[TextNode],
            self.markdown_parser.get_nodes_from_documents(documents),
        )

        # 긴 섹션에 대해 SemanticSplitter를 조건부로 적용하기 위한 리스트
        nodes: list[TextNode] = []
        for structural_node in structural_nodes:
            section_token_count = count_tokens(structural_node.text)
            existing_metadata = structural_node.metadata or {}

            is_semantic_split_needed = (
                self.semantic and section_token_count > self.max_tokens
            )
            method = (
                ChunkingMethod.HYBRID
                if is_semantic_split_needed
                else ChunkingMethod.MARKDOWN
            )

            # 토큰 수가 임계값을 초과하면 SemanticSplitter로 추가 분할
            if is_semantic_split_needed:
                sub_documents = [Document(text=structural_node.text)]
                semantic_nodes = cast(
                    list[TextNode],
                    self.semantic_parser.get_nodes_from_documents(
                        sub_documents
                    ),
                )
                nodes.extend(
                    tag_chunk(node, existing_metadata, method)
                    for node in semantic_nodes
                )
            else:
                nodes.append(
                    tag_chunk(
                        structural_node,
                        existing_metadata,
                        method,
                        token_count=section_token_count,
                    )
                )

        bounded = split_oversized(nodes, self.max_tokens)
        merged = merge_small_chunks(bounded, self.min_tokens, self.max_tokens)
        self._warn_about_oversized(merged)
        return merged

    def _warn_about_oversized(self, nodes: list[TextNode]) -> None:
        """의미 분할을 거치고도 상한을 넘는 조각이 있으면 알린다.

        SemanticSplitter 는 의미 경계로 자를 뿐 길이를 보장하지 않는다.
        그런 조각은 컨텍스트 예산을 혼자 먹거나 통째로 버려진다.
        """
        oversized = [
            count_tokens(node.get_content())
            for node in nodes
            if count_tokens(node.get_content()) > self.max_tokens
        ]
        if oversized:
            logger.warning(
                f"{len(oversized)} chunk(s) still exceed {self.max_tokens} "
                f"tokens after splitting (largest {max(oversized)})"
            )
