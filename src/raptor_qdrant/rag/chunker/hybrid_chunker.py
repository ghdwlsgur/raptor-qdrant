from abc import ABC, abstractmethod
from typing import cast

from llama_index.core import Document
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from raptor_qdrant.rag.chunker.models.chunk_metadata import (
    ChunkingMethod,
    ChunkMetadata,
)
from raptor_qdrant.rag.constants import (
    CHUNK_MAX_TOKENS,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_CHUNK_BUFFER_SIZE,
)
from raptor_qdrant.rag.embedding import BaseEmbeddingModel
from raptor_qdrant.rag.utils import count_tokens


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
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

        # 마크다운 구조 파서 생성
        self.markdown_parser = MarkdownNodeParser()

        # 의미 기반 파서 생성
        model_id = self.embedding_model.model_name
        llama_embed_model = HuggingFaceEmbedding(model_name=model_id)
        self.semantic_parser = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=llama_embed_model,
        )

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

            is_semantic_split_needed = section_token_count > self.max_tokens
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
        return nodes
