from abc import ABC, abstractmethod
from typing import List
import tiktoken

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    MarkdownNodeParser,
)
from src.rag.embedding import BaseEmbeddingModel


class BaseChunker(ABC):
    """청킹 전략의 기본 클래스"""

    @abstractmethod
    def chunk(self, text: str) -> List[TextNode]:
        """텍스트를 TextNode 리스트로 분할"""
        pass


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
    ):
        self.embedding_model = embedding_model
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

    def chunk(self, text: str) -> List[TextNode]:
        """텍스트를 의미론적으로 분할하여 TextNode 리스트 반환"""
        # HuggingFaceEmbedding 생성
        model_id = self.embedding_model.model_name
        llama_embed_model = HuggingFaceEmbedding(model_name=model_id)

        parser = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=llama_embed_model,
        )

        documents = [Document(text=text)]
        nodes = parser.get_nodes_from_documents(documents)

        return nodes


class HybridChunker(BaseChunker):
    """하이브리드 청킹 전략: MarkdownNodeParser + SemanticSplitter 조합"""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        max_tokens: int = 512,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
    ):
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.markdown_parser = MarkdownNodeParser()
        model_id = self.embedding_model.model_name
        llama_embed_model = HuggingFaceEmbedding(model_name=model_id)
        self.semantic_parser = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=llama_embed_model,
        )

    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 계산"""
        return len(self.tokenizer.encode(text))

    def chunk(self, text: str) -> List[TextNode]:
        """하이브리드 청킹: 1단계 구조 분할 + 2단계 의미 분할"""
        documents = [Document(text=text)]
        structural_nodes = self.markdown_parser.get_nodes_from_documents(
            documents
        )

        # 2단계: 긴 섹션에 대해 SemanticSplitter 적용
        final_nodes = []

        for node in structural_nodes:
            token_count = self._count_tokens(node.text)

            if token_count > self.max_tokens:
                # 토큰 수가 임계값을 초과하면 SemanticSplitter로 추가 분할
                sub_documents = [Document(text=node.text)]
                semantic_nodes = self.semantic_parser.get_nodes_from_documents(
                    sub_documents
                )

                # 원본 노드의 메타데이터를 하위 노드들에 상속
                for semantic_node in semantic_nodes:
                    if hasattr(node, 'metadata') and node.metadata:
                        semantic_node.metadata.update(node.metadata)
                    semantic_node.metadata['chunking_method'] = 'hybrid'
                    semantic_node.metadata['original_section_tokens'] = (
                        token_count
                    )

                final_nodes.extend(semantic_nodes)
            else:
                # 토큰 수가 적당하면 구조적 청크 그대로 사용
                if hasattr(node, 'metadata'):
                    node.metadata['chunking_method'] = 'structural'
                else:
                    node.metadata = {'chunking_method': 'structural'}
                node.metadata['section_tokens'] = token_count
                final_nodes.append(node)

        return final_nodes
