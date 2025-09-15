from abc import ABC, abstractmethod
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
