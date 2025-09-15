import logging
from typing import Optional, Tuple, List


from src.core.config import settings
from src.rag.summarizer import BaseSummarizationModel, BedrockSummarizer
from .embedding import BaseEmbeddingModel, KoreanEmbeddingModel
from .llm.bedrock import BaseChatbotModel, AmazonBedrock
from .builder.cluster.tree_builder import (
    ClusterTreeBuilder,
    ClusterTreeConfig,
)
from .retriever.qdrant_retriever import (
    QdrantRetriever,
    QdrantRetrieverConfig,
)

logger = logging.getLogger(__name__)

SUPPORTED_TREE_BUILDERS = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

"""
collection_name: 문서의 카테고리/도메인별 구분
    document_name: 각 개별 문서 구분
"""


class EngineConfig:
    def __init__(
        self,
        collection_name: str,
        tree_builder_type: str = "cluster",
        embedding_model: Optional[BaseEmbeddingModel] = None,
        summarization_model: Optional[BaseSummarizationModel] = None,
        llm: Optional[BaseChatbotModel] = None,
        max_tokens_per_chunk: int = 512,
        max_context_tokens: int = 3500,
        top_k: int = 5,
        hybrid_alpha: float = 0.7,
    ):
        if not collection_name:
            raise ValueError("collection_name must be provided and non-empty")

        self.tree_builder_type = tree_builder_type
        self.embedding_model = embedding_model or KoreanEmbeddingModel()
        self.summarization_model = summarization_model or BedrockSummarizer()
        self.llm = llm or AmazonBedrock()

        self.retriever_config = QdrantRetrieverConfig(
            max_tokens=max_tokens_per_chunk,
            max_context_tokens=max_context_tokens,
            embedding_model=self.embedding_model,
            embedding_model_string=settings.EMBEDDING_MODEL_STRING,
            top_k=top_k,
            collection_name=collection_name,
            hybrid_alpha=hybrid_alpha,
        )

        if tree_builder_type in SUPPORTED_TREE_BUILDERS:
            _, tree_builder_config_class = SUPPORTED_TREE_BUILDERS[
                tree_builder_type
            ]
            self.tree_builder_config = tree_builder_config_class(
                embedding_models={
                    settings.EMBEDDING_MODEL_STRING: self.embedding_model
                },
                cluster_embedding_model=settings.EMBEDDING_MODEL_STRING,
                summarization_model=self.summarization_model,
            )
        else:
            raise ValueError(
                f"unsupported tree_builder_type: {tree_builder_type}"
            )


class RaptorEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        if config is None:
            config = EngineConfig()

        self.config = config
        self.llm = self.config.llm
        self.retriever = QdrantRetriever(self.config.retriever_config)
        tree_builder_class, _ = SUPPORTED_TREE_BUILDERS[
            self.config.tree_builder_type
        ]
        self.tree_builder = tree_builder_class(self.config.tree_builder_config)

        logger.info(
            f"raptor engine initialized for collection '{self.retriever.collection_name}'"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[dict]]:
        if self.retriever is None:
            raise ValueError("retriever is not initialized.")

        logger.info(f"retrieving context for query: '{query}'")
        return self.retriever.retrieve(
            query,
            top_k=top_k,
            max_tokens=max_tokens,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> str:
        context, _ = self.retrieve(
            question,
            top_k=top_k,
            max_tokens=max_tokens,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )
        if not context:
            logger.warning("no relevant context found to answer the question.")
            return "no relevant context found to answer the question."

        logger.info("context retrieved. generating answer...")
        answer = self.llm.answer(context, question)
        return answer

    def get_document_points(self, document_name: str) -> List[dict]:
        """특정 문서의 모든 포인트를 반환"""
        return self.retriever.get_points_by_document_name(document_name)

    def add_document(self, text: str, document_name: Optional[str] = None):
        logger.info("building raptor tree from document...")
        tree = self.tree_builder.build_from_text(text, use_multithreading=True)

        logger.info(
            f"indexing all {len(tree.all_nodes)} nodes from the tree into Qdrant..."
        )
        self.retriever.build_from_tree(tree, document_name=document_name)
        logger.info("document processing and indexing complete.")

    def update_document(self, text: str, document_name: str) -> int:
        """기존 문서를 업데이트, 업데이트된 포인트 수를 반환"""
        # 1. 기존 문서 삭제
        deleted_count = self.delete_document(document_name)
        if deleted_count > 0:
            logger.info(
                f"deleted {deleted_count} existing points for '{document_name}'"
            )

        # 2. 새로운 문서 추가
        logger.info("building updated raptor tree from document...")
        tree = self.tree_builder.build_from_text(text, use_multithreading=True)

        logger.info(
            f"indexing {len(tree.all_nodes)} new nodes for '{document_name}'..."
        )
        self.retriever.build_from_tree(
            tree, document_name=document_name, append_mode=True
        )

        logger.info(f"document '{document_name}' update complete.")
        return len(tree.all_nodes)

    def delete_document(self, document_name: str) -> int:
        """특정 문서의 모든 데이터를 삭제, 삭제된 포인트 수 반환"""
        return self.retriever.delete_points_by_document_name(document_name)

    def list_documents(self) -> List[str]:
        """저장된 모든 문서명 목록을 반환"""
        # 모든 포인트에서 document_name 필드 수집
        scroll_result = self.retriever.client.scroll(
            collection_name=self.retriever.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=10000,
        )

        document_names = set()
        for point in scroll_result[0]:
            if "document_name" in point.payload:
                document_names.add(point.payload["document_name"])

        return sorted(list(document_names))

    def list_collections(self) -> List[str]:
        """모든 컬렉션명 목록을 반환"""
        try:
            responses = self.retriever.client.get_collections()
            return [collection.name for collection in responses.collections]
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """특정 컬렉션을 삭제"""
        try:
            self.retriever.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete collection '{collection_name}': {e}"
            )
            return False
