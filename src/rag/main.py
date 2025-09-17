import logging
from typing import Optional, Tuple, List


from src.core.config import settings
from src.rag.summarizer import BaseSummarizationModel, BedrockSummarizer
from .embedding import BaseEmbeddingModel, KoreanEmbeddingModel
from .llm.bedrock import BaseChatbotModel, AmazonBedrock
from .builder.cluster.cluster_builder import (
    ClusterTreeBuilder,
    ClusterTreeConfig,
)
from .retriever.qdrant_retriever import (
    QdrantRetriever,
    QdrantRetrieverConfig,
)

logger = logging.getLogger(__name__)


"""
collection_name: 문서의 카테고리/도메인별 구분
    document_name: 각 개별 문서 구분
"""


class EngineConfig:
    def __init__(
        self,
        collection_name: str,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        summarization_model: Optional[BaseSummarizationModel] = None,
        llm: Optional[BaseChatbotModel] = None,
    ):
        if not collection_name:
            raise ValueError("collection_name must be provided and non-empty")

        self.embedding_model = embedding_model or KoreanEmbeddingModel()
        self.summarization_model = summarization_model or BedrockSummarizer()
        self.llm = llm or AmazonBedrock()

        self.retriever_config = QdrantRetrieverConfig(
            embedding_model=self.embedding_model,
            collection_name=collection_name,
        )

        self.tree_builder_config = ClusterTreeConfig(
            embedding_models={
                settings.EMBEDDING_MODEL_STRING: self.embedding_model
            },
            cluster_embedding_model=settings.EMBEDDING_MODEL_STRING,
            summarization_model=self.summarization_model,
        )


class RaptorEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        if config is None:
            raise ValueError("config with collection_name must be provided")

        self.config = config
        self.llm = self.config.llm
        self.retriever = QdrantRetriever(self.config.retriever_config)
        self.tree_builder = ClusterTreeBuilder(self.config.tree_builder_config)

    def retrieve(
        self,
        query: str,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[dict]]:
        if self.retriever is None:
            raise ValueError("retriever is not initialized.")

        return self.retriever.retrieve(
            query,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )

    def answer(
        self,
        question: str,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> str:
        context, _ = self.retrieve(
            question,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )
        if not context:
            logger.warning("no relevant context found to answer the question.")
            return "no relevant context found to answer the question."

        answer = self.llm.answer(context, question)
        return answer

    def get_document_points(self, document_name: str) -> List[dict]:
        """특정 문서의 모든 포인트를 반환"""
        return self.retriever.manager.get_points_by_document_name(
            self.retriever.collection_name, document_name
        )

    def add_document(self, text: str, document_name: Optional[str] = None):
        tree = self.tree_builder.build_from_text(text, use_multithreading=True)
        self.retriever.build_from_tree(tree, document_name=document_name)

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
        return self.retriever.manager.delete_points_by_document_name(
            self.retriever.collection_name, document_name
        )

    def list_documents(self) -> List[str]:
        """저장된 모든 문서명 목록을 반환"""
        # 모든 포인트에서 document_name 필드 수집
        client = self.retriever.manager.get_client()
        scroll_result = client.scroll(
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
            client = self.retriever.manager.get_client()
            responses = client.get_collections()
            return [collection.name for collection in responses.collections]
        except Exception as e:
            logger.error(f"failed to get collections: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """특정 컬렉션을 삭제"""
        try:
            client = self.retriever.manager.get_client()
            client.delete_collection(collection_name)
            logger.info(f"collection '{collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(
                f"failed to delete collection '{collection_name}': {e}"
            )
            return False
