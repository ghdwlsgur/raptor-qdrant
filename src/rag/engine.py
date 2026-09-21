import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import settings
from src.database.qdrant_manager import QdrantManager
from src.rag.summarizer import BaseSummarizationModel, LLMSummarizer
from .builder.cluster.cluster_builder import (
    ClusterTreeBuilder,
    ClusterTreeConfig,
)
from .embedding import BaseEmbeddingModel, KoreanEmbeddingModel
from .llm import BaseChatbotModel, create_chatbot
from .retriever.qdrant_retriever import (
    QdrantRetriever,
    QdrantRetrieverConfig,
)

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "no relevant context found to answer the question."


@dataclass
class QueryResult:
    question: str
    answer: str
    context: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return bool(self.context)


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
        self.llm = llm or create_chatbot()
        self.summarization_model = summarization_model or LLMSummarizer(
            self.llm
        )

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
        self.llm = config.llm
        self.retriever = QdrantRetriever(config.retriever_config)
        self.tree_builder = ClusterTreeBuilder(config.tree_builder_config)

    @property
    def collection_name(self) -> str:
        return self.retriever.collection_name

    @property
    def manager(self) -> QdrantManager:
        return self.retriever.manager

    def retrieve(
        self,
        query: str,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        return self.retriever.retrieve(
            query,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )

    def query(
        self,
        question: str,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> QueryResult:
        context, chunks = self.retrieve(
            question,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )

        if not context:
            logger.warning(NO_CONTEXT_MESSAGE)
            return QueryResult(
                question=question,
                answer=NO_CONTEXT_MESSAGE,
                context="",
                chunks=[],
            )

        return QueryResult(
            question=question,
            answer=self.llm.answer(context, question),
            context=context,
            chunks=chunks,
        )

    def answer(
        self,
        question: str,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> str:
        return self.query(
            question,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        ).answer

    def add_document(
        self,
        text: str,
        document_name: Optional[str] = None,
        recreate_collection: bool = False,
    ) -> int:
        """문서를 인덱싱하고 적재한 노드 수를 반환한다."""
        if document_name and not recreate_collection:
            self._reject_if_already_indexed(document_name)

        logger.info("building raptor tree from document...")
        tree = self.tree_builder.build_from_text(text, use_multithreading=True)

        return self.retriever.build_from_tree(
            tree,
            document_name=document_name,
            recreate_collection=recreate_collection,
        )

    def update_document(self, text: str, document_name: str) -> int:
        """기존 문서를 교체하고 새로 적재한 노드 수를 반환한다."""
        if not document_name:
            raise ValueError("document_name must be provided and non-empty")

        deleted = self.delete_document(document_name)
        if deleted:
            logger.info(
                f"deleted {deleted} existing points for '{document_name}'"
            )

        indexed = self.add_document(text, document_name=document_name)
        logger.info(f"document '{document_name}' update complete.")
        return indexed

    def delete_document(self, document_name: str) -> int:
        """문서에 속한 포인트를 모두 지우고 지운 개수를 반환한다."""
        return self.manager.delete_points_by_document_name(
            self.collection_name, document_name
        )

    def get_document_points(self, document_name: str) -> List[dict]:
        return self.manager.get_points_by_document_name(
            self.collection_name, document_name
        )

    def list_documents(self) -> List[str]:
        return self.manager.list_document_names(self.collection_name)

    def list_collections(self) -> List[str]:
        try:
            return self.manager.list_collections()
        except Exception as e:
            logger.error(f"failed to get collections: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        try:
            return self.manager.drop_collection(collection_name)
        except Exception as e:
            logger.error(
                f"failed to delete collection '{collection_name}': {e}"
            )
            return False

    def _reject_if_already_indexed(self, document_name: str) -> None:
        existing = self.manager.count_points_by_document_name(
            self.collection_name, document_name
        )
        if existing:
            raise ValueError(
                f"document '{document_name}' already has {existing} points in "
                f"'{self.collection_name}'. use update_document() to replace it, "
                "or pick a different document_name"
            )
