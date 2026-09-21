import logging
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from raptor_qdrant.core.config import settings
from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant.rag.constants import (
    CONTENT_HASH_KEY,
    SOURCE_KEY,
    TREE_DRIFT_WARN_RATIO,
    TREE_GENERATION_KEY,
)
from raptor_qdrant.rag.summarizer import BaseSummarizationModel, LLMSummarizer

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
LEAF_LAYER = 0


@dataclass(frozen=True)
class IndexHealth:
    """요약 레이어가 잎에 비해 얼마나 낡았는지."""

    leaf_nodes: int
    summary_nodes: int
    leaves_outside_tree: int

    @property
    def drift(self) -> float:
        if not self.leaf_nodes:
            return 0.0
        return self.leaves_outside_tree / self.leaf_nodes

    @property
    def needs_rebuild(self) -> bool:
        return self.drift > TREE_DRIFT_WARN_RATIO

    @property
    def summary(self) -> str:
        return (
            f"leaves {self.leaf_nodes}, summaries {self.summary_nodes}, "
            f"outside the current tree {self.leaves_outside_tree} "
            f"({self.drift:.0%})"
        )


@dataclass
class QueryResult:
    question: str
    answer: str
    context: str
    chunks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return bool(self.context)

    @property
    def sources(self) -> list[str]:
        ordered: dict[str, None] = {}
        for chunk in self.chunks:
            for source in chunk.get("sources") or []:
                ordered.setdefault(source, None)
        return list(ordered)


class EngineConfig:
    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbeddingModel | None = None,
        summarization_model: BaseSummarizationModel | None = None,
        llm: BaseChatbotModel | None = None,
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
    def __init__(self, config: EngineConfig | None = None):
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
        start_layer: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        return self.retriever.retrieve(
            query,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )

    def query(
        self,
        question: str,
        collapse_tree: bool = True,
        start_layer: int | None = None,
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
        start_layer: int | None = None,
    ) -> str:
        return self.query(
            question,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        ).answer

    def add_document(
        self,
        text: str,
        document_name: str | None = None,
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

    def add_corpus(
        self,
        documents: Mapping[str, str],
        recreate_collection: bool = False,
        note_hashes: Mapping[str, str] | None = None,
    ) -> int:
        """여러 문서 위에 트리 하나를 올려 적재하고 노드 수를 반환한다.

        문서마다 따로 트리를 세우면 문서 하나가 청크 열 몇 개뿐이라 요약 레이어가
        만들어지지 않는다. 코퍼스 전체를 한 트리로 묶어야 상위 요약이 생긴다.
        """
        logger.info(f"building raptor tree from {len(documents)} documents...")
        tree = self.tree_builder.build_from_documents(
            documents, use_multithreading=True
        )

        return self.retriever.build_from_tree(
            tree,
            recreate_collection=recreate_collection,
            extra_payload_by_source=self._hash_payload(note_hashes),
            common_payload={TREE_GENERATION_KEY: uuid.uuid4().hex},
        )

    def upsert_notes(
        self,
        documents: Mapping[str, str],
        note_hashes: Mapping[str, str] | None = None,
    ) -> int:
        """바뀐 노트의 잎만 즉시 교체한다. 요약 레이어는 건드리지 않는다.

        트리 재구축은 LLM 호출이 붙어 느리다. 저장할 때마다 그걸 돌릴 수는
        없으므로 잎만 갈아끼운다. 트리 간선은 적재되지 않으므로 기존 요약
        노드가 깨지지는 않고, 내용만 그만큼 낡는다. drift() 로 추적한다.
        """
        if not documents:
            return 0

        tree = self.tree_builder.build_leaves_only(documents)

        return self.retriever.build_from_tree(
            tree,
            replace_sources=documents.keys(),
            extra_payload_by_source=self._hash_payload(note_hashes),
            common_payload={TREE_GENERATION_KEY: None},
        )

    def remove_notes(self, paths: Iterable[str]) -> int:
        """삭제된 노트의 포인트를 즉시 제거하고 제거한 개수를 반환한다."""
        return sum(self.delete_document(path) for path in paths)

    def drift(self) -> IndexHealth:
        """요약 레이어가 현재 잎 집합을 얼마나 반영하는지 계산한다."""
        leaves = summaries = orphan_leaves = 0
        generations: set[str] = set()

        for payload in self.manager.iter_payloads(self.collection_name):
            generation = payload.get(TREE_GENERATION_KEY)
            if payload.get("layer") == LEAF_LAYER:
                leaves += 1
                if generation is None:
                    orphan_leaves += 1
                else:
                    generations.add(generation)
            else:
                summaries += 1

        return IndexHealth(
            leaf_nodes=leaves,
            summary_nodes=summaries,
            leaves_outside_tree=orphan_leaves,
        )

    @staticmethod
    def _hash_payload(
        note_hashes: Mapping[str, str] | None,
    ) -> dict[str, dict[str, str]] | None:
        if not note_hashes:
            return None
        return {
            name: {CONTENT_HASH_KEY: digest}
            for name, digest in note_hashes.items()
        }

    def indexed_content_hashes(self) -> dict[str, str]:
        """적재된 문서별 내용 해시를 반환한다. 증분 판단의 기준이다."""
        hashes: dict[str, str] = {}
        for point in self.manager.iter_payloads(self.collection_name):
            name = point.get(SOURCE_KEY)
            digest = point.get(CONTENT_HASH_KEY)
            if name and digest:
                hashes.setdefault(name, digest)
        return hashes

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

    def get_document_points(self, document_name: str) -> list[dict]:
        return self.manager.get_points_by_document_name(
            self.collection_name, document_name
        )

    def list_documents(self) -> list[str]:
        return self.manager.list_document_names(self.collection_name)

    def list_collections(self) -> list[str]:
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
