import json
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
    SOURCE_SET_KEY,
    STALE_KEY,
    TREE_DRIFT_WARN_RATIO,
    TREE_GENERATION_KEY,
)
from raptor_qdrant.rag.summarizer import BaseSummarizationModel, LLMSummarizer

from .builder.cluster.cluster_builder import (
    ClusterTreeBuilder,
    ClusterTreeConfig,
)
from .builder.models.structure import Node
from .embedding import BaseEmbeddingModel, KoreanEmbeddingModel
from .llm import BaseChatbotModel, LazyChatbot
from .retriever.context_window import ordered_sources
from .retriever.qdrant_retriever import (
    QdrantRetriever,
    QdrantRetrieverConfig,
)

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "no relevant context found to answer the question."
LEAF_LAYER = 0
# llama-index 는 본문을 평평한 text 필드가 아니라 이 JSON 안에 넣는다
NODE_CONTENT_KEY = "_node_content"


def _stored_text(payload: dict[str, Any]) -> str:
    raw = payload.get(NODE_CONTENT_KEY)
    if not isinstance(raw, str):
        return ""
    try:
        return str(json.loads(raw).get("text", ""))
    except (ValueError, TypeError):
        return ""


@dataclass(frozen=True)
class IndexHealth:
    """요약 레이어가 잎에 비해 얼마나 낡았는지, 빌드가 몇 세대 섞여 있는지."""

    leaf_nodes: int
    summary_nodes: int
    leaves_outside_tree: int
    generations: int = 0
    stale_summaries: int = 0

    @property
    def drift(self) -> float:
        if not self.leaf_nodes:
            return 0.0
        return self.leaves_outside_tree / self.leaf_nodes

    @property
    def needs_rebuild(self) -> bool:
        return self.drift > TREE_DRIFT_WARN_RATIO

    @property
    def has_mixed_generations(self) -> bool:
        """완성된 트리는 세대가 하나다. 둘 이상이면 끝나지 않은 빌드가 남아 있다."""
        return self.generations > 1

    @property
    def summary(self) -> str:
        text = (
            f"leaves {self.leaf_nodes}, summaries {self.summary_nodes}, "
            f"outside the current tree {self.leaves_outside_tree} "
            f"({self.drift:.0%})"
        )
        if self.has_mixed_generations:
            text += f", {self.generations} tree generations mixed"
        if self.stale_summaries:
            text += f", {self.stale_summaries} summaries retired"
        return text


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
        return ordered_sources(self.chunks)


class EngineConfig:
    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbeddingModel | None = None,
        summarization_model: BaseSummarizationModel | None = None,
        llm: BaseChatbotModel | None = None,
        provider: str | None = None,
    ):
        if not collection_name:
            raise ValueError("collection_name must be provided and non-empty")

        self.provider = provider
        self.embedding_model = embedding_model or KoreanEmbeddingModel()
        # 잎만 갱신하거나 현황만 보는 명령은 LLM 을 한 번도 부르지 않는다
        self.llm = llm or LazyChatbot(provider)
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
            provider=provider,
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
        note_hashes: Mapping[str, str] | None = None,
        generation: str | None = None,
    ) -> int:
        """여러 문서 위에 트리 하나를 올려 적재하고 노드 수를 반환한다.

        문서마다 따로 트리를 세우면 문서 하나가 청크 열 몇 개뿐이라 요약 레이어가
        만들어지지 않는다. 코퍼스 전체를 한 트리로 묶어야 상위 요약이 생긴다.

        적재는 레이어가 완성될 때마다 바로 한다. 잎은 임베딩이 끝나는 즉시,
        요약 레이어는 하나씩 끝날 때마다 Qdrant 에 들어간다. 트리를 다 세운
        뒤 한 번에 쓰면 중간에 Qdrant 가 죽거나 프로세스가 끊길 때 몇 시간이
        통째로 날아간다. 새 노드는 모두 같은 tree_generation 을 달고, 끝나면
        그 세대가 아닌 포인트(이전 트리, 그 사이 watch 가 넣은 잎)를 지운다.
        """
        generation = generation or uuid.uuid4().hex
        extra = self._hash_payload(note_hashes)
        common = {TREE_GENERATION_KEY: generation}
        stored = 0

        def persist(layer: int, nodes: list[Node]) -> None:
            nonlocal stored
            stored += self.retriever.add_nodes(
                nodes,
                layer,
                extra_payload_by_source=extra,
                common_payload=common,
            )
            logger.info(
                f"layer {layer}: stored {len(nodes)} nodes "
                f"({stored} so far, generation {generation[:8]})"
            )

        logger.info(
            f"building raptor tree from {len(documents)} documents "
            f"(generation {generation[:8]})..."
        )
        self.tree_builder.build_from_documents(
            documents, use_multithreading=True, on_layer_built=persist
        )

        retired = self.retriever.retire_generations_except(generation)
        if retired:
            logger.info(f"retired {retired} points from earlier builds")

        self.retriever.finalize()
        return stored

    def discard_generation(self, generation: str) -> int:
        """끝나지 않은 빌드가 남긴 포인트를 지운다."""
        return self.retriever.drop_generation(generation)

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
        """삭제된 노트의 포인트를 제거하고, 그 노트를 덮던 요약을 빼둔다."""
        removed = 0
        retired = 0
        for path in paths:
            removed += self.delete_document(path)
            retired += self.retire_summaries_of(path)

        if retired:
            logger.info(
                f"{retired} summary nodes covered deleted notes and were "
                "taken out of search until the next rebuild"
            )
        return removed

    def retire_summaries_of(self, path: str) -> int:
        """지워진 노트를 덮던 요약을 검색에서 빼둔다.

        요약은 원문이 사라져도 그 내용을 그대로 물고 있다. 낡은 것과 없는
        것은 다르다. 재구축 전까지 이 요약이 없는 노트를 근거로 내놓지
        않도록 표시해 둔다.
        """
        return self.manager.set_payload_where(
            self.collection_name, SOURCE_SET_KEY, path, {STALE_KEY: True}
        )

    def summaries_covering_many(
        self, minimum: int = 1
    ) -> list[tuple[str, list[str]]]:
        """요약 노드의 본문과 그것이 덮는 노트 목록.

        평가셋이 트리를 겨냥한 질문을 만들 때 쓴다. 요약이 묶어 둔 노트
        집합이 곧 "이 질문에 나와야 할 것들"의 정답지가 된다.
        """
        found: list[tuple[str, list[str]]] = []

        for payload in self.manager.iter_payloads(
            self.collection_name,
            fields=("layer", NODE_CONTENT_KEY, SOURCE_SET_KEY, STALE_KEY),
        ):
            if not payload.get("layer") or payload.get(STALE_KEY):
                continue

            sources = payload.get(SOURCE_SET_KEY) or []
            text = _stored_text(payload)
            if text and len(sources) >= minimum:
                found.append((text, list(sources)))

        return found

    def drift(self) -> IndexHealth:
        """요약 레이어가 현재 잎 집합을 얼마나 반영하는지 계산한다."""
        leaves = summaries = orphan_leaves = stale = 0
        generations: set[str] = set()

        for payload in self.manager.iter_payloads(
            self.collection_name,
            fields=("layer", TREE_GENERATION_KEY, STALE_KEY),
        ):
            generation = payload.get(TREE_GENERATION_KEY)
            if generation is not None:
                generations.add(generation)
            if payload.get("layer") == LEAF_LAYER:
                leaves += 1
                if generation is None:
                    orphan_leaves += 1
            else:
                summaries += 1
                if payload.get(STALE_KEY):
                    stale += 1

        return IndexHealth(
            leaf_nodes=leaves,
            summary_nodes=summaries,
            leaves_outside_tree=orphan_leaves,
            generations=len(generations),
            stale_summaries=stale,
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
        for point in self.manager.iter_payloads(
            self.collection_name, fields=(SOURCE_KEY, CONTENT_HASH_KEY)
        ):
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
