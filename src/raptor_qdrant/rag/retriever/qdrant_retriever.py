import logging
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, cast

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import (
    BaseRetriever as LlamaBaseRetriever,
)
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from raptor_qdrant.core.config import settings
from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant.rag.builder.models.structure import Node, Tree
from raptor_qdrant.rag.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SUMMARY_QUOTA,
    DEFAULT_TOP_K,
    SOURCE_KEY,
    STALE_KEY,
    TREE_GENERATION_KEY,
)
from raptor_qdrant.rag.embedding import (
    BaseEmbeddingModel,
    KoreanEmbeddingModel,
    llama_embedding,
)
from raptor_qdrant.rag.utils import TOKEN_COUNT_KEY, resolve_token_count

from .base_retriever import BaseRetriever
from .context_window import ContextWindow, assemble_context
from .layer_mix import balance_layers

logger = logging.getLogger(__name__)

LAYER_KEY = "layer"

Payload = Mapping[str, Any]


class QdrantRetrieverConfig:
    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        embedding_model: BaseEmbeddingModel | None = None,
        top_k: int = DEFAULT_TOP_K,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
        summary_quota: int = DEFAULT_SUMMARY_QUOTA,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ):
        """Retriever 설정 객체

        Args:
            max_tokens (int, optional): 검색 컨텍스트의 최대 토큰 수
            embedding_model (Optional[BaseEmbeddingModel], optional): 임베딩 모델
            top_k (int, optional): DB에서 검색할 가장 유사한 문서 개수
            collection_name (str, optional): Qdrant 컬렉션 이름
            hybrid_alpha (float, optional): 하이브리드 검색 가중치 (0: 키워드, 1: 벡터)
            summary_quota (int, optional): 상위 top_k 안에 남겨 둘 요약 노드 자리
            candidate_multiplier (int, optional): top_k 의 몇 배를 후보로 받아올지
        """
        self._validate_parameters(
            max_tokens, top_k, hybrid_alpha, embedding_model, summary_quota
        )

        self.top_k = top_k
        self.max_tokens = max_tokens
        self.embedding_model = embedding_model or KoreanEmbeddingModel()
        self.embedding_model_string = settings.EMBEDDING_MODEL_STRING
        self.collection_name = collection_name
        # 0에 가까울 수록 텍스트 유사도 기반 검색, 1에 가까울 수록 의미 기반 검색
        self.hybrid_alpha = hybrid_alpha
        self.summary_quota = min(summary_quota, top_k)
        self.candidate_multiplier = max(1, candidate_multiplier)

    def _validate_parameters(
        self,
        max_tokens: int,
        top_k: int,
        hybrid_alpha: float,
        embedding_model: BaseEmbeddingModel | None,
        summary_quota: int = 0,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if summary_quota < 0:
            raise ValueError("summary_quota must not be negative")
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )

    @property
    def candidate_pool(self) -> int:
        """레이어를 섞으려면 top_k 보다 넉넉히 받아 와야 한다."""
        return self.top_k * self.candidate_multiplier

    @property
    def vector_size(self) -> int:
        return self.embedding_model.embedding_dimension

    @property
    def vector_store_config(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "enable_hybrid": True,
            "batch_size": DEFAULT_BATCH_SIZE,
        }


class QdrantRetriever(BaseRetriever):
    def __init__(self, config: QdrantRetrieverConfig):
        self.config = config
        self.embedding_model = config.embedding_model
        self.manager = QdrantManager()
        self.client: QdrantClient = self.manager.get_client()
        self._llama_embed_model: HuggingFaceEmbedding | None = None
        self._forget_collection_handles()

    def _forget_collection_handles(self) -> None:
        self.vector_store: QdrantVectorStore | None = None
        self.index: VectorStoreIndex | None = None
        self.retriever: LlamaBaseRetriever | None = None

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    def _setup_llama_embedding(self) -> None:
        if self._llama_embed_model is None:
            try:
                model_id = self.embedding_model.model_name
                self._llama_embed_model = llama_embedding(model_id)
                logger.debug(f"embedding model loaded: {model_id}")
            except Exception as e:
                logger.error(
                    f"failed to setup LlamaIndex embedding model: {e}"
                )
                raise

        Settings.embed_model = self._llama_embed_model

    def _get_vector_store(self) -> QdrantVectorStore:
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore(
                client=self.client, **self.config.vector_store_config
            )
        return self.vector_store

    def _initialize_retriever(self) -> None:
        try:
            self._setup_llama_embedding()

            if self.index is None:
                if not self.manager.collection_exists(self.collection_name):
                    raise ValueError(
                        f"collection '{self.collection_name}' does not exist. "
                        "index a document first with add_document()"
                    )
                self.index = VectorStoreIndex.from_vector_store(
                    self._get_vector_store()
                )
                logger.debug("loaded existing collection")

            self.retriever = self._build_retriever()
            logger.debug("retriever initialized successfully")
        except Exception as e:
            logger.error(f"failed to initialize retriever: {e}")
            raise

    def _build_retriever(
        self, filters: MetadataFilters | None = None
    ) -> LlamaBaseRetriever:
        assert self.index is not None
        return self.index.as_retriever(
            similarity_top_k=self.config.candidate_pool,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=self.config.hybrid_alpha,
            filters=filters,
        )

    @staticmethod
    def _layer_filter(layer: int) -> MetadataFilters:
        return MetadataFilters(
            filters=[
                MetadataFilter(
                    key=LAYER_KEY, value=layer, operator=FilterOperator.EQ
                )
            ]
        )

    def _candidates(
        self, query: str, collapse_tree: bool, start_layer: int | None
    ) -> list[NodeWithScore]:
        """후보를 받아온다. 레이어 지정은 Qdrant 에 필터로 내려보낸다.

        받아온 뒤에 레이어로 거르면 그 레이어의 좋은 후보는 애초에 후보에
        없다. 상위 마흔 개가 전부 잎이면 start_layer=2 는 빈손으로 끝난다.
        """
        if collapse_tree or start_layer is None:
            if self.retriever is None:
                self._initialize_retriever()
            assert self.retriever is not None
            return self.retriever.retrieve(query)

        if self.index is None:
            self._initialize_retriever()
        return self._build_retriever(self._layer_filter(start_layer)).retrieve(
            query
        )

    @staticmethod
    def _live(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        return [node for node in nodes if not node.metadata.get(STALE_KEY)]

    def _text_node(
        self,
        node: Node,
        layer: int | None,
        document_name: str | None,
        extra_payload_by_source: Mapping[str, Payload] | None,
        common_payload: Payload | None,
    ) -> TextNode:
        metadata: dict[str, Any] = {"layer": layer, "node_index": node.index}

        if document_name:
            metadata[SOURCE_KEY] = document_name

        if node.metadata:
            metadata.update(node.metadata)

        metadata[TOKEN_COUNT_KEY] = resolve_token_count(metadata, node.text)

        extra = (extra_payload_by_source or {}).get(
            metadata.get(SOURCE_KEY, "")
        )
        if extra:
            metadata.update(extra)
        if common_payload:
            metadata.update(common_payload)

        return TextNode(
            text=node.text,
            id_=str(uuid.uuid4()),
            embedding=node.embeddings[self.config.embedding_model_string],
            metadata=metadata,
        )

    def add_nodes(
        self,
        nodes: Iterable[Node],
        layer: int,
        document_name: str | None = None,
        extra_payload_by_source: Mapping[str, Payload] | None = None,
        common_payload: Payload | None = None,
    ) -> int:
        """한 레이어의 노드를 바로 적재하고 적재한 수를 돌려준다.

        컬렉션이 없으면 첫 적재 때 벡터 스토어가 만든다. 검색 준비는
        finalize() 가 한다. 레이어마다 인덱스를 다시 여는 것은 낭비다.
        """
        self._setup_llama_embedding()

        text_nodes = [
            self._text_node(
                node,
                layer,
                document_name,
                extra_payload_by_source,
                common_payload,
            )
            for node in nodes
        ]
        if text_nodes:
            self._get_vector_store().add(cast(list[BaseNode], text_nodes))

        logger.debug(f"layer {layer}: added {len(text_nodes)} points")
        return len(text_nodes)

    def finalize(self) -> None:
        """적재를 마친 컬렉션을 검색 가능한 상태로 연다."""
        self.index = VectorStoreIndex.from_vector_store(
            self._get_vector_store()
        )
        self._initialize_retriever()

    def retire_generations_except(self, generation: str) -> int:
        """이 세대가 아닌 포인트를 모두 지운다. 이전 트리와 세대 없는 잎이 대상이다."""
        return self.manager.delete_points_where(
            self.collection_name, TREE_GENERATION_KEY, generation, negate=True
        )

    def drop_generation(self, generation: str) -> int:
        """특정 세대의 포인트만 지운다. 끝나지 않은 빌드를 치울 때 쓴다."""
        return self.manager.delete_points_where(
            self.collection_name, TREE_GENERATION_KEY, generation
        )

    def build_from_tree(
        self,
        tree: Tree,
        document_name: str | None = None,
        recreate_collection: bool = False,
        extra_payload_by_source: Mapping[str, Payload] | None = None,
        common_payload: Payload | None = None,
        replace_sources: Iterable[str] = (),
    ) -> int:
        """완성된 트리를 레이어 순서로 적재하고 적재한 노드 수를 반환한다."""
        logger.info(
            f"building index from tree with {len(tree.all_nodes)} nodes"
        )

        try:
            if recreate_collection:
                logger.warning(
                    f"dropping collection '{self.collection_name}' before indexing"
                )
                self.manager.drop_collection(self.collection_name)
                self._forget_collection_handles()

            for source in replace_sources:
                self.manager.delete_points_by_document_name(
                    self.collection_name, source
                )

            stored = 0
            for layer, nodes in sorted(tree.layer_to_nodes.items()):
                stored += self.add_nodes(
                    nodes,
                    layer,
                    document_name,
                    extra_payload_by_source,
                    common_payload,
                )

            self.finalize()
            logger.info("tree indexing completed successfully")
            return stored
        except Exception as e:
            logger.error(f"failed to build index from tree: {e}")
            raise

    def retrieve(
        self,
        query: str,
        collapse_tree: bool = True,
        start_layer: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        try:
            retrieved_nodes = self._candidates(
                query, collapse_tree, start_layer
            )
            candidates = self._live(retrieved_nodes)
            selected = (
                balance_layers(
                    candidates,
                    self.config.top_k,
                    self.config.summary_quota,
                )
                if collapse_tree
                else candidates[: self.config.top_k]
            )

            window = assemble_context(selected, self.config.max_tokens)
            self._log_window(window, len(retrieved_nodes))

            return window.text, window.chunk_info

        except Exception as e:
            logger.error(f"failed to retrieve context: {e}")
            raise

    def _log_window(self, window: ContextWindow, retrieved_count: int) -> None:
        if window.skipped:
            logger.info(
                f"skipped {len(window.skipped)} node(s) that did not fit in the "
                f"remaining context budget "
                f"({window.total_tokens}/{self.config.max_tokens} tokens used): "
                f"{window.skipped}"
            )

        if retrieved_count and window.is_empty:
            logger.warning(
                f"all {retrieved_count} retrieved nodes were dropped. every node "
                f"is larger than max_tokens ({self.config.max_tokens}). raise it "
                "or chunk smaller"
            )

        logger.info(
            f"retrieved {len(window.chunk_info)} chunks "
            f"with {window.total_tokens} tokens"
        )
