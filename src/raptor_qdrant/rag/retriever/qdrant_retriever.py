import logging
import uuid
from typing import Any, cast

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import (
    BaseRetriever as LlamaBaseRetriever,
)
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from raptor_qdrant.core.config import settings
from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant.rag.builder.models.structure import Tree
from raptor_qdrant.rag.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_K,
)
from raptor_qdrant.rag.embedding import (
    BaseEmbeddingModel,
    KoreanEmbeddingModel,
)
from raptor_qdrant.rag.utils import TOKEN_COUNT_KEY, resolve_token_count

from .base_retriever import BaseRetriever
from .context_window import ContextWindow, assemble_context

logger = logging.getLogger(__name__)

TEXT_PAYLOAD_FIELD = "text"


class QdrantRetrieverConfig:
    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        embedding_model: BaseEmbeddingModel | None = None,
        top_k: int = DEFAULT_TOP_K,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
    ):
        """Retriever 설정 객체

        Args:
            max_tokens (int, optional): 검색 컨텍스트의 최대 토큰 수
            embedding_model (Optional[BaseEmbeddingModel], optional): 임베딩 모델
            top_k (int, optional): DB에서 검색할 가장 유사한 문서 개수
            collection_name (str, optional): Qdrant 컬렉션 이름
            hybrid_alpha (float, optional): 하이브리드 검색 가중치 (0: 키워드, 1: 벡터)
        """
        self._validate_parameters(
            max_tokens, top_k, hybrid_alpha, embedding_model
        )

        self.top_k = top_k
        self.max_tokens = max_tokens
        self.embedding_model = embedding_model or KoreanEmbeddingModel()
        self.embedding_model_string = settings.EMBEDDING_MODEL_STRING
        self.collection_name = collection_name
        # 0에 가까울 수록 텍스트 유사도 기반 검색, 1에 가까울 수록 의미 기반 검색
        self.hybrid_alpha = hybrid_alpha
        self.vector_size = self.embedding_model.embedding_dimension

    def _validate_parameters(
        self,
        max_tokens: int,
        top_k: int,
        hybrid_alpha: float,
        embedding_model: BaseEmbeddingModel | None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )

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
                self._llama_embed_model = HuggingFaceEmbedding(
                    model_name=model_id
                )
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

            self.retriever = self.index.as_retriever(
                similarity_top_k=self.config.top_k,
                vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                alpha=self.config.hybrid_alpha,
            )
            logger.debug("retriever initialized successfully")
        except Exception as e:
            logger.error(f"failed to initialize retriever: {e}")
            raise

    def _create_text_nodes(
        self, tree: Tree, document_name: str | None
    ) -> list[TextNode]:
        text_nodes = []
        for node in tree.all_nodes.values():
            metadata: dict[str, Any] = {
                "layer": tree.get_node_layer(node.index),
                "node_index": node.index,
            }

            if document_name:
                metadata["document_name"] = document_name

            if node.metadata:
                metadata.update(node.metadata)

            metadata[TOKEN_COUNT_KEY] = resolve_token_count(
                metadata, node.text
            )

            text_nodes.append(
                TextNode(
                    text=node.text,
                    id_=str(uuid.uuid4()),
                    embedding=node.embeddings[
                        self.config.embedding_model_string
                    ],
                    metadata=metadata,
                )
            )

        logger.debug(f"created {len(text_nodes)} text nodes")
        return text_nodes

    def _should_include_node(
        self,
        node: NodeWithScore,
        collapse_tree: bool,
        start_layer: int | None,
    ) -> bool:
        if collapse_tree or start_layer is None:
            return True

        return node.metadata.get("layer") == start_layer

    def _create_keyword_search_index(self) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=TEXT_PAYLOAD_FIELD,
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    lowercase=True,
                ),
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug("full-text index already exists for 'text' field")
            else:
                logger.warning(f"failed to create text index: {e}")

    def build_from_tree(
        self,
        tree: Tree,
        document_name: str | None = None,
        recreate_collection: bool = False,
    ) -> int:
        """트리를 적재하고 적재한 노드 수를 반환한다."""
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

            self._setup_llama_embedding()

            text_nodes = self._create_text_nodes(tree, document_name)
            self._get_vector_store().add(cast(list[BaseNode], text_nodes))
            self.index = VectorStoreIndex.from_vector_store(
                self._get_vector_store()
            )

            self._create_keyword_search_index()
            self._initialize_retriever()

            logger.info("tree indexing completed successfully")
            return len(text_nodes)
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
            if self.retriever is None:
                self._initialize_retriever()
            assert self.retriever is not None

            retrieved_nodes = self.retriever.retrieve(query)
            eligible_nodes = [
                node
                for node in retrieved_nodes
                if self._should_include_node(node, collapse_tree, start_layer)
            ]

            window = assemble_context(eligible_nodes, self.config.max_tokens)
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
