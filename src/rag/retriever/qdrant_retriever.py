import uuid
import logging
from typing import Optional, List, Tuple, Dict, Any

from qdrant_client import QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .base_retriever import BaseRetriever
from src.database.qdrant_manager import QdrantManager
from src.rag.embedding import BaseEmbeddingModel, KoreanEmbeddingModel
from src.rag.builder.models.structure import Tree
from src.rag.utils import TOKEN_COUNT_KEY, resolve_token_count
from src.rag.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_K,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_BATCH_SIZE,
)
from src.core.config import settings

logger = logging.getLogger(__name__)

TEXT_PAYLOAD_FIELD = "text"


class QdrantRetrieverConfig:
    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        embedding_model: Optional[BaseEmbeddingModel] = None,
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
        embedding_model: Optional[BaseEmbeddingModel],
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
    def vector_store_config(self) -> Dict[str, Any]:
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
        self._llama_embed_model: Optional[HuggingFaceEmbedding] = None
        self._forget_collection_handles()

    def _forget_collection_handles(self) -> None:
        self.vector_store: Optional[QdrantVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        self.retriever = None

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
        self, tree: Tree, document_name: Optional[str]
    ) -> List[TextNode]:
        text_nodes = []
        for node in tree.all_nodes.values():
            metadata = {
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
        start_layer: Optional[int],
    ) -> bool:
        if collapse_tree or start_layer is None:
            return True

        return node.metadata.get('layer') == start_layer

    def _create_keyword_search_index(self) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=TEXT_PAYLOAD_FIELD,
                field_schema=models.TextIndexParams(
                    type="text",
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
        document_name: Optional[str] = None,
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
            self._get_vector_store().add(text_nodes)
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
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        try:
            if self.retriever is None:
                self._initialize_retriever()

            retrieved_nodes = self.retriever.retrieve(query)

            chunks: List[str] = []
            total_tokens = 0
            tree_layer: List[Dict[str, Any]] = []
            skipped: List[Tuple[Any, int]] = []

            for node in retrieved_nodes:
                if not self._should_include_node(
                    node, collapse_tree, start_layer
                ):
                    continue

                tokens = resolve_token_count(node.metadata, node.text)

                if total_tokens + tokens > self.config.max_tokens:
                    skipped.append((node.metadata.get('node_index'), tokens))
                    continue

                chunks.append(node.text)
                total_tokens += tokens
                tree_layer.append(
                    {
                        "node_index": node.metadata.get('node_index'),
                        "layer_number": node.metadata.get('layer'),
                        "chunked_by": node.metadata.get('chunked_by'),
                        "token_count": tokens,
                        "score": getattr(node, 'score', 0.0) or 0.0,
                    }
                )

            if skipped:
                logger.info(
                    f"skipped {len(skipped)} node(s) that did not fit in the "
                    f"remaining context budget "
                    f"({total_tokens}/{self.config.max_tokens} tokens used): "
                    f"{skipped}"
                )
            if retrieved_nodes and not chunks:
                logger.warning(
                    f"all {len(retrieved_nodes)} retrieved nodes were dropped. "
                    f"every node is larger than max_tokens "
                    f"({self.config.max_tokens}) — raise it or chunk smaller"
                )

            logger.info(
                f"retrieved {len(tree_layer)} chunks with {total_tokens} tokens"
            )
            return "\n\n".join(chunks), tree_layer

        except Exception as e:
            logger.error(f"failed to retrieve context: {e}")
            raise
