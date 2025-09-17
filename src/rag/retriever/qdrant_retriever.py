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
from src.rag.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_K,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_BATCH_SIZE,
)
from src.core.config import settings

logger = logging.getLogger(__name__)


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
        self._validate_parameters(max_tokens, top_k, embedding_model)

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
        embedding_model: Optional[BaseEmbeddingModel],
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

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
    """Qdrant 벡터 데이터베이스를 사용하는 RAPTOR RAG 시스템의 Retriever 클래스"""

    def __init__(self, config: QdrantRetrieverConfig):
        self.config = config
        self.embedding_model = config.embedding_model
        self._initialize_qdrant_components()
        self._initialize_llama_index_components()

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    def _initialize_qdrant_components(self) -> None:
        self.manager = QdrantManager()
        self.client: QdrantClient = self.manager.get_client()
        self.manager.create_collection_if_not_exists(
            self.config.collection_name, self.config.vector_size
        )

    def _initialize_llama_index_components(self) -> None:
        self.vector_store = None
        self.index = None
        self.retriever = None

    def _initialize_retriever(self) -> None:
        """Initialize LlamaIndex Retriever."""
        try:
            self._setup_llama_embedding()

            if self.index is None:
                self._load_existing_collection()

            self.retriever = self.index.as_retriever(
                similarity_top_k=self.config.top_k,
                vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                alpha=self.config.hybrid_alpha,
            )
            logger.debug("retriever initialized successfully")
        except Exception as e:
            logger.error(f"failed to initialize retriever: {e}")
            raise

    def _load_existing_collection(self) -> None:
        """Qdrant에 이미 존재하는 컬렉션을 로드"""
        self.vector_store = QdrantVectorStore(
            client=self.client, **self.config.vector_store_config
        )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        logger.debug("loaded existing collection")

    def _setup_llama_embedding(self) -> None:
        """LlamaIndex의 임베딩 모델 등록"""
        try:
            model_id = self.embedding_model.model_name
            llama_embed_model = HuggingFaceEmbedding(model_name=model_id)
            Settings.embed_model = llama_embed_model
            logger.debug(f"embedding model set to: {model_id}")
        except Exception as e:
            logger.error(f"failed to setup LlamaIndex embedding model: {e}")
            raise

    def _create_text_nodes(
        self, all_nodes: List[Any], tree: Tree, document_name: Optional[str]
    ) -> List[TextNode]:
        text_nodes = []
        for node in all_nodes:
            layer = tree.get_node_layer(node.index)
            metadata = {
                "layer": layer,
                "node_index": node.index,
            }

            if document_name:
                metadata["document_name"] = document_name

            if hasattr(node, 'metadata') and node.metadata:
                metadata.update(node.metadata)

            text_node = TextNode(
                text=node.text,
                id_=str(uuid.uuid4()),
                embedding=node.embeddings[self.config.embedding_model_string],
                metadata=metadata,
            )
            text_nodes.append(text_node)

        logger.debug(f"created {len(text_nodes)} text nodes")
        return text_nodes

    def _should_include_node(
        self,
        node: NodeWithScore,
        collapse_tree: bool,
        start_layer: Optional[int],
    ) -> bool:
        """Check if node should be included based on layer filtering."""
        if collapse_tree or start_layer is None:
            return True

        node_layer = node.metadata.get('layer')
        return node_layer == start_layer

    def _create_text_index(self) -> None:
        """하이브리드 검색의 키워드 검색을 위해 'text' 필드에 대한 full-text 인덱스 생성"""
        try:
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                # 어떤 필드에 대해 텍스트 인덱스를 생성할지 지정
                field_name="text",
                # 어떤 규칙으로 색인을 생성할지 지정
                field_schema=models.TextIndexParams(
                    type="text",
                    # 다국어를 지원하는 토크나이저
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    # 검색 시 대소문자 구분하지 않도록 모두 소문자로 변환
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
        append_mode: bool = False,
    ) -> None:
        """RAPTOR Tree 객체로부터 Qdrant 컬렉션을 빌드"""
        all_nodes = list(tree.all_nodes.values())
        logger.info(f"building index from tree with {len(all_nodes)} nodes")

        try:
            if not append_mode:
                self.manager.recreate_collection(
                    self.config.collection_name, self.config.vector_size
                )

            self._setup_llama_embedding()

            text_nodes = self._create_text_nodes(all_nodes, tree, document_name)
            self.vector_store = QdrantVectorStore(
                client=self.client, **self.config.vector_store_config
            )
            self.vector_store.add(text_nodes)
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)

            self._create_text_index()
            self._initialize_retriever()

            logger.info("tree indexing completed successfully")
        except Exception as e:
            logger.error(f"failed to build index from tree: {e}")
            raise

    def retrieve(
        self,
        query: str,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        try:
            if self.retriever is None:
                self._initialize_retriever()

            retriever = self.retriever
            retrieved_nodes = retriever.retrieve(query)

            context = ""
            total_tokens = 0
            tree_layer = []

            for node in retrieved_nodes:
                if not self._should_include_node(
                    node, collapse_tree, start_layer
                ):
                    continue

                chunk = node.text
                tokens = node.metadata.get('token_count')

                if total_tokens + tokens <= self.config.max_tokens:
                    context += chunk + "\n\n"
                    total_tokens += tokens
                    tree_layer.append(
                        {
                            "node_index": node.metadata.get('node_index'),
                            "layer_number": node.metadata.get('layer'),
                            "chunked_by": node.metadata.get('chunked_by'),
                            "token_count": tokens,
                            "score": getattr(node, 'score', 0.0),
                        }
                    )
                else:
                    break
            logger.info(f"retrieved context with {total_tokens} tokens")
            return context.strip(), tree_layer

        except Exception as e:
            logger.error(f"failed to retrieve context: {e}")
            raise
