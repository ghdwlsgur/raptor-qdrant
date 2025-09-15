import uuid
import logging
from typing import Optional, List, Tuple

import tiktoken
from qdrant_client import QdrantClient, models
from tiktoken.core import Encoding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .base_retriever import BaseRetriever
from src.database.qdrant_manager import QdrantManager
from src.rag.embedding import BaseEmbeddingModel, KoreanEmbeddingModel
from src.rag.builder.models.structure import Tree

logger = logging.getLogger(__name__)


class QdrantRetrieverConfig:
    def __init__(
        self,
        max_tokens: int = 512,
        max_context_tokens: int = 3500,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        question_embedding_model: Optional[BaseEmbeddingModel] = None,
        top_k: int = 5,
        tokenizer: Optional[Encoding] = None,
        embedding_model_string: Optional[str] = None,
        collection_name: str = "default_collection",
        hybrid_alpha: float = 0.7,
    ):
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
        if question_embedding_model is not None and not isinstance(
            question_embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "question_embedding_model must be an instance of BaseEmbeddingModel"
            )

        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.embedding_model = embedding_model or KoreanEmbeddingModel()
        self.question_embedding_model = (
            question_embedding_model or self.embedding_model
        )
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self.embedding_model_string = embedding_model_string or "SBert"
        self.collection_name = collection_name
        self.hybrid_alpha = hybrid_alpha  # 0: 키워드 검색, 1: 벡터 검색

        self.vector_size = len(
            self.embedding_model.create_embedding("test vector size")
        )
        logger.info(
            f"Vector size for collection '{self.collection_name}' is set to {self.vector_size}"
        )


class QdrantRetriever(BaseRetriever):
    def __init__(self, config: QdrantRetrieverConfig):
        self.config = config
        self.tokenizer = config.tokenizer
        self.question_embedding_model = config.question_embedding_model
        self.embedding_model = config.embedding_model

        self.manager = QdrantManager()
        self.client: QdrantClient = self.manager.get_client()
        self.collection_name = config.collection_name
        self.vector_size = config.vector_size

        # LlamaIndex
        self.vector_store = None
        self.index = None
        self.retriever = None

        logger.info(
            f"initialized qdrant retriever for collection: '{self.collection_name}'"
        )
        self.manager.create_collection_if_not_exists(
            self.collection_name, self.vector_size
        )

    def build_from_tree(
        self,
        tree: Tree,
        document_name: Optional[str] = None,
        append_mode: bool = False,
    ):
        """
        RAPTOR Tree 객체로부터 Qdrant 컬렉션을 구축합니다.
        append_mode=False: 기존 컬렉션을 삭제하고 새로 만듭니다.
        append_mode=True: 기존 컬렉션에 추가합니다.
        """
        all_nodes = list(tree.all_nodes.values())
        logger.info(
            f"Building index from a tree with {len(all_nodes)} total nodes."
        )

        if not append_mode:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
                sparse_vectors_config={
                    "text-sparse-new": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )

        # LlamaIndex Settings에 임베딩 모델 설정
        model_id = self.embedding_model.model_name
        llama_embed_model = HuggingFaceEmbedding(model_name=model_id)
        Settings.embed_model = llama_embed_model

        # LlamaIndex TextNode 객체들 생성
        text_nodes = []
        for node in all_nodes:
            layer = tree.get_node_layer(node.index)
            metadata = {
                "layer": layer,
                "node_index": node.index,
            }

            # document_name이 제공된 경우 메타데이터에 추가
            if document_name:
                metadata["document_name"] = document_name

            text_node = TextNode(
                text=node.text,
                id_=str(uuid.uuid4()),
                embedding=node.embeddings[self.config.embedding_model_string],
                metadata=metadata,
            )
            text_nodes.append(text_node)

        # QdrantVectorStore 생성 (하이브리드 검색 지원)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            enable_hybrid=True,  # 하이브리드 검색 활성화
            batch_size=64,
        )

        # VectorStore에 직접 노드를 추가
        self.vector_store.add(text_nodes)

        # 데이터가 저장된 vector_store로부터 Index를 로드
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        logger.info("tree indexing complete")

        # Full-Text Index 생성 (하이브리드 검색을 위해)
        self._create_text_index()

        # Retriever 초기화
        self._initialize_retriever()

    def _initialize_retriever(self):
        """LlamaIndex Retriever 초기화"""
        # LlamaIndex Settings에 임베딩 모델 설정
        model_id = self.embedding_model.model_name
        llama_embed_model = HuggingFaceEmbedding(model_name=model_id)
        Settings.embed_model = llama_embed_model

        if self.index is None:
            # 기존 컬렉션이 있으면 로드
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                enable_hybrid=True,  # 하이브리드 검색 활성화
                batch_size=64,
            )
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        # 기본적으로 하이브리드 검색 사용
        self.retriever = self.index.as_retriever(
            similarity_top_k=self.config.top_k,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=self.config.hybrid_alpha,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[dict]]:
        """
        쿼리를 기반으로 Qdrant에서 관련 컨텍스트를 검색합니다.
        """
        # 파라미터가 제공되지 않으면 config의 기본값을 사용합니다.
        top_k = top_k if top_k is not None else self.config.top_k
        max_tokens = (
            max_tokens
            if max_tokens is not None
            else self.config.max_context_tokens
        )

        # Retriever가 초기화되지 않은 경우 초기화
        if self.retriever is None:
            self._initialize_retriever()

        # top_k가 기본값과 다르면 새로운 retriever 생성
        if top_k != self.config.top_k:
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                alpha=self.config.hybrid_alpha,
            )
        else:
            retriever = self.retriever

        # LlamaIndex retriever로 검색
        retrieved_nodes = retriever.retrieve(query)

        # 결과 처리
        context = ""
        total_tokens = 0
        layer_information = []

        for node in retrieved_nodes:
            # 레이어 필터링 적용
            if not collapse_tree and start_layer is not None:
                node_layer = node.metadata.get('layer')
                if node_layer != start_layer:
                    continue

            chunk = node.text
            tokens = len(self.tokenizer.encode(chunk))
            if total_tokens + tokens <= max_tokens:
                context += chunk + "\n\n"
                total_tokens += tokens
                layer_information.append(
                    {
                        "node_index": node.metadata.get('node_index'),
                        "layer_number": node.metadata.get('layer'),
                        "score": node.score if hasattr(node, 'score') else 0.0,
                    }
                )
            else:
                break

        logger.info(
            f"Hybrid search retrieved context with {total_tokens} tokens from {len(layer_information)} chunks."
        )
        return context.strip(), layer_information

    def get_points_by_document_name(self, document_name: str) -> List[dict]:
        """
        특정 document_name을 가진 모든 포인트를 반환합니다.
        """
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_name",
                    match=models.MatchValue(value=document_name),
                )
            ]
        )

        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=search_filter,
            with_payload=True,
            with_vectors=False,
            limit=10000,  # 충분히 큰 수로 설정
        )

        return [
            {
                "id": point.id,
                "payload": point.payload,
            }
            for point in scroll_result[0]
        ]

    def delete_points_by_document_name(self, document_name: str) -> int:
        """
        특정 document_name을 가진 모든 포인트를 삭제합니다.
        삭제된 포인트 수를 반환합니다.
        """
        points = self.get_points_by_document_name(document_name)
        if not points:
            logger.info(f"No points found for document_name: {document_name}")
            return 0

        point_ids = [point["id"] for point in points]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )

        logger.info(
            f"Deleted {len(point_ids)} points for document_name: {document_name}"
        )
        return len(point_ids)

    def _create_text_index(self):
        """하이브리드 검색을 위한 Full-Text Index 생성"""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.MULTILINGUAL,  # 한국어 지원
                    lowercase=True,
                ),
            )
            logger.info("Full-text index created for 'text' field")
        except Exception as e:
            # 이미 인덱스가 존재하는 경우 무시
            if "already exists" in str(e).lower():
                logger.info("Full-text index already exists for 'text' field")
            else:
                logger.warning(f"Failed to create text index: {e}")
